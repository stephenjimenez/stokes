"""Tools for exporting finite element solutions to ParaView VTU/PVD files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from stokes.rheology import interpolate_stress

if TYPE_CHECKING:
    from stokes.config import SimulationConfig

VTK_QUADRATIC_QUAD = 23


class ParaViewWriter:
    """Manages time-series VTU and PVD exports for Q8-Q4 Stokes flow."""

    base_name: str
    output_dir: Path
    pvd_path: Path
    timesteps: list[tuple[float, str]]

    def __init__(self, output_dir: str | Path, base_name: str = "solution") -> None:
        """Initialize the writer and directory structure."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = base_name
        self.pvd_path = self.output_dir / f"{base_name}.pvd"
        self.timesteps = []

    def write_step(
        self,
        config: SimulationConfig,
        time: float,
        uvp: NDArray[np.float64],
    ) -> Path:
        """Write a single time step to a .vtu file and update the master .pvd file."""

        # extract 3D points
        points_3d = np.zeros((config.mesh.num_points, 3), dtype=np.float64)
        points_3d[:, :2] = config.mesh.points

        # extract velocity field
        n = 2 * config.mesh.num_points
        velocity_3d = np.zeros((config.mesh.num_points, 3), dtype=np.float64)
        velocity_3d[:, 0] = uvp[0:n:2]
        velocity_3d[:, 1] = uvp[1:n:2]

        # extract pressure (and compute at midpoints)
        pressure = np.zeros(config.mesh.num_points, dtype=np.float64)
        pressure[config.mesh.vertices] = uvp[n:]
        for i in range(4):
            a = config.mesh.cells[:, i]
            b = config.mesh.cells[:, (i + 1) % 4]
            edge_nodes = config.mesh.cells[:, 4 + i]
            pressure[edge_nodes] = 0.5 * (pressure[a] + pressure[b])

        # construct cell connectivity
        cells_with_count = np.empty((config.mesh.num_cells, 9), dtype=np.int64)
        cells_with_count[:, 0] = 8
        cells_with_count[:, 1:] = config.mesh.cells
        cell_types = np.full(config.mesh.num_cells, VTK_QUADRATIC_QUAD, dtype=np.uint8)

        # create PyVista UnstructuredGrid
        grid = pv.UnstructuredGrid(cells_with_count.ravel(), cell_types, points_3d)

        # save point data
        grid.point_data["Velocity"] = velocity_3d
        grid.point_data["Pressure"] = pressure

        # save cell data
        epsilon, tau, sigma = interpolate_stress(
            config=config,
            uvp=uvp,
            xi=0.0,
            eta=0.0,
        )
        if config.plotting.plot_cauchy_stress:
            grid.cell_data["CauchyStress"] = sigma
            vtk_arr = grid.cell_data["CauchyStress"].VTKObject
            vtk_arr.SetComponentName(0, "XX")
            vtk_arr.SetComponentName(1, "YY")
            vtk_arr.SetComponentName(2, "XY")
        if config.plotting.plot_deviatoric_stress:
            grid.cell_data["DeviatoricStress"] = tau
            vtk_arr = grid.cell_data["DeviatoricStress"].VTKObject
            vtk_arr.SetComponentName(0, "XX")
            vtk_arr.SetComponentName(1, "YY")
            vtk_arr.SetComponentName(2, "XY")
        if config.plotting.plot_strain_rate:
            grid.cell_data["StrainRate"] = epsilon
            vtk_arr = grid.cell_data["StrainRate"].VTKObject
            vtk_arr.SetComponentName(0, "XX")
            vtk_arr.SetComponentName(1, "YY")
            vtk_arr.SetComponentName(2, "XY")

        # save the VTU file
        vtu_filename = f"{self.base_name}_{len(self.timesteps):05d}.vtu"
        vtu_path = self.output_dir / vtu_filename
        grid.save(str(vtu_path))

        # record and write master PVD file
        self.timesteps.append((time, vtu_filename))
        self._write_pvd()

        return vtu_path

    def _write_pvd(self) -> None:
        """Write or update the XML .pvd index collection."""
        root = ET.Element(
            "VTKFile",
            type="Collection",
            version="0.1",
            byte_order="LittleEndian",
        )
        collection = ET.SubElement(root, "Collection")
        for time, filename in self.timesteps:
            ET.SubElement(
                collection,
                "DataSet",
                timestep=f"{time:.6e}",
                group="",
                part="0",
                file=filename,
            )
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(self.pvd_path, encoding="utf-8", xml_declaration=True)
