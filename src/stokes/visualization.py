"""Tools for exporting finite element solutions to ParaView VTU/PVD files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from stokes.fe.shape import Q9_GAUSS_POINTS, q8_shape_functions
from stokes.rheology import (
    interpolate_cauchy_stress,
    interpolate_strain_rate,
    tensile_strain_energy,
)

if TYPE_CHECKING:
    from stokes.config import SimulationConfig

VTK_QUADRATIC_QUAD = 23

YEAR_TO_SECOND = 31536000.0


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

    @staticmethod
    def _save_weighted_tensor(
        grid: pv.UnstructuredGrid,
        tensor: NDArray[np.float64],
        w: NDArray[np.float64],
        name: str,
    ) -> None:
        a = w / w.sum(axis=1, keepdims=True)
        grid.cell_data[name] = np.einsum("ijk,ij->ik", tensor, a)
        vtk_arr = grid.cell_data[name].VTKObject
        vtk_arr.SetComponentName(0, "XX")
        vtk_arr.SetComponentName(1, "YY")
        vtk_arr.SetComponentName(2, "ZZ")
        vtk_arr.SetComponentName(3, "XY")

    def write_step(
        self,
        config: SimulationConfig,
        time: float,
        uvp: NDArray[np.float64],
        phi_nodes: NDArray[np.float64] | None = None,
    ) -> Path:
        """Write a single time step to a .vtu file and update the master .pvd file."""
        print("\nPlotting")

        # extract 3D points
        points_3d = np.zeros((config.mesh.num_points, 3), dtype=np.float64)
        points_3d[:, :2] = config.mesh.points

        # extract velocity field
        n = 2 * config.mesh.num_points
        velocity = np.zeros((config.mesh.num_points, 3), dtype=np.float64)
        velocity[:, 0] = uvp[0:n:2] / YEAR_TO_SECOND
        velocity[:, 1] = uvp[1:n:2] / YEAR_TO_SECOND

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
        grid.point_data["Velocity"] = velocity
        grid.point_data["Pressure"] = pressure

        # evaluate phase field and effective damage
        vol = config.mesh.gauss_point_volumes()
        if config.phase_field is not None and phi_nodes is not None:
            n8_gp = np.array(
                [q8_shape_functions(xi, eta) for xi, eta in Q9_GAUSS_POINTS],
                dtype=np.float64,
            )
            phi_gp = phi_nodes[config.mesh.cells] @ n8_gp.T
            eta_res = config.phase_field.residual_stiffness
            dmg_eff = 1.0 - ((1.0 - phi_gp) ** 2 + eta_res)

            if config.plotting.plot_phase_field:
                grid.point_data["PhaseField"] = phi_nodes

            if config.plotting.plot_effective_damage:
                dmg_eff_nodes = 1.0 - ((1.0 - phi_nodes) ** 2 + eta_res)
                grid.point_data["EffectiveDamage"] = dmg_eff_nodes
        else:
            dmg_eff = np.zeros((config.mesh.num_cells, 9), dtype=np.float64)

        # evaluate Cauchy stress at Gauss points
        sigma = np.empty((config.mesh.num_cells, 9, 4), dtype=np.float64)
        for gp, (xi, eta) in enumerate(Q9_GAUSS_POINTS):
            sigma[:, gp, :] = interpolate_cauchy_stress(
                config=config,
                uvp=uvp,
                dmg=dmg_eff[:, gp],
                xi=xi,
                eta=eta,
            )

        # save cell data
        if config.plotting.plot_cauchy_stress:
            self._save_weighted_tensor(
                grid=grid,
                tensor=sigma,
                w=vol,
                name="CauchyStress",
            )
        if config.plotting.plot_deviatoric_stress:
            tau = np.empty_like(sigma)
            p = -(sigma[..., 0] + sigma[..., 1] + sigma[..., 2]) / 3.0
            tau[..., 0] = sigma[..., 0] + p
            tau[..., 1] = sigma[..., 1] + p
            tau[..., 2] = 0.0
            tau[..., 3] = sigma[..., 3]
            self._save_weighted_tensor(
                grid=grid,
                tensor=tau,
                w=vol,
                name="DeviatoricStress",
            )
        if config.plotting.plot_max_principal_stress:
            half_diff = 0.5 * (sigma[..., 0] - sigma[..., 1])
            sig1 = 0.5 * (sigma[..., 0] + sigma[..., 1]) + np.sqrt(
                half_diff * half_diff + sigma[..., 3] ** 2
            )
            sig_max = np.maximum(sig1, sigma[..., 2])
            grid.cell_data["MaxPrincipalStress"] = np.average(
                sig_max, weights=vol, axis=1
            )
        if (
            config.plotting.plot_strain_energy_density
            and config.phase_field is not None
            and phi_nodes is not None
        ):
            psi_plus = tensile_strain_energy(config=config, uvp=uvp, phi_gp=phi_gp)
            grid.cell_data["StrainEnergyDensity"] = np.average(
                psi_plus, weights=vol, axis=1
            )
        if config.plotting.plot_strain_rate:
            d = np.empty((config.mesh.num_cells, 9, 4), dtype=np.float64)
            for gp, (xi, eta) in enumerate(Q9_GAUSS_POINTS):
                d[:, gp, :] = interpolate_strain_rate(
                    config=config,
                    uvp=uvp,
                    xi=xi,
                    eta=eta,
                )
            self._save_weighted_tensor(grid=grid, tensor=d, w=vol, name="StrainRate")

        # save the VTU file
        vtu_filename = f"{self.base_name}_{len(self.timesteps):05d}.vtu"
        vtu_path = self.output_dir / vtu_filename
        grid.save(str(vtu_path))
        try:
            display_path = vtu_path.relative_to(Path.cwd())
        except ValueError:
            display_path = vtu_path
        print(f"- Saved: {display_path}")

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
