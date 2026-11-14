"""Configuration parser for TOML simulation input files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NotRequired, Self, TypedDict

import numpy as np
from numpy.typing import NDArray

from stokes.fe.bc import BoundaryConditions
from stokes.fe.mesh import Mesh
from stokes.meshing import ConformingQuadMesh

GRAVITY_ACCELERATION = -9.80665

BCType = Literal["free slip"]


class BoundaryConditionsFormat(TypedDict):
    """Input format for boundary conditions."""

    left: NotRequired[BCType]
    right: NotRequired[BCType]
    top: NotRequired[BCType]
    bottom: NotRequired[BCType]


class PhaseFieldFormat(TypedDict):
    """Input format for phase-field fracture parameters."""

    gc: NotRequired[float]
    length_scale: NotRequired[float]
    youngs_modulus: NotRequired[float]
    poissons_ratio: NotRequired[float]
    critical_stress: NotRequired[float]
    residual_stiffness: NotRequired[float]
    erosion_threshold: NotRequired[float]


class MeshBoxFormat(TypedDict):
    """Input format for mesh refinements and notches."""

    lower_corner: list[float]
    upper_corner: list[float]


class MeshFormat(TypedDict):
    """Input format for the finite element mesh."""

    shape: list[int]
    size: list[float]
    refinement: NotRequired[list[MeshBoxFormat]]
    notch: NotRequired[list[MeshBoxFormat]]


class PlottingFormat(TypedDict):
    """Input format for plotting configuration."""

    output_interval: NotRequired[int]
    plot_cauchy_stress: NotRequired[bool]
    plot_deviatoric_stress: NotRequired[bool]
    plot_effective_damage: NotRequired[bool]
    plot_phase_field: NotRequired[bool]
    plot_strain_rate: NotRequired[bool]


class RheologyFormat(TypedDict):
    """Input format for rheology configuration."""

    apply_gravity: NotRequired[bool]
    ice_density: NotRequired[float]
    ice_temperature: NotRequired[float]


class SolverFormat(TypedDict):
    """Input format for solver configuration."""

    convergence_tolerance: NotRequired[float]
    max_iterations: NotRequired[int]
    num_steps: NotRequired[int]
    strict_convergence: NotRequired[bool]
    time_step: NotRequired[float]


@dataclass
class PhaseFieldConfig:
    """Container for phase-field fracture parameters."""

    gc: float = 50.0  # J/m^2
    length_scale: float = 10.0  # m
    youngs_modulus: float = 9.0e9  # Pa
    poissons_ratio: float = 0.33
    critical_stress: float = 0.0  # Pa
    residual_stiffness: float = 1e-4
    erosion_threshold: float | None = None

    @classmethod
    def from_dict(cls, d: PhaseFieldFormat) -> Self:
        """Unpack phase-field configuration from a dictionary."""
        return cls(
            gc=d.get("gc", cls.gc),
            length_scale=d.get("length_scale", cls.length_scale),
            youngs_modulus=d.get("youngs_modulus", cls.youngs_modulus),
            poissons_ratio=d.get("poissons_ratio", cls.poissons_ratio),
            critical_stress=d.get("critical_stress", cls.critical_stress),
            residual_stiffness=d.get("residual_stiffness", cls.residual_stiffness),
            erosion_threshold=d.get("erosion_threshold", cls.erosion_threshold),
        )


@dataclass
class PlottingConfig:
    """Plotting parameters."""

    output_interval: int = 1
    plot_cauchy_stress: bool = True
    plot_deviatoric_stress: bool = True
    plot_effective_damage: bool = True
    plot_phase_field: bool = True
    plot_strain_rate: bool = True

    @classmethod
    def from_dict(cls, d: PlottingFormat) -> Self:
        """Unpack plotting configuration from a dictionary."""
        return PlottingConfig(
            output_interval=d.get("output_interval", cls.output_interval),
            plot_cauchy_stress=d.get("plot_cauchy_stress", cls.plot_cauchy_stress),
            plot_deviatoric_stress=d.get(
                "plot_deviatoric_stress",
                cls.plot_deviatoric_stress,
            ),
            plot_effective_damage=d.get(
                "plot_effective_damage",
                cls.plot_effective_damage,
            ),
            plot_phase_field=d.get("plot_phase_field", cls.plot_phase_field),
            plot_strain_rate=d.get("plot_strain_rate", cls.plot_strain_rate),
        )


@dataclass
class RheologyConfig:
    """Container for rheology parameters."""

    gravity: bool = False
    rho_ice: float = 917.0
    temperature: float = 263.15

    def body_force(self) -> NDArray[np.float64]:
        """Evaluate the body force from gravity."""
        if self.gravity:
            return np.array(
                [0.0, GRAVITY_ACCELERATION * self.rho_ice],
                dtype=np.float64,
            )
        return np.zeros(2, dtype=np.float64)

    @classmethod
    def from_dict(cls, d: RheologyFormat) -> Self:
        """Unpack rheology configuration from a dictionary."""
        return cls(
            gravity=d.get("apply_gravity", cls.gravity),
            rho_ice=d.get("ice_density", cls.rho_ice),
            temperature=d.get("ice_temperature", cls.temperature),
        )


@dataclass
class SolverConfig:
    """Nonlinear solver and time-stepping parameters."""

    max_iterations: int = 50
    num_steps: int = 100
    strict_convergence: bool = False
    tolerance: float = 1e-6
    time_step: float = 0.05

    @classmethod
    def from_dict(cls, d: SolverFormat) -> Self:
        """Unpack solver configuration from a dictionary."""
        return cls(
            max_iterations=d.get("max_iterations", cls.max_iterations),
            num_steps=d.get("num_steps", cls.num_steps),
            strict_convergence=d.get("strict_convergence", cls.strict_convergence),
            tolerance=d.get("convergence_tolerance", cls.tolerance),
            time_step=d.get("time_step", cls.time_step),
        )


@dataclass
class SimulationConfig:
    """Complete simulation setup parsed from a TOML file."""

    bc: BoundaryConditions
    mesh: Mesh
    phase_field: PhaseFieldConfig | None
    plotting: PlottingConfig
    rheology: RheologyConfig
    solver: SolverConfig
    bc_edges: list[str] = field(default_factory=list)

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load simulation configuration from a TOML file."""
        with Path(path).open(mode="rb") as f:
            raw = tomllib.load(f)

        bc_raw: BoundaryConditionsFormat = raw.get("boundary_conditions", {})
        pf_raw: PhaseFieldFormat | None = raw.get("phase_field")
        mesh_raw: MeshFormat = raw.get("mesh")
        plotting_raw: PlottingFormat = raw.get("plotting", {})
        rheology_raw: RheologyFormat = raw.get("rheology", {})
        solver_raw: SolverFormat = raw.get("solver", {})

        mesh = initialize_mesh(mesh_raw)

        bc_edges = [
            k.lower()
            for k, v in bc_raw.items()
            if isinstance(v, str) and v.lower() == "free slip"
        ]
        bc = BoundaryConditions.free_slip_bc(
            mesh=mesh,
            edges=bc_edges,
        )

        pf = None if pf_raw is None else PhaseFieldConfig.from_dict(pf_raw)
        return cls(
            bc=bc,
            mesh=mesh,
            phase_field=pf,
            plotting=PlottingConfig.from_dict(plotting_raw),
            rheology=RheologyConfig.from_dict(rheology_raw),
            solver=SolverConfig.from_dict(solver_raw),
            bc_edges=bc_edges,
        )


def initialize_mesh(d: MeshFormat) -> Mesh:
    """Initialize the finite element mesh."""
    quad_mesh = ConformingQuadMesh.from_grid(
        shape=tuple(d["shape"]),
        size=tuple(d["size"]),
    )

    for notch in d.get("notch", []):
        lower = tuple(notch["lower_corner"])
        upper = tuple(notch["upper_corner"])
        patch = quad_mesh.elements_in_box(lower=lower, upper=upper)
        quad_mesh.remove_elements(patch)

    refinements = d.get("refinement", [])
    for ref in refinements:
        lower = tuple(ref["lower_corner"])
        upper = tuple(ref["upper_corner"])
        patch = quad_mesh.elements_in_box(lower=lower, upper=upper)
        quad_mesh.refine_patch(patch)
    q4_cells, vertices = quad_mesh.to_solver_arrays()
    return Mesh.from_q4_mesh(q4_cells=q4_cells, vertices=vertices)
