"""Configuration parser for TOML simulation input files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, Self, TypedDict

import numpy as np
from numpy.typing import NDArray

from stokes.fe.bc import BoundaryConditions
from stokes.fe.mesh import Mesh
from stokes.meshing import ConformingQuadMesh

GRAVITY_ACCELERATION = -9.80665

BCType = Literal["free slip"]


class BoundaryConditionFormat(TypedDict):
    """Input format for boundary conditions."""

    left: NotRequired[BCType]
    right: NotRequired[BCType]
    top: NotRequired[BCType]
    bottom: NotRequired[BCType]


class MeshFormat(TypedDict):
    """Input format for the finite element mesh."""

    shape: list[int]
    size: list[float]
    refinement: NotRequired[list[RefinementFormat]]


class PlottingFormat(TypedDict):
    """Input format for plotting configuration."""

    output_interval: NotRequired[int]
    plot_cauchy_stress: NotRequired[bool]
    plot_deviatoric_stress: NotRequired[bool]
    plot_strain_rate: NotRequired[bool]


class RefinementFormat(TypedDict):
    """Input format for mesh refinements."""

    lower_corner: list[float]
    upper_corner: list[float]


class RheologyFormat(TypedDict):
    """Input format for rheology configuration."""

    apply_gravity: NotRequired[float]
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
class RheologyConfig:
    """Rheology parameters."""

    gravity: bool = False
    rho_ice: float = 917.0
    temperature: float = 263.15

    @property
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
class PlottingConfig:
    """Plotting parameters."""

    output_interval: int = 1
    plot_cauchy_stress: bool = True
    plot_deviatoric_stress: bool = True
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
            plot_strain_rate=d.get("plot_strain_rate", cls.plot_strain_rate),
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

    mesh: Mesh
    bc: BoundaryConditions
    plotting: PlottingConfig
    rheology: RheologyConfig
    solver: SolverConfig

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load simulation configuration from a TOML file."""
        with Path(path).open(mode="rb") as f:
            raw = tomllib.load(f)

        mesh = initialize_mesh(raw.get("mesh"))

        bc_cfg: dict[str, str] = raw.get("boundary_conditions", {})
        bc = BoundaryConditions.free_slip_bc(
            mesh=mesh,
            edges=[k.lower() for k, v in bc_cfg.items() if v.lower() == "free slip"],
        )

        plotting = PlottingConfig.from_dict(raw.get("plotting", {}))
        rheology = RheologyConfig.from_dict(raw.get("rheology", {}))
        solver = SolverConfig.from_dict(raw.get("solver", {}))

        return cls(
            mesh=mesh,
            plotting=plotting,
            rheology=rheology,
            bc=bc,
            solver=solver,
        )


def initialize_mesh(d: MeshFormat) -> Mesh:
    """Initialize the finite element mesh."""
    quad_mesh = ConformingQuadMesh.from_grid(
        shape=tuple(d["shape"]),
        size=tuple(d["size"]),
    )
    refinements = d.get("refinement", [])
    if any(refinements):
        patch = set()
        for ref in refinements:
            lower = tuple(ref["lower_corner"])
            upper = tuple(ref["upper_corner"])
            patch |= quad_mesh.refine_box(lower=lower, upper=upper)
        quad_mesh.refine_patch(patch)
    q4_cells, vertices = quad_mesh.to_solver_arrays()
    return Mesh.from_q4_mesh(q4_cells=q4_cells, vertices=vertices)
