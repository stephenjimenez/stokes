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


class BoundaryConditionsFormat(TypedDict):
    """Input format for boundary conditions."""

    left: NotRequired[BCType]
    right: NotRequired[BCType]
    top: NotRequired[BCType]
    bottom: NotRequired[BCType]


class DamageFormat(TypedDict):
    """Input format for damage parameters."""

    b_hat: NotRequired[float]
    r: NotRequired[float]
    k1: NotRequired[float]
    k2: NotRequired[float]
    alpha: NotRequired[float]
    beta: NotRequired[float]
    length_scale: NotRequired[float]
    kappa: NotRequired[float]
    max_damage: NotRequired[float]
    critical_damage: NotRequired[float]


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
    plot_hayhurst_stress: NotRequired[bool]
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
class DamageConfig:
    """Container for damage parameters."""

    b_hat: float = 5.23e-7  # MPa^-r s^-1
    r: float = 0.43
    k1: float = -2.63
    k2: float = 7.24  # MPa^-1
    alpha: float = 0.21
    beta: float = 0.63
    length_scale: float = 10.0  # m
    kappa: float = 2.0
    max_damage: float = 0.99
    critical_damage: float = 0.63

    @classmethod
    def from_dict(cls, d: DamageFormat) -> Self:
        """Unpack damage configuration from a dictionary."""
        return cls(
            b_hat=d.get("b_hat", cls.b_hat),
            r=d.get("r", cls.r),
            k1=d.get("k1", cls.k1),
            k2=d.get("k2", cls.k2),
            alpha=d.get("alpha", cls.alpha),
            beta=d.get("beta", cls.beta),
            length_scale=d.get("length_scale", cls.length_scale),
            kappa=d.get("kappa", cls.kappa),
            max_damage=d.get("max_damage", cls.max_damage),
            critical_damage=d.get("critical_damage", cls.critical_damage),
        )


@dataclass
class PlottingConfig:
    """Plotting parameters."""

    output_interval: int = 1
    plot_cauchy_stress: bool = True
    plot_deviatoric_stress: bool = True
    plot_hayhurst_stress: bool = True
    plot_strain_rate: bool = True

    @classmethod
    def from_dict(cls, d: PlottingFormat) -> Self:
        """Unpack plotting configuration from a dictionary."""
        return PlottingConfig(
            output_interval=d.get("output_interval", cls.output_interval),
            plot_cauchy_stress=d.get("plot_cauchy_stress", cls.plot_cauchy_stress),
            plot_hayhurst_stress=d.get(
                "plot_hayhurst_stress",
                cls.plot_hayhurst_stress,
            ),
            plot_deviatoric_stress=d.get(
                "plot_deviatoric_stress",
                cls.plot_deviatoric_stress,
            ),
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
    damage: DamageConfig | None
    mesh: Mesh
    plotting: PlottingConfig
    rheology: RheologyConfig
    solver: SolverConfig

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load simulation configuration from a TOML file."""
        with Path(path).open(mode="rb") as f:
            raw = tomllib.load(f)

        # unpack sub dictionaries
        bc_raw: BoundaryConditionsFormat = raw.get("boundary_conditions", {})
        damage_raw: DamageFormat = raw.get("damage")
        mesh_raw: MeshFormat = raw.get("mesh")
        plotting_raw: PlottingFormat = raw.get("plotting", {})
        rheology_raw: RheologyFormat = raw.get("rheology", {})
        solver_raw: SolverFormat = raw.get("solver", {})

        # initialize the mesh
        mesh = initialize_mesh(mesh_raw)

        # boundary conditions
        bc = BoundaryConditions.free_slip_bc(
            mesh=mesh,
            edges=[
                k.lower()
                for k, v in bc_raw.items()
                if isinstance(v, str) and v.lower() == "free slip"
            ],
        )

        return cls(
            bc=bc,
            damage=None if damage_raw is None else DamageConfig.from_dict(damage_raw),
            mesh=mesh,
            plotting=PlottingConfig.from_dict(plotting_raw),
            rheology=RheologyConfig.from_dict(rheology_raw),
            solver=SolverConfig.from_dict(solver_raw),
        )


def initialize_mesh(d: MeshFormat) -> Mesh:
    """Initialize the finite element mesh."""
    # create the base mesh
    quad_mesh = ConformingQuadMesh.from_grid(
        shape=tuple(d["shape"]),
        size=tuple(d["size"]),
    )

    # remove notches
    for notch in d.get("notch", []):
        lower = tuple(notch["lower_corner"])
        upper = tuple(notch["upper_corner"])
        patch = quad_mesh.elements_in_box(lower=lower, upper=upper)
        quad_mesh.remove_elements(patch)

    # apply mesh refinements
    refinements = d.get("refinement", [])
    for ref in refinements:
        lower = tuple(ref["lower_corner"])
        upper = tuple(ref["upper_corner"])
        patch = quad_mesh.elements_in_box(lower=lower, upper=upper)
        quad_mesh.refine_patch(patch)
    q4_cells, vertices = quad_mesh.to_solver_arrays()
    return Mesh.from_q4_mesh(q4_cells=q4_cells, vertices=vertices)
