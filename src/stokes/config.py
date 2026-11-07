"""Configuration parser for TOML simulation input files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, Self, TypedDict

import numpy as np
from numpy.typing import NDArray

from stokes.fe.bc import BoundaryConditions
from stokes.fe.mesh import Mesh

GRAVITY_ACCELERATION = -9.80665

BCType = Literal["free slip"]


class BoundaryConditionFormat(TypedDict):
    """Input format for boundary conditions."""

    left: NotRequired[BCType]
    right: NotRequired[BCType]
    top: NotRequired[BCType]
    bottom: NotRequired[BCType]


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
    output_interval: NotRequired[int]
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
class SolverConfig:
    """Nonlinear solver and time-stepping parameters."""

    max_iterations: int = 50
    num_steps: int = 100
    output_interval: int = 1
    strict_convergence: bool = False
    tolerance: float = 1e-6
    time_step: float = 0.05

    @classmethod
    def from_dict(cls, d: SolverFormat) -> Self:
        """Unpack solver configuration from a dictionary."""
        return cls(
            max_iterations=d.get("max_iterations", cls.max_iterations),
            num_steps=d.get("num_steps", cls.num_steps),
            output_interval=d.get("output_interval", cls.output_interval),
            strict_convergence=d.get("strict_convergence", cls.strict_convergence),
            tolerance=d.get("convergence_tolerance", cls.tolerance),
            time_step=d.get("time_step", cls.time_step),
        )


@dataclass
class SimulationConfig:
    """Complete simulation setup parsed from a TOML file."""

    mesh: Mesh
    bc: BoundaryConditions
    rheology: RheologyConfig
    solver: SolverConfig

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load simulation configuration from a TOML file."""
        with Path(path).open(mode="rb") as f:
            raw = tomllib.load(f)

        mesh_cfg: dict[str, Any] = raw.get("mesh", {})
        mesh = Mesh.from_grid(  # [cite: 1]
            shape=tuple(mesh_cfg.get("shape", (5, 5))),  # [cite: 1]
            size=tuple(mesh_cfg.get("size", (1.0, 1.0))),  # [cite: 1]
        )

        bc_cfg: dict[str, str] = raw.get("boundary_conditions", {})
        bc = BoundaryConditions.free_slip_bc(
            mesh=mesh,
            edges=[k.lower() for k, v in bc_cfg.items() if v.lower() == "free slip"],
        )

        rheology = RheologyConfig.from_dict(raw.get("rheology"))
        solver = SolverConfig.from_dict(raw.get("solver", {}))

        return cls(mesh=mesh, rheology=rheology, bc=bc, solver=solver)
