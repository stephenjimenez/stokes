"""Tools for executing the Stokes flow simulation."""

from enum import Enum, auto
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve

from stokes.config import SimulationConfig
from stokes.fe import shape
from stokes.fe.assembly import DamageAssembler, StokesAssembler
from stokes.fe.bc import apply_bcs_inplace
from stokes.rheology import (
    damage_rate,
    glen_coefficient,
    hayhurst_driving_stress,
)
from stokes.visualization import ParaViewWriter

YEARS_TO_HOURS = 8760.0


class SolverStatus(Enum):
    """Enumerate solver status."""

    CONVERGED = auto()
    MAX_ITERATIONS = auto()


def execute(config: SimulationConfig) -> None:
    """Execute a Stokes flow simulation."""
    print("# Stokes flow")
    try:
        v = version("stokes")
        print(f"\nVersion {v:s}")
    except PackageNotFoundError:
        pass

    # setup
    t = 0.0
    dt = config.solver.time_step / YEARS_TO_HOURS
    b_glen = glen_coefficient(config.rheology.temperature)
    body_force = config.rheology.body_force()
    assembler = StokesAssembler(config.mesh)
    damage_assembler = DamageAssembler(config.mesh)
    writer = ParaViewWriter(output_dir=Path.cwd() / "output", base_name="stokes")

    # initial damage state
    dmg0 = np.zeros((config.mesh.num_cells, 9), dtype=np.float64)

    # initial guess
    uvp0 = np.zeros(sum(config.mesh.num_dofs), dtype=np.float64)

    # begin time stepping
    for step in range(config.solver.num_steps + 1):
        print(f"\n## Step {step:d} (t={t:g} hours)", flush=True)

        # solve the Stokes problem
        uvp, status = solve_stokes(
            config=config,
            assembler=assembler,
            uvp0=uvp0,
            dmg0=dmg0,
            b_glen=b_glen,
            body_force=body_force,
        )
        if config.solver.strict_convergence and status != SolverStatus.CONVERGED:
            msg = "Stokes flow did not converge."
            raise RuntimeError(msg)

        # update the damage variable
        dmg = update_damage(
            config=config,
            uvp=uvp,
            dmg0=dmg0,
            damage_assembler=damage_assembler,
        )

        # generate plot file
        if step % config.plotting.output_interval == 0:
            writer.write_step(config=config, time=t, uvp=uvp, dmg=dmg)

        # exit loop if we're at the last step
        if step == config.solver.num_steps:
            return

        # update the mesh coordinates
        if step == 0:
            config.mesh.update_points(uvp=uvp, dt=dt)
        else:
            config.mesh.update_points(uvp=1.5 * uvp - 0.5 * uvp0, dt=dt)

        # update time and solution
        t += config.solver.time_step
        uvp0 = uvp
        dmg0 = dmg


def solve_stokes(
    config: SimulationConfig,
    assembler: StokesAssembler,
    uvp0: NDArray[np.float64],
    dmg0: NDArray[np.float64],
    b_glen: float,
    body_force: NDArray[np.float64],
) -> tuple[NDArray[np.float64], SolverStatus]:
    """Solve the Stokes system of equations."""
    k = 0
    err_uv = 1.0
    t_assembly = 0.0
    t_solve = 0.0
    n = config.mesh.num_dofs[0]
    while k < config.solver.max_iterations and err_uv > config.solver.tolerance:
        alpha = 0.0 if k % 4 == 0 else 1.0

        # assemble the system of equations
        tic = perf_counter()
        ae, be, fe = assembler.assemble_system(
            uvp=uvp0,
            b_glen=b_glen,
            damage=dmg0,
            body_force=body_force,
            alpha=alpha,
        )
        apply_bcs_inplace(ae=ae, be=be, fe=fe, bc=config.bc)
        a, b = assembler.assemble_csc(ae=ae, be=be, fe=fe)
        toc = perf_counter()
        t_assembly += toc - tic

        # solve the system of equations
        tic = perf_counter()
        uvp = spsolve(a, b)
        toc = perf_counter()
        t_solve += toc - tic

        # compute error
        uv0, p0 = uvp0[:n], uvp0[n:]
        uv, p = uvp[:n], uvp[n:]
        err_uv = float(np.linalg.norm(uv - uv0) / max(1e-16, np.linalg.norm(uv)))
        err_p = float(np.linalg.norm(p - p0) / max(1e-16, np.linalg.norm(p)))

        # update solution
        k += 1
        uvp0 = uvp
        print(
            f"- k={k:02d}, err_uv={err_uv:e}, err_p={err_p:e}, alpha={alpha:g}",
            flush=True,
        )

    print(
        f"\nTiming\n- t_assembly: {t_assembly:g} seconds"
        f"\n- t_solve:    {t_solve:g} seconds"
    )
    if k < config.solver.max_iterations:
        return uvp, SolverStatus.CONVERGED
    return uvp, SolverStatus.MAX_ITERATIONS


def update_damage(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
    dmg0: NDArray[np.float64],
    damage_assembler: DamageAssembler | None = None,
) -> NDArray[np.float64]:
    """Update the damage variable using implicit gradient regularization."""
    if config.damage is None:
        return dmg0

    if damage_assembler is None:
        damage_assembler = DamageAssembler(config.mesh)

    dt = config.solver.time_step * 3600.0

    # 1. Compute local Hayhurst stress chi of shape (num_cells, 9) at the 9 Gauss points
    chi, skk = hayhurst_driving_stress(
        config=config, uvp=uvp, dmg0=dmg0, return_skk=True
    )

    # 2. Assemble and solve the Q4 linear system a, b
    ae, fe = damage_assembler.assemble_system(
        chi=chi, length_scale=config.damage.length_scale
    )
    a, b = damage_assembler.assemble_csc(ae=ae, fe=fe)
    chi_nodes = spsolve(a, b)
    chi_nodes = np.fmax(chi_nodes, 0.0)

    # 3. Interpolate chi_nodes from corner nodes back to the 9 Gauss points
    corner_idx = damage_assembler.corner_vertices
    chi_cell_nodes = chi_nodes[corner_idx]
    n4_gp = np.array(
        [shape.q4_shape_functions(xi, eta) for xi, eta in shape.Q9_GAUSS_POINTS],
        dtype=np.float64,
    )
    chi_bar = chi_cell_nodes @ n4_gp.T
    chi_bar = np.fmax(chi_bar, 0.0)

    # 4. Evaluate the local damage rate and advance
    ddmg = damage_rate(config=config, chi=chi_bar, dmg0=dmg0, skk=skk)
    dmg = dmg0 + ddmg * dt
    dmg[dmg > config.damage.critical_damage] = config.damage.max_damage
    return dmg
