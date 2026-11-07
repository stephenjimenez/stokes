"""Tools for executing the Stokes flow simulation."""

from enum import Enum, auto
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve

from stokes.config import SimulationConfig
from stokes.fe.assembly import StokesAssembler
from stokes.fe.bc import apply_bcs_inplace
from stokes.rheology import glen_coefficient
from stokes.visualization import ParaViewWriter

YEARS_TO_HOURS = 8760.0


class SolverStatus(Enum):
    """Enumerate solver status."""

    CONVERGED = auto()
    MAX_ITERATIONS = auto()


def execute(config: SimulationConfig) -> None:
    """Execute a Stokes flow simulation."""

    # setup
    t = 0.0
    dt = config.solver.time_step / YEARS_TO_HOURS
    b_glen = glen_coefficient(config.rheology.temperature)
    body_force = config.rheology.body_force
    assembler = StokesAssembler(config.mesh)
    writer = ParaViewWriter(output_dir=Path.cwd() / "output", base_name="stokes")

    # initial guess
    uvp0 = np.zeros(sum(config.mesh.num_dofs), dtype=np.float64)

    # begin time stepping
    print("# Stokes flow")
    for step in range(config.solver.num_steps + 1):
        print(f"\n## Step {step:d} (t={t:g} hours)", flush=True)

        # solve the Stokes problem
        uvp, status = solve_stokes(
            config=config,
            assembler=assembler,
            uvp0=uvp0,
            b_glen=b_glen,
            body_force=body_force,
        )
        if config.solver.strict_convergence and status != SolverStatus.CONVERGED:
            msg = "Stokes flow did not converge."
            raise RuntimeError(msg)

        # generate plots
        if step % config.solver.output_interval == 0:
            writer.write_step(config=config, time=t, uvp=uvp)

        # update the mesh coordinates
        if step == 0:
            config.mesh.update_points(uvp=uvp, dt=dt)
        else:
            config.mesh.update_points(uvp=1.5 * uvp - 0.5 * uvp0, dt=dt)

        # update time and solution
        t += config.solver.time_step
        uvp0 = uvp


def solve_stokes(
    config: SimulationConfig,
    assembler: StokesAssembler,
    uvp0: NDArray[np.float64],
    b_glen: float,
    body_force: NDArray[np.float64],
) -> tuple[NDArray[np.float64], SolverStatus]:
    """Solve the Stokes system of equations."""
    k = 0
    err_uv = 1.0
    t_assembly = 0.0
    t_solve = 0.0

    n = config.mesh.num_dofs[0]
    warm_start = bool(np.any(uvp0[:n] != 0.0))

    while k < config.solver.max_iterations and err_uv > config.solver.tolerance:
        alpha = 1.0 if (warm_start or k > 2) else 0.0

        # assemble the system of equations
        tic = perf_counter()
        ae, be, fe = assembler.assemble_system(
            u_global=uvp0,
            b_glen=b_glen,
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
        err_uv = float(np.linalg.norm(uv - uv0) / np.linalg.norm(uv))
        err_p = float(np.linalg.norm(p - p0) / np.linalg.norm(p))

        # update solution
        k += 1
        uvp0 = uvp
        print(f"- k={k:02d}, err_uv={err_uv:e}, err_p={err_p:e}", flush=True)

    print(f"\nt_assembly: {t_assembly:g} s\nt_solve:    {t_solve:g} s")
    if k < config.solver.max_iterations:
        return uvp, SolverStatus.CONVERGED
    return uvp, SolverStatus.MAX_ITERATIONS
