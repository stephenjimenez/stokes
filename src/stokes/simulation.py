"""Tools for executing the Stokes flow simulation."""

from enum import Enum, auto
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree

from stokes.config import SimulationConfig
from stokes.fe.assembly import StokesAssembler
from stokes.fe.bc import apply_bcs_inplace
from stokes.rheology import glen_coefficient, local_damage_rate
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
        dmg = update_damage(config=config, uvp=uvp, dmg0=dmg0)

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


def regularize_damage_rate(
    config: SimulationConfig,
    ddmg_loc: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Regularize the damage rate using a k-d tree."""
    # compute Gauss point volumes and coordinates
    vol = config.mesh.gauss_point_volumes().ravel()
    x = config.mesh.gauss_point_coordinates()

    # compute distance-based weights
    tree = KDTree(x)
    neighbors = tree.query_ball_point(x=x, r=3.0 * config.damage.length_scale)
    inv_l2 = 1.0 / config.damage.length_scale**2
    ddmg = np.empty(9 * config.mesh.num_cells, dtype=np.float64)
    for i, nbr in enumerate(neighbors):
        dx = x[i, 0] - x[nbr, 0]
        dy = x[i, 1] - x[nbr, 1]
        u2 = (dx * dx + dy * dy) * inv_l2
        w = np.exp(-config.damage.kappa * u2) * vol[nbr]
        ddmg[i] = np.average(ddmg_loc.ravel()[nbr], weights=w)

    return ddmg.reshape(-1, 9)


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
) -> NDArray[np.float64]:
    """Update the damage variable."""
    if config.damage is None:
        return dmg0
    dt = config.solver.time_step * 3600.0
    ddmg_loc = local_damage_rate(config=config, uvp=uvp, dmg0=dmg0)
    ddmg = regularize_damage_rate(config=config, ddmg_loc=ddmg_loc)
    dmg = dmg0 + ddmg * dt
    dmg[dmg > config.damage.critical_damage] = config.damage.max_damage
    return dmg
