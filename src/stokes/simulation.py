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
    dmg0 = np.zeros(config.mesh.num_cells, dtype=np.float64)

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

        # generate plots
        if step % config.plotting.output_interval == 0:
            writer.write_step(config=config, time=t, uvp=uvp, dmg=dmg)

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
    # compute element areas and centroids
    corners = config.mesh.points[config.mesh.cells[:, :4]]
    x0, y0 = corners[:, 0, 0], corners[:, 0, 1]
    x1, y1 = corners[:, 1, 0], corners[:, 1, 1]
    x2, y2 = corners[:, 2, 0], corners[:, 2, 1]
    x3, y3 = corners[:, 3, 0], corners[:, 3, 1]
    areas = 0.5 * np.abs(
        (x0 * y1 - x1 * y0)
        + (x1 * y2 - x2 * y1)
        + (x2 * y3 - x3 * y2)
        + (x3 * y0 - x0 * y3)
    )
    x = corners.mean(axis=1)

    # compute distance-based weights
    tree = KDTree(x)
    neighbors = tree.query_ball_point(x=x, r=3.0 * config.damage.length_scale)
    inv_l2 = 1.0 / config.damage.length_scale**2
    ddmg = np.empty_like(ddmg_loc)
    for i, nbr in enumerate(neighbors):
        dx = x[i, 0] - x[nbr, 0]
        dy = x[i, 1] - x[nbr, 1]
        u2 = (dx * dx + dy * dy) * inv_l2
        w = np.exp(-config.damage.kappa * u2) * areas[nbr]
        ddmg[i] = np.average(ddmg_loc[nbr], weights=w)

    return ddmg


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
        alpha = 0.0 if k % 3 == 0 else 1.0

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
        err_uv = float(np.linalg.norm(uv - uv0) / np.linalg.norm(uv))
        err_p = float(np.linalg.norm(p - p0) / np.linalg.norm(p))

        # update solution
        k += 1
        uvp0 = uvp
        print(
            f"- k={k:02d}, err_uv={err_uv:e}, err_p={err_p:e}, alpha={alpha:g}",
            flush=True,
        )

    print(
        f"\nTiming\n- t_assembly: {t_assembly:g} seconds"
        "\n- t_solve:    {t_solve:g} seconds"
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
