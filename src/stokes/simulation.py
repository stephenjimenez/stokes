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
from stokes.fe.assembly import PhaseFieldAssembler, StokesAssembler
from stokes.fe.bc import BoundaryConditions, apply_bcs_inplace
from stokes.fe.mesh import Mesh
from stokes.rheology import (
    glen_coefficient,
    tensile_strain_energy,
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
    stokes_assembler = StokesAssembler(config.mesh)
    pf_assembler = (
        PhaseFieldAssembler(config.mesh) if config.phase_field is not None else None
    )
    writer = ParaViewWriter(output_dir=Path.cwd() / "output", base_name="stokes")

    # initial phase-field and history states
    phi_nodes = np.zeros(config.mesh.num_points, dtype=np.float64)
    history = np.zeros((config.mesh.num_cells, 9), dtype=np.float64)

    # initial guess
    uvp0 = np.zeros(sum(config.mesh.num_dofs), dtype=np.float64)

    n8_gp = np.array(
        [shape.q8_shape_functions(xi, eta) for xi, eta in shape.Q9_GAUSS_POINTS],
        dtype=np.float64,
    )

    # begin time stepping
    for step in range(config.solver.num_steps + 1):
        print(f"\n## Step {step:d} (t={t:g} hours)", flush=True)

        if config.phase_field is not None:
            phi_cell = phi_nodes[config.mesh.cells]
            phi_gp = phi_cell @ n8_gp.T
            eta_res = config.phase_field.residual_stiffness
            g_phi = (1.0 - phi_gp) ** 2 + eta_res
            dmg_eff = 1.0 - g_phi
        else:
            phi_gp = np.zeros((config.mesh.num_cells, 9), dtype=np.float64)
            dmg_eff = np.zeros((config.mesh.num_cells, 9), dtype=np.float64)

        # solve the Stokes problem
        uvp, status = solve_stokes(
            config=config,
            stokes_assembler=stokes_assembler,
            uvp0=uvp0,
            dmg0=dmg_eff,
            b_glen=b_glen,
            body_force=body_force,
        )
        if config.solver.strict_convergence and status != SolverStatus.CONVERGED:
            msg = "Stokes flow did not converge."
            raise RuntimeError(msg)

        # update the phase-field variable
        if config.phase_field is not None and pf_assembler is not None:
            psi_plus = tensile_strain_energy(config=config, uvp=uvp, phi_gp=phi_gp)
            np.maximum(history, psi_plus, out=history)
            ae, fe = pf_assembler.assemble_system(
                history=history,
                gc=config.phase_field.gc,
                length_scale=config.phase_field.length_scale,
            )
            a_mat, f_vec = pf_assembler.assemble_csc(ae=ae, fe=fe)
            phi_nodes = spsolve(a_mat, f_vec)
            phi_nodes = np.clip(phi_nodes, 0.0, 1.0)

            # check erosion threshold
            if config.phase_field.erosion_threshold is not None:
                phi_cell = phi_nodes[config.mesh.cells]
                phi_gp = phi_cell @ n8_gp.T
                bar_phi_e = np.mean(phi_gp, axis=1)
                eroded = bar_phi_e >= config.phase_field.erosion_threshold
                if np.any(eroded) and not np.all(eroded):
                    keep = ~eroded
                    surviving_cells = config.mesh.cells[keep]
                    used_nodes, new_cells_flat = np.unique(
                        surviving_cells, return_inverse=True
                    )
                    new_cells = new_cells_flat.reshape(surviving_cells.shape).astype(
                        np.int32
                    )
                    new_points = config.mesh.points[used_nodes]
                    config.mesh = Mesh(cells=new_cells, points=new_points)
                    config.bc = BoundaryConditions.free_slip_bc(
                        mesh=config.mesh, edges=config.bc_edges
                    )
                    stokes_assembler = StokesAssembler(config.mesh)
                    pf_assembler = PhaseFieldAssembler(config.mesh)
                    history = history[keep]
                    phi_nodes = phi_nodes[used_nodes]
                    uvp0 = np.zeros(sum(config.mesh.num_dofs), dtype=np.float64)

        # generate plot file
        if step % config.plotting.output_interval == 0:
            writer.write_step(
                config=config,
                time=t,
                uvp=uvp,
                phi_nodes=phi_nodes if config.phase_field is not None else None,
            )

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


def solve_stokes(
    config: SimulationConfig,
    stokes_assembler: StokesAssembler,
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
        ae, be, fe = stokes_assembler.assemble_system(
            uvp=uvp0,
            b_glen=b_glen,
            damage=dmg0,
            body_force=body_force,
            alpha=alpha,
        )
        apply_bcs_inplace(ae=ae, be=be, fe=fe, bc=config.bc)
        a, b = stokes_assembler.assemble_csc(ae=ae, be=be, fe=fe)
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


def update_phase_field(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
    phi_nodes: NDArray[np.float64],
    history: NDArray[np.float64],
    pf_assembler: PhaseFieldAssembler | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Update phase field and history variable."""
    if config.phase_field is None:
        return phi_nodes, history

    if pf_assembler is None:
        pf_assembler = PhaseFieldAssembler(config.mesh)

    n8_gp = np.array(
        [shape.q8_shape_functions(xi, eta) for xi, eta in shape.Q9_GAUSS_POINTS],
        dtype=np.float64,
    )
    phi_cell = phi_nodes[config.mesh.cells]
    phi_gp = phi_cell @ n8_gp.T

    psi_plus = tensile_strain_energy(config=config, uvp=uvp, phi_gp=phi_gp)
    history = np.maximum(history, psi_plus)

    ae, fe = pf_assembler.assemble_system(
        history=history,
        gc=config.phase_field.gc,
        length_scale=config.phase_field.length_scale,
    )
    a, b = pf_assembler.assemble_csc(ae=ae, fe=fe)
    phi_nodes = spsolve(a, b)
    phi_nodes = np.clip(phi_nodes, 0.0, 1.0)
    return phi_nodes, history
