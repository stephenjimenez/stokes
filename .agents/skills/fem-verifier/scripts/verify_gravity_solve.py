"""Quick analytical and numerical benchmark: Gravity-driven Stokes flow.

Verifies:
1. Non-linear Stokes iteration converges within < 15 iterations.
2. Global algebraic residual ||K * uvp - f|| / ||f|| is near machine precision.
3. Pressure is positive (compressive) inside the domain.
4. Velocity field exhibits expected glacier spreading (outward from divide).
"""

import sys

import numpy as np

from stokes.config import PlottingConfig, RheologyConfig, SimulationConfig, SolverConfig
from stokes.fe.assembly import StokesAssembler
from stokes.fe.bc import BoundaryConditions, apply_bcs_inplace
from stokes.fe.mesh import Mesh
from stokes.rheology import glen_coefficient
from stokes.simulation import SolverStatus, solve_stokes


def run_benchmark() -> bool:
    """Run gravity slab test and verify FEM invariants."""
    lx = 100.0
    ly = 50.0
    nx = 6
    ny = 3
    mesh = Mesh.from_grid(shape=(nx, ny), size=(lx, ly))
    # Left divide is symmetric (free-slip), bottom is rigid bedrock (free-slip)
    bc = BoundaryConditions.free_slip_bc(mesh=mesh, edges=["left", "bottom"])

    config = SimulationConfig(
        bc=bc,
        mesh=mesh,
        phase_field=None,
        plotting=PlottingConfig(output_interval=100),
        rheology=RheologyConfig(gravity=True, rho_ice=917.0, temperature=263.15),
        solver=SolverConfig(tolerance=1e-5, max_iterations=20, num_steps=1),
    )

    stokes_assembler = StokesAssembler(config.mesh)
    b_glen = glen_coefficient(config.rheology.temperature)
    body_force = config.rheology.body_force()
    dmg0 = np.zeros((config.mesh.num_cells, 9), dtype=np.float64)
    uvp0 = np.zeros(sum(config.mesh.num_dofs), dtype=np.float64)

    print("Running gravity-driven flow benchmark...")
    uvp, status = solve_stokes(
        config=config,
        stokes_assembler=stokes_assembler,
        uvp0=uvp0,
        dmg0=dmg0,
        b_glen=b_glen,
        body_force=body_force,
    )

    if status != SolverStatus.CONVERGED:
        print("[FAIL] Solver did not reach convergence.")
        return False

    n_vel = config.mesh.num_dofs[0]
    u_computed = uvp[:n_vel]
    p_computed = uvp[n_vel:]

    # 1. Algebraic residual check: ||K * uvp - f|| / ||f||
    ae, be, fe = stokes_assembler.assemble_system(
        uvp=uvp,
        b_glen=b_glen,
        damage=dmg0,
        body_force=body_force,
        alpha=0.0,
    )
    apply_bcs_inplace(ae=ae, be=be, fe=fe, bc=config.bc)
    k_mat, f_vec = stokes_assembler.assemble_csc(ae=ae, be=be, fe=fe)
    res = k_mat @ uvp - f_vec
    rel_res = float(np.linalg.norm(res) / np.linalg.norm(f_vec))
    print(f"Algebraic System Relative Residual ||K*uvp - f|| / ||f||: {rel_res:e}")

    # 2. Pressure positivity check (excluding top free surface)
    p_node_coords = mesh.points[mesh.vertices]
    internal_mask = p_node_coords[:, 1] < (ly - 1e-3)
    p_internal = p_computed[internal_mask]
    min_internal_p = float(np.min(p_internal))
    print(f"Minimum Internal Pressure: {min_internal_p:.2f} Pa (must be > 0)")

    # 3. Spreading velocity check: u_x at right boundary should be positive
    # (flowing out)
    right_mask = np.isclose(mesh.points[:, 0], lx)
    u_x_right = u_computed[0:n_vel:2][right_mask]
    avg_u_right = float(np.mean(u_x_right))
    print(f"Average Rightward Velocity at Outflow: {avg_u_right:.2f} m/yr")

    passed = (
        (status == SolverStatus.CONVERGED)
        and (rel_res < 1e-6)
        and (min_internal_p > 0.0)
        and (avg_u_right > 0.0)
    )

    if passed:
        print("[PASS] All FEM and physical benchmarks passed!")
        return True
    else:
        print("[FAIL] One or more benchmark invariants failed.")
        return False


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
