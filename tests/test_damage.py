"""Tests for implicit gradient damage assembler and evolution."""

import numpy as np
from scipy.sparse.linalg import spsolve

from stokes.config import (
    DamageConfig,
    PlottingConfig,
    RheologyConfig,
    SimulationConfig,
    SolverConfig,
)
from stokes.fe import shape
from stokes.fe.assembly import DamageAssembler
from stokes.fe.bc import BoundaryConditions
from stokes.fe.mesh import Mesh
from stokes.rheology import damage_rate, hayhurst_driving_stress, interpolate_stress
from stokes.simulation import update_damage


def make_config(mesh: Mesh, damage: DamageConfig | None = None) -> SimulationConfig:
    """Helper to create SimulationConfig with test defaults."""
    return SimulationConfig(
        bc=BoundaryConditions.free_slip_bc(mesh=mesh, edges=[]),
        damage=damage,
        mesh=mesh,
        plotting=PlottingConfig(),
        rheology=RheologyConfig(),
        solver=SolverConfig(),
    )


def test_damage_assembler_patch_test_conservation() -> None:
    """Verify that a uniform Hayhurst stress field chi0 is preserved exactly
    (bar{chi} = chi0) across different length scales and mesh resolutions.
    """
    for nx, ny in [(2, 2), (4, 3), (5, 5)]:
        mesh = Mesh.from_grid(shape=(nx, ny), size=(10.0, 8.0))
        assembler = DamageAssembler(mesh)

        chi_val = 42.5  # MPa
        chi = np.full((mesh.num_cells, 9), chi_val, dtype=np.float64)

        for ell in [0.0, 0.5, 2.0, 10.0]:
            ae, fe = assembler.assemble_system(chi=chi, length_scale=ell)
            a, b = assembler.assemble_csc(ae=ae, fe=fe)

            chi_nodes = spsolve(a, b)
            assert np.allclose(chi_nodes, chi_val, atol=1e-11, rtol=1e-11)

            # Interpolate to Gauss points
            corner_idx = assembler.corner_vertices
            chi_cell_nodes = chi_nodes[corner_idx]
            n4_gp = np.array(
                [
                    shape.q4_shape_functions(xi, eta)
                    for xi, eta in shape.Q9_GAUSS_POINTS
                ],
                dtype=np.float64,
            )
            chi_bar = chi_cell_nodes @ n4_gp.T
            assert np.allclose(chi_bar, chi_val, atol=1e-11, rtol=1e-11)


def test_damage_assembler_csc_properties() -> None:
    """Verify sparsity, symmetry, and positive definiteness of DamageAssembler."""
    mesh = Mesh.from_grid(shape=(4, 4), size=(5.0, 5.0))
    assembler = DamageAssembler(mesh)

    chi = np.ones((mesh.num_cells, 9), dtype=np.float64)
    ae, fe = assembler.assemble_system(chi=chi, length_scale=1.5)
    a, b = assembler.assemble_csc(ae=ae, fe=fe)

    assert a.shape == (mesh.num_vertices, mesh.num_vertices)
    assert b.shape == (mesh.num_vertices,)

    # Symmetry check: A == A.T
    diff = a - a.T
    if diff.nnz > 0:
        assert np.max(np.abs(diff.data)) < 1e-14

    # Positive definiteness: x^T A x > 0 for non-zero x
    np.random.seed(42)
    for _ in range(5):
        v = np.random.randn(mesh.num_vertices)
        quad_form = float(v @ (a @ v))
        assert quad_form > 0.0


def test_damage_assembler_smoothing_effect() -> None:
    """Verify that length_scale > 0 diffuses a localized spike compared to ell = 0."""
    mesh = Mesh.from_grid(shape=(5, 5), size=(5.0, 5.0))
    assembler = DamageAssembler(mesh)

    # Localized source: non-zero only at center element (index 12)
    chi = np.zeros((mesh.num_cells, 9), dtype=np.float64)
    center_cell = 12
    chi[center_cell, 4] = 100.0  # Spike at center Gauss point

    # Solve with ell = 0 (no diffusion)
    ae0, fe0 = assembler.assemble_system(chi=chi, length_scale=0.0)
    a0, b0 = assembler.assemble_csc(ae=ae0, fe=fe0)
    chi_nodes_0 = spsolve(a0, b0)

    # Solve with ell = 1.0 (diffusion)
    ae1, fe1 = assembler.assemble_system(chi=chi, length_scale=1.0)
    a1, b1 = assembler.assemble_csc(ae=ae1, fe=fe1)
    chi_nodes_1 = spsolve(a1, b1)

    # With diffusion, the peak should be lower and non-zero values spread outward
    assert np.max(chi_nodes_1) < np.max(chi_nodes_0)
    assert np.count_nonzero(chi_nodes_1 > 0.01) > np.count_nonzero(chi_nodes_0 > 0.01)


def test_hayhurst_driving_stress_screening() -> None:
    """Verify driving stress is zeroed where tr(sigma) <= 0 or D >= D_crit."""
    mesh = Mesh.from_grid(shape=(2, 2), size=(2.0, 2.0))
    config = make_config(mesh=mesh, damage=DamageConfig(critical_damage=0.63))

    n_dofs = sum(mesh.num_dofs)
    uvp = np.zeros(n_dofs, dtype=np.float64)

    # Apply positive pressure (compression -> tr(sigma) < 0)
    p_dofs = mesh.p_dofs.ravel()
    uvp[p_dofs] = 1e5  # 100 kPa compression (p > 0)

    dmg0 = np.zeros((mesh.num_cells, 9), dtype=np.float64)
    chi, _ = hayhurst_driving_stress(config=config, uvp=uvp, dmg0=dmg0)
    # Under hydrostatic compression, chi should be screened to 0
    assert np.all(chi == 0.0)

    # Now apply negative pressure (tension -> tr(sigma) > 0)
    uvp[p_dofs] = -1e5  # 100 kPa tension (p < 0)
    chi_tensile, skk = hayhurst_driving_stress(config=config, uvp=uvp, dmg0=dmg0)
    assert np.all(skk > 0.0)
    assert np.all(chi_tensile > 0.0)

    # Mark element 0 as failed (D >= critical_damage)
    dmg0[0, :] = 0.70
    chi_failed, _ = hayhurst_driving_stress(config=config, uvp=uvp, dmg0=dmg0)
    assert np.all(chi_failed[0, :] == 0.0)
    assert np.all(chi_failed[1:, :] > 0.0)


def test_interpolate_stress_exact_q4_pressure() -> None:
    """Verify interpolate_stress evaluates pressure via Q4 shape functions
    at (xi, eta).
    """
    mesh = Mesh.from_grid(shape=(1, 1), size=(2.0, 2.0))
    config = make_config(mesh=mesh)

    n_u, n_p = mesh.num_dofs
    uvp = np.zeros(n_u + n_p, dtype=np.float64)

    # Set linear corner pressures: p0=0, p1=10, p2=30, p3=20
    # Q4 corners: (-1,-1)->0, (1,-1)->10, (1,1)->30, (-1,1)->20
    p_indices = mesh.cell_dofs()[0, 16:]
    uvp[p_indices] = [0.0, 10.0, 30.0, 20.0]

    dmg = np.zeros(1, dtype=np.float64)

    # Center (0, 0): pressure should be average = 15.0
    _, _, sigma_center = interpolate_stress(config, uvp, dmg, xi=0.0, eta=0.0)
    assert np.isclose(-sigma_center[0, 0], 15.0)  # p = 15.0 (with zero strain rate)

    # Corner 0 (-1, -1): pressure should be 0.0
    _, _, sigma_c0 = interpolate_stress(config, uvp, dmg, xi=-1.0, eta=-1.0)
    assert np.isclose(-sigma_c0[0, 0], 0.0)

    # Corner 2 (1, 1): pressure should be 30.0
    _, _, sigma_c2 = interpolate_stress(config, uvp, dmg, xi=1.0, eta=1.0)
    assert np.isclose(-sigma_c2[0, 0], 30.0)


def test_update_damage_advancement() -> None:
    """Verify update_damage advances damage in active zones and caps at max_damage."""
    mesh = Mesh.from_grid(shape=(2, 2), size=(2.0, 2.0))
    config = make_config(
        mesh=mesh,
        damage=DamageConfig(
            critical_damage=0.63,
            max_damage=0.99,
            b_hat=1.0,  # Fast damage rate for testing
            length_scale=1.0,
        ),
    )
    damage_assembler = DamageAssembler(mesh)

    n_dofs = sum(mesh.num_dofs)
    uvp = np.zeros(n_dofs, dtype=np.float64)
    # Tensile pressure: p = -1e5 Pa -> chi > 0
    uvp[mesh.p_dofs.ravel()] = -1e5

    dmg0 = np.zeros((mesh.num_cells, 9), dtype=np.float64)
    dmg1 = update_damage(
        config=config,
        uvp=uvp,
        dmg0=dmg0,
        damage_assembler=damage_assembler,
    )

    # Damage should have advanced
    assert np.all(dmg1 > 0.0)
    assert np.all(dmg1 <= config.damage.max_damage)

    # Test capping at max_damage when exceeding critical_damage
    dmg_near_crit = np.full((mesh.num_cells, 9), 0.62, dtype=np.float64)
    dmg_capped = update_damage(
        config=config,
        uvp=uvp,
        dmg0=dmg_near_crit,
        damage_assembler=damage_assembler,
    )
    # Because b_hat = 1.0, dmg should surpass critical_damage and be set to max_damage
    assert np.all(dmg_capped == config.damage.max_damage)


def test_update_damage_none() -> None:
    """Verify update_damage is a no-op when config.damage is None."""
    mesh = Mesh.from_grid(shape=(2, 2), size=(2.0, 2.0))
    config = make_config(mesh=mesh, damage=None)
    uvp = np.zeros(sum(mesh.num_dofs), dtype=np.float64)
    dmg0 = np.zeros((mesh.num_cells, 9), dtype=np.float64)

    dmg1 = update_damage(config=config, uvp=uvp, dmg0=dmg0)
    assert np.array_equal(dmg1, dmg0)


def test_damage_rate_no_active() -> None:
    """Verify damage_rate returns zeros when no points are active."""
    mesh = Mesh.from_grid(shape=(1, 1), size=(1.0, 1.0))
    config = make_config(mesh=mesh, damage=DamageConfig())

    chi = np.zeros((1, 9), dtype=np.float64)
    dmg0 = np.zeros((1, 9), dtype=np.float64)
    ddmg = damage_rate(config, chi, dmg0)
    assert np.all(ddmg == 0.0)
