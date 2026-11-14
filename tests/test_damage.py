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
from stokes.rheology import interpolate_stress


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
