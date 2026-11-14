"""Tests for Q8 variational phase-field fracture model and assembly."""

import numpy as np
from scipy.sparse.linalg import spsolve

from stokes.config import (
    PhaseFieldConfig,
    PlottingConfig,
    RheologyConfig,
    SimulationConfig,
    SolverConfig,
)
from stokes.fe import shape
from stokes.fe.assembly import PhaseFieldAssembler
from stokes.fe.bc import BoundaryConditions
from stokes.fe.mesh import Mesh
from stokes.rheology import (
    tensile_strain_energy,
)


def make_config(
    mesh: Mesh, phase_field: PhaseFieldConfig | None = None
) -> SimulationConfig:
    """Helper to create SimulationConfig with test defaults."""
    return SimulationConfig(
        bc=BoundaryConditions.free_slip_bc(mesh=mesh, edges=[]),
        mesh=mesh,
        phase_field=phase_field,
        plotting=PlottingConfig(),
        rheology=RheologyConfig(),
        solver=SolverConfig(),
    )


def test_phase_field_assembler_patch_test_exact() -> None:
    """Patch Test: For uniform driving energy H0, the Q8 phase-field solver
    reproduces the exact analytical constant phi = 2*H0 / (Gc / ell + 2*H0)
    to machine precision (< 1e-12).
    """
    for nx, ny in [(2, 2), (4, 3), (5, 5)]:
        mesh = Mesh.from_grid(shape=(nx, ny), size=(10.0, 8.0))
        assembler = PhaseFieldAssembler(mesh)

        gc = 50.0  # J/m^2
        h0_val = 120.0  # Pa (virtual tensile energy)
        history = np.full((mesh.num_cells, 9), h0_val, dtype=np.float64)

        for ell in [0.5, 2.0, 10.0]:
            expected_phi = (2.0 * h0_val) / (gc / ell + 2.0 * h0_val)

            ae, fe = assembler.assemble_system(history=history, gc=gc, length_scale=ell)
            a, f = assembler.assemble_csc(ae=ae, fe=fe)

            phi_nodes = spsolve(a, f)
            assert np.allclose(phi_nodes, expected_phi, atol=1e-12, rtol=1e-12)

            # Interpolate to Gauss points
            n8_gp = np.array(
                [
                    shape.q8_shape_functions(xi, eta)
                    for xi, eta in shape.Q9_GAUSS_POINTS
                ],
                dtype=np.float64,
            )
            phi_gp = phi_nodes[mesh.cells] @ n8_gp.T
            assert np.allclose(phi_gp, expected_phi, atol=1e-12, rtol=1e-12)


def test_phase_field_assembler_csc_properties() -> None:
    """Verify sparsity, symmetry, and positive definiteness of PhaseFieldAssembler."""
    mesh = Mesh.from_grid(shape=(4, 4), size=(5.0, 5.0))
    assembler = PhaseFieldAssembler(mesh)

    history = np.ones((mesh.num_cells, 9), dtype=np.float64)
    ae, fe = assembler.assemble_system(history=history, gc=50.0, length_scale=1.5)
    a, f = assembler.assemble_csc(ae=ae, fe=fe)

    assert a.shape == (mesh.num_points, mesh.num_points)
    assert f.shape == (mesh.num_points,)

    # Symmetry check: A == A.T
    diff = a - a.T
    if diff.nnz > 0:
        assert np.max(np.abs(diff.data)) < 1e-14

    # Positive definiteness: v^T A v > 0 for non-zero v
    np.random.seed(42)
    for _ in range(5):
        v = np.random.randn(mesh.num_points)
        quad_form = float(v @ (a @ v))
        assert quad_form > 0.0


def test_tensile_strain_energy_decomposition() -> None:
    """Verify tensile stress decomposition, Macaulay brackets, and critical stress."""
    mesh = Mesh.from_grid(shape=(2, 2), size=(1.0, 1.0))
    e_mod = 1.0e10
    nu = 0.25

    # 1. Zero stress -> zero energy
    cfg = make_config(
        mesh=mesh,
        phase_field=PhaseFieldConfig(
            youngs_modulus=e_mod,
            poissons_ratio=nu,
            critical_stress=0.0,
        ),
    )
    n_total = sum(mesh.num_dofs)
    uvp = np.zeros(n_total, dtype=np.float64)
    phi_gp = np.zeros((mesh.num_cells, 9), dtype=np.float64)

    psi = tensile_strain_energy(cfg, uvp, phi_gp)
    assert np.allclose(psi, 0.0)

    # 2. Pure hydrostatic compression: pressure p > 0, velocity = 0
    # Cauchy stress is sigma = -p I (all negative). Macaulay brackets should be 0.
    p_compressive = 1.0e6  # Pa
    uvp[mesh.num_dofs[0] :] = p_compressive
    psi_comp = tensile_strain_energy(cfg, uvp, phi_gp)
    assert np.allclose(psi_comp, 0.0)

    # 3. Pure hydrostatic tension: pressure p < 0 (-p > 0 in compression convention)
    p_tensile = -1.0e6  # Pa
    uvp[mesh.num_dofs[0] :] = p_tensile
    # sigma_1 = sigma_2 = sigma_3 = -p = 1e6 Pa
    # psi+ = (0.5/E)*(3*(1e6)^2) - (nu/E)*(3*(1e6)^2) = (3*(1e6)^2 / (2E)) * (1 - 2*nu)
    # For nu=0.25: (1 - 2*0.25) = 0.5. psi+ = 1.5 * (1e6)^2 / (2E)
    expected_psi = (0.5 / e_mod) * (3.0 * (1.0e6**2)) - (nu / e_mod) * (
        3.0 * (1.0e6**2)
    )
    psi_tens = tensile_strain_energy(cfg, uvp, phi_gp)
    assert np.allclose(psi_tens, expected_psi, rtol=1e-10)

    # 4. Critical threshold: sigma_c = 2.0e6 Pa -> psi_c = 200 > 75 -> clamped to 0
    cfg_crit = make_config(
        mesh=mesh,
        phase_field=PhaseFieldConfig(
            youngs_modulus=e_mod,
            poissons_ratio=nu,
            critical_stress=2.0e6,
        ),
    )
    psi_clamped = tensile_strain_energy(cfg_crit, uvp, phi_gp)
    assert np.allclose(psi_clamped, 0.0)


def test_gamma_convergence_crack_surface_energy() -> None:
    """1D/2D Crack Surface Energy (Gamma-Convergence):
    A prescribed crack profile phi(x, y) = exp(-|y|/ell) across length L
    yields discrete surface energy approaching L as h <= ell / 2.
    """
    length_x = 10.0
    width_y = 12.0
    ell = 1.5

    # Test with mesh resolution h <= ell / 2
    # ny = 24 elements -> hy = 12.0 / 24 = 0.5 <= ell / 2 (0.75)
    for ny in [16, 24]:
        nx = 10
        mesh = Mesh.from_grid(
            shape=(nx, ny),
            size=(length_x, width_y),
        )
        # Shift y-coordinates so crack is at y=0: [-width_y/2, width_y/2]
        mesh.points[:, 1] -= width_y / 2.0

        # Exact 1D analytical profile at nodes: phi = exp(-|y|/ell)
        phi_nodes = np.exp(-np.abs(mesh.points[:, 1]) / ell)

        # Compute Gamma_ell numerically via Gauss quadrature over all cells
        discrete_gamma = 0.0
        for gp in range(9):
            xi, eta = shape.Q9_GAUSS_POINTS[gp]
            w = shape.Q9_GAUSS_WEIGHTS[gp]
            n8 = shape.q8_shape_functions(xi, eta)
            dndxi = shape.q8_shape_gradients(xi, eta)

            # Evaluate coordinates, Jacobian and gradients
            xe = mesh.points[mesh.cells]  # (n_cells, 8, 2)
            jac = np.einsum("ia,nak->nik", dndxi, xe)
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            inv = 1.0 / det
            ijac = np.empty_like(jac)
            ijac[:, 0, 0] = jac[:, 1, 1] * inv
            ijac[:, 0, 1] = -jac[:, 0, 1] * inv
            ijac[:, 1, 0] = -jac[:, 1, 0] * inv
            ijac[:, 1, 1] = jac[:, 0, 0] * inv

            # dndx: Cartesian gradients for each cell (n_cells, 2, 8)
            dndx = np.einsum("nji,ja->nia", ijac, dndxi)

            phi_cell = phi_nodes[mesh.cells]  # (n_cells, 8)
            phi_gp = phi_cell @ n8  # (n_cells,)
            grad_x = np.einsum("na,na->n", dndx[:, 0, :], phi_cell)
            grad_y = np.einsum("na,na->n", dndx[:, 1, :], phi_cell)
            grad_sq = grad_x**2 + grad_y**2

            energy_density = (0.5 / ell) * (phi_gp**2 + (ell**2) * grad_sq)
            discrete_gamma += np.sum(energy_density * w * det)

        # Analytical value for domain [-W/2, W/2]:
        # integral_{-W/2}^{W/2} 1/ell * exp(-2|y|/ell) dy = 1 - exp(-W/ell)
        analytical_gamma = length_x * (1.0 - np.exp(-width_y / ell))
        rel_error = abs(discrete_gamma - analytical_gamma) / analytical_gamma
        assert rel_error < 0.05, f"ny={ny}, rel_error={rel_error}"


def test_single_edge_notch_mesh_convergence() -> None:
    """Single-Edge Notched Crevasse Benchmark:
    Halving element size (h -> h/2 at fixed length scale ell) produces
    consistent, convergent phase-field response at the notch tip.
    """
    ell = 2.0
    gc = 50.0

    phi_max_list = []
    for nx, ny in [(6, 6), (12, 12)]:
        mesh = Mesh.from_grid(shape=(nx, ny), size=(10.0, 10.0))
        assembler = PhaseFieldAssembler(mesh)

        # Concentrated driving energy at notch tip (e.g. x=5, y=5)
        coords = mesh.gauss_point_coordinates().reshape(mesh.num_cells, 9, 2)
        dist_sq = (coords[..., 0] - 5.0) ** 2 + (coords[..., 1] - 5.0) ** 2
        history = 500.0 * np.exp(-dist_sq / (2.0 * (ell**2)))

        ae, fe = assembler.assemble_system(history=history, gc=gc, length_scale=ell)
        a, f = assembler.assemble_csc(ae=ae, fe=fe)
        phi_nodes = spsolve(a, f)

        # Phase field should be strictly bounded in [0, 1]
        assert np.all(phi_nodes >= -1e-12)
        assert np.all(phi_nodes <= 1.0 + 1e-12)

        phi_max_list.append(np.max(phi_nodes))

    # Halving h produces consistent peak phase-field value (< 5% difference)
    assert abs(phi_max_list[1] - phi_max_list[0]) / phi_max_list[1] < 0.05


def test_element_erosion_mechanism() -> None:
    """Verify that when erosion_threshold is reached, damaged elements are removed
    and the mesh/assemblers remain consistent.
    """
    mesh = Mesh.from_grid(shape=(4, 2), size=(4.0, 2.0))
    config = make_config(
        mesh=mesh,
        phase_field=PhaseFieldConfig(
            gc=50.0,
            length_scale=1.0,
            erosion_threshold=0.8,
        ),
    )
    config.bc_edges = ["left", "bottom"]

    # Initial state
    phi_nodes = np.zeros(mesh.num_points, dtype=np.float64)
    # Set high phase field on cell 0
    phi_nodes[mesh.cells[0]] = 0.95

    n8_gp = np.array(
        [shape.q8_shape_functions(xi, eta) for xi, eta in shape.Q9_GAUSS_POINTS],
        dtype=np.float64,
    )
    phi_cell = phi_nodes[config.mesh.cells]
    phi_gp = phi_cell @ n8_gp.T
    bar_phi_e = np.mean(phi_gp, axis=1)

    eroded = bar_phi_e >= config.phase_field.erosion_threshold
    assert eroded[0]
    assert not eroded[1]

    # Perform erosion rebuild
    keep = ~eroded
    surviving_cells = config.mesh.cells[keep]
    used_nodes, new_cells_flat = np.unique(surviving_cells, return_inverse=True)
    new_cells = new_cells_flat.reshape(surviving_cells.shape).astype(np.int32)
    new_points = config.mesh.points[used_nodes]

    new_mesh = Mesh(cells=new_cells, points=new_points)
    assert new_mesh.num_cells == mesh.num_cells - 1
    new_pf = PhaseFieldAssembler(new_mesh)
    assert new_pf.num_points == new_mesh.num_points


def test_vtu_export_effective_damage(tmp_path) -> None:
    """Verify ParaView VTU export contains PhaseField and EffectiveDamage fields."""
    import pyvista as pv

    from stokes.visualization import ParaViewWriter

    mesh = Mesh.from_grid(shape=(2, 2), size=(2.0, 2.0))
    cfg = make_config(
        mesh=mesh,
        phase_field=PhaseFieldConfig(residual_stiffness=1e-4),
    )
    writer = ParaViewWriter(output_dir=tmp_path, base_name="test_out")

    uvp = np.zeros(sum(mesh.num_dofs), dtype=np.float64)
    phi_nodes = np.linspace(0.0, 0.8, mesh.num_points, dtype=np.float64)

    vtu_file = writer.write_step(config=cfg, time=0.0, uvp=uvp, phi_nodes=phi_nodes)
    assert vtu_file.exists()

    grid = pv.read(str(vtu_file))
    # PhaseField and EffectiveDamage in point data
    assert "PhaseField" in grid.point_data
    assert "EffectiveDamage" in grid.point_data
    assert np.allclose(grid.point_data["PhaseField"], phi_nodes)

    # EffectiveDamage in cell data
    assert "EffectiveDamage" in grid.cell_data
    assert len(grid.cell_data["EffectiveDamage"]) == mesh.num_cells
