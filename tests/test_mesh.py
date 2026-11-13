"""Tests for finite element mesh construction and properties."""

import numpy as np

from stokes.fe.mesh import Mesh


def test_mesh_from_grid_dimensions() -> None:
    """Verify mesh geometry, cell count, and total area."""
    nx, ny = 4, 3
    lx, ly = 8.0, 6.0
    mesh = Mesh.from_grid(shape=(nx, ny), size=(lx, ly))

    assert mesh.num_cells == nx * ny
    areas = mesh.cell_areas()
    assert np.isclose(np.sum(areas), lx * ly)
    assert np.all(areas > 0)


def test_mesh_dof_ordering() -> None:
    """Verify 20 DOFs per cell: 16 velocity, 4 corner pressure DOFs."""
    mesh = Mesh.from_grid(shape=(3, 2), size=(3.0, 2.0))
    dofs = mesh.cell_dofs()

    assert dofs.shape == (mesh.num_cells, 20)
    # Velocity DOFs are interleaved u0, v0, u1, v1 ...
    for c in range(mesh.num_cells):
        u_dofs = dofs[c, 0:16:2]
        v_dofs = dofs[c, 1:16:2]
        assert np.array_equal(v_dofs, u_dofs + 1)
        # Pressure DOFs must be >= 2 * num_points
        assert np.all(dofs[c, 16:] >= 2 * mesh.num_points)


def test_mesh_gauss_volumes() -> None:
    """Verify sum of Gauss point integration weights equals mesh area."""
    mesh = Mesh.from_grid(shape=(5, 5), size=(10.0, 10.0))
    gp_vols = mesh.gauss_point_volumes()
    assert np.isclose(np.sum(gp_vols), 100.0)


def test_mesh_boundaries() -> None:
    """Verify external boundary edges are detected."""
    mesh = Mesh.from_grid(shape=(2, 2), size=(2.0, 2.0))
    boundaries = mesh.find_boundaries()
    # A 2x2 grid has 2*2 + 2*2 = 8 boundary edges
    assert len(boundaries) == 8
