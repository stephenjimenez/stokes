"""Assembly wrapper calling the compiled C kernel."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

from stokes._core import assemble_phase_field, assemble_stokes
from stokes.fe.mesh import Mesh


class StokesAssembler:
    """Precompiles the CSC sparsity pattern for zero-allocation O(nnz) assembly."""

    mesh: Mesh
    n_total: int
    nnz: int
    coo_to_csc_map: NDArray[np.intp]
    indptr: NDArray[np.int32]
    indices: NDArray[np.int32]
    uv_dofs: NDArray[np.int32]
    _raw_vals: NDArray[np.float64]

    def __init__(self, mesh: Mesh) -> None:
        """Initialize the assembler and precompute the CSC index map."""
        self.mesh = mesh
        n_u, n_p = mesh.num_dofs
        self.n_total = n_u + n_p

        # build COO index arrays for [Ae (16x16), Be (4x16), Be^T (4x16 transposed)]
        uv_dofs = mesh.dofs[:, :16]
        p_dofs = mesh.dofs[:, 16:]
        r_ae = np.repeat(uv_dofs[:, :, None], 16, axis=2).ravel()
        c_ae = np.repeat(uv_dofs[:, None, :], 16, axis=1).ravel()
        r_be = np.repeat(p_dofs[:, :, None], 16, axis=2).ravel()
        c_be = np.repeat(uv_dofs[:, None, :], 4, axis=1).ravel()
        rows = np.concatenate([r_ae, r_be, c_be])
        cols = np.concatenate([c_ae, c_be, r_be])

        # sort primarily by col, secondarily by row (exact canonical CSC order)
        linear_idx = rows.astype(np.intp) + cols.astype(np.intp) * self.n_total
        unique_linear, self.coo_to_csc_map = np.unique(linear_idx, return_inverse=True)
        unique_cols = (unique_linear // self.n_total).astype(np.int32)
        unique_rows = (unique_linear % self.n_total).astype(np.int32)

        # build CSC indptr and indices directly from sorted unique_cols/unique_rows
        self.indices = unique_rows
        self.indptr = np.searchsorted(
            unique_cols, np.arange(self.n_total + 1, dtype=np.int32)
        ).astype(np.int32)
        self.nnz = len(unique_linear)
        self.uv_dofs = uv_dofs

        # preallocate buffer for [ae (256*N), -be (64*N), -be (64*N)]
        self._raw_vals = np.empty(384 * mesh.num_cells, dtype=np.float64)

    def assemble_csc(
        self,
        ae: NDArray[np.float64],
        be: NDArray[np.float64],
        fe: NDArray[np.float64],
    ) -> tuple[csc_matrix, NDArray[np.float64]]:
        """Assemble global CSC matrix and RHS vector without sorting or temp buffers."""
        n_ae = ae.size
        n_be = be.size

        # pack into preallocated buffer without np.concatenate allocations
        self._raw_vals[:n_ae] = ae.ravel()
        np.negative(be.ravel(), out=self._raw_vals[n_ae : n_ae + n_be])
        self._raw_vals[n_ae + n_be :] = self._raw_vals[n_ae : n_ae + n_be]

        # sum duplicate element contributions directly into CSC data array
        csc_data = np.bincount(
            self.coo_to_csc_map,
            weights=self._raw_vals,
            minlength=self.nnz,
        )

        # assemble global RHS vector
        f = np.zeros(self.n_total, dtype=np.float64)
        np.add.at(f, self.uv_dofs, fe)

        # construct CSC matrix directly from pre-verified canonical buffers (no copy)
        k = csc_matrix(
            (csc_data, self.indices, self.indptr),
            shape=(self.n_total, self.n_total),
            copy=False,
        )
        return k, f

    def assemble_system(
        self,
        b_glen: float,
        uvp: NDArray[np.float64],
        damage: NDArray[np.float64],
        body_force: tuple[float, float] = (0.0, 0.0),
        alpha: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Assemble element matrices for the Stokes system via compiled C."""
        cells = np.ascontiguousarray(self.mesh.cells, dtype=np.int32)
        dofs = np.ascontiguousarray(self.mesh.dofs, dtype=np.int32)
        x = np.ascontiguousarray(self.mesh.points, dtype=np.float64)
        u = np.ascontiguousarray(uvp, dtype=np.float64)
        dmg = np.ascontiguousarray(damage, dtype=np.float64)
        bf = np.ascontiguousarray(body_force, dtype=np.float64)

        ae = np.empty((self.mesh.num_cells, 16, 16), dtype=np.float64)
        be = np.empty((self.mesh.num_cells, 4, 16), dtype=np.float64)
        fe = np.empty((self.mesh.num_cells, 16), dtype=np.float64)

        assemble_stokes(cells, dofs, x, u, dmg, bf, b_glen, alpha, ae, be, fe)

        return ae, be, fe


class PhaseFieldAssembler:
    """Precompiles the CSC sparsity pattern for zero-allocation O(nnz) Q8
    phase-field assembly on mesh.num_points DOFs.
    """

    mesh: Mesh
    num_points: int
    nnz: int
    coo_to_csc_map: NDArray[np.intp]
    indptr: NDArray[np.int32]
    indices: NDArray[np.int32]
    _raw_vals: NDArray[np.float64]

    def __init__(self, mesh: Mesh) -> None:
        """Initialize the assembler and precompute the CSC index map."""
        self.mesh = mesh
        self.num_points = mesh.num_points

        # build COO index arrays for [Ae (8x8)]
        r_ae = np.repeat(mesh.cells[:, :, None], 8, axis=2).ravel()
        c_ae = np.repeat(mesh.cells[:, None, :], 8, axis=1).ravel()

        # sort primarily by col, secondarily by row (exact canonical CSC order)
        linear_idx = r_ae.astype(np.intp) + c_ae.astype(np.intp) * self.num_points
        unique_linear, self.coo_to_csc_map = np.unique(linear_idx, return_inverse=True)
        unique_cols = (unique_linear // self.num_points).astype(np.int32)
        unique_rows = (unique_linear % self.num_points).astype(np.int32)

        # build CSC indptr and indices directly from sorted unique_cols/unique_rows
        self.indices = unique_rows
        self.indptr = np.searchsorted(
            unique_cols, np.arange(self.num_points + 1, dtype=np.int32)
        ).astype(np.int32)
        self.nnz = len(unique_linear)

        # preallocate buffer for ae (64 * N)
        self._raw_vals = np.empty(64 * mesh.num_cells, dtype=np.float64)

    def assemble_csc(
        self,
        ae: NDArray[np.float64],
        fe: NDArray[np.float64],
    ) -> tuple[csc_matrix, NDArray[np.float64]]:
        """Assemble global CSC matrix and RHS vector without sorting or temp buffers."""
        self._raw_vals[:] = ae.ravel()

        csc_data = np.bincount(
            self.coo_to_csc_map,
            weights=self._raw_vals,
            minlength=self.nnz,
        )

        f = np.zeros(self.num_points, dtype=np.float64)
        np.add.at(f, self.mesh.cells, fe)

        a = csc_matrix(
            (csc_data, self.indices, self.indptr),
            shape=(self.num_points, self.num_points),
            copy=False,
        )
        return a, f

    def assemble_system(
        self,
        history: NDArray[np.float64],
        gc: float,
        length_scale: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Assemble element matrices for the Q8 phase field system via compiled C."""
        cells = np.ascontiguousarray(self.mesh.cells, dtype=np.int32)
        x = np.ascontiguousarray(self.mesh.points, dtype=np.float64)
        hist = np.ascontiguousarray(history, dtype=np.float64)

        ae = np.empty((self.mesh.num_cells, 8, 8), dtype=np.float64)
        fe = np.empty((self.mesh.num_cells, 8), dtype=np.float64)

        assemble_phase_field(cells, x, hist, float(gc), float(length_scale), ae, fe)

        return ae, fe
