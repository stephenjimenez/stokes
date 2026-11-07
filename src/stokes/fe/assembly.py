"""Assembly wrapper calling the compiled C kernel."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

from stokes._c_assembly import assemble_stokes
from stokes.fe.mesh import Mesh


class StokesAssembler:
    """Precompiles the CSC sparsity pattern for zero-allocation O(nnz) assembly."""

    mesh: Mesh
    n_total: int
    nnz: int
    coo_to_csc_map: NDArray[np.intp]
    indptr: NDArray[np.int32]
    indices: NDArray[np.int32]
    v_dofs: NDArray[np.int32]
    _raw_vals: NDArray[np.float64]

    def __init__(self, mesh: Mesh) -> None:
        """Initialize the assembler and precompute the CSC index map."""
        self.mesh = mesh
        n_u, n_p = mesh.num_dofs
        self.n_total = n_u + n_p

        # build COO index arrays for [Ae (16x16), Be (4x16), Be^T (4x16 transposed)]
        dofs = mesh.cell_dofs()
        v_dofs = dofs[:, :16]
        p_dofs = dofs[:, 16:]
        r_ae = np.repeat(v_dofs[:, :, None], 16, axis=2).ravel()
        c_ae = np.repeat(v_dofs[:, None, :], 16, axis=1).ravel()
        r_be = np.repeat(p_dofs[:, :, None], 16, axis=2).ravel()
        c_be = np.repeat(v_dofs[:, None, :], 4, axis=1).ravel()
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
        self.v_dofs = v_dofs

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
        np.add.at(f, self.v_dofs, fe)

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
        u_global: NDArray[np.float64],
        body_force: tuple[float, float] = (0.0, 0.0),
        alpha: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Assemble element matrices for the Stokes system via compiled C."""

        # ensure arrays match the exact types and C-contiguity expected by PyArray_DATA
        cells = np.ascontiguousarray(self.mesh.cells, dtype=np.int32)
        dofs = np.ascontiguousarray(self.mesh.cell_dofs(), dtype=np.int32)
        x = np.ascontiguousarray(self.mesh.points, dtype=np.float64)
        u = np.ascontiguousarray(u_global, dtype=np.float64)
        bf = np.ascontiguousarray(body_force, dtype=np.float64)

        # pre-allocate output arrays (matching the shapes unpacked in python_module.c)
        ae = np.empty((self.mesh.num_cells, 16, 16), dtype=np.float64)
        be = np.empty((self.mesh.num_cells, 4, 16), dtype=np.float64)
        fe = np.empty((self.mesh.num_cells, 16), dtype=np.float64)

        # call into the C extension (modifies ae, be, fe in-place with zero memory copy)
        assemble_stokes(cells, dofs, x, u, bf, b_glen, alpha, ae, be, fe)

        return ae, be, fe
