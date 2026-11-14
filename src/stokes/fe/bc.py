"""Axis-aligned homogeneous Dirichlet and free-slip boundary conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stokes.fe.mesh import Mesh


@dataclass(frozen=True, slots=True)
class BoundaryConditions:
    """Boolean masks for homogeneous axis-aligned velocity constraints."""

    cell_mask: NDArray[np.bool_]
    global_mask: NDArray[np.bool_]

    @classmethod
    def free_slip_bc(
        cls,
        mesh: Mesh,
        edges: list[Literal["left", "right", "bottom", "top"]],
    ) -> Self:
        """Generate a free-slip boundary condition on an edge of the mesh."""
        flag_u = np.zeros(mesh.num_points, dtype=np.bool_)
        flag_v = np.zeros(mesh.num_points, dtype=np.bool_)
        for edge in edges:
            match edge:
                case "left":
                    flag_u |= np.isclose(mesh.points[:, 0], mesh.points[:, 0].min())
                case "right":
                    flag_u |= np.isclose(mesh.points[:, 0], mesh.points[:, 0].max())
                case "bottom":
                    flag_v |= np.isclose(mesh.points[:, 1], mesh.points[:, 1].min())
                case "top":
                    flag_v |= np.isclose(mesh.points[:, 1], mesh.points[:, 1].max())
                case _:
                    msg = f"Invalid edge: {edge}"
                    raise ValueError(msg)
        return cls.from_nodes(
            mesh=mesh,
            fix_u_nodes=np.where(flag_u)[0],
            fix_v_nodes=np.where(flag_v)[0],
        )

    @classmethod
    def from_nodes(
        cls,
        mesh: Mesh,
        fix_u_nodes: ArrayLike | None = None,
        fix_v_nodes: ArrayLike | None = None,
        no_slip_nodes: ArrayLike | None = None,
    ) -> Self:
        """Build constraint masks from node index arrays."""
        n_u_dofs = mesh.num_dofs[0]
        global_mask = np.zeros(n_u_dofs, dtype=np.bool_)
        if fix_u_nodes is not None:
            u_nodes = np.asarray(fix_u_nodes, dtype=np.int32)
            global_mask[2 * u_nodes] = True
        if fix_v_nodes is not None:
            v_nodes = np.asarray(fix_v_nodes, dtype=np.int32)
            global_mask[2 * v_nodes + 1] = True
        if no_slip_nodes is not None:
            ns_nodes = np.asarray(no_slip_nodes, dtype=np.int32)
            global_mask[2 * ns_nodes] = True
            global_mask[2 * ns_nodes + 1] = True
        uv_dofs = mesh.dofs[:, :16]
        cell_mask = np.ascontiguousarray(global_mask[uv_dofs])
        return cls(cell_mask=cell_mask, global_mask=global_mask)


def apply_bcs_inplace(
    ae: NDArray[np.float64],
    be: NDArray[np.float64],
    fe: NDArray[np.float64],
    bc: BoundaryConditions,
) -> None:
    """Zero out constrained rows and columns in-place for u=0 / v=0 BCs."""

    # save existing diagonal entries to preserve O(mu) matrix scaling
    idx = np.arange(16)
    diag = ae[:, idx, idx].copy()
    diag = np.where(diag > 0.0, diag, 1.0)

    # zero constrained rows and columns in ae, columns in be, and entries in fe
    ae[bc.cell_mask, :] = 0.0
    ae.swapaxes(1, 2)[bc.cell_mask, :] = 0.0
    be.swapaxes(1, 2)[bc.cell_mask, :] = 0.0
    fe[bc.cell_mask] = 0.0

    # restore diagonal on constrained DOFs (sets equation to diag * u_i = 0)
    ae[:, idx, idx] = np.where(bc.cell_mask, diag, ae[:, idx, idx])
