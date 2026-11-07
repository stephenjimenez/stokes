"""Tools for managing the finite element mesh."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import ArrayLike, NDArray


class Mesh:
    """Container for the finite element mesh."""

    cells: NDArray[np.int32]
    points: NDArray[np.float64]
    p_dofs: NDArray[np.int32]
    u_dofs: NDArray[np.int32]
    v_dofs: NDArray[np.int32]
    vertices: NDArray[np.int32]

    def __init__(self, cells: NDArray[np.int32], points: NDArray[np.float64]) -> None:
        """Initialize the mesh."""
        self.cells = np.ascontiguousarray(cells, dtype=np.int32)
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        u, idx = np.unique(self.cells[:, :4], return_inverse=True)
        self.p_dofs = np.ascontiguousarray(
            idx.reshape(self.cells.shape[0], 4) + 2 * len(points),
            dtype=np.int32,
        )
        self.u_dofs = np.ascontiguousarray(2 * self.cells, dtype=np.int32)
        self.v_dofs = np.ascontiguousarray(2 * self.cells + 1, dtype=np.int32)
        self.vertices = np.ascontiguousarray(u, dtype=np.int32)

        # disable overwriting of cell topology
        self.cells.setflags(write=False)
        self.p_dofs.setflags(write=False)
        self.u_dofs.setflags(write=False)
        self.v_dofs.setflags(write=False)
        self.vertices.setflags(write=False)

    def cell_dofs(self) -> NDArray[np.int32]:
        """Construct an array of cell degrees of freedom."""
        dofs = np.empty((self.num_cells, 20), dtype=np.int32)
        dofs[:, 0:16:2] = self.u_dofs
        dofs[:, 1:16:2] = self.v_dofs
        dofs[:, 16:] = self.p_dofs
        return dofs

    def find_boundaries(self) -> NDArray[np.int32]:
        """Construct an array of all cell edges that are external boundaries."""
        cols = np.array([[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]], dtype=np.int32)
        faces = self.cells[:, cols].reshape(-1, 3)
        _, index, counts = np.unique(
            np.sort(faces, axis=1),
            return_index=True,
            return_counts=True,
            axis=0,
        )
        return np.ascontiguousarray(faces[index[counts == 1]], dtype=np.int32)

    @classmethod
    def from_grid(
        cls,
        shape: tuple[int, int] = (5, 5),
        size: tuple[float, float] = (1.0, 1.0),
    ) -> Self:
        """Construct a Q8-Q4 mesh from a structured grid."""
        nx, ny = shape

        # generate cells
        stride = nx + 1
        q4_cells = [
            [
                i * stride + j,
                i * stride + j + 1,
                (i + 1) * stride + j + 1,
                (i + 1) * stride + j,
            ]
            for i in range(ny)
            for j in range(nx)
        ]

        # generate vertices
        lx, ly = size
        x = np.linspace(0.0, lx, nx + 1, dtype=np.float64)
        y = np.linspace(0.0, ly, ny + 1, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        vertices = np.column_stack([xx.ravel(), yy.ravel()])

        return cls.from_q4_mesh(q4_cells=q4_cells, vertices=vertices)

    @classmethod
    def from_q4_mesh(cls, q4_cells: ArrayLike, vertices: ArrayLike) -> Self:
        """Construct a Q8-Q4 mesh from Q4 cells."""
        _q4_cells = np.ascontiguousarray(q4_cells, dtype=np.int32)
        _vertices = np.ascontiguousarray(vertices, dtype=np.float64)

        num_cells = len(_q4_cells)
        cells = np.empty((num_cells, 8), dtype=np.int32)
        cells[:, :4] = _q4_cells

        edge_map: dict[tuple[int, int], int] = {}
        midpoints: list[NDArray[np.float64]] = []
        for i in range(num_cells):
            for j, key in enumerate(_iterate_cell(cells[i])):
                if key not in edge_map:
                    edge_map[key] = len(_vertices) + len(edge_map)
                    a, b = key
                    midpoints.append(0.5 * (_vertices[a] + _vertices[b]))
                cells[i, 4 + j] = edge_map[key]

        return cls(cells=cells, points=np.vstack([_vertices, midpoints]))

    @property
    def num_dofs(self) -> tuple[int, int]:
        """Return the number of degrees of freedom for (velocity, pressure)."""
        return 2 * self.num_points, self.num_vertices

    @property
    def num_cells(self) -> int:
        """Return the number of cells."""
        return self.cells.shape[0]

    @property
    def num_points(self) -> int:
        """Return the number of points."""
        return self.points.shape[0]

    @property
    def num_vertices(self) -> int:
        """Return the number of vertices."""
        return self.vertices.shape[0]

    def update_points(self, uvp: NDArray[np.float64], dt: float) -> None:
        """Update coordinates of points using a velocity solution."""
        n = self.num_dofs[0]
        self.points[:, 0] += uvp[0:n:2] * dt
        self.points[:, 1] += uvp[1:n:2] * dt

        # ensure edge nodes remain midpoints
        for i in range(4):
            v1 = self.cells[:, i]
            v2 = self.cells[:, (i + 1) % 4]
            edge = self.cells[:, 4 + i]
            self.points[edge] = 0.5 * (self.points[v1] + self.points[v2])


def _iterate_cell(cell: Sequence[int]) -> Iterator[tuple[int, int]]:
    for i in range(4):
        a = cell[i]
        b = cell[(i + 1) % 4]
        yield (a, b) if a < b else (b, a)
