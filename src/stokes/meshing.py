"""Conforming 3-to-1 quad mesh refinement generator (no hanging nodes)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

type Point = tuple[float, float]
type Edge = tuple[int, int]
type Quad = tuple[int, int, int, int]


def _canonical_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _interp(p0: Point, p1: Point, p2: Point, p3: Point, u: float, v: float) -> Point:
    """Bilinear interpolation on [0, 1]^2."""
    x = (
        (1.0 - u) * (1.0 - v) * p0[0]
        + u * (1.0 - v) * p1[0]
        + u * v * p2[0]
        + (1.0 - u) * v * p3[0]
    )
    y = (
        (1.0 - u) * (1.0 - v) * p0[1]
        + u * (1.0 - v) * p1[1]
        + u * v * p2[1]
        + (1.0 - u) * v * p3[1]
    )
    return (x, y)


@dataclass(slots=True)
class ConformingQuadMesh:
    """Container for a conforming quadrilateral mesh."""

    points: dict[int, Point] = field(default_factory=dict)
    quads: list[Quad] = field(default_factory=list)
    _next_pid: int = 0

    def add_point(self, x: float, y: float) -> int:
        pid = self._next_pid
        self.points[pid] = (float(x), float(y))
        self._next_pid += 1
        return pid

    def elements_in_box(self, lower: Point, upper: Point) -> set[int]:
        """Identify quads that fit within a box."""
        ax, ay = lower
        bx, by = upper
        patch = set()
        for i, (n0, n1, n2, n3) in enumerate(self.quads):
            x0, y0 = self.points[n0]
            x1, y1 = self.points[n1]
            x2, y2 = self.points[n2]
            x3, y3 = self.points[n3]
            x = 0.25 * (x0 + x1 + x2 + x3)
            y = 0.25 * (y0 + y1 + y2 + y3)
            if ax <= x <= bx and ay <= y <= by:
                patch.add(i)
        return patch

    @classmethod
    def from_grid(
        cls,
        shape: tuple[int, int],
        size: tuple[float, float] = (1.0, 1.0),
    ) -> Self:
        """Create a baseline structured grid of conforming quads."""
        mesh = cls()
        nx, ny = shape
        lx, ly = size
        hx, hy = lx / nx, ly / ny

        grid = [
            [mesh.add_point(j * hx, i * hy) for j in range(nx + 1)]
            for i in range(ny + 1)
        ]

        for i in range(ny):
            for j in range(nx):
                mesh.quads.append(
                    (grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j])
                )
        return mesh

    def get_edge_trisection(self, a: int, b: int) -> tuple[int, int]:
        """Return the two 1/3 and 2/3 points ordered from a -> b."""
        pa, pb = self.points[a], self.points[b]
        p1 = self.add_point((2.0 * pa[0] + pb[0]) / 3.0, (2.0 * pa[1] + pb[1]) / 3.0)
        p2 = self.add_point((pa[0] + 2.0 * pb[0]) / 3.0, (pa[1] + 2.0 * pb[1]) / 3.0)
        return p1, p2

    def remove_elements(self, patch: Sequence[int]) -> None:
        """Remove tagged cells."""
        to_remove = set(patch)
        self.quads = [quad for i, quad in enumerate(self.quads) if i not in to_remove]
        to_keep = set(i for quad in self.quads for i in quad)
        self.points = {k: v for k, v in self.points.items() if k in to_keep}

    def refine_patch(self, patch: Sequence[int]) -> None:
        """Refine tagged cells and transition neighbors with zero hanging nodes."""
        to_refine = set(patch)

        # 1. Close refinement set for cells with >= 3 split
        # edges or opposite split edges
        changed = True
        while changed:
            changed = False
            split_set: set[Edge] = set()
            for c_idx in to_refine:
                q = self.quads[c_idx]
                for j in range(4):
                    split_set.add(_canonical_edge(q[j], q[(j + 1) % 4]))

            for idx, q in enumerate(self.quads):
                if idx in to_refine:
                    continue
                s = [
                    _canonical_edge(q[j], q[(j + 1) % 4]) in split_set for j in range(4)
                ]
                num_s = sum(s)
                is_opposite = num_s == 2 and (s[0] == s[2])
                if num_s >= 3 or is_opposite:
                    to_refine.add(idx)
                    changed = True

        # 2. Allocate trisection points for every split edge (canonical a < b order)
        split_edges: dict[Edge, tuple[int, int]] = {}
        for c_idx in to_refine:
            q = self.quads[c_idx]
            for j in range(4):
                edge = _canonical_edge(q[j], q[(j + 1) % 4])
                if edge not in split_edges:
                    split_edges[edge] = self.get_edge_trisection(edge[0], edge[1])

        def oriented_edge_nodes(a: int, b: int) -> tuple[int, int]:
            m1, m2 = split_edges[_canonical_edge(a, b)]
            return (m1, m2) if a < b else (m2, m1)

        new_quads: list[Quad] = []

        # 3. Rebuild elements
        for idx, (n0, n1, n2, n3) in enumerate(self.quads):
            if idx in to_refine:
                # 3x3 uniform subdivision (9 sub-quads)
                p0, p1, p2, p3 = (
                    self.points[n0],
                    self.points[n1],
                    self.points[n2],
                    self.points[n3],
                )
                b1, b2 = oriented_edge_nodes(n0, n1)
                r1, r2 = oriented_edge_nodes(n1, n2)
                t2, t1 = oriented_edge_nodes(n2, n3)
                l2, l1 = oriented_edge_nodes(n3, n0)

                c11 = self.add_point(*_interp(p0, p1, p2, p3, 1.0 / 3.0, 1.0 / 3.0))
                c21 = self.add_point(*_interp(p0, p1, p2, p3, 2.0 / 3.0, 1.0 / 3.0))
                c22 = self.add_point(*_interp(p0, p1, p2, p3, 2.0 / 3.0, 2.0 / 3.0))
                c12 = self.add_point(*_interp(p0, p1, p2, p3, 1.0 / 3.0, 2.0 / 3.0))

                g = [
                    [n0, b1, b2, n1],
                    [l1, c11, c21, r1],
                    [l2, c12, c22, r2],
                    [n3, t1, t2, n2],
                ]
                for i in range(3):
                    for j in range(3):
                        new_quads.append(
                            (g[i][j], g[i][j + 1], g[i + 1][j + 1], g[i + 1][j])
                        )
                continue

            nodes = [n0, n1, n2, n3]
            edges_split = [
                _canonical_edge(nodes[j], nodes[(j + 1) % 4]) in split_edges
                for j in range(4)
            ]
            num_splits = sum(edges_split)

            if num_splits == 0:
                new_quads.append((n0, n1, n2, n3))

            elif num_splits == 1:
                # 1-edge transition template (4 quads, matching figure b)
                rot = edges_split.index(True)
                rn0, rn1, rn2, rn3 = (nodes[(k + rot) % 4] for k in range(4))
                ma, mb = oriented_edge_nodes(rn0, rn1)
                pr0, pr1, pr2, pr3 = (
                    self.points[rn0],
                    self.points[rn1],
                    self.points[rn2],
                    self.points[rn3],
                )
                c0 = self.add_point(*_interp(pr0, pr1, pr2, pr3, 1.0 / 3.0, 0.5))
                c1 = self.add_point(*_interp(pr0, pr1, pr2, pr3, 2.0 / 3.0, 0.5))

                new_quads.extend(
                    [
                        (rn0, ma, c0, rn3),
                        (ma, mb, c1, c0),
                        (mb, rn1, rn2, c1),
                        (c0, c1, rn2, rn3),
                    ]
                )

            elif num_splits == 2:
                # 2-adjacent-edge corner transition template (6 strictly convex quads)
                # Rotate so split edges are edge 0 (rn0->rn1) and edge 3 (rn3->rn0)
                if edges_split[0] and edges_split[3]:
                    rot = 0
                elif edges_split[0] and edges_split[1]:
                    rot = 1
                elif edges_split[1] and edges_split[2]:
                    rot = 2
                else:
                    rot = 3

                rn0, rn1, rn2, rn3 = (nodes[(k + rot) % 4] for k in range(4))
                b1, b2 = oriented_edge_nodes(rn0, rn1)
                l2, l1 = oriented_edge_nodes(rn3, rn0)
                pr0, pr1, pr2, pr3 = (
                    self.points[rn0],
                    self.points[rn1],
                    self.points[rn2],
                    self.points[rn3],
                )

                c11 = self.add_point(*_interp(pr0, pr1, pr2, pr3, 1.0 / 3.0, 1.0 / 3.0))
                c21 = self.add_point(*_interp(pr0, pr1, pr2, pr3, 2.0 / 3.0, 1.0 / 3.0))
                c12 = self.add_point(*_interp(pr0, pr1, pr2, pr3, 1.0 / 3.0, 2.0 / 3.0))

                new_quads.extend(
                    [
                        (rn0, b1, c11, l1),
                        (b1, b2, c21, c11),
                        (l1, c11, c12, l2),
                        (c11, c21, rn2, c12),
                        (b2, rn1, rn2, c21),
                        (l2, c12, rn2, rn3),
                    ]
                )

        self.quads = new_quads

    def to_solver_arrays(self) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        """Return contiguous Q4 arrays ready for Mesh.from_q4_mesh()."""
        remap: dict[int, int] = {}
        dense_points: list[Point] = []
        for new_idx, (old_id, pt) in enumerate(self.points.items()):
            remap[old_id] = new_idx
            dense_points.append(pt)

        dense_quads = [
            [remap[n0], remap[n1], remap[n2], remap[n3]]
            for n0, n1, n2, n3 in self.quads
        ]
        return (
            np.ascontiguousarray(dense_quads, dtype=np.int32),
            np.ascontiguousarray(dense_points, dtype=np.float64),
        )
