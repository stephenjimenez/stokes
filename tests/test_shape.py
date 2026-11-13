"""Tests for Q8 serendipity and Q4 bilinear shape functions and quadrature."""

import numpy as np

from stokes.fe.shape import (
    Q9_GAUSS_POINTS,
    Q9_GAUSS_WEIGHTS,
    q4_shape_functions,
    q4_shape_gradients,
    q8_shape_functions,
    q8_shape_gradients,
)


def test_gauss_quadrature_weights() -> None:
    """Verify Q9 Gauss weights sum to reference element area (4.0)."""
    assert np.isclose(sum(Q9_GAUSS_WEIGHTS), 4.0)
    assert len(Q9_GAUSS_POINTS) == 9
    assert len(Q9_GAUSS_WEIGHTS) == 9


def test_q4_partition_of_unity() -> None:
    """Verify Q4 shape functions sum to 1 everywhere on the reference element."""
    test_points = [
        (-1.0, -1.0),
        (0.0, 0.0),
        (0.3, -0.7),
        (1.0, 1.0),
        (-0.5, 0.5),
    ]
    for xi, eta in test_points:
        n = q4_shape_functions(xi, eta)
        assert np.isclose(np.sum(n), 1.0)


def test_q4_kronecker_delta() -> None:
    """Verify Q4 shape functions satisfy Kronecker delta at corner nodes."""
    nodes = [
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    ]
    for i, (xi, eta) in enumerate(nodes):
        n = q4_shape_functions(xi, eta)
        expected = np.zeros(4)
        expected[i] = 1.0
        assert np.allclose(n, expected)


def test_q4_gradients_constant_field() -> None:
    """Gradients of shape functions summed must equal zero (gradient of constant 1)."""
    for xi, eta in [(-0.3, 0.2), (0.0, 0.0), (0.7, -0.4)]:
        dn = q4_shape_gradients(xi, eta)
        assert np.allclose(np.sum(dn, axis=1), [0.0, 0.0])


def test_q8_partition_of_unity() -> None:
    """Verify Q8 shape functions sum to 1 everywhere on reference element."""
    sample_points = [
        (0.0, 0.0),
        (0.25, -0.35),
        (-0.8, 0.4),
        (0.5, 0.5),
        (-0.1, -0.9),
    ]
    for xi, eta in sample_points:
        n = q8_shape_functions(xi, eta)
        assert np.isclose(np.sum(n), 1.0)


def test_q8_kronecker_delta() -> None:
    """Verify Q8 shape functions satisfy Kronecker delta at all 8 nodes."""
    nodes = [
        (-1.0, -1.0),  # 0: bottom-left
        (1.0, -1.0),  # 1: bottom-right
        (1.0, 1.0),  # 2: top-right
        (-1.0, 1.0),  # 3: top-left
        (0.0, -1.0),  # 4: bottom-mid
        (1.0, 0.0),  # 5: right-mid
        (0.0, 1.0),  # 6: top-mid
        (-1.0, 0.0),  # 7: left-mid
    ]
    for i, (xi, eta) in enumerate(nodes):
        n = q8_shape_functions(xi, eta)
        expected = np.zeros(8)
        expected[i] = 1.0
        assert np.allclose(n, expected, atol=1e-12)


def test_q8_gradients_constant_field() -> None:
    """Gradients of Q8 shape functions summed must equal zero."""
    for xi, eta in [(-0.5, 0.3), (0.0, 0.0), (0.6, -0.2)]:
        dn = q8_shape_gradients(xi, eta)
        assert np.allclose(np.sum(dn, axis=1), [0.0, 0.0])
