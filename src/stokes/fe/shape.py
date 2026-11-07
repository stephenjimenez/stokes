"""Tools for finite element shape functions."""

import numpy as np
from numpy.typing import NDArray


def q4_shape_functions(xi: float, eta: float) -> NDArray[np.float64]:
    """Shape functions for a 4-node Lagrange (Q4) element."""
    return np.array(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )


def q4_shape_gradients(xi: float, eta: float) -> NDArray[np.float64]:
    """Shape function gradients for a 4-node Lagrange (Q4) element."""
    return np.array(
        [
            [
                -0.25 * (1.0 - eta),
                +0.25 * (1.0 - eta),
                +0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta),
            ],
            [
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                +0.25 * (1.0 + xi),
                +0.25 * (1.0 - xi),
            ],
        ],
        dtype=np.float64,
    )


def q8_shape_functions(xi: float, eta: float) -> NDArray[np.float64]:
    """Shape functions for an 8-node serendipity (Q8) element."""
    return np.array(
        [
            0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0),
            0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0),
            0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0),
            0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0),
            0.5 * (1.0 - xi * xi) * (1.0 - eta),
            0.5 * (1.0 + xi) * (1.0 - eta * eta),
            0.5 * (1.0 - xi * xi) * (1.0 + eta),
            0.5 * (1.0 - xi) * (1.0 - eta * eta),
        ],
        dtype=np.float64,
    )


def q8_shape_gradients(xi: float, eta: float) -> NDArray[np.float64]:
    """Shape function gradients for an 8-node serendipity (Q8) element."""
    return np.array(
        [
            [
                0.25 * (1.0 - eta) * (2.0 * xi + eta),
                0.25 * (1.0 - eta) * (2.0 * xi - eta),
                0.25 * (1.0 + eta) * (2.0 * xi + eta),
                0.25 * (1.0 + eta) * (2.0 * xi - eta),
                -xi * (1.0 - eta),
                0.5 * (1.0 - eta * eta),
                -xi * (1.0 + eta),
                -0.5 * (1.0 - eta * eta),
            ],
            [
                0.25 * (1.0 - xi) * (xi + 2.0 * eta),
                0.25 * (1.0 + xi) * (-xi + 2.0 * eta),
                0.25 * (1.0 + xi) * (xi + 2.0 * eta),
                0.25 * (1.0 - xi) * (-xi + 2.0 * eta),
                -0.5 * (1.0 - xi * xi),
                -eta * (1.0 + xi),
                0.5 * (1.0 - xi * xi),
                -eta * (1.0 - xi),
            ],
        ],
        dtype=np.float64,
    )
