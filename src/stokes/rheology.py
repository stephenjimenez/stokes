"""Tools for rheological relationships."""

from __future__ import annotations

from math import exp

import numpy as np
from numpy.typing import NDArray

from stokes.config import SimulationConfig
from stokes.fe import shape


def l2_projection(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Perform an L2 projection to get Gauss point data at nodes."""

    # perform lumped mass L2 projection
    b_glen = glen_coefficient(config.rheology.temperature)
    b = np.empty((3, 8), dtype=np.float64)
    dofs = config.mesh.cell_dofs()
    epsilon = np.zeros((config.mesh.num_points, 3), dtype=np.float64)
    sigma = np.zeros((config.mesh.num_points, 3), dtype=np.float64)
    tau = np.zeros((config.mesh.num_points, 2), dtype=np.float64)
    weights = np.zeros(config.mesh.num_points, dtype=np.float64)
    for i, cell in enumerate(config.mesh.cells):
        quad = cell[:4]
        for (xi, eta), w in zip(shape.Q9_GAUSS_POINTS, shape.Q9_GAUSS_WEIGHTS):
            n4 = shape.q4_shape_functions(xi, eta)
            dndxi = shape.q4_shape_gradients(xi, eta)
            jac = dndxi @ config.mesh.points[quad]
            w_det = w * np.linalg.det(jac)
            dndx = np.linalg.solve(jac, dndxi)
            b[0, 0::2] = b[2, 1::2] = dndx[0]
            b[1, 1::2] = b[2, 0::2] = dndx[1]
            b[1, 0::2] = b[0, 1::2] = 0.0
            d = b @ uvp[dofs[i, :8]]
            mu = glen_viscosity(b_glen=b_glen, d=d)
            t11 = 2.0 * mu * d[0]
            t22 = 2.0 * mu * d[1]
            t12 = mu * d[2]
            p = n4 @ uvp[dofs[i, 16:]]
            s11 = t11 - p
            s22 = t22 - p
            epsilon[quad, 0] += n4 * d[0] * w_det
            epsilon[quad, 1] += n4 * d[1] * w_det
            epsilon[quad, 2] += n4 * 0.5 * d[2] * w_det
            sigma[quad, 0] += n4 * s11 * w_det
            sigma[quad, 1] += n4 * s22 * w_det
            sigma[quad, 2] += n4 * t12 * w_det
            tau[quad, 0] += n4 * t11 * w_det
            tau[quad, 1] += n4 * t22 * w_det
            weights[quad] += n4 * w_det
    epsilon /= np.fmax(weights[:, None], np.finfo(np.float64).eps)
    sigma /= np.fmax(weights[:, None], np.finfo(np.float64).eps)
    tau /= np.fmax(weights[:, None], np.finfo(np.float64).eps)

    # compute average stress at cell edges
    for i in range(4):
        v1 = config.mesh.cells[:, i]
        v2 = config.mesh.cells[:, (i + 1) % 4]
        edge = config.mesh.cells[:, 4 + i]
        epsilon[edge] = 0.5 * (epsilon[v1] + epsilon[v2])
        sigma[edge] = 0.5 * (sigma[v1] + sigma[v2])
        tau[edge] = 0.5 * (tau[v1] + tau[v2])

    return epsilon, sigma, tau


def glen_coefficient(temperature: float = 263.15) -> float:
    """Apply Glen's law to compute the viscosity for ice."""

    B0 = 2.207  # Pa * year^(1/3)
    C = 0.16612  # Kelvin^K
    K = 1.17
    T0 = 3155.0  # Kelvin
    T_REF = 273.39  # Kelvin

    return B0 * exp(T0 / temperature - C / (T_REF - temperature) ** K)


def glen_viscosity(b_glen: float, d: NDArray[np.float64]) -> float:
    """Compute the Viscosity using Glen's law."""
    deq2 = 0.5 * (d[0] * d[0] + d[1] * d[1] + 0.5 * d[2] * d[2])
    return 0.5 * b_glen / np.cbrt(deq2)
