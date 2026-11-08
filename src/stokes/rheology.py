"""Tools for rheological relationships."""

from __future__ import annotations

from math import exp

import numpy as np
from numpy.typing import NDArray

from stokes._core import eval_strainrate
from stokes.config import SimulationConfig
from stokes.fe import shape


def interpolate_stress(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
    xi: float,
    eta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute strain rate, deviatoric stress, and Cauchy stress at a Gauss point."""

    # ensure arrays match the exact types and C-contiguity expected by PyArray_DATA
    cells = np.ascontiguousarray(config.mesh.cells, dtype=np.int32)
    dofs = np.ascontiguousarray(config.mesh.cell_dofs(), dtype=np.int32)
    x = np.ascontiguousarray(config.mesh.points, dtype=np.float64)
    u = np.ascontiguousarray(uvp, dtype=np.float64)

    # compute strain rates
    epsilon = np.empty((config.mesh.num_cells, 3), dtype=np.float64)
    eval_strainrate(cells, dofs, x, u, xi, eta, epsilon)

    # compute deviatoric stress
    b_glen = glen_coefficient(config.rheology.temperature)
    mu = glen_viscosity(b_glen=b_glen, d=epsilon)
    t11 = 2.0 * mu * epsilon[:, 0]
    t22 = 2.0 * mu * epsilon[:, 1]
    t12 = mu * epsilon[:, 2]
    tau = np.column_stack([t11, t22, t12])

    # compute Cauchy stress
    n = shape.q4_shape_functions(xi, eta)
    p = uvp[dofs[:, 16:]] @ n
    s11 = t11 - p
    s22 = t22 - p
    sigma = np.column_stack([s11, s22, t12])

    return epsilon, tau, sigma


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
    if len(d.shape) == 1:
        e11 = d[0]
        e22 = d[1]
        g12 = d[2]
        deq2 = 0.5 * (d[0] * d[0] + d[1] * d[1] + 0.5 * d[2] * d[2])
    else:
        e11 = d[..., 0]
        e22 = d[..., 1]
        g12 = d[..., 2]
    deq2 = 0.5 * (e11 * e11 + e22 * e22 + 0.5 * g12 * g12)
    return 0.5 * b_glen / np.cbrt(deq2)
