"""Tools for rheological relationships."""

from __future__ import annotations

from math import exp

import numpy as np
from numpy.typing import NDArray

from stokes.config import SimulationConfig
from stokes.fe import shape


def cauchy_stress(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the Cauchy stresses for each element."""
    b_glen = glen_coefficient(config.rheology.temperature)
    b = np.empty((3, 16), dtype=np.float64)
    dofs = config.mesh.cell_dofs()
    sigma = np.empty((config.mesh.num_cells, 3), dtype=np.float64)
    for i, cell in enumerate(config.mesh.cells):
        dndxi = shape.q8_shape_gradients(0.0, 0.0)
        jac = dndxi @ config.mesh.points[cell]
        dndx = np.linalg.solve(jac, dndxi)
        b[0, 0::2] = b[2, 1::2] = dndx[0]
        b[1, 1::2] = b[2, 0::2] = dndx[1]
        b[1, 0::2] = b[0, 1::2] = 0.0
        d = b @ uvp[dofs[i, :16]]
        mu = glen_viscosity(b_glen=b_glen, d=d)
        t11 = 2.0 * mu * d[0]
        t22 = 2.0 * mu * d[1]
        t12 = mu * d[2]
        p = uvp[dofs[i, 16:]].mean()
        sigma[i] = (t11 - p, t22 - p, t12)
    return sigma


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
