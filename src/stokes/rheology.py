"""Tools for rheological and material relationships."""

from __future__ import annotations

from math import exp

import numpy as np
from numpy.typing import NDArray

from stokes._core import eval_strainrate
from stokes.config import SimulationConfig

MIN_DEQ2 = 1e-14


def deviatoric_stress(
    config: SimulationConfig,
    d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the deviatoric stress components."""
    b_glen = glen_coefficient(config.rheology.temperature)
    mu = glen_viscosity(b_glen=b_glen, d=d)
    tau = np.empty_like(d)
    tau[:, 0] = 2.0 * mu * d[:, 0]
    tau[:, 1] = 2.0 * mu * d[:, 1]
    tau[:, 2] = mu * d[:, 2]
    return tau


def glen_coefficient(temperature: float = 263.15) -> float:
    """Apply Glen's law to compute the viscosity for ice."""
    B0 = 2.207  # Pa * year^(1/3)
    C = 0.16612  # Kelvin^K
    K = 1.17
    T0 = 3155.0  # Kelvin
    T_REF = 273.39  # Kelvin
    return B0 * exp(T0 / temperature - C / (T_REF - temperature) ** K)


def glen_viscosity(b_glen: float, d: NDArray[np.float64]) -> float:
    """Compute Glen viscosity."""
    d11 = d[..., 0]
    d22 = d[..., 1]
    d12 = 0.5 * d[..., 2]
    deq2 = 0.5 * (d11 * d11 + d22 * d22) + d12 * d12
    return 0.5 * b_glen / np.cbrt(np.fmax(deq2, MIN_DEQ2))


def hayhurst_stress(
    config: SimulationConfig,
    sigma: NDArray[np.float64],
    tau: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the Hayhurst stress."""
    alpha = config.damage.alpha
    beta = config.damage.beta
    t11 = tau[:, 0]
    t22 = tau[:, 1]
    t12 = tau[:, 2]
    s11 = sigma[:, 0]
    s22 = sigma[:, 1]
    s12 = sigma[:, 2]
    s33 = s11 - t11
    j2 = 0.5 * (t11 * t11 + t22 * t22) + t12 * t12
    mises = np.sqrt(3.0 * j2)
    tmp = 0.5 * (s11 - s22)
    sig1 = np.fmax(0.5 * (s11 + s22) + np.sqrt(tmp * tmp + s12 * s12), s33)
    skk = s11 + s22 + s33
    return alpha * sig1 + beta * mises + (1.0 - alpha - beta) * skk


def interpolate_stress(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
    dmg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate strain rate, true deviatoric stresses, and true Cauchy stresses at
    cell centroids.
    """
    # evaluate strain rate
    cells = np.ascontiguousarray(config.mesh.cells, dtype=np.int32)
    dofs = np.ascontiguousarray(config.mesh.cell_dofs(), dtype=np.int32)
    x = np.ascontiguousarray(config.mesh.points, dtype=np.float64)
    u = np.ascontiguousarray(uvp, dtype=np.float64)
    d = np.empty((config.mesh.num_cells, 3), dtype=np.float64)
    eval_strainrate(cells, dofs, x, u, 0.0, 0.0, d)

    # interpolate true deviatoric stress
    tau = (1.0 - dmg)[:, None] * deviatoric_stress(config=config, d=d)
    i = dmg >= config.damage.critical_damage
    tau[i] = 0.0

    # interpolate pressure at element centroids
    p = uvp[dofs[:, 16:]].mean(axis=1)
    p[i] = 0.0

    # compute true Cauchy stress
    sigma = np.empty_like(tau)
    sigma[:, 0] = tau[:, 0] - p
    sigma[:, 1] = tau[:, 1] - p
    sigma[:, 2] = tau[:, 2]

    return d, tau, sigma


def local_damage_rate(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
    dmg0: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the local damage rate."""
    _, tau, sigma = interpolate_stress(config=config, uvp=uvp, dmg=dmg0)
    chi = 1e-6 * hayhurst_stress(config=config, sigma=sigma, tau=tau)
    ddmg = np.zeros(config.mesh.num_cells, dtype=np.float64)
    p = tau[:, 0] - sigma[:, 0]
    skk = 1e-6 * (sigma[:, 0] + sigma[:, 1] - p)  # MPa
    i = (skk > 0.0) & (chi > 0.0) & (dmg0 < config.damage.max_damage)
    if i.any():
        chi_eff = chi[i] / (1.0 - dmg0[i])
        k_sig = config.damage.k1 + config.damage.k2 * skk[i]
        ddmg[i] = (
            config.damage.b_hat * (chi_eff**config.damage.r) / (1.0 - dmg0[i]) ** k_sig
        )

    return ddmg
