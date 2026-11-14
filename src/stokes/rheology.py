"""Tools for rheological and material relationships."""

from __future__ import annotations

from math import exp

import numpy as np
from numpy.typing import NDArray

from stokes._core import eval_strainrate
from stokes.config import SimulationConfig
from stokes.fe import shape

MIN_DEQ2 = 1e-14


def deviatoric_stress(
    config: SimulationConfig,
    d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the deviatoric stress components."""
    mu = glen_viscosity(temperature=config.rheology.temperature, d=d)
    tau = np.empty_like(d)
    tau[:, 0] = 2.0 * mu * d[:, 0]
    tau[:, 1] = 2.0 * mu * d[:, 1]
    tau[:, 2] = 2.0 * mu * d[:, 2]
    tau[:, 3] = mu * d[:, 3]
    return tau


def glen_coefficient(temperature: float = 263.15) -> float:
    """Apply Glen's law to compute the viscosity for ice."""
    B0 = 2.207  # Pa * year^(1/3)
    C = 0.16612  # Kelvin^K
    K = 1.17
    T0 = 3155.0  # Kelvin
    T_REF = 273.39  # Kelvin
    return B0 * exp(T0 / temperature - C / (T_REF - temperature) ** K)


def glen_viscosity(temperature: float, d: NDArray[np.float64]) -> float:
    """Compute Glen viscosity."""
    d11 = d[..., 0]
    d22 = d[..., 1]
    d12 = 0.5 * d[..., 3]
    deq2 = 0.5 * (d11 * d11 + d22 * d22) + d12 * d12
    return 0.5 * glen_coefficient(temperature) / np.cbrt(deq2 + MIN_DEQ2)


def hayhurst_stress(
    config: SimulationConfig,
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate Hayhurst driving stress."""
    alpha = config.damage.alpha
    beta = config.damage.beta
    s11 = sigma[..., 0]
    s22 = sigma[..., 1]
    s33 = sigma[..., 2]
    s12 = sigma[..., 3]
    skk_gp = s11 + s22 + s33
    p = -skk_gp / 3.0
    t11 = s11 + p
    t22 = s22 + p
    t12 = s12
    j2 = 0.5 * (t11 * t11 + t22 * t22) + t12 * t12
    mises = np.sqrt(3.0 * j2)
    tmp = 0.5 * (s11 - s22)
    sig1 = np.fmax(0.5 * (s11 + s22) + np.sqrt(tmp * tmp + s12 * s12), s33)
    return alpha * sig1 + beta * mises + (1.0 - alpha - beta) * skk_gp


def interpolate_cauchy_stress(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
    dmg: NDArray[np.float64],
    xi: float = 0.0,
    eta: float = 0.0,
) -> NDArray[np.float64]:
    """Interpolate the true Cauchy stress at isoparametric coordinates (xi, eta)."""
    d = interpolate_strain_rate(config=config, uvp=uvp, xi=xi, eta=eta)
    tau = (1.0 - dmg)[:, None] * deviatoric_stress(config=config, d=d)
    p = shape.q4_shape_functions(xi, eta) @ uvp[config.mesh.dofs[:, 16:]].T
    sigma = np.empty_like(tau)
    sigma[:, 0] = tau[:, 0] - p
    sigma[:, 1] = tau[:, 1] - p
    sigma[:, 2] = tau[:, 2] - p
    sigma[:, 3] = tau[:, 3]

    # zero out the stress in failed elements
    if config.damage is not None:
        sigma[dmg >= config.damage.critical_damage] = 0.0

    return sigma


def interpolate_strain_rate(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
    xi: float = 0.0,
    eta: float = 0.0,
) -> NDArray[np.float64]:
    """Interpolate the strain rate at isoparametric coordinates (xi, eta)."""
    cells = np.ascontiguousarray(config.mesh.cells, dtype=np.int32)
    dofs = np.ascontiguousarray(config.mesh.dofs, dtype=np.int32)
    x = np.ascontiguousarray(config.mesh.points, dtype=np.float64)
    u = np.ascontiguousarray(uvp, dtype=np.float64)
    d = np.empty((config.mesh.num_cells, 4), dtype=np.float64)
    eval_strainrate(cells, dofs, x, u, xi, eta, d)
    return d


def damage_rate(
    config: SimulationConfig,
    chi: NDArray[np.float64],
    dmg0: NDArray[np.float64],
    skk: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Evaluate damage rate dD/dt from Hayhurst driving stress."""
    if config.damage is None:
        return np.zeros_like(dmg0)

    ddmg = np.zeros_like(dmg0)
    active = (dmg0 < config.damage.critical_damage) & (chi > 0.0) & (skk > 0.0)
    if not active.any():
        return ddmg

    chi_eff = chi[active] / (1.0 - dmg0[active])
    chi_eff_mpa = 1e-6 * chi_eff
    skk_mpa = 1e-6 * skk
    if skk is None:
        k_sig = config.damage.k1
    else:
        k_sig = config.damage.k1 + config.damage.k2 * np.maximum(skk_mpa[active], 0.0)

    ddmg[active] = (
        config.damage.b_hat
        * (chi_eff_mpa**config.damage.r)
        / (1.0 - dmg0[active]) ** k_sig
    )
    return ddmg
