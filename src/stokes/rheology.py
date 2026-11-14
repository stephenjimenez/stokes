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
    return 0.5 * b_glen / np.cbrt(deq2 + MIN_DEQ2)


def hayhurst_driving_stress(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
    dmg0: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate Hayhurst driving stress (in MPa) at the 9 Gauss points."""
    if config.damage is None:
        chi_zeros = np.zeros((config.mesh.num_cells, 9), dtype=np.float64)
        return chi_zeros, np.zeros_like(chi_zeros)

    chi = np.zeros((config.mesh.num_cells, 9), dtype=np.float64)
    skk = np.zeros((config.mesh.num_cells, 9), dtype=np.float64)

    alpha = config.damage.alpha
    beta = config.damage.beta

    for gp in range(9):
        xi, eta = shape.Q9_GAUSS_POINTS[gp]
        _, tau, sigma = interpolate_stress(
            config=config,
            uvp=uvp,
            dmg=dmg0[:, gp],
            xi=xi,
            eta=eta,
        )
        t11 = tau[:, 0]
        t22 = tau[:, 1]
        t12 = tau[:, 2]
        s11 = sigma[:, 0]
        s22 = sigma[:, 1]
        s12 = sigma[:, 2]
        p = t11 - s11
        s33 = -p
        j2 = 0.5 * (t11 * t11 + t22 * t22) + t12 * t12
        mises = np.sqrt(3.0 * j2)
        tmp = 0.5 * (s11 - s22)
        skk_gp = -3.0 * p
        sig1 = np.fmax(0.5 * (s11 + s22) + np.sqrt(tmp * tmp + s12 * s12), s33)
        chi[:, gp] = alpha * sig1 + beta * mises + (1.0 - alpha - beta) * skk_gp
        skk[:, gp] = skk_gp

    return chi, skk


def interpolate_stress(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
    dmg: NDArray[np.float64],
    xi: float = 0.0,
    eta: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate strain rate, true deviatoric stresses, and true Cauchy stresses at
    isoparametric coordinates (xi, eta).
    """
    # evaluate strain rate
    cells = np.ascontiguousarray(config.mesh.cells, dtype=np.int32)
    dofs = np.ascontiguousarray(config.mesh.cell_dofs(), dtype=np.int32)
    x = np.ascontiguousarray(config.mesh.points, dtype=np.float64)
    u = np.ascontiguousarray(uvp, dtype=np.float64)
    d = np.empty((config.mesh.num_cells, 3), dtype=np.float64)
    eval_strainrate(cells, dofs, x, u, xi, eta, d)

    # interpolate true deviatoric stress and pressure
    tau = (1.0 - dmg)[:, None] * deviatoric_stress(config=config, d=d)
    p = shape.q4_shape_functions(xi, eta) @ uvp[dofs[:, 16:]].T

    # zero out the stress in failed elements
    if config.damage is not None:
        i = dmg >= config.damage.critical_damage
        tau[i] = 0.0
        p[i] = 0.0

    # compute true Cauchy stress
    sigma = np.empty_like(tau)
    sigma[:, 0] = tau[:, 0] - p
    sigma[:, 1] = tau[:, 1] - p
    sigma[:, 2] = tau[:, 2]

    return d, tau, sigma


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
