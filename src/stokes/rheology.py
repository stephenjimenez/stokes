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
    sigma[:, 2] = -p
    sigma[:, 3] = tau[:, 3]
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


def tensile_strain_energy(
    config: SimulationConfig,
    uvp: NDArray[np.float64],
    phi_gp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute virtual tensile elastic strain-energy density psi^+_gp."""
    if config.phase_field is None:
        return np.zeros((config.mesh.num_cells, 9), dtype=np.float64)

    pf = config.phase_field
    e_mod = pf.youngs_modulus
    nu = pf.poissons_ratio
    eta_res = pf.residual_stiffness
    crit_stress = pf.critical_stress
    psi_c = (crit_stress**2) / (2.0 * e_mod) if crit_stress > 0.0 else 0.0

    # g(phi) = (1 - phi)^2 + eta_res
    g_phi = (1.0 - phi_gp) ** 2 + eta_res

    # Q4 pressure nodal values per cell
    p_dofs = config.mesh.dofs[:, 16:]
    p_nodes = uvp[p_dofs]

    psi_plus = np.empty((config.mesh.num_cells, 9), dtype=np.float64)

    for gp, (xi, eta) in enumerate(shape.Q9_GAUSS_POINTS):
        d = interpolate_strain_rate(config=config, uvp=uvp, xi=xi, eta=eta)
        tau_tilde = deviatoric_stress(config=config, d=d)
        tau = g_phi[:, gp, None] * tau_tilde

        n4 = shape.q4_shape_functions(xi, eta)
        p = p_nodes @ n4

        # Degraded Cauchy stress components
        s_xx = tau[:, 0] - p
        s_yy = tau[:, 1] - p
        s_xy = tau[:, 3]
        s_zz = -p

        # 3D plane-strain principal Cauchy stresses
        half_diff = 0.5 * (s_xx - s_yy)
        rad = np.sqrt(half_diff * half_diff + s_xy * s_xy)
        s_avg = 0.5 * (s_xx + s_yy)
        sig1 = s_avg + rad
        sig2 = s_avg - rad
        sig3 = s_zz

        # Tensile Macaulay positive parts <x>+ = max(0, x)
        sig1_pos = np.maximum(0.0, sig1)
        sig2_pos = np.maximum(0.0, sig2)
        sig3_pos = np.maximum(0.0, sig3)

        # Virtual tensile elastic strain-energy density (Pa)
        psi_val = (0.5 / e_mod) * (sig1_pos**2 + sig2_pos**2 + sig3_pos**2) - (
            nu / e_mod
        ) * (sig1_pos * sig2_pos + sig2_pos * sig3_pos + sig3_pos * sig1_pos)

        if psi_c > 0.0:
            psi_val = np.maximum(0.0, psi_val - psi_c)

        psi_plus[:, gp] = psi_val

    return psi_plus
