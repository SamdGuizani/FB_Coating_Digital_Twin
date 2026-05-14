"""
Heat and mass transfer coefficient calculations.

Both use the Ranz-Marshall correlation, which is the Nusselt/Sherwood analogy
for a single sphere in a fluid stream — identical to the MATLAB implementation.
"""

from __future__ import annotations

import numpy as np

from .parameters import ProcessParameters


def _reynolds(params: ProcessParameters, stage: int) -> float:
    """
    Particle Reynolds number Re_p = ρ_air · u · d_eq / μ_air.

    The air velocity is derived from the mass flow rate and bed cross-section.
    For stage 1 (spraying), atomisation air is added to process air, matching MATLAB.
    """
    m_air = params.stage_air_flow(stage)
    velocity = m_air / (params.cross_section_bed * params.rho_air)
    return params.rho_air * velocity * params.diameter_eq / params.viscosity_air


def _ranz_marshall(Re: float, dimensionless_group: float) -> float:
    """
    Ranz-Marshall correlation.

    Nu (or Sh) = 2 + DG^(2/5) · (0.43·Re^0.5 + 0.06·Re^(2/3))

    where DG is either Prandtl (heat) or Schmidt (mass).
    """
    return 2.0 + dimensionless_group ** 0.4 * (0.43 * Re ** 0.5 + 0.06 * Re ** (2 / 3))


def calc_heat_transfer(params: ProcessParameters, stage: int) -> float:
    """
    Heat transfer coefficient α_h [W/m²/K] via Nusselt (Ranz-Marshall).

    α_h = Nu · λ_air / d_eq
    """
    Re = _reynolds(params, stage)
    Pr = params.cp_air * params.viscosity_air / params.conductivity_air
    Nu = _ranz_marshall(Re, Pr)
    return Nu * params.conductivity_air / params.diameter_eq


def calc_mass_transfer(params: ProcessParameters, stage: int) -> float:
    """
    Mass transfer coefficient α_m [m/s] via Sherwood (Ranz-Marshall).

    α_m = Sh · D_acetone / d_eq
    """
    Re = _reynolds(params, stage)
    Sc = params.viscosity_air / (params.rho_air * params.diffusivity_acetone)
    Sh = _ranz_marshall(Re, Sc)
    return Sh * params.diffusivity_acetone / params.diameter_eq


def gas_temperature_qs(params: ProcessParameters, stage: int, T_particle: float) -> float:
    """
    Quasi-steady gas temperature in the bed [K] via effectiveness-NTU method.

    Assumes gas residence time ≪ particle heating time constant, so the gas
    reaches a steady heat-exchange profile instantaneously.

        NTU = α_h · A_total / (ṁ_air · cp_air)
        T_g = (T_g_in + NTU · T_p) / (1 + NTU)
    """
    alpha_h = calc_heat_transfer(params, stage)
    m_air = params.stage_air_flow(stage)
    if m_air <= 0:
        return params.air_temperatures[stage]
    NTU = alpha_h * params.total_surface / (m_air * params.cp_air)
    T_g_in = params.air_temperatures[stage]
    return (T_g_in + NTU * T_particle) / (1.0 + NTU)


def convective_heat_flux(params: ProcessParameters, stage: int,
                         T_particle: float, T_gas: float) -> float:
    """
    Total convective heat flow from gas to particles [W].

    Q_conv = α_h · A_total · (T_g − T_p)
    """
    alpha_h = calc_heat_transfer(params, stage)
    return alpha_h * params.total_surface * (T_gas - T_particle)
