"""
Drying rate (solvent evaporation) calculation.

Direct port of Calcul_DryingRate.m, which calls Calcul_MassTransfer.m.

The drying rate R_D [kg_acetone / (kg_particle · s)] is derived from:
  - Antoine equation for acetone vapour pressure at the particle surface
  - Partial pressure of acetone in the gas phase
  - Ranz-Marshall mass transfer coefficient
  - Langmuir-type isotherm factor limiting evaporation when particles are nearly dry
"""

from __future__ import annotations

import numpy as np

from .parameters import ProcessParameters
from .transfer import calc_mass_transfer


def antoine_pressure(params: ProcessParameters, T_K: float) -> float:
    """
    Saturated vapour pressure of acetone [Pa] via Antoine equation.

    log10(P [mmHg]) = A − B / (C + T [°C])
    Converted to Pa:  P [Pa] = P [mmHg] × 133.322
    """
    T_C = T_K - 273.15
    log_p_mmhg = params.antoine_A - params.antoine_B / (params.antoine_C + T_C)
    return (10.0 ** log_p_mmhg) * 133.322


def gas_vapor_pressure(params: ProcessParameters, Y_gas: float) -> float:
    """
    Partial pressure of acetone in the gas phase [Pa].

    Derived from the ideal-gas mixture:
        p_ac = P_total · Y_gas / (MW_ac/MW_air + Y_gas)

    where Y_gas [kg_acetone / kg_dry_air].
    """
    ratio = params.mw_acetone / params.mw_air
    return params.pressure * Y_gas / (ratio + Y_gas)


def calc_drying_rate(
    params: ProcessParameters,
    Y_particle: float,   # kg acetone / kg dry particle
    Y_gas: float,        # kg acetone / kg dry air
    T_particle: float,   # K
    T_air: float,        # K (local gas temperature)
    stage: int,
) -> float:
    """
    Specific drying rate R_D [kg_acetone / (kg_particle · s)].

    R_D = isotherm · α_m · A_p · ΔP_vap / (m_p · (R/MW_ac) · T_avg)

    where:
      isotherm  = Y_p / (|Y_p| + α_langmuir)   [Langmuir-type, limits drying near zero moisture]
      α_m       = mass transfer coefficient [m/s]
      A_p       = single-particle surface area [m²]
      ΔP_vap    = P_vap(T_p) − P_vap_gas       [Pa]  (clamped to ≥ 0)
      m_p       = single-particle mass [kg]
      R/MW_ac   = specific gas constant for acetone [J/kg/K]
      T_avg     = 0.5·(T_particle + T_air)      [K]

    Total evaporation rate from the full bed: R_D · batch_size [kg/s].
    """
    alpha_m = calc_mass_transfer(params, stage)

    P_vap_particle = antoine_pressure(params, T_particle)
    P_vap_gas = gas_vapor_pressure(params, Y_gas)
    delta_P = max(P_vap_particle - P_vap_gas, 0.0)  # no condensation

    isotherm = Y_particle / (abs(Y_particle) + params.langmuir_alpha)
    T_avg = 0.5 * (T_particle + T_air)
    R_spec = params.specific_gas_constant_acetone()

    return (
        isotherm
        * alpha_m
        * params.surface_particle
        * delta_P
        / (params.mass_particle * R_spec * T_avg)
    )
