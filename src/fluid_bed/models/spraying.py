"""
Spraying stage model.

Spray solution (acetone + coating polymer) is atomised onto the heated particle bed.
Acetone evaporates; coating (e.g. ethylcellulose) deposits on particle surfaces.

State vector: [Y_p, Y_g, M_c, T_p]
    Y_p : particle acetone content  [kg_acetone / kg_dry_particle]
    Y_g : gas acetone content       [kg_acetone / kg_dry_air]
    M_c : cumulative coating mass   [kg]
    T_p : particle temperature      [K]

Gas temperature T_g is resolved quasi-steadily (effectiveness-NTU),
making T_g an algebraic output rather than an ODE state.

Gas moisture Y_g is modelled with a fast first-order dynamic:
    dY_g/dt = (R_D·batch_size − ṁ_air·(Y_g − Y_g_in)) / (ṁ_air·τ_gas)
so it tracks its quasi-steady value with time constant τ_gas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..parameters import ProcessParameters
from ..transfer import calc_heat_transfer, gas_temperature_qs
from ..drying import calc_drying_rate

STAGE = 1  # spraying stage index


def _rhs(t: float, state: list[float], params: ProcessParameters) -> list[float]:
    Y_p, Y_g, M_c, T_p = state

    # Clamp to physically valid range
    Y_p = max(Y_p, 0.0)
    Y_g = max(Y_g, 0.0)

    m_air = params.stage_air_flow(STAGE)
    Y_g_in = params.air_inlet_moisture[STAGE]
    T_g_in = params.air_temperatures[STAGE]

    # Quasi-steady gas temperature
    T_g = gas_temperature_qs(params, STAGE, T_p)

    # Transfer coefficients and drying rate
    alpha_h = calc_heat_transfer(params, STAGE)
    R_D = calc_drying_rate(params, Y_p, Y_g, T_p, T_g, STAGE)  # [kg_ac / (kg_p · s)]

    # Spray components
    spray_rate_dm = params.spray_rate * params.dry_matter_conc           # kg DM/s
    spray_rate_solvent = params.spray_rate * (1.0 - params.dry_matter_conc)  # kg solvent/s

    # ── ODEs ────────────────────────────────────────────────────────────────
    # Particle acetone balance
    dYp_dt = spray_rate_solvent / params.batch_size - R_D

    # Gas acetone balance (fast relaxation to quasi-steady)
    R_D_total = R_D * params.batch_size  # [kg_ac/s], total evaporation from bed
    tau = params.gas_residence_time
    dYg_dt = (R_D_total - m_air * (Y_g - Y_g_in)) / (m_air * tau)

    # Coating mass deposition (all dry matter deposits; efficiency = 1 simplification)
    dMc_dt = spray_rate_dm

    # Particle energy balance
    Q_conv = alpha_h * params.total_surface * (T_g - T_p)
    Q_evap = R_D * params.batch_size * params.latent_heat_acetone
    cp_eff = params.cp_particle + Y_p * params.cp_acetone_liquid
    dTp_dt = (Q_conv - Q_evap) / (params.batch_size * cp_eff)

    return [dYp_dt, dYg_dt, dMc_dt, dTp_dt]


@dataclass
class SprayingResult:
    t: np.ndarray            # s
    Y_particle: np.ndarray   # kg_acetone / kg_particle
    Y_gas: np.ndarray        # kg_acetone / kg_dry_air
    M_coating: np.ndarray    # kg, cumulative coating mass
    T_particle: np.ndarray   # K
    T_gas: np.ndarray        # K (quasi-steady)
    success: bool
    message: str


def run_spraying(
    params: ProcessParameters,
    duration: float,
    Y_particle_init: float = 0.0,
    Y_gas_init: float = 0.0,
    M_coating_init: float = 0.0,
    T_particle_init: float = 333.0,
    t_eval: np.ndarray | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> SprayingResult:
    """
    Integrate the spraying stage ODE.

    Initial conditions are typically taken from the end of pre-heating.

    Parameters
    ----------
    params : ProcessParameters
    duration : float
        Spraying duration [s].  Computed as: Qty_solution [kg] / spray_rate [kg/s].
    Y_particle_init : float
        Initial acetone content [kg/kg]. Usually 0 at start of spraying.
    Y_gas_init : float
        Initial gas moisture [kg/kg]. Usually ~0 (dry air).
    M_coating_init : float
        Pre-existing coating mass [kg]. Usually 0.
    T_particle_init : float
        Particle temperature at spraying start [K], from pre-heating result.
    """
    if t_eval is None:
        t_eval = np.linspace(0, duration, 500)

    sol = solve_ivp(
        _rhs,
        t_span=(0.0, duration),
        y0=[Y_particle_init, Y_gas_init, M_coating_init, T_particle_init],
        args=(params,),
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    T_p = sol.y[3]
    T_g = np.array([gas_temperature_qs(params, STAGE, tp) for tp in T_p])

    return SprayingResult(
        t=sol.t,
        Y_particle=np.maximum(sol.y[0], 0.0),
        Y_gas=np.maximum(sol.y[1], 0.0),
        M_coating=sol.y[2],
        T_particle=T_p,
        T_gas=T_g,
        success=sol.success,
        message=sol.message,
    )
