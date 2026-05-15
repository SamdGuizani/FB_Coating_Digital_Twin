"""
Drying stage model.

Spray is stopped; hot air removes residual acetone and particle-particle
collisions cause attrition of the freshly deposited coating layer.

State vector: [Y_p, Y_g, Mc, T_p]
    Y_p : particle acetone content  [kg_acetone / kg_dry_particle]
    Y_g : gas acetone content       [kg_acetone / kg_dry_air]
    Mc  : total coating mass in bed [kg]
    T_p : particle temperature      [K]

Gas temperature T_g is quasi-steady (effectiveness-NTU).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..parameters import ProcessParameters
from ..transfer import calc_heat_transfer, gas_temperature_qs
from ..drying import calc_drying_rate

STAGE = 2  # drying stage index


# ── Empirical correlation for r_drying ────────────────────────────────────────

def calc_r_drying(
    spray_rate_g_min: float,
    ec_level_pct: float,
    batch_size_kg: float,
    drying_time_min: float,
    ec_conc_pct: float,
    inlet_humidity_g_kg: float,
) -> float:
    """
    Empirical DoE correlation for the drying-stage coating attrition constant
    r_drying [kg/s], valid for order-1 attrition (n_drying = 1).

    dMc/dt = -r_drying · (Mc / batch_size)

    Parameters
    ----------
    spray_rate_g_min    : spray rate during the preceding coating step [g/min]
    ec_level_pct        : coating weight gain at end of spraying [%]
    batch_size_kg       : initial batch mass [kg]
    drying_time_min     : drying duration [min]
    ec_conc_pct         : ethylcellulose concentration in spray solution [%]
    inlet_humidity_g_kg : inlet air absolute humidity [g/kg dry air]

    Returns
    -------
    r_drying : float [kg/s]
    """
    sr  = spray_rate_g_min
    ecl = ec_level_pct
    bs  = batch_size_kg
    dt  = drying_time_min
    ecc = ec_conc_pct
    h   = inlet_humidity_g_kg

    val = (
        11.64667539
        + 0.06534760834226  * sr
        + 8.2738161258148   * ecl
        - 1.2288413170698   * bs
        - 0.27428071716161  * dt
        + 12.15084603659    * ecc
        + 4.7217504741835   * h
        - 0.053691042425364 * sr  * ecl
        + 0.006471800809636 * sr  * dt
        - 0.12232718407282  * sr  * ecc
    )
    return float(val ** -2)


# ── ODE right-hand side ────────────────────────────────────────────────────────

def _rhs(t: float, state: list[float], params: ProcessParameters) -> list[float]:
    Y_p, Y_g, Mc, T_p = state

    Y_p = max(Y_p, 0.0)
    Y_g = max(Y_g, 0.0)
    Mc  = max(Mc,  0.0)

    m_air   = params.stage_air_flow(STAGE)
    Y_g_in  = params.air_inlet_moisture[STAGE]

    T_g     = gas_temperature_qs(params, STAGE, T_p)
    alpha_h = calc_heat_transfer(params, STAGE)
    R_D     = calc_drying_rate(params, Y_p, Y_g, T_p, T_g, STAGE)

    # Particle acetone balance (evaporation only)
    dYp_dt = -R_D

    # Gas acetone balance
    R_D_total = R_D * params.batch_size
    tau = params.gas_residence_time
    dYg_dt = (R_D_total - m_air * (Y_g - Y_g_in)) / (m_air * tau)

    # Coating attrition:  dMc/dt = -r_drying · (Mc/batch)^n_drying
    Mc_norm = Mc / params.batch_size
    dMc_dt  = -params.r_drying * (Mc_norm ** params.n_drying)

    # Particle energy balance
    Q_conv  = alpha_h * params.total_surface * (T_g - T_p)
    Q_evap  = R_D * params.batch_size * params.latent_heat_acetone
    cp_eff  = params.cp_particle + Y_p * params.cp_acetone_liquid
    dTp_dt  = (Q_conv - Q_evap) / (params.batch_size * cp_eff)

    return [dYp_dt, dYg_dt, dMc_dt, dTp_dt]


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class DryingResult:
    t: np.ndarray            # s
    Y_particle: np.ndarray   # kg_acetone / kg_particle
    Y_gas: np.ndarray        # kg_acetone / kg_dry_air
    M_coating: np.ndarray    # kg, coating mass remaining in bed
    T_particle: np.ndarray   # K
    T_gas: np.ndarray        # K (quasi-steady)
    success: bool
    message: str


# ── Public API ─────────────────────────────────────────────────────────────────

def run_drying(
    params: ProcessParameters,
    duration: float,
    Y_particle_init: float,
    Y_gas_init: float = 0.0,
    M_coating_init: float = 0.0,
    T_particle_init: float = 333.0,
    t_eval: np.ndarray | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> DryingResult:
    """
    Integrate the drying stage ODE.

    Parameters
    ----------
    params : ProcessParameters
    duration : float
        Drying duration [s].
    Y_particle_init : float
        Residual acetone at end of spraying [kg/kg].
    Y_gas_init : float
        Gas acetone content at start of drying [kg/kg].
    M_coating_init : float
        Coating mass on particles at start of drying [kg].
        Typically taken from SprayingResult.M_coating[-1].
    T_particle_init : float
        Particle temperature at start of drying [K].
    """
    if t_eval is None:
        t_eval = np.linspace(0, duration, 300)

    sol = solve_ivp(
        _rhs,
        t_span=(0.0, duration),
        y0=[Y_particle_init, Y_gas_init, M_coating_init, T_particle_init],
        args=(params,),
        method="BDF",     # equivalent to MATLAB ode15s; handles Y_g stiffness (τ=0.5 s)
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    T_p = sol.y[3]
    T_g = np.array([gas_temperature_qs(params, STAGE, tp) for tp in T_p])

    return DryingResult(
        t=sol.t,
        Y_particle=np.maximum(sol.y[0], 0.0),
        Y_gas=np.maximum(sol.y[1], 0.0),
        M_coating=np.maximum(sol.y[2], 0.0),
        T_particle=T_p,
        T_gas=T_g,
        success=sol.success,
        message=sol.message,
    )
