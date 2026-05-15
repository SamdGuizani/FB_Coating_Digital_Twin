"""
Drying stage model.

Spray is stopped; hot air continues to remove residual acetone from the coated particles.
The coating mass is fixed at its end-of-spraying value.

State vector: [Y_p, Y_g, T_p]
    Y_p : particle acetone content  [kg_acetone / kg_dry_particle]
    Y_g : gas acetone content       [kg_acetone / kg_dry_air]
    T_p : particle temperature      [K]

Gas temperature T_g is again quasi-steady (effectiveness-NTU).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..parameters import ProcessParameters
from ..transfer import calc_heat_transfer, gas_temperature_qs
from ..drying import calc_drying_rate

STAGE = 2  # drying stage index


def _rhs(t: float, state: list[float], params: ProcessParameters) -> list[float]:
    Y_p, Y_g, T_p = state

    Y_p = max(Y_p, 0.0)
    Y_g = max(Y_g, 0.0)

    m_air = params.stage_air_flow(STAGE)
    Y_g_in = params.air_inlet_moisture[STAGE]

    T_g = gas_temperature_qs(params, STAGE, T_p)
    alpha_h = calc_heat_transfer(params, STAGE)
    R_D = calc_drying_rate(params, Y_p, Y_g, T_p, T_g, STAGE)

    # Particle acetone balance (evaporation only, no spray)
    dYp_dt = -R_D

    # Gas acetone balance
    R_D_total = R_D * params.batch_size
    tau = params.gas_residence_time
    dYg_dt = (R_D_total - m_air * (Y_g - Y_g_in)) / (m_air * tau)

    # Particle energy balance
    Q_conv = alpha_h * params.total_surface * (T_g - T_p)
    Q_evap = R_D * params.batch_size * params.latent_heat_acetone
    cp_eff = params.cp_particle + Y_p * params.cp_acetone_liquid
    dTp_dt = (Q_conv - Q_evap) / (params.batch_size * cp_eff)

    return [dYp_dt, dYg_dt, dTp_dt]


@dataclass
class DryingResult:
    t: np.ndarray            # s
    Y_particle: np.ndarray   # kg_acetone / kg_particle
    Y_gas: np.ndarray        # kg_acetone / kg_dry_air
    T_particle: np.ndarray   # K
    T_gas: np.ndarray        # K (quasi-steady)
    success: bool
    message: str


def run_drying(
    params: ProcessParameters,
    duration: float,
    Y_particle_init: float,
    Y_gas_init: float = 0.0,
    T_particle_init: float = 333.0,
    t_eval: np.ndarray | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> DryingResult:
    """
    Integrate the drying stage ODE.

    Initial conditions are taken from the end of the spraying result.

    Parameters
    ----------
    params : ProcessParameters
    duration : float
        Drying duration [s].
    Y_particle_init : float
        Residual acetone in particles at end of spraying [kg/kg].
    Y_gas_init : float
        Gas acetone content at start of drying [kg/kg].
    T_particle_init : float
        Particle temperature at start of drying [K].
    """
    if t_eval is None:
        t_eval = np.linspace(0, duration, 300)

    sol = solve_ivp(
        _rhs,
        t_span=(0.0, duration),
        y0=[Y_particle_init, Y_gas_init, T_particle_init],
        args=(params,),
        method="LSODA",   # Y_g has τ=0.5 s vs minutes-long simulation → stiff
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    T_p = sol.y[2]
    T_g = np.array([gas_temperature_qs(params, STAGE, tp) for tp in T_p])

    return DryingResult(
        t=sol.t,
        Y_particle=np.maximum(sol.y[0], 0.0),
        Y_gas=np.maximum(sol.y[1], 0.0),
        T_particle=T_p,
        T_gas=T_g,
        success=sol.success,
        message=sol.message,
    )
