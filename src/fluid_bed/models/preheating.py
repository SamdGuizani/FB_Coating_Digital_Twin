"""
Pre-heating stage model.

Hot process air heats the particle bed before spraying begins.
No solvent, no coating — only heat exchange between gas and particles.

State vector: [T_p]
    T_p : particle (product) temperature [K]

Gas temperature is handled quasi-steadily via the effectiveness-NTU method
so the ODE has only one degree of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..parameters import ProcessParameters
from ..transfer import calc_heat_transfer, gas_temperature_qs

STAGE = 0  # pre-heating stage index


def _rhs(t: float, state: list[float], params: ProcessParameters) -> list[float]:
    (T_p,) = state

    T_g = gas_temperature_qs(params, STAGE, T_p)
    alpha_h = calc_heat_transfer(params, STAGE)
    Q_conv = alpha_h * params.total_surface * (T_g - T_p)

    dTp_dt = Q_conv / (params.batch_size * params.cp_particle)
    return [dTp_dt]


@dataclass
class PreheatingResult:
    t: np.ndarray          # s
    T_particle: np.ndarray  # K
    T_gas: np.ndarray       # K (quasi-steady)
    success: bool
    message: str


def run_preheating(
    params: ProcessParameters,
    duration: float,            # s
    T_particle_init: float,     # K, initial particle temperature
    t_eval: np.ndarray | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> PreheatingResult:
    """
    Integrate the pre-heating ODE over `duration` seconds.

    Parameters
    ----------
    params : ProcessParameters
    duration : float
        Total pre-heating time [s].
    T_particle_init : float
        Initial particle temperature [K]. Typically ambient (~293 K).
    t_eval : array-like, optional
        Time points at which to store the solution. Defaults to 200 points.

    Returns
    -------
    PreheatingResult
        Time array and state trajectories.
    """
    if t_eval is None:
        t_eval = np.linspace(0, duration, 200)

    sol = solve_ivp(
        _rhs,
        t_span=(0.0, duration),
        y0=[T_particle_init],
        args=(params,),
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    T_p = sol.y[0]
    T_g = np.array([gas_temperature_qs(params, STAGE, tp) for tp in T_p])

    return PreheatingResult(
        t=sol.t,
        T_particle=T_p,
        T_gas=T_g,
        success=sol.success,
        message=sol.message,
    )
