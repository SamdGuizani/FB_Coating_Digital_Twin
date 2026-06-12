"""
Full-process orchestration: pre-heating → spraying → drying.

`run_full_process` is the single implementation of the digital-twin chain that
both front-ends (Streamlit app.py and notebook 05b) consume. It takes operating
parameters in the units the UIs expose (mm, °C, m³/h, g/min, %), converts to SI,
derives the coating-loss rates from the DoE-fitted empirical correlations, runs
the three stage ODEs with end-state chaining, and returns per-stage results
plus the concatenated time series ready for plotting (time in min, temperatures
in °C, solvent contents in wt %).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import default_coater_params, wg_max_noloss
from .models.preheating import run_preheating, PreheatingResult
from .models.spraying import run_spraying, SprayingResult
from .models.drying_stage import run_drying, DryingResult
from .models.coating_correlations import calc_r_spraying, calc_r_drying_empirical


@dataclass
class FullProcessResult:
    """Per-stage ODE results plus plot-ready concatenated series."""

    # Stage results (SI units, as returned by the stage solvers)
    preheating: PreheatingResult
    spraying: SprayingResult
    drying: DryingResult

    # Concatenated series over the whole process (plot units)
    t_all: np.ndarray        # min
    T_product: np.ndarray    # °C
    T_gas: np.ndarray        # °C (quasi-steady)
    Y_particle: np.ndarray   # wt % acetone on particles
    Y_gas: np.ndarray        # wt % acetone in gas
    WG: np.ndarray           # % coating weight gain
    WG_noloss: np.ndarray    # % theoretical no-loss reference

    # Stage boundaries [min]
    ph_end: float
    sp_end: float
    t_end: float

    # Derived scalars
    r_spraying: float        # kg/s, from empirical correlation
    r_drying: float          # kg/s, from empirical correlation
    dm_ratio_g_kg: float     # g dry coating / kg particles
    qty_sol_kg: float        # kg coating solution sprayed
    sp_dur_s: float          # s, spraying duration
    wg_max_noloss: float     # %, theoretical maximum WG
    wg_end_spray: float      # %, WG at end of spraying
    wg_final: float          # %, WG at discharge


def run_full_process(
    d_mm: float, ssa_cm2g: float, T0_C: float, batch_kg: float,
    humidity_g_kg: float, dmc_pct: float, coating_level: float,
    ph_T_C: float, ph_flow_m3h: float, ph_dur_min: float,
    sp_T_C: float, sp_flow_m3h: float, sp_rate_g_min: float,
    dr_T_C: float, dr_flow_m3h: float, dr_dur_min: float,
) -> FullProcessResult:
    """
    Run the full PH → SP → DR chain from UI-unit operating parameters.

    Parameters
    ----------
    d_mm           : equivalent particle diameter [mm]
    ssa_cm2g       : particle specific surface area [cm²/g]
    T0_C           : initial particle temperature [°C]
    batch_kg       : batch mass [kg]
    humidity_g_kg  : measured inlet air absolute humidity [g/kg dry air]
                     (enters only the r_drying correlation, not the ODEs)
    dmc_pct        : coating solution dry-matter concentration [wt %]
    coating_level  : coded DoE coating-level factor [-1 … 1]; sets the
                     solution quantity via the DoE recipe
                     qty_sol = (0.0017·level + 0.0064)·batch/dmc_frac
    ph/sp/dr_T_C   : stage inlet air temperatures [°C]
    ph/sp/dr_flow_m3h : stage air flows [m³/h]
    ph_dur_min, dr_dur_min : stage durations [min]
    sp_rate_g_min  : spray rate [g solution/min]; spraying duration follows
                     from qty_sol / spray rate
    """
    dmc_frac = dmc_pct / 100.0
    qty_sol_kg = (0.0017 * coating_level + 0.0064) * batch_kg / dmc_frac
    sp_rate_kgs = sp_rate_g_min / 60_000.0
    sp_dur_s = qty_sol_kg / sp_rate_kgs
    dm_ratio_g_kg = qty_sol_kg * dmc_frac / batch_kg * 1000.0

    rho_air = default_coater_params().rho_air
    ph_m = ph_flow_m3h / 3600.0 * rho_air
    sp_m = sp_flow_m3h / 3600.0 * rho_air
    dr_m = dr_flow_m3h / 3600.0 * rho_air
    ph_K, sp_K, dr_K = ph_T_C + 273.15, sp_T_C + 273.15, dr_T_C + 273.15

    # Empirical correlations for coating loss rates
    r_spray = calc_r_spraying(sp_rate_g_min, dmc_pct, dm_ratio_g_kg)
    r_dry = calc_r_drying_empirical(batch_kg, dm_ratio_g_kg, ssa_cm2g, humidity_g_kg)

    phys = dict(diameter_eq=d_mm * 1e-3, ssa_cm2_g=ssa_cm2g, batch_size=batch_kg)

    # Inlet air carries no acetone; humidity enters only via r_dry above.
    p_ph = default_coater_params(
        **phys, air_flow_rates=(ph_m,) * 3,
        air_temperatures=(ph_K,) * 3, air_inlet_moisture=(0.0, 0.0, 0.0))
    res_ph = run_preheating(p_ph, duration=ph_dur_min * 60.0,
                            T_particle_init=T0_C + 273.15)

    p_sp = default_coater_params(
        **phys, air_flow_rates=(sp_m,) * 3,
        air_temperatures=(sp_K,) * 3, air_inlet_moisture=(0.0, 0.0, 0.0),
        spray_rate=sp_rate_kgs, dry_matter_conc=dmc_frac, r_spraying=r_spray)
    res_sp = run_spraying(p_sp, duration=sp_dur_s,
                          T_particle_init=res_ph.T_particle[-1])

    p_dr = default_coater_params(
        **phys, air_flow_rates=(dr_m,) * 3,
        air_temperatures=(dr_K,) * 3, air_inlet_moisture=(0.0, 0.0, 0.0),
        r_drying=r_dry)
    res_dr = run_drying(
        p_dr, duration=dr_dur_min * 60.0,
        Y_particle_init=res_sp.Y_particle[-1],
        Y_gas_init=res_sp.Y_gas[-1],
        M_coating_init=res_sp.M_coating[-1],
        T_particle_init=res_sp.T_particle[-1],
    )

    # ── Concatenated plot series ──────────────────────────────────────────────
    t_ph = res_ph.t / 60.0
    t_sp = (res_sp.t + res_ph.t[-1]) / 60.0
    t_dr = (res_dr.t + res_ph.t[-1] + res_sp.t[-1]) / 60.0
    t_all = np.concatenate([t_ph, t_sp, t_dr])

    wg_max = wg_max_noloss(qty_sol_kg, dmc_frac, batch_kg)
    # No-loss WG reference: linear ramp during spray, flat plateau during dry
    WG_noloss = np.concatenate([
        np.zeros_like(t_ph),
        np.linspace(0.0, wg_max, len(t_sp)),
        np.full(len(t_dr), wg_max),
    ])
    WG = np.concatenate([
        np.zeros_like(t_ph),
        res_sp.M_coating / batch_kg * 100,
        res_dr.M_coating / batch_kg * 100,
    ])
    T_prod = np.concatenate([
        res_ph.T_particle - 273.15,
        res_sp.T_particle - 273.15,
        res_dr.T_particle - 273.15,
    ])
    T_gas = np.concatenate([
        res_ph.T_gas - 273.15,
        res_sp.T_gas - 273.15,
        res_dr.T_gas - 273.15,
    ])
    Y_part = np.concatenate([
        np.zeros_like(t_ph),
        res_sp.Y_particle * 100,
        res_dr.Y_particle * 100,
    ])
    Y_gas_ = np.concatenate([
        np.zeros_like(t_ph),
        res_sp.Y_gas * 100,
        res_dr.Y_gas * 100,
    ])

    return FullProcessResult(
        preheating=res_ph, spraying=res_sp, drying=res_dr,
        t_all=t_all, T_product=T_prod, T_gas=T_gas,
        Y_particle=Y_part, Y_gas=Y_gas_, WG=WG, WG_noloss=WG_noloss,
        ph_end=float(t_ph[-1]), sp_end=float(t_sp[-1]), t_end=float(t_dr[-1]),
        r_spraying=r_spray, r_drying=r_dry,
        dm_ratio_g_kg=dm_ratio_g_kg, qty_sol_kg=qty_sol_kg, sp_dur_s=sp_dur_s,
        wg_max_noloss=wg_max,
        wg_end_spray=float(res_sp.M_coating[-1] / batch_kg * 100),
        wg_final=float(res_dr.M_coating[-1] / batch_kg * 100),
    )
