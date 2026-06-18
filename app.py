"""
Fluid Bed Coating Digital Twin — Streamlit app
Replicates the 05b notebook interactive dashboard.
Deploy on HuggingFace Spaces (SDK: streamlit).

The simulation itself (constants, correlations, PH→SP→DR chaining, dissolution
model) lives in the fluid_bed package — this file is UI only.
"""

import sys
import os

# Make the fluid_bed package importable whether or not it is pip-installed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend required on HF Spaces
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import streamlit as st

from fluid_bed.config import DISSOLUTION
from fluid_bed.simulate import run_full_process
from fluid_bed.models.dissolution import dissolution_curve

# ── Stage colour map ───────────────────────────────────────────────────────────
_SC = {"PH": "royalblue", "SP": "darkorange", "DR": "seagreen"}


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _shade(ax, ph_end, sp_end, t_end):
    ax.axvspan(0, ph_end,      alpha=0.07, color=_SC["PH"])
    ax.axvspan(ph_end, sp_end, alpha=0.07, color=_SC["SP"])
    ax.axvspan(sp_end, t_end,  alpha=0.07, color=_SC["DR"])
    ax.axvline(ph_end, color="grey", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(sp_end, color="grey", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Time (min)")


def _stxt(ax, ph_end, sp_end, t_end):
    lo, hi = ax.get_ylim()
    yp = lo + (hi - lo) * 0.97
    for pos, lbl in [
        ((0 + ph_end) / 2,      "PH"),
        ((ph_end + sp_end) / 2, "SP"),
        ((sp_end + t_end) / 2,  "DR"),
    ]:
        ax.text(pos, yp, lbl, ha="center", va="top",
                fontsize=8, color=_SC[lbl], fontweight="bold")


def _stage_at(t, ph_end, sp_end):
    if t <= ph_end:
        return "PH", _SC["PH"]
    if t <= sp_end:
        return "SP", _SC["SP"]
    return "DR", _SC["DR"]


# ── Core simulation (results cached per unique parameter set) ─────────────────

@st.cache_data(show_spinner="Running ODE simulation…")
def run_simulation(
    d_mm, ssa_cm2g, T0_C, batch_kg, humidity_g_kg, dmc_pct, coating_level,
    ph_T_C, ph_flow, ph_dur_min,
    sp_T_C, sp_flow, sp_rate_g_min,
    dr_T_C, dr_flow, dr_dur_min,
):
    return run_full_process(
        d_mm, ssa_cm2g, T0_C, batch_kg, humidity_g_kg, dmc_pct, coating_level,
        ph_T_C, ph_flow, ph_dur_min,
        sp_T_C, sp_flow, sp_rate_g_min,
        dr_T_C, dr_flow, dr_dur_min,
    )


# ── Figure builders ────────────────────────────────────────────────────────────

def make_process_figure(sim, ph_T_C, sp_T_C, dr_T_C):
    t_all = sim.t_all
    ph_end, sp_end, t_end = sim.ph_end, sim.sp_end, sim.t_end
    t_step = [0, ph_end, ph_end, sp_end, sp_end, t_end]
    T_step = [ph_T_C, ph_T_C, sp_T_C, sp_T_C, dr_T_C, dr_T_C]

    fig, axes = plt.subplots(2, 2, figsize=(11, 4.25))
    fig.suptitle(
        f"Empirical correlations  |  "
        f"r_spray = {sim.r_spraying*1e6:.1f} ×10⁻⁶ kg/s  |  "
        f"r_dry = {sim.r_drying*1e3:.2f} ×10⁻³ kg/s  |  "
        f"DM ratio = {sim.dm_ratio_g_kg:.2f} g/kg",
        fontsize=11, fontweight="bold",
    )

    ax = axes[0, 0]
    _shade(ax, ph_end, sp_end, t_end)
    ax.step(t_step, T_step, color="tomato", lw=1.5, ls="--",
            where="post", label="T inlet")
    ax.plot(t_all, sim.T_gas,     color="tomato",    lw=1.5, alpha=0.5, label="T outlet (qs)")
    ax.plot(t_all, sim.T_product, color="steelblue", lw=2.5, label="T product")
    ax.set_ylabel("Temperature (°C)"); ax.set_title("Temperature profiles")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(True, alpha=0.3)
    _stxt(ax, ph_end, sp_end, t_end)

    ax = axes[0, 1]
    _shade(ax, ph_end, sp_end, t_end)
    ax.plot(t_all, sim.Y_particle, color="darkorange", lw=2.5)
    ax.set_ylabel("Acetone on particles (wt %)"); ax.set_title("Particle acetone")
    ax.grid(True, alpha=0.3); _stxt(ax, ph_end, sp_end, t_end)

    ax = axes[1, 0]
    _shade(ax, ph_end, sp_end, t_end)
    ax.plot(t_all, sim.Y_gas, color="cadetblue", lw=2.5)
    ax.set_ylabel("Acetone in gas (wt %)"); ax.set_title("Gas-phase acetone")
    ax.grid(True, alpha=0.3); _stxt(ax, ph_end, sp_end, t_end)

    ax = axes[1, 1]
    _shade(ax, ph_end, sp_end, t_end)
    ax.plot(t_all, sim.WG_noloss, color="mediumpurple", lw=1.5,
            ls=":", alpha=0.8, label="WG no loss")
    ax.plot(t_all, sim.WG, color="mediumpurple", lw=2.5, label="WG model")
    ax.set_ylabel("Coating WG (%)"); ax.set_title("Coating weight gain")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _stxt(ax, ph_end, sp_end, t_end)

    plt.tight_layout()
    return fig


def make_dissolution_figure(sim, t_proc_min, ssa_cm2g):
    t_all = sim.t_all
    WG, WG_noloss = sim.WG, sim.WG_noloss
    ph_end, sp_end, t_end = sim.ph_end, sim.sp_end, sim.t_end

    t_s        = float(np.clip(t_proc_min, 0.0, t_end))
    wg_at_t    = float(np.interp(t_s, t_all, WG))
    wg_nl_at_t = float(np.interp(t_s, t_all, WG_noloss))
    stage_name, stage_color = _stage_at(t_s, ph_end, sp_end)

    t_diss, F_diss, k = dissolution_curve(wg_at_t / 100.0, ssa_cm2g)
    _, F_noloss, _    = dissolution_curve(wg_nl_at_t / 100.0, ssa_cm2g)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        f"Virtual Sample — t = {t_s:.1f} min  [{stage_name}]  "
        f"WG = {wg_at_t:.3f}%  (no-loss: {wg_nl_at_t:.3f}%)   k = {k:.4e} s⁻¹",
        fontsize=11, fontweight="bold",
    )

    # Left: WG profile + sample marker
    ax = axes[0]
    _shade(ax, ph_end, sp_end, t_end)
    ax.plot(t_all, WG_noloss, color="mediumpurple", lw=1.5,
            ls=":", alpha=0.8, label="WG no loss")
    ax.plot(t_all, WG, color="mediumpurple", lw=2.0, label="WG model")
    bw = max(t_end * 0.008, 0.3)
    ax.axvspan(t_s - bw, t_s + bw, alpha=0.55,
               color=stage_color, label=f"Sample [{stage_name}]")
    ax.axvline(t_s, color=stage_color, lw=1.5)
    ax.axhline(wg_at_t, color=stage_color, lw=1.0, ls=":", alpha=0.6)
    ax.annotate(
        f"{wg_at_t:.3f}%", xy=(t_s, wg_at_t),
        xytext=(t_s + t_end * 0.04, wg_at_t),
        fontsize=9, color=stage_color, va="center",
    )
    _stxt(ax, ph_end, sp_end, t_end)
    ax.set_xlabel("Process time (min)"); ax.set_ylabel("Coating WG (%)")
    ax.set_title("Coating evolution — sample point")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Right: dissolution curve
    ax = axes[1]
    ax.set_facecolor(to_rgba(stage_color, 0.06))
    ax.plot(t_diss, F_noloss, color="dimgrey", lw=1.5,
            ls=":", alpha=0.8, label="No loss")
    ax.plot(t_diss, F_diss, color=stage_color, lw=2.5, label="Model")
    for tgt, ls in [(50, "--"), (80, ":")]:
        if F_diss[-1] >= tgt:
            t_m = float(np.interp(tgt, F_diss, t_diss))
            ax.axhline(tgt, color="grey", lw=0.8, ls=ls, alpha=0.5)
            ax.axvline(t_m, color="grey", lw=0.8, ls=ls, alpha=0.5)
            ax.text(t_m + 3, tgt + 1.5, f"t{tgt} = {t_m:.0f} min",
                    fontsize=8, color="dimgrey")
    ax.set_xlim(0, DISSOLUTION["Total_min"]); ax.set_ylim(0, 105)
    ax.set_xlabel("Dissolution time (min)"); ax.set_ylabel("Drug released (%)")
    ax.set_title(f"First-order dissolution  [{stage_name}]")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, wg_at_t, wg_nl_at_t, k, stage_name, F_diss, t_diss


# ── Dissolution panel (fragment: t_sample reruns only this section) ───────────

@st.fragment
def dissolution_panel(sim, ssa_cm2g):
    st.markdown("### Dissolution — Virtual Sample Withdrawal")
    st.caption(
        "Move the slider to any process time. "
        "The dissolution curve updates instantly (simulation result is cached)."
    )

    t_max = float(sim.t_end)
    if "t_sample" not in st.session_state:
        st.session_state["t_sample"] = min(float(sim.sp_end), t_max)
    elif st.session_state["t_sample"] > t_max:
        st.session_state["t_sample"] = t_max

    t_proc_min = st.slider(
        "Sample time (min)",
        min_value=0.0,
        max_value=t_max,
        step=0.5,
        key="t_sample",
    )

    fig_diss, wg_at_t, wg_nl_at_t, k, stage_name, F_diss, t_diss = (
        make_dissolution_figure(sim, t_proc_min, ssa_cm2g)
    )
    st.pyplot(fig_diss, use_container_width=True)
    plt.close(fig_diss)

    # Dissolution metrics
    t50 = float(np.interp(50, F_diss, t_diss)) if F_diss[-1] >= 50 else float("nan")
    t80 = float(np.interp(80, F_diss, t_diss)) if F_diss[-1] >= 80 else float("nan")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Stage",      stage_name)
    d2.metric("t₅₀ (min)", f"{t50:.0f}" if not np.isnan(t50) else "> 240")
    d3.metric("t₈₀ (min)", f"{t80:.0f}" if not np.isnan(t80) else "> 240")
    d4.metric("k (s⁻¹)",   f"{k:.4e}")


# ── Page layout ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fluid Bed Coating Digital Twin",
    page_icon="🧪",
    layout="wide",
)

st.title("Fluid Bed Coating Digital Twin")
st.caption(
    "**Pre-heating → Spraying → Drying → Dissolution prediction.** "
    "r_spray and r_dry are derived from the AICc-selected empirical correlations "
    "(r_spray R²=0.844, r_dry R²=0.851; Bosch 2018 DoE, 19 runs)."
)
st.caption(
    "**Full model development**: [GitHub repo](https://github.com/SamdGuizani/FB_Coating_Digital_Twin)"
)

# ── Sidebar sliders ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Parameters")

    st.markdown("**Particle / Batch**")
    d_mm          = st.slider("Diameter (mm)",     0.80, 1.50, 1.00, 0.05)
    ssa_cm2g      = st.slider("SSA (cm²/g)",       55.0, 75.0, 65.0, 1.0)
    T0_C          = st.slider("T₀ particle (°C)", 15.0, 30.0, 20.0, 1.0)
    batch_kg      = st.slider("Batch size (kg)",   3.0,  6.0,  4.6,  0.1)
    humidity_g_kg = st.slider("Humidity (g/kg)",   0.0, 25.0, 13.4,  0.5)
    dmc_pct       = st.slider("DMC (% m/m)",       1.0,  2.0,  1.5,  0.1,
        help="Coating solution dry-matter concentration. Feeds the "
             "correlations as CC (coating concentration).")
    coating_level = st.slider("Coating level",    -1.0,  1.0,  0.5,  0.05,
        help="Coded DoE coating level (−1…+1). Converted to the dry-matter "
             "ratio DM = 1.7·level + 6.4 g/kg, which feeds the correlations.")

    st.markdown("**Pre-heating**")
    ph_T_C     = st.slider("[PH] T inlet (°C)",   40.0, 60.0, 50.0, 2.0)
    ph_flow    = st.slider("[PH] Flow (m³/h)",    200,  400,  250,  10)
    ph_dur_min = st.slider("[PH] Duration (min)", 10.0, 30.0, 20.0, 1.0)

    st.markdown("**Spraying** *(r_spray from correlation)*")
    sp_T_C        = st.slider("[SP] T inlet (°C)",      30.0, 60.0, 40.0, 2.0)
    sp_flow       = st.slider("[SP] Flow (m³/h)",       200,  400,  250,  10)
    sp_rate_g_min = st.slider("[SP] Spray rate (g/min)", 95,  150,  120,   5)

    st.markdown("**Drying** *(r_dry from correlation)*")
    dr_T_C     = st.slider("[DR] T inlet (°C)",   30.0, 60.0, 40.0, 2.0)
    dr_flow    = st.slider("[DR] Flow (m³/h)",    200,  400,  250,  10)
    dr_dur_min = st.slider("[DR] Duration (min)",  3.0, 15.0,  9.0, 1.0)

# ── Run simulation (result cached per unique slider combination) ──────────────
sim = run_simulation(
    d_mm, ssa_cm2g, T0_C, batch_kg, humidity_g_kg, dmc_pct, coating_level,
    ph_T_C, ph_flow, ph_dur_min,
    sp_T_C, sp_flow, sp_rate_g_min,
    dr_T_C, dr_flow, dr_dur_min,
)

# ── Summary metrics row ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("r_spray (×10⁻⁶ kg/s)", f"{sim.r_spraying*1e6:.2f}")
c2.metric("r_dry (×10⁻³ kg/s)",   f"{sim.r_drying*1e3:.3f}")
c3.metric("DM ratio (g/kg)",       f"{sim.dm_ratio_g_kg:.2f}")
c4.metric("WG end-spray",          f"{sim.wg_end_spray:.3f}%")
c5.metric("WG final",              f"{sim.wg_final:.3f}%")

# ── Process plots ─────────────────────────────────────────────────────────────
fig_proc = make_process_figure(sim, ph_T_C, sp_T_C, dr_T_C)
st.pyplot(fig_proc, use_container_width=True)
plt.close(fig_proc)

# ── Dissolution panel ─────────────────────────────────────────────────────────
st.divider()
dissolution_panel(sim, ssa_cm2g)
