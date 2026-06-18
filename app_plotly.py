"""
Fluid Bed Coating Digital Twin — Streamlit app (Plotly rendering variant).

Functionally identical to app.py, but charts are rendered with interactive
Plotly figures instead of Matplotlib. The simulation itself (constants,
correlations, PH→SP→DR chaining, dissolution model) is reused unchanged from the
fluid_bed package; only the rendering layer differs and lives in plotly_figures.py.

Run with:  streamlit run app_plotly.py
"""

import sys
import os

# Make the fluid_bed package importable whether or not it is pip-installed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import streamlit as st

from fluid_bed.simulate import run_full_process
from fluid_bed.models.dissolution import dissolution_curve

from plotly_figures import make_process_figure, make_dissolution_figure


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
        make_dissolution_figure(sim, t_proc_min, ssa_cm2g, dissolution_curve)
    )
    # theme=None → use the figure's own (white) template instead of Streamlit's
    # dark theme, which would otherwise force title/tick fonts to white.
    st.plotly_chart(fig_diss, use_container_width=True, theme=None)

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
    page_title="Fluid Bed Coating Digital Twin (Plotly)",
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
st.plotly_chart(fig_proc, use_container_width=True, theme=None)

# ── Dissolution panel ─────────────────────────────────────────────────────────
st.divider()
dissolution_panel(sim, ssa_cm2g)
