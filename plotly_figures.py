"""
Plotly rendering layer for the Streamlit digital-twin app (app_plotly.py).

This module is the Plotly counterpart of the Matplotlib figure builders embedded
in app.py. It is deliberately kept OUTSIDE the fluid_bed package: it contains no
simulation logic, only chart construction. All numbers come from a
``FullProcessResult`` (fluid_bed.simulate) and from dissolution_curve()
(fluid_bed.models.dissolution) — exactly the same inputs app.py uses.

Returned figures are plotly.graph_objects.Figure instances, rendered in the app
with ``st.plotly_chart``.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fluid_bed.config import DISSOLUTION

# ── Stage colour map (mirrors app.py) ────────────────────────────────────────
_SC = {"PH": "royalblue", "SP": "darkorange", "DR": "seagreen"}

# rgba shade used behind each stage (a touch stronger so the bands stay visible
# over the light-gray ggplot-style panel background)
_SHADE = {
    "PH": "rgba(65,105,225,0.14)",   # royalblue
    "SP": "rgba(255,140,0,0.14)",    # darkorange
    "DR": "rgba(46,139,87,0.14)",    # seagreen
}


def _stage_at(t, ph_end, sp_end):
    if t <= ph_end:
        return "PH", _SC["PH"]
    if t <= sp_end:
        return "SP", _SC["SP"]
    return "DR", _SC["DR"]


def _add_stage_shading(fig, ph_end, sp_end, t_end, row, col):
    """Stage background bands + divider lines on a single subplot cell."""
    for x0, x1, key in [
        (0.0, ph_end, "PH"),
        (ph_end, sp_end, "SP"),
        (sp_end, t_end, "DR"),
    ]:
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=_SHADE[key], line_width=0,
            layer="below", row=row, col=col,
        )
    for x in (ph_end, sp_end):
        fig.add_vline(
            x=x, line=dict(color="grey", width=0.7, dash="dash"),
            opacity=0.5, row=row, col=col,
        )


def _add_stage_labels(fig, ph_end, sp_end, t_end, row, col):
    """PH / SP / DR text labels near the top of a subplot cell.

    The y position is given in the subplot's *domain* coordinates (fractional,
    0-1) rather than data coordinates, so the label does not stretch the axis
    range and crush small-valued curves toward the bottom.
    """
    ax_x, ax_y = _ax_ids(row, col)
    for pos, lbl in [
        ((0 + ph_end) / 2, "PH"),
        ((ph_end + sp_end) / 2, "SP"),
        ((sp_end + t_end) / 2, "DR"),
    ]:
        fig.add_annotation(
            x=pos, y=0.97, xref=ax_x, yref=f"{ax_y} domain",
            text=f"<b>{lbl}</b>", showarrow=False,
            font=dict(size=20, color=_SC[lbl]),
            yanchor="top",
        )


def _ax_ids(row, col):
    """Subplot axis names for a 2-column grid (1-indexed, row-major).

    (1,1)->('x','y'), (1,2)->('x2','y2'), (2,1)->('x3','y3'), (2,2)->('x4','y4').
    """
    n = (row - 1) * 2 + col
    s = "" if n == 1 else str(n)
    return f"x{s}", f"y{s}"


# Forced light canvas so the charts keep good contrast under Streamlit dark mode.
# ggplot-style look: white paper, light-gray plot panel, white gridlines (set on
# the axes in each figure), dark font.
_PANEL_GRAY = "#F5F5F5"
_LIGHT_CANVAS = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor=_PANEL_GRAY,
    font=dict(color="#222222"),
)


# ── Process figure (2×2) ──────────────────────────────────────────────────────

def make_process_figure(sim, ph_T_C, sp_T_C, dr_T_C):
    t_all = sim.t_all
    ph_end, sp_end, t_end = sim.ph_end, sim.sp_end, sim.t_end
    t_step = [0, ph_end, ph_end, sp_end, sp_end, t_end]
    T_step = [ph_T_C, ph_T_C, sp_T_C, sp_T_C, dr_T_C, dr_T_C]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Temperature profiles", "Particle acetone",
            "Gas-phase acetone", "Coating weight gain",
        ),
        horizontal_spacing=0.09, vertical_spacing=0.16,
    )

    # (1,1) Temperatures
    fig.add_trace(go.Scatter(
        x=t_step, y=T_step, mode="lines", name="T inlet",
        line=dict(color="tomato", width=2.0, dash="dash", shape="hv"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_all, y=sim.T_gas, mode="lines", name="T outlet (qs)",
        line=dict(color="tomato", width=2.0), opacity=0.5,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_all, y=sim.T_product, mode="lines", name="T product",
        line=dict(color="steelblue", width=3.2),
    ), row=1, col=1)

    # (1,2) Particle acetone
    fig.add_trace(go.Scatter(
        x=t_all, y=sim.Y_particle, mode="lines", name="Particle acetone",
        line=dict(color="darkorange", width=3.2), showlegend=False,
    ), row=1, col=2)

    # (2,1) Gas acetone
    fig.add_trace(go.Scatter(
        x=t_all, y=sim.Y_gas, mode="lines", name="Gas acetone",
        line=dict(color="cadetblue", width=3.2), showlegend=False,
    ), row=2, col=1)

    # (2,2) Coating WG
    fig.add_trace(go.Scatter(
        x=t_all, y=sim.WG_noloss, mode="lines", name="WG no loss",
        line=dict(color="mediumpurple", width=2.0, dash="dot"), opacity=0.8,
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=t_all, y=sim.WG, mode="lines", name="WG model",
        line=dict(color="mediumpurple", width=3.2),
    ), row=2, col=2)

    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        _add_stage_shading(fig, ph_end, sp_end, t_end, r, c)
        _add_stage_labels(fig, ph_end, sp_end, t_end, r, c)

    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Acetone on particles (wt %)", row=1, col=2)
    fig.update_yaxes(title_text="Acetone in gas (wt %)", row=2, col=1)
    fig.update_yaxes(title_text="Coating WG (%)", row=2, col=2)
    for r, c in [(2, 1), (2, 2)]:
        fig.update_xaxes(title_text="Time (min)", row=r, col=c)

    # Dark-gray box around every subplot, white gridlines (ggplot look)
    fig.update_xaxes(showline=True, linewidth=1.4, linecolor="dimgray",
                     mirror=True, gridcolor="white")
    fig.update_yaxes(showline=True, linewidth=1.4, linecolor="dimgray",
                     mirror=True, gridcolor="white")

    fig.update_layout(
        height=560,
        title=dict(
            text=(
                f"<b>Empirical correlations  |  "
                f"r_spray = {sim.r_spraying * 1e6:.1f} ×10⁻⁶ kg/s  |  "
                f"r_dry = {sim.r_drying * 1e3:.2f} ×10⁻³ kg/s  |  "
                f"DM ratio = {sim.dm_ratio_g_kg:.2f} g/kg</b>"
            ),
            x=0.5, xanchor="center", font=dict(size=25),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="right", x=1.0),
        margin=dict(t=120, b=50, l=60, r=30),
        hovermode="x unified",
        **_LIGHT_CANVAS,
    )
    return fig


# ── Dissolution figure (1×2) ──────────────────────────────────────────────────

def make_dissolution_figure(sim, t_proc_min, ssa_cm2g, dissolution_curve):
    """
    Build the 1×2 dissolution figure.

    ``dissolution_curve`` is passed in (the fluid_bed.models.dissolution function)
    so this rendering module stays free of simulation imports beyond config.
    Returns the same tuple shape app.py's matplotlib builder returns.
    """
    t_all = sim.t_all
    WG, WG_noloss = sim.WG, sim.WG_noloss
    ph_end, sp_end, t_end = sim.ph_end, sim.sp_end, sim.t_end

    t_s = float(np.clip(t_proc_min, 0.0, t_end))
    wg_at_t = float(np.interp(t_s, t_all, WG))
    wg_nl_at_t = float(np.interp(t_s, t_all, WG_noloss))
    stage_name, stage_color = _stage_at(t_s, ph_end, sp_end)

    t_diss, F_diss, k = dissolution_curve(wg_at_t / 100.0, ssa_cm2g)
    _, F_noloss, _ = dissolution_curve(wg_nl_at_t / 100.0, ssa_cm2g)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Coating evolution — sample point",
            f"First-order dissolution  [{stage_name}]",
        ),
        horizontal_spacing=0.10,
    )

    # ── Left: WG profile + sample marker ──────────────────────────────────────
    # Traces first: add_vrect/add_vline with row/col are dropped by Plotly if the
    # target subplot has no trace yet, so shading must follow the traces.
    fig.add_trace(go.Scatter(
        x=t_all, y=WG_noloss, mode="lines", name="WG no loss",
        line=dict(color="mediumpurple", width=2.0, dash="dot"), opacity=0.8,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_all, y=WG, mode="lines", name="WG model",
        line=dict(color="mediumpurple", width=2.8),
    ), row=1, col=1)
    _add_stage_shading(fig, ph_end, sp_end, t_end, 1, 1)
    _add_stage_labels(fig, ph_end, sp_end, t_end, 1, 1)

    bw = max(t_end * 0.008, 0.3)
    fig.add_vrect(
        x0=t_s - bw, x1=t_s + bw, fillcolor=stage_color, opacity=0.55,
        line_width=0, layer="below", row=1, col=1,
    )
    fig.add_vline(x=t_s, line=dict(color=stage_color, width=1.5), row=1, col=1)
    fig.add_hline(
        y=wg_at_t, line=dict(color=stage_color, width=1.0, dash="dot"),
        opacity=0.6, row=1, col=1,
    )
    fig.add_annotation(
        x=t_s + t_end * 0.04, y=wg_at_t, xref="x", yref="y",
        text=f"{wg_at_t:.3f}%", showarrow=False,
        font=dict(size=11, color=stage_color), xanchor="left", yanchor="middle",
        row=1, col=1,
    )

    # ── Right: dissolution curve ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t_diss, y=F_noloss, mode="lines", name="No loss",
        line=dict(color="dimgrey", width=2.0, dash="dot"), opacity=0.8,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=t_diss, y=F_diss, mode="lines", name="Model",
        line=dict(color=stage_color, width=3.2),
    ), row=1, col=2)
    for tgt, dash in [(50, "dash"), (80, "dot")]:
        if F_diss[-1] >= tgt:
            t_m = float(np.interp(tgt, F_diss, t_diss))
            fig.add_hline(y=tgt, line=dict(color="grey", width=0.8, dash=dash),
                          opacity=0.5, row=1, col=2)
            fig.add_vline(x=t_m, line=dict(color="grey", width=0.8, dash=dash),
                          opacity=0.5, row=1, col=2)
            fig.add_annotation(
                x=t_m + 3, y=tgt + 1.5, xref="x2", yref="y2",
                text=f"t{tgt} = {t_m:.0f} min", showarrow=False,
                font=dict(size=10, color="dimgrey"),
                xanchor="left", yanchor="bottom", row=1, col=2,
            )

    fig.update_xaxes(title_text="Process time (min)", row=1, col=1)
    fig.update_yaxes(title_text="Coating WG (%)", row=1, col=1)
    fig.update_xaxes(title_text="Dissolution time (min)",
                     range=[0, DISSOLUTION["Total_min"]], row=1, col=2)
    fig.update_yaxes(title_text="Drug released (%)", range=[0, 105], row=1, col=2)

    # Dark-gray box around every subplot, white gridlines (ggplot look)
    fig.update_xaxes(showline=True, linewidth=1.4, linecolor="dimgray",
                     mirror=True, gridcolor="white")
    fig.update_yaxes(showline=True, linewidth=1.4, linecolor="dimgray",
                     mirror=True, gridcolor="white")

    fig.update_layout(
        height=440,
        title=dict(
            text=(
                f"<b>Virtual Sample — t = {t_s:.1f} min  [{stage_name}]  "
                f"WG = {wg_at_t:.3f}%  (no-loss: {wg_nl_at_t:.3f}%)   "
                f"k = {k:.4e} s⁻¹</b>"
            ),
            x=0.5, xanchor="center", font=dict(size=25),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="right", x=1.0),
        margin=dict(t=115, b=50, l=60, r=30),
        **_LIGHT_CANVAS,
    )
    return fig, wg_at_t, wg_nl_at_t, k, stage_name, F_diss, t_diss
