"""
Process parameter visualisation.

Key plots
---------
plot_coating_run
    Three-panel time series for a single coating run:
    (1) product temperature [°C], (2) acetone content [% w/w], (3) coating weight gain [%].
    Phase boundaries (pre-heating / spraying / drying) are marked with vertical lines.

plot_run_overlay
    Overlay a single variable across multiple DoE runs for comparison.

plot_phase_summary
    Concatenate pre-heating, spraying, and drying results from the three model
    outputs and produce a unified plot of the full coating run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Colour palette ─────────────────────────────────────────────────────────────
_PHASE_COLORS = {
    "preheating": "#F5A623",   # amber
    "spraying":   "#4A90D9",   # blue
    "drying":     "#7ED321",   # green
}
_VAR_COLORS = {
    "temperature": "#D0021B",  # warm red
    "acetone":     "#4A90D9",  # blue
    "coating":     "#417505",  # dark green
}


# ── Helper ─────────────────────────────────────────────────────────────────────

def _shade_phases(ax: plt.Axes, phase_boundaries: dict[str, float]) -> None:
    """Add translucent phase bands and vertical dashed lines."""
    t_start = 0.0
    phases = []
    if "preheating_end" in phase_boundaries:
        phases.append(("preheating", t_start, phase_boundaries["preheating_end"]))
        t_start = phase_boundaries["preheating_end"]
    if "spraying_end" in phase_boundaries:
        phases.append(("spraying", t_start, phase_boundaries["spraying_end"]))
        t_start = phase_boundaries["spraying_end"]
    if "drying_end" in phase_boundaries:
        phases.append(("drying", t_start, phase_boundaries["drying_end"]))

    for name, t0, t1 in phases:
        ax.axvspan(t0, t1, alpha=0.07, color=_PHASE_COLORS[name], zorder=0)
        ax.axvline(t1, color=_PHASE_COLORS[name], lw=1.0, ls="--", alpha=0.6, zorder=1)


def _phase_legend_patches() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=_PHASE_COLORS[k], alpha=0.5, label=k.capitalize())
        for k in _PHASE_COLORS
    ]


# ── Main plotting functions ─────────────────────────────────────────────────────

def plot_coating_run(
    t: np.ndarray,
    T_particle: np.ndarray,
    Y_acetone: np.ndarray,
    M_coating: np.ndarray,
    batch_size: float,
    phase_boundaries: dict[str, float] | None = None,
    run_label: str | None = None,
    T_gas: np.ndarray | None = None,
    fig: plt.Figure | None = None,
    axes: Sequence[plt.Axes] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Three-panel time series for a single coating run.

    Parameters
    ----------
    t : array [s]
        Time axis (will be converted to minutes for display).
    T_particle : array [K]
        Product (particle) temperature.
    Y_acetone : array [kg/kg]
        Particle acetone content.
    M_coating : array [kg]
        Cumulative coating mass.
    batch_size : float [kg]
        Dry batch mass (used to compute weight gain %).
    phase_boundaries : dict, optional
        Keys: 'preheating_end', 'spraying_end', 'drying_end' — values in **minutes**.
    run_label : str, optional
        Legend label.
    T_gas : array [K], optional
        Gas temperature (plotted as dashed line on temperature panel).
    fig, axes : optional
        Existing figure/axes to draw on (for overlaying multiple runs).

    Returns
    -------
    fig, axes : the Figure and list of three Axes.
    """
    if fig is None or axes is None:
        fig, axes = plt.subplots(
            3, 1, figsize=(10, 8), sharex=True,
            gridspec_kw={"hspace": 0.08},
        )
    ax_T, ax_ac, ax_coat = axes

    t_min = t / 60.0

    # ── Panel 1: Product temperature ──
    ax_T.plot(t_min, T_particle - 273.15,
              color=_VAR_COLORS["temperature"], lw=1.8, label=run_label)
    if T_gas is not None:
        ax_T.plot(t_min, T_gas - 273.15,
                  color=_VAR_COLORS["temperature"], lw=1.0, ls="--", alpha=0.5,
                  label=f"{run_label} (gas)" if run_label else "Gas T")
    ax_T.set_ylabel("Product Temp. (°C)", fontsize=11)
    ax_T.grid(True, alpha=0.25, lw=0.5)

    # ── Panel 2: Acetone content ──
    ax_ac.plot(t_min, Y_acetone * 100.0,
               color=_VAR_COLORS["acetone"], lw=1.8)
    ax_ac.set_ylabel("Acetone (% w/w)", fontsize=11)
    ax_ac.grid(True, alpha=0.25, lw=0.5)

    # ── Panel 3: Coating weight gain ──
    weight_gain = M_coating / batch_size * 100.0
    ax_coat.plot(t_min, weight_gain,
                 color=_VAR_COLORS["coating"], lw=1.8)
    ax_coat.set_ylabel("Coating WG (%)", fontsize=11)
    ax_coat.set_xlabel("Time (min)", fontsize=11)
    ax_coat.grid(True, alpha=0.25, lw=0.5)

    # ── Phase shading ──
    if phase_boundaries:
        for ax in axes:
            _shade_phases(ax, phase_boundaries)

    # ── Legend on top panel ──
    handles = []
    if run_label:
        handles.append(plt.Line2D([0], [0], color=_VAR_COLORS["temperature"], lw=1.8,
                                  label=run_label))
    handles += _phase_legend_patches()
    if handles:
        ax_T.legend(handles=handles, fontsize=9, loc="upper left", framealpha=0.85)

    title = f"Coating Run — {run_label}" if run_label else "Coating Run"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, list(axes)


def plot_phase_summary(
    preheating_result,
    spraying_result,
    drying_result,
    batch_size: float,
    run_label: str | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Concatenate results from the three model stages and produce a unified run plot.

    Parameters
    ----------
    preheating_result : PreheatingResult
    spraying_result : SprayingResult
    drying_result : DryingResult
    batch_size : float [kg]
    run_label : str, optional
    """
    t_ph = preheating_result.t
    t_sp = spraying_result.t + t_ph[-1]
    t_dr = drying_result.t + t_sp[-1]
    t_all = np.concatenate([t_ph, t_sp, t_dr])

    # Temperature
    T_p_all = np.concatenate([
        preheating_result.T_particle,
        spraying_result.T_particle,
        drying_result.T_particle,
    ])
    T_g_all = np.concatenate([
        preheating_result.T_gas,
        spraying_result.T_gas,
        drying_result.T_gas,
    ])

    # Acetone (pre-heating: zeros; spraying/drying: Y_particle)
    Y_ph = np.zeros_like(t_ph)
    Y_all = np.concatenate([Y_ph, spraying_result.Y_particle, drying_result.Y_particle])

    # Coating mass (pre-heating: 0; spraying: grows; drying: constant at end-of-spray value)
    M_sp_end = spraying_result.M_coating[-1]
    M_ph = np.zeros_like(t_ph)
    M_dr = np.full_like(t_dr, M_sp_end)
    M_all = np.concatenate([M_ph, spraying_result.M_coating, M_dr])

    phase_boundaries = {
        "preheating_end": t_ph[-1] / 60.0,
        "spraying_end":   t_sp[-1] / 60.0,
        "drying_end":     t_dr[-1] / 60.0,
    }

    return plot_coating_run(
        t=t_all,
        T_particle=T_p_all,
        Y_acetone=Y_all,
        M_coating=M_all,
        batch_size=batch_size,
        phase_boundaries=phase_boundaries,
        run_label=run_label,
        T_gas=T_g_all,
    )


def plot_run_overlay(
    results_list: list[dict],
    labels: list[str] | None = None,
    variable: str = "T_particle",
    t_key: str = "t",
    y_label: str | None = None,
    title: str | None = None,
    unit_scale: float = 1.0,
    offset: float = 0.0,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Overlay a single variable across multiple DoE run results.

    Parameters
    ----------
    results_list : list of dicts
        Each dict has at least 't' [s] and the variable to plot.
    labels : list of str, optional
        Legend labels for each run.
    variable : str
        Key in result dicts to plot (e.g. 'T_particle', 'Y_particle', 'M_coating').
    unit_scale : float
        Multiply the variable by this factor (e.g. -273.15 offset for K→°C is handled
        via `offset`).
    offset : float
        Add this value after scaling (e.g. set offset=-273.15 to convert K to °C).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab20")

    for i, res in enumerate(results_list):
        t_min = np.asarray(res[t_key]) / 60.0
        y = np.asarray(res[variable]) * unit_scale + offset
        lbl = labels[i] if labels else f"Run {i + 1}"
        ax.plot(t_min, y, lw=1.5, color=cmap(i / max(len(results_list), 1)), label=lbl)

    ax.set_xlabel("Time (min)", fontsize=11)
    ax.set_ylabel(y_label or variable, fontsize=11)
    ax.set_title(title or f"DoE Comparison — {variable}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    return fig, ax
