"""
Dissolution visualisation.

Key plots
---------
plot_dissolution_profile
    Single panel: model curve(s) overlaid on experimental scatter.

plot_dissolution_doe_grid
    Grid of subplots, one per DoE run, each showing model vs. experiment.

plot_model_comparison
    Overlay all four dissolution model fits on a single panel to compare quality.

plot_doe_f2_heatmap
    Heatmap of f₂ similarity factor between runs (model vs. experiment).
"""

from __future__ import annotations

from math import ceil, sqrt

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Colour conventions ─────────────────────────────────────────────────────────
_MODEL_STYLES = {
    "zero_order":        {"color": "#E67E22", "ls": "-",  "lw": 2.0},
    "first_order":       {"color": "#2980B9", "ls": "-",  "lw": 2.0},
    "higuchi":           {"color": "#27AE60", "ls": "--", "lw": 2.0},
    "korsmeyer_peppas":  {"color": "#8E44AD", "ls": "-.", "lw": 2.0},
}
_EXP_STYLE = {"s": 55, "zorder": 5, "edgecolors": "k", "linewidths": 0.5}


def _format_axis(ax: plt.Axes, xlabel: str = "Time (min)",
                 ylabel: str = "Drug Released (%)") -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(-2, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(True, alpha=0.25, lw=0.5)


# ── Core single-panel plot ─────────────────────────────────────────────────────

def plot_dissolution_profile(
    t_model: np.ndarray,
    F_model: np.ndarray,
    t_exp: np.ndarray | None = None,
    F_exp: np.ndarray | None = None,
    model_name: str = "model",
    run_label: str | None = None,
    color: str | None = None,
    rmse: float | None = None,
    r_squared: float | None = None,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Overlay a model dissolution curve and optional experimental scatter.

    Parameters
    ----------
    t_model : array [min]
        Dense time array for the model curve.
    F_model : array [%]
        Dissolved fraction predicted by the model.
    t_exp : array [min], optional
        Experimental time points.
    F_exp : array [%], optional
        Experimental dissolved fractions.
    model_name : str
        Key from DISSOLUTION_MODELS or a custom label.
    run_label : str, optional
        Run identifier added to the legend.
    rmse, r_squared : float, optional
        Goodness-of-fit metrics shown in the legend label.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    style = _MODEL_STYLES.get(model_name, {"color": color or "#333333", "ls": "-", "lw": 2.0})
    c = color or style["color"]

    # Build legend label
    lbl_parts = [run_label or "", model_name.replace("_", " ").title()]
    if rmse is not None:
        lbl_parts.append(f"RMSE={rmse:.1f}%")
    if r_squared is not None:
        lbl_parts.append(f"R²={r_squared:.3f}")
    model_lbl = " | ".join(p for p in lbl_parts if p)

    ax.plot(t_model, F_model, color=c, ls=style["ls"], lw=style["lw"], label=model_lbl)

    if t_exp is not None and F_exp is not None:
        exp_lbl = f"{run_label} — exp." if run_label else "Experimental"
        ax.scatter(t_exp, F_exp, color=c, label=exp_lbl, **_EXP_STYLE)

    _format_axis(ax)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig, ax


# ── Multi-model comparison on one panel ───────────────────────────────────────

def plot_model_comparison(
    fit_results: dict,
    t_exp: np.ndarray | None = None,
    F_exp: np.ndarray | None = None,
    run_label: str | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot all available dissolution model fits on one panel.

    Parameters
    ----------
    fit_results : dict
        Mapping model_name → DissolutionFitResult (from fit_all_models).
    t_exp, F_exp : arrays, optional
        Experimental data for scatter overlay.
    run_label : str, optional
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, res in fit_results.items():
        style = _MODEL_STYLES.get(name, {"color": "#666", "ls": "-", "lw": 1.5})
        lbl = f"{name.replace('_', ' ').title()} (R²={res.r_squared:.3f})"
        ax.plot(res.t_fit, res.F_model, label=lbl,
                color=style["color"], ls=style["ls"], lw=style["lw"])

    if t_exp is not None and F_exp is not None:
        ax.scatter(t_exp, F_exp, color="black", label="Experimental",
                   zorder=6, s=60, edgecolors="k", linewidths=0.5)

    _format_axis(ax)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.set_title(title or (f"Dissolution Model Comparison — {run_label}" if run_label
                           else "Dissolution Model Comparison"),
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, ax


# ── DoE grid ──────────────────────────────────────────────────────────────────

def plot_dissolution_doe_grid(
    exp_data: dict[str, tuple[np.ndarray, np.ndarray]],
    fit_results: dict[str, object] | None = None,
    model_name: str = "first_order",
    ncols: int = 4,
    fig_width: float = 14.0,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Grid of dissolution subplots — one per DoE run.

    Parameters
    ----------
    exp_data : dict
        Mapping run_label → (t_exp [min], F_exp [%]).
    fit_results : dict, optional
        Mapping run_label → DissolutionFitResult (or dict of them).
        If provided, the model curve is overlaid.
    model_name : str
        Which model to pull from fit_results if it contains dicts.
    ncols : int
        Number of subplot columns.
    """
    run_labels = sorted(exp_data.keys())
    n = len(run_labels)
    nrows = ceil(n / ncols)
    fig_height = 3.2 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                             sharey=True, sharex=False)
    axes_flat = axes.flatten() if n > 1 else [axes]

    style = _MODEL_STYLES.get(model_name, {"color": "#2980B9", "ls": "-", "lw": 1.8})

    for i, label in enumerate(run_labels):
        ax = axes_flat[i]
        t_e, F_e = exp_data[label]
        ax.scatter(t_e, F_e, color=style["color"], **_EXP_STYLE)

        if fit_results and label in fit_results:
            res = fit_results[label]
            # Support both a single result and a dict of model results
            if hasattr(res, "t_fit"):
                single = res
            elif isinstance(res, dict) and model_name in res:
                single = res[model_name]
            else:
                single = None

            if single is not None:
                ax.plot(single.t_fit, single.F_model,
                        color=style["color"], ls=style["ls"], lw=style["lw"])
                ax.set_title(f"{label}\nR²={single.r_squared:.3f}", fontsize=8)
            else:
                ax.set_title(label, fontsize=8)
        else:
            ax.set_title(label, fontsize=8)

        ax.set_ylim(-2, 105)
        ax.grid(True, alpha=0.2, lw=0.5)
        if i % ncols == 0:
            ax.set_ylabel("Dissolved (%)", fontsize=8)
        ax.set_xlabel("Time (min)", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Dissolution Profiles — DoE Runs ({model_name.replace('_', ' ').title()})",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig, axes_flat[:n]


# ── f₂ similarity factor ──────────────────────────────────────────────────────

def f2_similarity(F_ref: np.ndarray, F_test: np.ndarray) -> float:
    """
    FDA f₂ similarity factor.

    f₂ = 50 · log10(100 / sqrt(1 + mean((F_ref − F_test)²)))

    Values ≥ 50 indicate similarity between the two profiles.
    """
    n = len(F_ref)
    msd = np.sum((F_ref - F_test) ** 2) / n
    return 50.0 * np.log10(100.0 / np.sqrt(1.0 + msd))
