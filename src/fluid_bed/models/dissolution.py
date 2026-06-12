"""
Coating dissolution models.

Predicts the fraction of drug released over time when coated particles are
placed in a dissolution medium (e.g. USP paddle apparatus).

Available models
----------------
- Zero-order      : constant release rate
- First-order     : release proportional to remaining undissolved fraction
- Higuchi         : square-root-of-time (matrix diffusion)
- Korsmeyer-Peppas: power-law (generalised, n determines mechanism)

Each model returns F(t) in % (0–100).

Fitting
-------
`fit_dissolution` uses scipy.optimize.curve_fit to estimate model parameters
from experimental (t, F%) data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from ..config import DISSOLUTION


# ── Model functions ────────────────────────────────────────────────────────────

def model_zero_order(t: np.ndarray, k: float) -> np.ndarray:
    """F(t) = k·t, clipped to [0, 100] %."""
    return np.clip(k * t, 0.0, 100.0)


def model_first_order(t: np.ndarray, k: float) -> np.ndarray:
    """F(t) = 100·(1 − exp(−k·t)) %."""
    return 100.0 * (1.0 - np.exp(-k * t))


def model_higuchi(t: np.ndarray, k: float) -> np.ndarray:
    """F(t) = k·√t, clipped to [0, 100] %."""
    return np.clip(k * np.sqrt(t), 0.0, 100.0)


def model_korsmeyer_peppas(t: np.ndarray, k: float, n: float) -> np.ndarray:
    """
    F(t) = 100·k·t^n, clipped to [0, 100] %.

    n < 0.45: Fickian diffusion
    0.45 < n < 0.89: anomalous transport
    n ≈ 0.89: Case II transport (erosion)
    """
    return np.clip(100.0 * k * np.power(np.maximum(t, 0.0), n), 0.0, 100.0)


# ── Coating-mass ↔ rate-constant link (diffusion through the EC film) ─────────

def dissolution_k(wg_fraction: float, ssa_cm2g: float) -> float:
    """
    First-order dissolution rate constant k [1/s] implied by a coating level.

    Diffusion-through-film relation (MODELLING_BACKGROUND_README.md §5):

        k = m_sample · SSA² · P · ρ_EC / (V_disso · x_EC)

    Parameters
    ----------
    wg_fraction : EC mass fraction x_EC = M_coating / M_batch [g/g]
    ssa_cm2g    : particle specific surface area [cm²/g]

    Returns
    -------
    k [1/s]; ``inf`` when wg_fraction <= 0 (no coating → instant release).
    """
    if wg_fraction <= 0:
        return float("inf")
    S = DISSOLUTION["Mass_sample"] * ssa_cm2g
    return (S * DISSOLUTION["Permeability"] * DISSOLUTION["rho_EC"] * ssa_cm2g
            / (DISSOLUTION["Volume_disso"] * wg_fraction))


def dissolution_curve(wg_fraction: float, ssa_cm2g: float):
    """
    Predicted first-order dissolution profile for a given coating level.

    Returns
    -------
    (t_min, F_pct, k) : time grid [min] over the standard test duration,
    released fraction [%] via :func:`model_first_order`, and the rate
    constant k [1/s] from :func:`dissolution_k`.
    """
    k = dissolution_k(wg_fraction, ssa_cm2g)
    t_s = np.arange(1, DISSOLUTION["Total_min"] + 1) * 60.0
    if np.isinf(k):
        F = np.full_like(t_s, 100.0)
    else:
        F = model_first_order(t_s, k)
    return t_s / 60.0, F, k


# Registry of available models for programmatic access
DISSOLUTION_MODELS = {
    "zero_order": (model_zero_order, ["k"]),
    "first_order": (model_first_order, ["k"]),
    "higuchi": (model_higuchi, ["k"]),
    "korsmeyer_peppas": (model_korsmeyer_peppas, ["k", "n"]),
}


# ── Fitting ────────────────────────────────────────────────────────────────────

@dataclass
class DissolutionFitResult:
    model_name: str
    params: dict[str, float]    # fitted parameter values
    params_std: dict[str, float]  # standard errors
    F_model: np.ndarray         # fitted curve evaluated on t_fit
    t_fit: np.ndarray
    rmse: float                 # root-mean-square error [%]
    r_squared: float
    converged: bool


def fit_dissolution(
    t_exp: np.ndarray,        # time points [min]
    F_exp: np.ndarray,        # dissolved fraction [%]
    model_name: str = "first_order",
    p0: list[float] | None = None,
    bounds: tuple | None = None,
    t_fit: np.ndarray | None = None,
) -> DissolutionFitResult:
    """
    Fit a dissolution model to experimental data.

    Parameters
    ----------
    t_exp : array
        Experimental time points [min].
    F_exp : array
        Experimental dissolved fraction [%].
    model_name : str
        One of 'zero_order', 'first_order', 'higuchi', 'korsmeyer_peppas'.
    p0 : list, optional
        Initial parameter guess. Defaults depend on model.
    bounds : tuple, optional
        Parameter bounds passed to curve_fit. Defaults to (0, np.inf).
    t_fit : array, optional
        Dense time array for plotting the fitted curve.
        Defaults to linspace(0, max(t_exp), 300).

    Returns
    -------
    DissolutionFitResult
    """
    if model_name not in DISSOLUTION_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(DISSOLUTION_MODELS)}")

    fn, param_names = DISSOLUTION_MODELS[model_name]

    if p0 is None:
        p0 = [0.1] * len(param_names)
    if bounds is None:
        bounds = (0, np.inf)
    if t_fit is None:
        t_fit = np.linspace(0, t_exp.max(), 300)

    converged = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(fn, t_exp, F_exp, p0=p0, bounds=bounds, maxfev=5000)
    except RuntimeError:
        popt = np.array(p0)
        pcov = np.full((len(p0), len(p0)), np.nan)
        converged = False

    perr = np.sqrt(np.diag(np.abs(pcov)))
    F_pred_exp = fn(t_exp, *popt)
    residuals = F_exp - F_pred_exp
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((F_exp - F_exp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    F_model = fn(t_fit, *popt)

    return DissolutionFitResult(
        model_name=model_name,
        params=dict(zip(param_names, popt.tolist())),
        params_std=dict(zip(param_names, perr.tolist())),
        F_model=F_model,
        t_fit=t_fit,
        rmse=rmse,
        r_squared=r2,
        converged=converged,
    )


def fit_all_models(
    t_exp: np.ndarray,
    F_exp: np.ndarray,
    t_fit: np.ndarray | None = None,
) -> dict[str, DissolutionFitResult]:
    """Fit all four dissolution models and return a dict of results."""
    return {
        name: fit_dissolution(t_exp, F_exp, model_name=name, t_fit=t_fit)
        for name in DISSOLUTION_MODELS
    }


def best_model(results: dict[str, DissolutionFitResult]) -> DissolutionFitResult:
    """Return the model with the highest R² among fitted results."""
    return max(results.values(), key=lambda r: r.r_squared if np.isfinite(r.r_squared) else -np.inf)
