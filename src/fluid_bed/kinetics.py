"""
Coating deposition kinetics — 1st and 2nd order models.

Port of ordre_1.m and ordre_2.m (Minimisation/ folder).

These models fit the experimental end-of-spraying coating mass by optimising
the rate constant k_spray using scipy.optimize.minimize_scalar (equivalent to
MATLAB's fminbnd / fminsearch).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


# ── Analytical coating deposition models ──────────────────────────────────────

def coating_mass_order1(
    t_end: float,
    k_spray: float,
    spray_rate: float,
    dry_matter_conc: float,
    batch_size: float,
) -> float:
    """
    First-order deposition model (port of ordre_1.m).

    Y_model = (DMC · ṁ_spray / k) · (1 − exp(−k · t / batch_size))

    Returns predicted coating mass [kg] at end of spraying.
    """
    DMC = dry_matter_conc
    SR = spray_rate
    return (DMC * SR / k_spray) * (1.0 - np.exp(-k_spray * t_end / batch_size))


def coating_mass_order2(
    t_end: float,
    k_spray: float,
    spray_rate: float,
    dry_matter_conc: float,
    batch_size: float,
) -> float:
    """
    Second-order deposition model (port of ordre_2.m).

    Y_model = sqrt(DMC·ṁ/k) · tan(sqrt(k·DMC·ṁ) · t / batch_size)

    Returns predicted coating mass [kg] at end of spraying.
    """
    DMC = dry_matter_conc
    SR = spray_rate
    arg = np.sqrt(k_spray * DMC * SR) * t_end / batch_size
    # tan diverges at π/2; guard against unphysical parameter space
    if abs(arg) >= np.pi / 2:
        return np.inf
    return np.sqrt(DMC * SR / k_spray) * np.tan(arg)


# ── Fitting utilities ──────────────────────────────────────────────────────────

@dataclass
class KineticsFitResult:
    k_spray: float          # fitted rate constant [1/s]
    coating_mass_model: float   # predicted coating mass [kg]
    coating_mass_exp: float     # experimental coating mass [kg]
    residual: float         # (model − exp)² [kg²]
    order: int              # 1 or 2
    converged: bool


def fit_order1(
    t_end_spray: float,
    spray_rate: float,
    dry_matter_conc: float,
    batch_size: float,
    coating_mass_exp: float,
    k_bounds: tuple[float, float] = (1e-6, 10.0),
) -> KineticsFitResult:
    """
    Fit first-order rate constant k_spray to experimental coating mass.

    Uses bounded scalar minimisation (equivalent to MATLAB's fminbnd).
    """
    def objective(k):
        pred = coating_mass_order1(t_end_spray, k, spray_rate, dry_matter_conc, batch_size)
        return (pred - coating_mass_exp) ** 2

    result = minimize_scalar(objective, bounds=k_bounds, method="bounded")
    k_opt = result.x
    pred = coating_mass_order1(t_end_spray, k_opt, spray_rate, dry_matter_conc, batch_size)
    return KineticsFitResult(
        k_spray=k_opt,
        coating_mass_model=pred,
        coating_mass_exp=coating_mass_exp,
        residual=result.fun,
        order=1,
        converged=result.success,
    )


def fit_order2(
    t_end_spray: float,
    spray_rate: float,
    dry_matter_conc: float,
    batch_size: float,
    coating_mass_exp: float,
    k_bounds: tuple[float, float] = (1e-6, 10.0),
) -> KineticsFitResult:
    """
    Fit second-order rate constant k_spray to experimental coating mass.
    """
    def objective(k):
        pred = coating_mass_order2(t_end_spray, k, spray_rate, dry_matter_conc, batch_size)
        if not np.isfinite(pred):
            return 1e12
        return (pred - coating_mass_exp) ** 2

    result = minimize_scalar(objective, bounds=k_bounds, method="bounded")
    k_opt = result.x
    pred = coating_mass_order2(t_end_spray, k_opt, spray_rate, dry_matter_conc, batch_size)
    return KineticsFitResult(
        k_spray=k_opt,
        coating_mass_model=pred,
        coating_mass_exp=coating_mass_exp,
        residual=result.fun,
        order=2,
        converged=result.success,
    )


def fit_all_runs(
    runs: dict,
    order: int = 1,
    k_bounds: tuple[float, float] = (1e-6, 10.0),
) -> dict[str, KineticsFitResult]:
    """
    Fit kinetics for each DoE run.

    `runs` is a dict mapping run label → dict with keys:
        't_end_spray', 'spray_rate', 'dry_matter_conc',
        'batch_size', 'coating_mass_exp'

    Returns dict mapping run label → KineticsFitResult.
    """
    fit_fn = fit_order1 if order == 1 else fit_order2
    results = {}
    for label, r in runs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results[label] = fit_fn(
                t_end_spray=r["t_end_spray"],
                spray_rate=r["spray_rate"],
                dry_matter_conc=r["dry_matter_conc"],
                batch_size=r["batch_size"],
                coating_mass_exp=r["coating_mass_exp"],
                k_bounds=k_bounds,
            )
    return results
