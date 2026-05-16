"""
Step 1 - Fit first-order dissolution rate constant k from observed profiles.

For each of the 20 DoE runs, fit k [1/s] to the observed average dissolution
curves at two sampling times (end-of-spray and discharge) using nonlinear
least squares:

    F(t) = 100 * (1 - exp(-k * t))       t in seconds

Outputs
-------
- Printed table of k values and fit quality (R2) for all 20 runs
- data/k_dissolution_fitted.csv   -- results table for use in Step 2
- data/k_dissolution_fits.png     -- 4x5 grid of fit curves vs observations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Paths
ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA    = os.path.join(ROOT, "data")
DSP_DIR = os.path.join(DATA, "Disso_end_spray")
DDC_DIR = os.path.join(DATA, "Disso_discharge")
DOE_CSV = os.path.join(DATA, "DoE_recipe_and_inputs.csv")
OUT_CSV = os.path.join(DATA, "k_dissolution_fitted.csv")
OUT_PNG = os.path.join(DATA, "k_dissolution_fits.png")

doe = pd.read_csv(DOE_CSV)
doe["Run"] = doe["Run"].str.strip()


# Dissolution model and fit helpers
def _model(t_min, k_per_s):
    """First-order model: F [%] as function of t [min], k [1/s]."""
    return 100.0 * (1.0 - np.exp(-k_per_s * t_min * 60.0))


def fit_k(t_min, F_pct):
    """
    Fit k [1/s] to observed (t_min, F_pct) data.
    Returns (k_best, R2).
    """
    t_nonzero = t_min[t_min > 0]
    F_nonzero = F_pct[t_min > 0]
    if len(t_nonzero) == 0 or F_nonzero.max() < 5:
        return np.nan, np.nan

    # Rough initial guess from linear log transform
    finite = F_nonzero < 99.9
    if finite.sum() >= 2:
        k0 = float(np.median(
            -np.log(1.0 - F_nonzero[finite] / 100.0) / (t_nonzero[finite] * 60.0)
        ))
    else:
        k0 = 1e-4

    try:
        popt, _ = curve_fit(
            _model, t_min, F_pct,
            p0=[k0], bounds=(1e-8, 1e-1),
            maxfev=5000,
        )
        k_fit = float(popt[0])
    except RuntimeError:
        return np.nan, np.nan

    F_pred = _model(t_min, k_fit)
    ss_res = np.sum((F_pct - F_pred) ** 2)
    ss_tot = np.sum((F_pct - F_pct.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return k_fit, r2


def load_disso(folder, run_num, tag):
    fname = f"Bosch_DoE_(Ded_2018)_-_Run{run_num:02d}_disso_{tag}.csv"
    df = pd.read_csv(os.path.join(folder, fname))
    return df["Time"].values.astype(float), df["Average_Disso"].values.astype(float)


# Fit all 20 runs
records = []

fig, axes = plt.subplots(4, 5, figsize=(18, 14), sharex=True, sharey=True)
fig.suptitle("Step 1 -- First-order dissolution fits (avg of 3 aliquots)",
             fontsize=13, fontweight="bold")
t_plot = np.linspace(0, 240, 300)

for idx, run_num in enumerate(range(1, 21)):
    run_id = f"Run_{run_num:02d}"
    doe_row = doe[doe["Run"] == run_id].iloc[0]

    t_sp, F_sp = load_disso(DSP_DIR, run_num, "end_spray")
    t_dc, F_dc = load_disso(DDC_DIR, run_num, "discharge")

    k_sp, r2_sp = fit_k(t_sp, F_sp)
    k_dc, r2_dc = fit_k(t_dc, F_dc)

    records.append({
        "Run"              : run_id,
        "Spray_rate_g_min" : doe_row["sp_spray_rate_g_min"],
        "Coating_level"    : doe_row["Coating_level"],
        "Batch_size_kg"    : doe_row["Batch_size"],
        "Drying_time_min"  : doe_row["dr_duration_min"],
        "DMC_pct"          : doe_row["Coating_Solution_Concentration"],
        "Humidity_coded"   : doe_row["Inlet_air_humidity"],
        "SSA_cm2g"         : doe_row["SSA"],
        "k_sp_1_per_s"     : k_sp,
        "R2_sp"            : r2_sp,
        "k_dc_1_per_s"     : k_dc,
        "R2_dc"            : r2_dc,
        "k_dc_over_k_sp"   : k_dc / k_sp if (not np.isnan(k_sp) and not np.isnan(k_dc) and k_sp > 0) else np.nan,
    })

    ax = axes[idx // 5, idx % 5]
    ax.scatter(t_sp, F_sp, color="darkorange", s=20, zorder=3, label="End-spray obs")
    ax.scatter(t_dc, F_dc, color="seagreen",   s=20, zorder=3, label="Discharge obs")
    if not np.isnan(k_sp):
        ax.plot(t_plot, _model(t_plot, k_sp), color="darkorange", lw=1.5,
                label=f"k={k_sp:.2e}")
    if not np.isnan(k_dc):
        ax.plot(t_plot, _model(t_plot, k_dc), color="seagreen", lw=1.5,
                label=f"k={k_dc:.2e}")
    ax.set_title(run_id, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 240); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=6, loc="lower right")

for ax in axes[-1]:
    ax.set_xlabel("Time (min)")
for ax in axes[:, 0]:
    ax.set_ylabel("Released (%)")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=120)
plt.close(fig)
print(f"Fit plot saved: {OUT_PNG}")

# Results table
results = pd.DataFrame(records)
results.to_csv(OUT_CSV, index=False)
print(f"Results saved: {OUT_CSV}\n")

# Print summary
print("=" * 100)
print(f"{'Run':<10} {'k_sp (1/s)':>12} {'R2_sp':>7} {'k_dc (1/s)':>12} "
      f"{'R2_dc':>7} {'k_dc/k_sp':>10}  note")
print("-" * 100)
for _, r in results.iterrows():
    k_ratio = r["k_dc_over_k_sp"]
    note = ""
    if not np.isnan(k_ratio):
        if k_ratio > 1.05:
            note = "<-- k increases: coating lost during drying"
        elif k_ratio < 0.95:
            note = "<-- k decreases: coating gained during drying (?)"
    ratio_str = f"{k_ratio:>10.3f}" if not np.isnan(k_ratio) else f"{'nan':>10}"
    print(f"{r['Run']:<10} {r['k_sp_1_per_s']:>12.4e} {r['R2_sp']:>7.4f} "
          f"{r['k_dc_1_per_s']:>12.4e} {r['R2_dc']:>7.4f} "
          f"{ratio_str}  {note}")
print("=" * 100)
print(f"\nMean R2 end-spray : {results['R2_sp'].mean():.4f}")
print(f"Mean R2 discharge : {results['R2_dc'].mean():.4f}")
print(f"\nk_dc/k_sp > 1.05 (coating attrition during drying): "
      f"{(results['k_dc_over_k_sp'] > 1.05).sum()} / 20 runs")
print(f"k_dc/k_sp < 0.95 (anomalous WG gain during drying) : "
      f"{(results['k_dc_over_k_sp'] < 0.95).sum()} / 20 runs")
