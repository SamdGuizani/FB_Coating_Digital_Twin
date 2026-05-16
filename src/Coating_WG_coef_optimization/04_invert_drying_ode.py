"""
Step 4 - Invert the drying ODE to recover r_drying per run.

For order-1 attrition the drying mass balance integrates analytically:

    dMc/dt = -r_drying * (Mc / batch)
    Mc(t)  = Mc_0 * exp(-r_drying * t_dry / batch)

Inverting given implied WG values from Step 2:

    r_drying = -(batch / t_dry) * ln(WG_discharge / WG_end_spray)

Run 20 is excluded (WG inconsistent with sprayed quantity).

Three r_drying values are compared for each run:
  1. Inferred  : derived here from dissolution-implied WG values
  2. Corr.     : calc_r_drying() with DoE coded inputs (current notebooks)
  3. Fixed     : constant 3.19e-3 kg/s (default in ProcessParameters)

Outputs
-------
- data/r_drying_inferred.csv   -- r_drying per run, for use in Step 6
- printed three-way comparison table
- data/r_drying_inferred.png   -- scatter plots vs key process variables
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path so fluid_bed package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fluid_bed.models.drying_stage import calc_r_drying

DATA    = os.path.join(ROOT, "data")
WG_CSV  = os.path.join(DATA, "WG_implied.csv")
DOE_CSV = os.path.join(DATA, "DoE_recipe_and_inputs.csv")
OUT_CSV = os.path.join(DATA, "r_drying_inferred.csv")
OUT_PNG = os.path.join(DATA, "r_drying_inferred.png")

R_DRY_FIXED = 3.19e-3   # kg/s — default in ProcessParameters

wg  = pd.read_csv(WG_CSV)
doe = pd.read_csv(DOE_CSV)
doe["Run"] = doe["Run"].str.strip()
wg["Run"]  = wg["Run"].str.strip()

# Exclude Run 20
wg  = wg[wg["Run"]  != "Run_20"].copy()
doe = doe[doe["Run"] != "Run_20"].copy()

merged = wg.merge(doe[["Run", "Batch_size", "Coating_Solution_Concentration",
                         "Coating_level", "Inlet_air_humidity",
                         "sp_spray_rate_g_min", "dr_duration_min"]],
                  on="Run", how="left")

records = []
for _, row in merged.iterrows():
    batch      = row["Batch_size"]          # kg
    t_dry_s    = row["dr_duration_min"] * 60.0  # s
    wg_sp      = row["WG_end_spray_pct"]    # % (from Step 2)
    wg_dc      = row["WG_discharge_pct"]    # % (from Step 2)
    dmc_pct    = row["Coating_Solution_Concentration"]
    coat_lvl   = row["Coating_level"]
    h_coded    = row["Inlet_air_humidity"]
    sr_g_min   = row["sp_spray_rate_g_min"]

    # Invert the order-1 drying ODE analytically
    if wg_sp > 0 and wg_dc > 0 and wg_dc < wg_sp:
        r_dry_inferred = -(batch / t_dry_s) * np.log(wg_dc / wg_sp)
    else:
        r_dry_inferred = np.nan   # guard against numerical edge cases

    # Correlation-predicted value (current approach in notebooks)
    r_dry_corr = calc_r_drying(sr_g_min, coat_lvl, batch,
                               row["dr_duration_min"], dmc_pct, h_coded)

    records.append({
        "Run"               : row["Run"],
        "Spray_rate_g_min"  : sr_g_min,
        "Coating_level"     : coat_lvl,
        "Batch_size_kg"     : batch,
        "Drying_time_min"   : row["dr_duration_min"],
        "DMC_pct"           : dmc_pct,
        "Humidity_coded"    : h_coded,
        "WG_end_spray_pct"  : wg_sp,
        "WG_discharge_pct"  : wg_dc,
        "DeltaWG_rel_pct"   : row["DeltaWG_rel_pct"],
        "r_dry_inferred"    : r_dry_inferred,
        "r_dry_corr"        : r_dry_corr,
        "r_dry_fixed"       : R_DRY_FIXED,
        "ratio_inf_vs_corr" : r_dry_inferred / r_dry_corr  if r_dry_corr > 0 else np.nan,
        "ratio_inf_vs_fixed": r_dry_inferred / R_DRY_FIXED,
    })

results = pd.DataFrame(records)
results.to_csv(OUT_CSV, index=False)
print(f"Results saved: {OUT_CSV}\n")

# Print three-way comparison table
print("=" * 110)
print(f"{'Run':<10} {'WG_sp':>6} {'WG_dc':>6} {'dWG%':>6} "
      f"{'r_inf(1e-3)':>12} {'r_corr(1e-3)':>13} {'r_fix(1e-3)':>12} "
      f"{'inf/corr':>9} {'inf/fix':>8}")
print(f"{'':10} {'%':>6} {'%':>6} {'%':>6} "
      f"{'kg/s':>12} {'kg/s':>13} {'kg/s':>12} "
      f"{'ratio':>9} {'ratio':>8}")
print("-" * 110)
for _, r in results.iterrows():
    print(f"{r['Run']:<10} {r['WG_end_spray_pct']:>6.3f} {r['WG_discharge_pct']:>6.3f} "
          f"{r['DeltaWG_rel_pct']:>6.1f} "
          f"{r['r_dry_inferred']*1e3:>12.4f} "
          f"{r['r_dry_corr']*1e3:>13.4f} "
          f"{R_DRY_FIXED*1e3:>12.4f} "
          f"{r['ratio_inf_vs_corr']:>9.2f} "
          f"{r['ratio_inf_vs_fixed']:>8.2f}")
print("=" * 110)

valid = results.dropna(subset=["r_dry_inferred"])
print(f"\nInferred r_drying (x 10^-3 kg/s):")
print(f"  min    = {valid['r_dry_inferred'].min()*1e3:.4f}")
print(f"  max    = {valid['r_dry_inferred'].max()*1e3:.4f}")
print(f"  mean   = {valid['r_dry_inferred'].mean()*1e3:.4f}")
print(f"  median = {valid['r_dry_inferred'].median()*1e3:.4f}")
print(f"\nCorrelation r_drying (x 10^-3 kg/s):")
print(f"  min    = {valid['r_dry_corr'].min()*1e3:.4f}")
print(f"  max    = {valid['r_dry_corr'].max()*1e3:.4f}")
print(f"  mean   = {valid['r_dry_corr'].mean()*1e3:.4f}")
print(f"  median = {valid['r_dry_corr'].median()*1e3:.4f}")
print(f"\nFixed r_drying : {R_DRY_FIXED*1e3:.4f} x 10^-3 kg/s")
print(f"\nCorrelation between inferred and correlation-predicted r_drying:")
corr = valid[["r_dry_inferred","r_dry_corr"]].corr().iloc[0,1]
print(f"  Pearson r = {corr:.3f}")

# Scatter plots vs key process variables and parity plot
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Step 4 -- Inferred r_drying vs process variables (Run 20 excluded)",
             fontsize=12, fontweight="bold")

plot_vars = [
    ("Spray_rate_g_min", "Spray rate (g/min)"),
    ("Coating_level",    "Coating level (coded)"),
    ("Batch_size_kg",    "Batch size (kg)"),
    ("Drying_time_min",  "Drying time (min)"),
    ("DMC_pct",          "EC concentration (%)"),
    ("Humidity_coded",   "Humidity (coded)"),
    ("WG_end_spray_pct", "WG end-spray (%)"),
]

for ax, (col, lbl) in zip(axes.flatten()[:7], plot_vars):
    x = valid[col]
    y = valid["r_dry_inferred"] * 1e3
    ax.scatter(x, y, color="steelblue", s=60, zorder=3, label="Inferred")
    ax.axhline(R_DRY_FIXED * 1e3, color="grey", lw=1.2, ls=":",
               label=f"Fixed ({R_DRY_FIXED*1e3:.2f})")
    for xi, yi, run in zip(x, y, valid["Run"]):
        ax.annotate(run.replace("Run_", ""), (xi, yi),
                    textcoords="offset points", xytext=(4, 2), fontsize=6)
    ax.set_xlabel(lbl, fontsize=9)
    ax.set_ylabel("r_drying (x 10^-3 kg/s)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

# Last panel: parity plot inferred vs correlation
ax = axes.flatten()[7]
x = valid["r_dry_corr"] * 1e3
y = valid["r_dry_inferred"] * 1e3
ax.scatter(x, y, color="darkorange", s=60, zorder=3)
lim = max(x.max(), y.max()) * 1.1
ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
ax.axhline(R_DRY_FIXED * 1e3, color="grey", lw=1, ls=":", label="Fixed")
ax.axvline(R_DRY_FIXED * 1e3, color="grey", lw=1, ls=":")
for xi, yi, run in zip(x, y, valid["Run"]):
    ax.annotate(run.replace("Run_", ""), (xi, yi),
                textcoords="offset points", xytext=(4, 2), fontsize=6)
ax.set_xlabel("r_drying correlation (x 10^-3 kg/s)", fontsize=9)
ax.set_ylabel("r_drying inferred (x 10^-3 kg/s)", fontsize=9)
ax.set_title(f"Parity: inferred vs correlation  (r={corr:.2f})", fontsize=9)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=120)
plt.close(fig)
print(f"\nPlot saved: {OUT_PNG}")
