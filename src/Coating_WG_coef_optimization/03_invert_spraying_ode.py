"""
Step 3 - Invert the spraying ODE to recover r_spraying per run.

For order-0 coating loss the spraying mass balance integrates analytically:

    dMc/dt = spray_rate_dm - r_spraying
    Mc(t_spray) = (spray_rate_dm - r_spraying) * t_spray    [Mc(0) = 0]

Inverting given the implied WG from Step 2:

    r_spraying = spray_rate_dm - (WG_end_spray/100 * batch) / t_spray

where
    spray_rate_dm = spray_rate_kgs * dmc_frac   [kg DM / s]
    t_spray       = qty_solution / spray_rate    [s]

Run 20 is excluded (WG_end_spray inconsistent with sprayed quantity).

Outputs
-------
- data/r_spraying_inferred.csv  -- r_spraying per run, for use in Step 6
- printed comparison vs current fixed value (6.7e-6 kg/s)
- data/r_spraying_inferred.png  -- scatter plots vs key process variables
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA    = os.path.join(ROOT, "data")
WG_CSV  = os.path.join(DATA, "WG_implied.csv")
DOE_CSV = os.path.join(DATA, "DoE_recipe_and_inputs.csv")
OUT_CSV = os.path.join(DATA, "r_spraying_inferred.csv")
OUT_PNG = os.path.join(DATA, "r_spraying_inferred.png")

R_SPRAY_CURRENT = 6.7e-6   # kg/s — current fixed value used in notebooks

wg  = pd.read_csv(WG_CSV)
doe = pd.read_csv(DOE_CSV)
doe["Run"] = doe["Run"].str.strip()
wg["Run"]  = wg["Run"].str.strip()

# Exclude Run 20
wg  = wg[wg["Run"] != "Run_20"].copy()
doe = doe[doe["Run"] != "Run_20"].copy()

merged = wg.merge(doe[["Run", "Qty_solution", "Batch_size",
                         "Coating_Solution_Concentration",
                         "sp_spray_rate_g_min", "Spray_Air_Pressure",
                         "Coating_level"]],
                  on="Run", how="left")

records = []
for _, row in merged.iterrows():
    batch       = row["Batch_size"]            # kg
    dmc_frac    = row["Coating_Solution_Concentration"] / 100.0
    qty_sol     = row["Qty_solution"]          # kg solution
    sr_kgs      = row["sp_spray_rate_g_min"] / 60_000.0   # kg/s

    spray_rate_dm = sr_kgs * dmc_frac          # kg DM/s
    t_spray       = qty_sol / sr_kgs           # s
    wg_sp         = row["WG_end_spray_pct"]    # % (from Step 2)
    mc_end        = wg_sp / 100.0 * batch      # kg

    r_spray = spray_rate_dm - mc_end / t_spray  # kg/s

    records.append({
        "Run"               : row["Run"],
        "Spray_rate_g_min"  : row["sp_spray_rate_g_min"],
        "DMC_pct"           : row["Coating_Solution_Concentration"],
        "Batch_size_kg"     : batch,
        "Qty_solution_kg"   : qty_sol,
        "Spray_Air_Pres_bar": row["Spray_Air_Pressure"],
        "Coating_level"     : row["Coating_level"],
        "spray_rate_dm_kgs" : spray_rate_dm,
        "t_spray_s"         : round(t_spray, 1),
        "WG_max_pct"        : row["WG_max_pct"],
        "WG_end_spray_pct"  : wg_sp,
        "r_spraying_kgs"    : r_spray,
        "r_spray_current"   : R_SPRAY_CURRENT,
        "ratio_vs_current"  : r_spray / R_SPRAY_CURRENT,
    })

results = pd.DataFrame(records)
results.to_csv(OUT_CSV, index=False)
print(f"Results saved: {OUT_CSV}\n")

# Flag anomalous values
neg  = results["r_spraying_kgs"] < 0
over = results["r_spraying_kgs"] > results["spray_rate_dm_kgs"]

# Print table
print("=" * 105)
print(f"{'Run':<10} {'sr_dm(1e-6)':>12} {'t_spray':>8} {'WG_max':>7} "
      f"{'WG_sp':>7} {'r_spr(1e-6)':>12} {'vs_curr':>8}  flag")
print(f"{'':10} {'kg/s':>12} {'s':>8} {'%':>7} "
      f"{'%':>7} {'kg/s':>12} {'ratio':>8}")
print("-" * 105)
for _, r in results.iterrows():
    flag = ""
    if r["r_spraying_kgs"] < 0:
        flag = "NEGATIVE -- WG implied > WG_max"
    elif r["r_spraying_kgs"] > r["spray_rate_dm_kgs"]:
        flag = "r > spray_rate_dm -- WG would be negative"
    print(f"{r['Run']:<10} {r['spray_rate_dm_kgs']*1e6:>12.2f} {r['t_spray_s']:>8.0f} "
          f"{r['WG_max_pct']:>7.3f} {r['WG_end_spray_pct']:>7.3f} "
          f"{r['r_spraying_kgs']*1e6:>12.2f} {r['ratio_vs_current']:>8.2f}  {flag}")
print("=" * 105)
print(f"\nCurrent fixed r_spraying : {R_SPRAY_CURRENT*1e6:.2f} x 10^-6 kg/s")
valid = results[results["r_spraying_kgs"] > 0]
print(f"Inferred r_spraying (valid runs): "
      f"min={valid['r_spraying_kgs'].min()*1e6:.2f}  "
      f"max={valid['r_spraying_kgs'].max()*1e6:.2f}  "
      f"mean={valid['r_spraying_kgs'].mean()*1e6:.2f}  "
      f"median={valid['r_spraying_kgs'].median()*1e6:.2f}  (x 10^-6 kg/s)")
print(f"Negative r_spraying (WG > WG_max): {neg.sum()} runs -- {list(results.loc[neg,'Run'])}")

# Scatter plots vs key process variables
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Step 3 -- Inferred r_spraying vs process variables (Run 20 excluded)",
             fontsize=12, fontweight="bold")

plot_vars = [
    ("Spray_rate_g_min",   "Spray rate (g/min)"),
    ("DMC_pct",            "EC concentration (%)"),
    ("Batch_size_kg",      "Batch size (kg)"),
    ("Spray_Air_Pres_bar", "Atomization pressure (bar)"),
    ("Coating_level",      "Coating level (coded)"),
    ("WG_max_pct",         "WG_max (%)"),
]

for ax, (col, lbl) in zip(axes.flatten(), plot_vars):
    x = results[col]
    y = results["r_spraying_kgs"] * 1e6
    colors = ["tomato" if v < 0 else "steelblue" for v in y]
    ax.scatter(x, y, c=colors, s=60, zorder=3)
    ax.axhline(R_SPRAY_CURRENT * 1e6, color="darkorange", lw=1.5,
               ls="--", label=f"Current ({R_SPRAY_CURRENT*1e6:.1f})")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    for xi, yi, run in zip(x, y, results["Run"]):
        ax.annotate(run.replace("Run_", ""), (xi, yi),
                    textcoords="offset points", xytext=(4, 2), fontsize=6)
    ax.set_xlabel(lbl, fontsize=9)
    ax.set_ylabel("r_spraying (x 10^-6 kg/s)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=120)
plt.close(fig)
print(f"\nPlot saved: {OUT_PNG}")
