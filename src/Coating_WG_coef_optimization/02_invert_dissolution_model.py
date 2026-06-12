"""
Step 2 - Invert the dissolution model to recover implied coating weight gains.

Given the fitted k values from Step 1, invert the first-order dissolution model
analytically to recover the implied coating weight gain (WG) at each sampling
time for all 20 DoE runs.

Dissolution model (forward):
    k = (Mass_sample * SSA^2 * Permeability * rho_EC) / (Volume_disso * EC_mass)
    where EC_mass = Mc / batch  [g/g, dimensionless]

Inverted:
    EC_mass = (Mass_sample * SSA^2 * Permeability * rho_EC) / (Volume_disso * k)
    WG [%]  = EC_mass * 100

Also computes for context:
  - WG_max [%]       : theoretical maximum WG (100% spray efficiency, r_spraying=0)
  - Spray_eff [%]    : WG_end_spray / WG_max x 100
  - DeltaWG_abs [%]  : WG_end_spray - WG_discharge  (absolute coating lost in drying)
  - DeltaWG_rel [%]  : (WG_end_spray - WG_discharge) / WG_end_spray x 100

Outputs
-------
- data/WG_implied.csv    -- implied WG values table for use in Steps 3 & 4
- printed summary table
"""

import os
import sys
import numpy as np
import pandas as pd

# Paths
ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA    = os.path.join(ROOT, "data")
IN_K    = os.path.join(DATA, "k_dissolution_fitted.csv")
DOE_CSV = os.path.join(DATA, "DoE_recipe_and_inputs.csv")
OUT_CSV = os.path.join(DATA, "WG_implied.csv")

# Dissolution model calibration constants — shared with the twin front-ends
from fluid_bed.config import DISSOLUTION as DISS


def k_to_WG(k_per_s, ssa_cm2g):
    """
    Invert the dissolution model: given k [1/s] and SSA [cm^2/g],
    return implied WG [%] (= EC_mass * 100).
    Returns NaN if k is NaN or zero.
    """
    if np.isnan(k_per_s) or k_per_s <= 0:
        return np.nan
    S      = DISS["Mass_sample"] * ssa_cm2g                    # cm^2
    num    = S * DISS["Permeability"] * DISS["rho_EC"] * ssa_cm2g
    denom  = DISS["Volume_disso"] * k_per_s
    EC_mass = num / denom                                       # g/g
    return EC_mass * 100.0                                      # %


# Load inputs
k_df  = pd.read_csv(IN_K)
doe   = pd.read_csv(DOE_CSV)
doe["Run"] = doe["Run"].str.strip()
k_df["Run"] = k_df["Run"].str.strip()

merged = k_df.merge(doe[["Run", "Qty_solution", "Batch_size",
                           "Coating_Solution_Concentration", "SSA",
                           "sp_spray_rate_g_min", "dr_duration_min"]],
                    on="Run", how="left")

# Compute implied WG and contextual metrics
records = []
for _, row in merged.iterrows():
    ssa     = row["SSA"]                            # cm^2/g
    batch   = row["Batch_size"]                     # kg
    dmc_frac= row["Coating_Solution_Concentration"] / 100.0
    qty_sol = row["Qty_solution"]                   # kg solution
    sr_gmin = row["sp_spray_rate_g_min"]            # g/min

    # Implied WG from fitted k
    wg_sp = k_to_WG(row["k_sp_1_per_s"], ssa)
    wg_dc = k_to_WG(row["k_dc_1_per_s"], ssa)

    # Theoretical maximum WG (100% coating efficiency)
    qty_dm_kg  = qty_sol * dmc_frac                 # kg dry matter sprayed
    wg_max     = qty_dm_kg / batch * 100.0          # %

    # Spray efficiency
    spray_eff  = wg_sp / wg_max * 100.0 if wg_max > 0 else np.nan

    # Coating change during drying
    delta_wg_abs = wg_sp - wg_dc                   # % (positive = loss)
    delta_wg_rel = delta_wg_abs / wg_sp * 100.0 if wg_sp > 0 else np.nan

    records.append({
        "Run"             : row["Run"],
        "SSA_cm2g"        : ssa,
        "k_sp_1_per_s"    : row["k_sp_1_per_s"],
        "k_dc_1_per_s"    : row["k_dc_1_per_s"],
        "WG_max_pct"      : round(wg_max,       3),
        "WG_end_spray_pct": round(wg_sp,        3),
        "Spray_eff_pct"   : round(spray_eff,    1),
        "WG_discharge_pct": round(wg_dc,        3),
        "DeltaWG_abs_pct" : round(delta_wg_abs, 3),
        "DeltaWG_rel_pct" : round(delta_wg_rel, 1),
    })

results = pd.DataFrame(records)
results.to_csv(OUT_CSV, index=False)
print(f"Results saved: {OUT_CSV}\n")

# Print summary
print("=" * 110)
print(f"{'Run':<10} {'SSA':>6} {'WG_max':>8} {'WG_sp':>8} {'Eff%':>6} "
      f"{'WG_dc':>8} {'dWG_abs':>8} {'dWG_rel%':>9}  interpretation")
print(f"{'':10} {'cm2/g':>6} {'%':>8} {'%':>8} {'':>6} "
      f"{'%':>8} {'%':>8} {'':>9}")
print("-" * 110)
for _, r in results.iterrows():
    if r["Spray_eff_pct"] > 100:
        interp = "! eff > 100%: WG implies more coating than sprayed"
    elif r["Spray_eff_pct"] < 50:
        interp = "low efficiency (<50%)"
    elif r["DeltaWG_rel_pct"] > 30:
        interp = "heavy drying attrition (>30% relative WG lost)"
    else:
        interp = ""
    print(f"{r['Run']:<10} {r['SSA_cm2g']:>6.1f} {r['WG_max_pct']:>8.3f} "
          f"{r['WG_end_spray_pct']:>8.3f} {r['Spray_eff_pct']:>6.1f} "
          f"{r['WG_discharge_pct']:>8.3f} {r['DeltaWG_abs_pct']:>8.3f} "
          f"{r['DeltaWG_rel_pct']:>9.1f}  {interp}")
print("=" * 110)

print(f"\nSummary statistics")
print(f"  WG_max         : {results['WG_max_pct'].min():.3f} -- {results['WG_max_pct'].max():.3f} %")
print(f"  WG_end_spray   : {results['WG_end_spray_pct'].min():.3f} -- {results['WG_end_spray_pct'].max():.3f} %")
print(f"  Spray_eff      : {results['Spray_eff_pct'].min():.1f} -- {results['Spray_eff_pct'].max():.1f} %   (mean {results['Spray_eff_pct'].mean():.1f}%)")
print(f"  WG_discharge   : {results['WG_discharge_pct'].min():.3f} -- {results['WG_discharge_pct'].max():.3f} %")
print(f"  DeltaWG_abs    : {results['DeltaWG_abs_pct'].min():.3f} -- {results['DeltaWG_abs_pct'].max():.3f} %")
print(f"  DeltaWG_rel    : {results['DeltaWG_rel_pct'].min():.1f} -- {results['DeltaWG_rel_pct'].max():.1f} %   (mean {results['DeltaWG_rel_pct'].mean():.1f}%)")
print(f"\n  Runs with spray_eff > 100%: {(results['Spray_eff_pct'] > 100).sum()} (check dissolution model consistency)")
