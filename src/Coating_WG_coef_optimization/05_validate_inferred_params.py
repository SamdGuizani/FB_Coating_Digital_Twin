"""
Step 5 - Validate inferred r_spraying and r_drying against observed dissolution.

Re-runs the full PH->SP->DR simulation chain for all 19 valid DoE runs using
two sets of coating loss parameters:

  A. Inferred  : r_spraying from Step 3, r_drying from Step 4
  B. Current   : r_spraying = 6.7e-6 (fixed), r_drying from calc_r_drying()

For each run and each parameter set, the predicted dissolution profiles at
end-of-spray and discharge are compared to the observed averages.

Metrics reported per run:
  - RMSE [%] on the dissolution curve at both sample times
  - Delta-RMSE: improvement of inferred params vs current params

Outputs
-------
- data/validation_results.csv     -- RMSE table for all runs and both approaches
- data/validation_dissolution.png -- 4x5 overlay plot (inferred vs current vs obs)
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fluid_bed.parameters          import ProcessParameters
from fluid_bed.models.preheating   import run_preheating
from fluid_bed.models.spraying     import run_spraying
from fluid_bed.models.drying_stage import run_drying, calc_r_drying
from fluid_bed.config import (RHO_AIR, CP_AIR, RHO_PARTICLE as RHO_PART,
                              CP_PARTICLE as CP_PART, D_BED,
                              DISSOLUTION as DISS)
from fluid_bed.models.dissolution import dissolution_k

DATA      = os.path.join(ROOT, "data")
DOE_CSV   = os.path.join(DATA, "DoE_recipe_and_inputs.csv")
RSP_CSV   = os.path.join(DATA, "r_spraying_inferred.csv")
RDR_CSV   = os.path.join(DATA, "r_drying_inferred.csv")
DSP_DIR   = os.path.join(DATA, "Disso_end_spray")
DDC_DIR   = os.path.join(DATA, "Disso_discharge")
OUT_CSV   = os.path.join(DATA, "validation_results.csv")
OUT_PNG   = os.path.join(DATA, "validation_dissolution.png")

# Run-level constants (rig constants RHO_AIR…D_BED come from fluid_bed.config)
D_PART   = 0.001           # m
T0_PART  = 293.15          # K
HUMIDITY = 0.0             # kg/kg, inlet moisture for the ODEs
R_SPRAY_CURRENT = 6.7e-6   # kg/s

def disso_curve(Mc_kg, batch_kg, ssa):
    EC = Mc_kg / batch_kg
    if EC <= 0:
        return np.zeros(DISS["Total_min"] + 1)
    k = dissolution_k(EC, ssa)
    t_s = np.arange(0, DISS["Total_min"] + 1) * 60.0
    return 100.0 * (1.0 - np.exp(-k * t_s))

def load_disso(folder, run_num, tag):
    fname = f"Bosch_DoE_(Ded_2018)_-_Run{run_num:02d}_disso_{tag}.csv"
    df = pd.read_csv(os.path.join(folder, fname))
    return df["Time"].values.astype(float), df["Average_Disso"].values.astype(float)

def rmse(F_obs, F_pred_full, t_obs):
    """RMSE of F_pred interpolated at t_obs points."""
    t_full = np.arange(0, DISS["Total_min"] + 1, dtype=float)
    F_interp = np.interp(t_obs, t_full, F_pred_full)
    return float(np.sqrt(np.mean((F_obs - F_interp) ** 2)))

def run_simulation(params_ph, params_sp, params_dr, ph_dur_s, sp_dur_s, dr_dur_s):
    res_ph = run_preheating(params_ph, duration=ph_dur_s, T_particle_init=T0_PART)
    res_sp = run_spraying(params_sp, duration=sp_dur_s,
                          T_particle_init=res_ph.T_particle[-1])
    res_dr = run_drying(params_dr, duration=dr_dur_s,
                        Y_particle_init=res_sp.Y_particle[-1],
                        Y_gas_init=res_sp.Y_gas[-1],
                        M_coating_init=res_sp.M_coating[-1],
                        T_particle_init=res_sp.T_particle[-1])
    return res_sp.M_coating[-1], res_dr.M_coating[-1]

# Load inputs
doe  = pd.read_csv(DOE_CSV);  doe["Run"]  = doe["Run"].str.strip()
rsp  = pd.read_csv(RSP_CSV);  rsp["Run"]  = rsp["Run"].str.strip()
rdr  = pd.read_csv(RDR_CSV);  rdr["Run"]  = rdr["Run"].str.strip()

# Exclude Run 20
doe = doe[doe["Run"] != "Run_20"].copy()
rsp = rsp[rsp["Run"] != "Run_20"].copy()
rdr = rdr[rdr["Run"] != "Run_20"].copy()

merged = doe.merge(rsp[["Run", "r_spraying_kgs"]], on="Run") \
            .merge(rdr[["Run", "r_dry_inferred"]], on="Run")

records = []
t_plot  = np.arange(0, DISS["Total_min"] + 1, dtype=float)

fig, axes = plt.subplots(4, 5, figsize=(20, 16), sharex=True, sharey=True)
fig.suptitle("Step 5 -- Dissolution validation: inferred (solid) vs current (dashed) vs observed (+)",
             fontsize=12, fontweight="bold")

print(f"Running simulations for {len(merged)} runs...")

for idx, (_, row) in enumerate(merged.iterrows()):
    run_id  = row["Run"]
    run_num = int(run_id.split("_")[1])
    print(f"  {run_id}", end="  ", flush=True)

    # Process parameters
    batch    = row["Batch_size"]
    ssa      = row["SSA"]
    dmc_frac = row["Coating_Solution_Concentration"] / 100.0
    qty_sol  = row["Qty_solution"]
    sr_kgs   = row["sp_spray_rate_g_min"] / 60_000.0
    sp_dur_s = qty_sol / sr_kgs
    ph_dur_s = row["ph_duration_min"] * 60.0
    dr_dur_s = row["dr_duration_min"] * 60.0
    dr_dur_min = row["dr_duration_min"]
    h_coded  = row["Inlet_air_humidity"]
    coat_lvl = row["Coating_level"]
    dmc_pct  = row["Coating_Solution_Concentration"]

    ph_m = row["ph_airflow_m3h"] / 3600.0 * RHO_AIR
    sp_m = row["sp_airflow_m3h"] / 3600.0 * RHO_AIR
    dr_m = row["dr_airflow_m3h"] / 3600.0 * RHO_AIR
    ph_K = row["ph_inlet_T_C"] + 273.15
    sp_K = row["sp_inlet_T_C"] + 273.15
    dr_K = row["dr_inlet_T_C"] + 273.15

    PHYS = dict(diameter_eq=D_PART, ssa_cm2_g=ssa,
                particle_density=RHO_PART, cp_particle=CP_PART,
                diameter_bed=D_BED, rho_air=RHO_AIR, cp_air=CP_AIR,
                batch_size=batch)

    p_ph = ProcessParameters(**PHYS,
        air_flow_rates=(ph_m, ph_m, ph_m), air_temperatures=(ph_K, ph_K, ph_K),
        air_inlet_moisture=(HUMIDITY,)*3)

    # ── A: Inferred parameters ────────────────────────────────────────────────
    r_sp_inf = row["r_spraying_kgs"]
    r_dr_inf = row["r_dry_inferred"]

    p_sp_inf = ProcessParameters(**PHYS,
        air_flow_rates=(sp_m, sp_m, sp_m), air_temperatures=(sp_K, sp_K, sp_K),
        air_inlet_moisture=(HUMIDITY,)*3,
        spray_rate=sr_kgs, dry_matter_conc=dmc_frac, r_spraying=r_sp_inf)
    p_dr_inf = ProcessParameters(**PHYS,
        air_flow_rates=(dr_m, dr_m, dr_m), air_temperatures=(dr_K, dr_K, dr_K),
        air_inlet_moisture=(HUMIDITY,)*3, r_drying=r_dr_inf)

    mc_sp_inf, mc_dc_inf = run_simulation(p_ph, p_sp_inf, p_dr_inf,
                                          ph_dur_s, sp_dur_s, dr_dur_s)
    F_sp_inf = disso_curve(mc_sp_inf, batch, ssa)
    F_dc_inf = disso_curve(mc_dc_inf, batch, ssa)

    # ── B: Current parameters ─────────────────────────────────────────────────
    r_dr_cur = calc_r_drying(row["sp_spray_rate_g_min"], coat_lvl, batch,
                             dr_dur_min, dmc_pct, h_coded)

    p_sp_cur = ProcessParameters(**PHYS,
        air_flow_rates=(sp_m, sp_m, sp_m), air_temperatures=(sp_K, sp_K, sp_K),
        air_inlet_moisture=(HUMIDITY,)*3,
        spray_rate=sr_kgs, dry_matter_conc=dmc_frac, r_spraying=R_SPRAY_CURRENT)
    p_dr_cur = ProcessParameters(**PHYS,
        air_flow_rates=(dr_m, dr_m, dr_m), air_temperatures=(dr_K, dr_K, dr_K),
        air_inlet_moisture=(HUMIDITY,)*3, r_drying=r_dr_cur)

    mc_sp_cur, mc_dc_cur = run_simulation(p_ph, p_sp_cur, p_dr_cur,
                                          ph_dur_s, sp_dur_s, dr_dur_s)
    F_sp_cur = disso_curve(mc_sp_cur, batch, ssa)
    F_dc_cur = disso_curve(mc_dc_cur, batch, ssa)

    # ── Observed data ─────────────────────────────────────────────────────────
    t_sp_obs, F_sp_obs = load_disso(DSP_DIR, run_num, "end_spray")
    t_dc_obs, F_dc_obs = load_disso(DDC_DIR, run_num, "discharge")

    # ── RMSE ─────────────────────────────────────────────────────────────────
    rmse_sp_inf = rmse(F_sp_obs, F_sp_inf, t_sp_obs)
    rmse_dc_inf = rmse(F_dc_obs, F_dc_inf, t_dc_obs)
    rmse_sp_cur = rmse(F_sp_obs, F_sp_cur, t_sp_obs)
    rmse_dc_cur = rmse(F_dc_obs, F_dc_cur, t_dc_obs)

    records.append({
        "Run"              : run_id,
        "WG_sp_inf_pct"    : round(mc_sp_inf / batch * 100, 3),
        "WG_dc_inf_pct"    : round(mc_dc_inf / batch * 100, 3),
        "WG_sp_cur_pct"    : round(mc_sp_cur / batch * 100, 3),
        "WG_dc_cur_pct"    : round(mc_dc_cur / batch * 100, 3),
        "RMSE_sp_inf"      : round(rmse_sp_inf, 2),
        "RMSE_dc_inf"      : round(rmse_dc_inf, 2),
        "RMSE_sp_cur"      : round(rmse_sp_cur, 2),
        "RMSE_dc_cur"      : round(rmse_dc_cur, 2),
        "dRMSE_sp"         : round(rmse_sp_cur - rmse_sp_inf, 2),
        "dRMSE_dc"         : round(rmse_dc_cur - rmse_dc_inf, 2),
    })

    # ── Plot ──────────────────────────────────────────────────────────────────
    ax = axes[idx // 5, idx % 5]
    # End-spray
    ax.plot(t_plot, F_sp_inf, color="darkorange", lw=1.8, label="SP inf")
    ax.plot(t_plot, F_sp_cur, color="darkorange", lw=1.2, ls="--", alpha=0.7, label="SP cur")
    ax.plot(t_sp_obs, F_sp_obs, marker="+", ls="none", color="darkorange",
            ms=7, mew=1.5, label="SP obs")
    # Discharge
    ax.plot(t_plot, F_dc_inf, color="seagreen", lw=1.8, label="DC inf")
    ax.plot(t_plot, F_dc_cur, color="seagreen", lw=1.2, ls="--", alpha=0.7, label="DC cur")
    ax.plot(t_dc_obs, F_dc_obs, marker="+", ls="none", color="seagreen",
            ms=7, mew=1.5, label="DC obs")

    ax.set_title(f"{run_id}\nRMSE sp {rmse_sp_inf:.1f}/{rmse_sp_cur:.1f} "
                 f"dc {rmse_dc_inf:.1f}/{rmse_dc_cur:.1f}",
                 fontsize=7, fontweight="bold")
    ax.set_xlim(0, 240); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=5, loc="lower right", ncol=2)

    print(f"RMSE sp={rmse_sp_inf:.1f}/{rmse_sp_cur:.1f}  dc={rmse_dc_inf:.1f}/{rmse_dc_cur:.1f}")

for ax in axes[-1]:
    ax.set_xlabel("Time (min)")
for ax in axes[:, 0]:
    ax.set_ylabel("Released (%)")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=110)
plt.close(fig)
print(f"\nPlot saved: {OUT_PNG}")

# Summary table
results = pd.DataFrame(records)
results.to_csv(OUT_CSV, index=False)
print(f"Results saved: {OUT_CSV}\n")

print("=" * 95)
print(f"{'Run':<10} {'WGsp_inf':>9} {'WGsp_cur':>9} {'WGdc_inf':>9} {'WGdc_cur':>9} "
      f"{'RMsp_inf':>9} {'RMsp_cur':>9} {'RMdc_inf':>9} {'RMdc_cur':>9}")
print(f"{'':10} {'%':>9} {'%':>9} {'%':>9} {'%':>9} "
      f"{'%':>9} {'%':>9} {'%':>9} {'%':>9}")
print("-" * 95)
for _, r in results.iterrows():
    print(f"{r['Run']:<10} {r['WG_sp_inf_pct']:>9.3f} {r['WG_sp_cur_pct']:>9.3f} "
          f"{r['WG_dc_inf_pct']:>9.3f} {r['WG_dc_cur_pct']:>9.3f} "
          f"{r['RMSE_sp_inf']:>9.2f} {r['RMSE_sp_cur']:>9.2f} "
          f"{r['RMSE_dc_inf']:>9.2f} {r['RMSE_dc_cur']:>9.2f}")
print("=" * 95)
print(f"\n{'Mean RMSE':}")
print(f"  End-spray  -- inferred: {results['RMSE_sp_inf'].mean():.2f}%   "
      f"current: {results['RMSE_sp_cur'].mean():.2f}%   "
      f"improvement: {results['dRMSE_sp'].mean():.2f}%")
print(f"  Discharge  -- inferred: {results['RMSE_dc_inf'].mean():.2f}%   "
      f"current: {results['RMSE_dc_cur'].mean():.2f}%   "
      f"improvement: {results['dRMSE_dc'].mean():.2f}%")
print(f"\n  Runs where inferred RMSE_sp < current: "
      f"{(results['dRMSE_sp'] > 0).sum()} / {len(results)}")
print(f"  Runs where inferred RMSE_dc < current: "
      f"{(results['dRMSE_dc'] > 0).sum()} / {len(results)}")
