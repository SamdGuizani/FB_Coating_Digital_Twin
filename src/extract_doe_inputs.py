"""
Extract process parameters from DoE_Process_Parameter.csv and the 20 per-run
process variable CSVs, then save a combined DoE_recipe_and_inputs.csv.

Phase assignment rules:
  Pre-heating : rows AFTER the last "Charge" row up to (not including) the
                first "Spraying" row  — i.e. "Preheat/dry 3" in the phase
                sequence, which is when the charged batch is actually heated.
                (Rows before Charge belong to machine warm-up and are excluded.)
  Spraying    : all rows whose ProcessPhase starts with "Spraying"
  Drying      : all rows after the last Spraying row (including Discharge)

Steady-state extraction:
  - Pre-heating : median of all post-Charge rows (temperature already stable)
  - Spraying    : median of rows where Spray_rate >= 50 g/min (active spray)
  - Drying      : skip first 3 rows (spray wind-down) then median

Also extracts Inlet_air_humidity_abs [g/kg] from the pre-heating phase and
prints a comparison of r_drying computed with humidity=0 vs measured humidity.

Usage:
    python src/extract_doe_inputs.py
"""

import os
import numpy as np
import pandas as pd

from fluid_bed.models.drying_stage import calc_r_drying

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOE_CSV  = os.path.join(DATA_DIR, "DoE_Process_Parameter.csv")
PV_DIR   = os.path.join(DATA_DIR, "Process_variables")
OUT_CSV  = os.path.join(DATA_DIR, "DoE_recipe_and_inputs.csv")

SPRAY_THRESHOLD = 50  # g/min — below this the spray pump is not yet active


def _med(df_phase: pd.DataFrame, col: str) -> float:
    v = df_phase[col].replace([np.inf, -np.inf], np.nan).dropna()
    return float(v.median()) if len(v) else np.nan


def extract_run(run_num: int) -> dict:
    fname = f"Bosch_DoE_(Ded_2018)_-_Run{run_num:02d}_process_variables.csv"
    fpath = os.path.join(PV_DIR, fname)
    run_id = f"Run_{run_num:02d}"

    df = pd.read_csv(fpath, low_memory=False)
    df.columns = df.columns.str.strip()

    phase_raw = df["ProcessPhase"].str.strip()
    is_spray  = phase_raw.str.startswith("Spraying")

    if not is_spray.any():
        raise ValueError(f"{run_id}: no Spraying rows found in {fname}")

    # Positional indices of first / last spraying row
    first_sp = int(np.argmax(is_spray.values))
    last_sp  = int(len(is_spray) - 1 - np.argmax(is_spray.values[::-1]))

    # Pre-heating starts after the last "Charge" row (batch loading step).
    # Rows before Charge are machine warm-up and should not be included.
    is_charge = phase_raw.str.startswith("Charge")
    if is_charge.any():
        last_charge = int(len(is_charge) - 1 - np.argmax(is_charge.values[::-1]))
        ph_start = last_charge + 1
    else:
        ph_start = 0

    # Phase slices
    ph_df = df.iloc[ph_start : first_sp].copy()         # after Charge, before Spraying
    sp_df = df.iloc[first_sp : last_sp + 1].copy()      # Spraying rows
    dr_df = df.iloc[last_sp + 1:].copy()                # after last Spraying (all)

    # ── Pre-heating ──────────────────────────────────────────────────────────
    # Temperature and humidity are already stable after charging — use all rows.
    n_ph  = len(ph_df)

    ph_inlet_T   = _med(ph_df, "Inlet_air_temp")
    ph_airflow   = _med(ph_df, "Inlet_air_volume_komp")
    ph_humidity  = _med(ph_df, "Inlet_air_humidity_abs")   # g/kg dry air
    ph_duration  = (float(ph_df["Time"].iloc[-1]) - float(ph_df["Time"].iloc[0])
                    if n_ph > 1 else 0.0)

    # ── Spraying ─────────────────────────────────────────────────────────────
    sp_active = sp_df[sp_df["Spray_rate"] >= SPRAY_THRESHOLD]
    sp_src    = sp_active if len(sp_active) >= 3 else sp_df  # fallback

    sp_inlet_T       = _med(sp_src, "Inlet_air_temp")
    sp_airflow       = _med(sp_src, "Inlet_air_volume_komp")
    sp_spray_rate    = _med(sp_src, "Spray_rate")
    sp_atom_pressure = _med(sp_src, "Spray_air_pressure")

    # ── Drying (all rows after last Spraying, including Discharge) ────────────
    n_dr  = len(dr_df)
    skip3 = min(3, n_dr)               # skip first 3 rows (spray wind-down)
    dr_ss = dr_df.iloc[skip3:] if n_dr > skip3 else dr_df

    dr_inlet_T  = _med(dr_ss, "Inlet_air_temp")
    dr_airflow  = _med(dr_ss, "Inlet_air_volume_komp")
    dr_duration = (float(dr_df["Time"].iloc[-1]) - float(dr_df["Time"].iloc[0])
                   if n_dr > 1 else 0.0)

    print(f"{run_id}  PH={ph_duration:.1f} min  humidity={ph_humidity:.1f} g/kg  "
          f"SP={sp_spray_rate:.0f} g/min @{sp_atom_pressure:.2f} bar  "
          f"DR={dr_duration:.1f} min")

    return {
        "Run"                   : run_id,
        "ph_inlet_T_C"          : round(ph_inlet_T,   1),
        "ph_airflow_m3h"        : round(ph_airflow,   0),
        "ph_humidity_g_kg"      : round(ph_humidity,  2),
        "ph_duration_min"       : round(ph_duration,  2),
        "sp_inlet_T_C"          : round(sp_inlet_T,   1),
        "sp_airflow_m3h"        : round(sp_airflow,   0),
        "sp_spray_rate_g_min"   : round(sp_spray_rate,    1),
        "sp_atom_pressure_bar"  : round(sp_atom_pressure, 2),
        "dr_inlet_T_C"          : round(dr_inlet_T,   1),
        "dr_airflow_m3h"        : round(dr_airflow,   0),
        "dr_duration_min"       : round(dr_duration,  2),
    }


def main() -> None:
    doe = pd.read_csv(DOE_CSV)
    doe["Run"] = doe["Run"].str.strip()

    records = [extract_run(i) for i in range(1, 21)]

    proc_df = pd.DataFrame(records)
    merged  = doe.merge(proc_df, on="Run", how="left")

    doe_cols  = list(doe.columns)
    proc_cols = [c for c in proc_df.columns if c != "Run"]
    merged    = merged[doe_cols + proc_cols]

    merged.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(merged)} rows to {OUT_CSV}")

    # ── r_drying comparison: humidity=0 vs DoE coded humidity factor ─────────
    # The correlation was fitted against the DoE coded humidity factor
    # (Inlet_air_humidity column: levels -1, 0, 0.5, 1) — not raw g/kg values.
    print("\n" + "="*85)
    print("r_drying comparison: humidity = 0  vs  DoE coded Inlet_air_humidity factor")
    print("="*85)
    header = (f"{'Run':<10} {'sr':>6} {'ecl':>6} {'batch':>6} {'dr_min':>7} "
              f"{'ec%':>5} {'hum_coded':>10} {'hum_g/kg':>9} "
              f"{'r_dry(h=0)':>12} {'r_dry(coded)':>13} {'ratio':>7}")
    print(header)
    print("-" * len(header))

    for _, row in merged.iterrows():
        sr        = row["sp_spray_rate_g_min"]
        ecl       = row["Coating_level"]
        bs        = row["Batch_size"]
        dt        = row["dr_duration_min"]
        ecc       = row["Coating_Solution_Concentration"]
        h_coded   = row["Inlet_air_humidity"]   # DoE coded factor (-1, 0, 0.5, 1)
        h_meas    = row["ph_humidity_g_kg"]      # measured g/kg (for reference)

        r0      = calc_r_drying(sr, ecl, bs, dt, ecc, 0.0)
        r_coded = calc_r_drying(sr, ecl, bs, dt, ecc, h_coded)
        ratio   = r_coded / r0

        print(f"{row['Run']:<10} {sr:>6.0f} {ecl:>6.2f} {bs:>6.1f} {dt:>7.1f} "
              f"{ecc:>5.1f} {h_coded:>10.2f} {h_meas:>9.1f} "
              f"{r0:>12.4e} {r_coded:>13.4e} {ratio:>7.3f}")


if __name__ == "__main__":
    main()
