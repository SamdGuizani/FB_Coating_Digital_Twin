"""
I/O helpers for DoE CSV files, dissolution Excel data, and MATLAB .mat files.

CSV format note
---------------
The process CSV files are Bosch Solidlab exports with a non-standard layout:
  - A $$Header section with machine/batch metadata
  - A $$Protocol section with 1-minute time-series data
Each protocol section contains phase marker rows, data rows, and summary rows
(Min / Max / Average / StdDev). Numbers may use a European comma-as-decimal
format inside quoted fields (e.g. "18,341" → 18.341 kg).

Use `parse_bosch_csv` to get a clean, labelled DataFrame from one run file.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── Bosch Solidlab CSV parser ──────────────────────────────────────────────────

# Column index → output name + unit
# Index is 0-based within the pandas DataFrame (col 0 = first CSV column = empty/marker)
_COL_MAP = {
    1:  "timestamp",
    4:  "T_inlet_C",          # TIR 3.04  Inlet air temperature     [°C]
    5:  "T_product_C",        # TIR 1.01  Product temperature        [°C]
    6:  "T_outlet_C",         # TIR 3.41  Outlet air temperature     [°C]
    7:  "spray_rate_g_min",   # FIR 2.01  Spray rate                 [g/min]
    8:  "weight_kg",          # WIR 2.01  Weight scale               [kg]
    9:  "spray_qty_kg",       # QIR 2.01  Cumulative spray quantity   [kg]
    10: "inlet_flow_m3h",     # FIR 3.01  Inlet air volume flow      [m³/h]
    11: "spray_pressure_bar", # PIR 2.21  Atomising air pressure      [bar]
    13: "inlet_rh_pct",       # MIR 3.02r Inlet relative humidity    [%rH]
    14: "inlet_abs_g_kg",     # MIR 3.02a Inlet absolute humidity    [g/kg]
    15: "outlet_rh_pct",      # MIR 3.41r Outlet relative humidity   [%rH]
    16: "outlet_abs_g_kg",    # MIR 3.41a Outlet absolute humidity   [g/kg]
    17: "gas_meas_pct_lel",   # QIR 3.41  Measured solvent conc.     [%LEL]
    18: "gas_calc_pct_lel",   # QIR 3.41c Calculated solvent conc.   [%LEL]
}

_SUMMARY_LABELS = {"Min", "Max", "Average", "StdDev"}
_OFFLINE_SENTINEL = -10000.0   # Solidlab value for "sensor offline / not applicable"


def _to_float(val: object) -> float:
    """
    Convert a Solidlab string value to float.

    Handles:
    - European comma-as-decimal:  "18,341"  → 18.341
    - Sensor offline sentinel:    -10000    → NaN
    - Empty / non-numeric:        ""        → NaN
    """
    s = str(val).strip().replace('"', "")
    if s in ("", "nan"):
        return np.nan
    # European format: comma decimal, no dot → replace comma with dot
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        f = float(s)
        return np.nan if f <= _OFFLINE_SENTINEL else f
    except ValueError:
        return np.nan


def _detect_phase(label: str, seen_spraying: bool) -> str:
    """Map a Solidlab phase label to a model stage name."""
    lo = label.lower()
    if "spray" in lo:
        return "spraying"
    if "preheat" in lo or "dry" in lo:
        return "drying" if seen_spraying else "preheating"
    return "other"


def parse_bosch_csv(path: Path | str) -> pd.DataFrame:
    """
    Parse one Bosch Solidlab export CSV into a clean time-series DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp         : datetime
        elapsed_s         : float  — seconds from first data point
        phase             : str    — 'preheating' | 'spraying' | 'drying'
        T_inlet_C         : float  — inlet air temperature [°C]
        T_product_C       : float  — product (particle) temperature [°C]
        T_outlet_C        : float  — outlet air temperature [°C]
        spray_rate_g_min  : float  — spray rate [g/min]
        weight_kg         : float  — weight scale reading [kg]
        spray_qty_kg      : float  — cumulative spray quantity [kg]
        inlet_flow_m3h    : float  — inlet air flow [m³/h]
        spray_pressure_bar: float  — atomising air pressure [bar]
        inlet_rh_pct      : float  — inlet relative humidity [%]
        inlet_abs_g_kg    : float  — inlet absolute humidity [g/kg dry air]
        outlet_rh_pct     : float  — outlet relative humidity [%]
        outlet_abs_g_kg   : float  — outlet absolute humidity [g/kg dry air]
        gas_meas_pct_lel  : float  — measured solvent concentration [%LEL]
        gas_calc_pct_lel  : float  — calculated solvent concentration [%LEL]

    Rows with phase 'other' (Charge, Discharge, Batch Control) are excluded.
    Summary rows (Min / Max / Average / StdDev) are excluded.
    Sensor-offline sentinel values (-10000) are replaced with NaN.
    """
    path = Path(path)
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)

    # Find the $$Protocol section
    protocol_idx = raw.index[raw.iloc[:, 0].str.strip() == "$$Protocol"]
    if protocol_idx.empty:
        raise ValueError(f"No $$Protocol section in {path.name}")
    data_rows = raw.iloc[protocol_idx[0] + 1 :].reset_index(drop=True)

    records: list[dict] = []
    current_phase = "preheating"
    seen_spraying = False

    for _, row in data_rows.iterrows():
        col1 = str(row.iloc[1]).strip() if len(row) > 1 else ""
        col2 = str(row.iloc[2]).strip() if len(row) > 2 else ""

        # Skip summary rows
        if col1 in _SUMMARY_LABELS:
            continue

        # Phase marker row — col2 contains the Solidlab phase name
        if any(kw in col2 for kw in ("Preheat", "Spray", "Charge", "Discharge", "Batch")):
            phase = _detect_phase(col2, seen_spraying)
            if phase == "spraying":
                seen_spraying = True
            current_phase = phase
            continue

        # Data row — try parsing the timestamp
        try:
            ts = pd.to_datetime(col1, format="%d.%m.%y %H:%M:%S")
        except (ValueError, TypeError):
            continue

        if current_phase == "other":
            continue

        record: dict = {"timestamp": ts, "phase": current_phase}
        for idx, name in _COL_MAP.items():
            if name == "timestamp":
                continue
            val = row.iloc[idx] if idx < len(row) else np.nan
            record[name] = _to_float(val)

        records.append(record)

    if not records:
        raise ValueError(f"No usable data rows found in {path.name}")

    df = pd.DataFrame(records).reset_index(drop=True)
    t0 = df["timestamp"].iloc[0]
    df.insert(1, "elapsed_s", (df["timestamp"] - t0).dt.total_seconds())

    return df


def load_doe_run(path: Path | str) -> pd.DataFrame:
    """
    Load one DoE run CSV. Thin wrapper around `parse_bosch_csv`.

    Example
    -------
    >>> from fluid_bed.data_loader import load_doe_run
    >>> df = load_doe_run("MATLAB_Legacy/Coater 1 stage/Input Files/...Run01.csv")
    >>> df.head()
    """
    return parse_bosch_csv(path)


def load_all_doe_runs(
    input_dir: Path | str,
    pattern: str = "*Run*.csv",
) -> dict[str, pd.DataFrame]:
    """
    Load all DoE run CSV files from `input_dir` matching `pattern`.

    Returns a dict mapping run label (e.g. 'Run01') to its parsed DataFrame.

    Example
    -------
    >>> runs = load_all_doe_runs("MATLAB_Legacy/Coater 1 stage/Input Files/")
    >>> list(runs.keys())
    ['Run01', 'Run02', ..., 'Run20']
    >>> runs['Run01']['phase'].unique()
    array(['preheating', 'spraying', 'drying'], dtype=object)
    """
    input_dir = Path(input_dir)
    runs: dict[str, pd.DataFrame] = {}
    for p in sorted(input_dir.glob(pattern)):
        m = re.search(r"Run(\d+)", p.name, re.IGNORECASE)
        label = f"Run{m.group(1).zfill(2)}" if m else p.stem
        runs[label] = parse_bosch_csv(p)
    return runs


# ── Dissolution Excel data ─────────────────────────────────────────────────────

def load_dissolution_data(path: Path | str, sheet_name: int | str = 0) -> pd.DataFrame:
    """
    Load dissolution experimental data from the Excel file.

    Returns the raw DataFrame for the requested sheet. Inspect column names
    with `df.columns` and adapt `dissolution_by_run` accordingly.

    Example
    -------
    >>> df = load_dissolution_data("MATLAB_Legacy/.../Bosch DoE Dec-2018_Dissolution data.xlsx")
    >>> print(df.columns.tolist())
    >>> print(df.head())
    """
    return pd.read_excel(path, sheet_name=sheet_name)


def dissolution_by_run(
    df: pd.DataFrame,
    time_col: str,
    dissolved_col: str,
    run_col: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Split a dissolution DataFrame into per-run (t, F%) pairs.

    Parameters
    ----------
    df           : DataFrame from load_dissolution_data
    time_col     : name of the time column [min]
    dissolved_col: name of the % dissolved column
    run_col      : name of the run identifier column

    Returns
    -------
    dict mapping run label → (t [min array], F [% array])

    Example
    -------
    >>> by_run = dissolution_by_run(df, 'Time_min', 'Dissolved_pct', 'Run')
    >>> t, F = by_run['Run01']
    """
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run, grp in df.groupby(run_col):
        grp_sorted = grp.sort_values(time_col)
        result[str(run)] = (
            grp_sorted[time_col].to_numpy(dtype=float),
            grp_sorted[dissolved_col].to_numpy(dtype=float),
        )
    return result


# ── MATLAB .mat files ──────────────────────────────────────────────────────────

def load_mat(path: Path | str) -> dict:
    """
    Load a MATLAB .mat file using scipy.io.loadmat.

    MATLAB meta-keys (__header__, __version__, __globals__) are stripped.
    squeeze_me=True flattens single-element arrays; struct_as_record=False
    returns MATLAB structs as Python objects with attribute access.

    Limitation
    ----------
    The DoE Process Parameter.mat and DoE Coating Mass Exp.mat files store
    MATLAB `table` objects (MCOS format), which scipy cannot deserialise —
    they will appear as MatlabOpaque entries. To work around this, re-export
    from MATLAB as a plain struct:

        save('DoE_Process_Parameter_struct.mat', '-struct', 'Parameter')

    or export to CSV directly from MATLAB.

    Example
    -------
    >>> data = load_mat("MATLAB_Legacy/.../DoE Process Parameter.mat")
    >>> print(list(data.keys()))               # see variable names
    >>> param = data['Parameter']              # access a variable
    >>> print(param._fieldnames)              # if it's a struct
    >>> print(param.Batch_size)               # access a struct field
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to load .mat files.") from exc

    raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    cleaned = {k: v for k, v in raw.items() if not k.startswith("__")}

    # Warn if any variable is an unreadable MCOS table object
    for k, v in cleaned.items():
        if type(v).__name__ == "MatlabOpaque":
            import warnings
            warnings.warn(
                f"Variable '{k}' in {Path(path).name} is a MATLAB table object "
                f"(MCOS format) and cannot be read by scipy. "
                f"Re-export from MATLAB as a plain struct or CSV to use it in Python.",
                UserWarning,
                stacklevel=2,
            )
    return cleaned


def load_process_parameters_mat(path: Path | str) -> dict:
    """
    Load `DoE Process Parameter.mat`.

    Note: this file stores a MATLAB `table` object which scipy cannot read
    (see load_mat docstring for the workaround). The process parameters can
    alternatively be extracted directly from the Bosch CSV files using
    `extract_run_parameters`.

    Example
    -------
    >>> data = load_process_parameters_mat("MATLAB_Legacy/.../DoE Process Parameter.mat")
    """
    return load_mat(path)


def load_coating_mass_exp_mat(path: Path | str) -> dict:
    """
    Load `DoE Coating Mass Exp.mat`.

    Note: this file stores a MATLAB `table` object which scipy cannot read
    (see load_mat docstring for the workaround). Once re-exported from MATLAB
    as a struct or CSV, the coating mass values can be used with kinetics.py.

    Example
    -------
    >>> data = load_coating_mass_exp_mat("MATLAB_Legacy/.../DoE Coating Mass Exp.mat")
    """
    return load_mat(path)


def extract_run_parameters(df: pd.DataFrame) -> dict:
    """
    Extract key operating parameters for one run from its parsed DataFrame.

    This is a Python-native alternative to reading DoE Process Parameter.mat.
    Values are derived directly from the Bosch CSV time-series.

    Returns
    -------
    dict with keys:
        T_inlet_preheating_C, T_inlet_spraying_C, T_inlet_drying_C  [°C]
        spray_rate_mean_g_min                                         [g/min]
        spray_qty_total_kg                                            [kg]
        duration_preheating_s, duration_spraying_s, duration_drying_s [s]
        inlet_flow_mean_m3h                                           [m³/h]

    Example
    -------
    >>> df = load_doe_run("...Run01.csv")
    >>> p = extract_run_parameters(df)
    >>> print(p['T_inlet_spraying_C'], p['spray_rate_mean_g_min'])
    """
    out = {}
    for phase in ("preheating", "spraying", "drying"):
        sub = df[df["phase"] == phase]
        label = phase
        out[f"T_inlet_{label}_C"] = sub["T_inlet_C"].mean()
        out[f"T_product_mean_{label}_C"] = sub["T_product_C"].mean()
        t = sub["elapsed_s"]
        out[f"duration_{label}_s"] = float(t.max() - t.min()) if len(t) > 1 else 0.0

    spray = df[df["phase"] == "spraying"]
    out["spray_rate_mean_g_min"] = spray["spray_rate_g_min"].mean()
    out["spray_qty_total_kg"] = spray["spray_qty_kg"].max() - spray["spray_qty_kg"].min()
    out["inlet_flow_mean_m3h"] = df["inlet_flow_m3h"].mean()
    return out


# ── Simulation results I/O ─────────────────────────────────────────────────────

def save_results(results: dict, path: Path | str) -> None:
    """
    Save a simulation results dict to Parquet.

    `results` should be a dict of equal-length arrays, e.g.:
        {'t': sol.t, 'T_particle': sol.T_particle, ...}

    Example
    -------
    >>> save_results({'t': t, 'T_particle': T_p}, 'outputs/run01_spraying.parquet')
    """
    pd.DataFrame(results).to_parquet(path, index=False)


def load_results(path: Path | str) -> pd.DataFrame:
    """
    Load simulation results from a Parquet file.

    Example
    -------
    >>> df = load_results('outputs/run01_spraying.parquet')
    >>> df.columns
    Index(['t', 'T_particle', 'Y_particle', ...])
    """
    return pd.read_parquet(path)
