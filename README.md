# FB_Coating_Digital_Twin

Physics-based digital twin for a fluid bed coating process - Python port
of a MATLAB legacy model, extended with empirical correlations for coating
loss rates fitted on a 20-run Design of Experiments (2018).

---

## Overview

The model simulates a three-stage pharmaceutical coating process:

```
Pre-heating  →  Spraying  →  Drying
```

<img width="1252" height="797" alt="image" src="https://github.com/user-attachments/assets/8e2b3af7-d7ca-4203-a21d-58b7a2e2b018" />


Each stage solves a system of ODEs for particle temperature, gas
temperature, solvent (acetone) content on particles and in the gas phase,
and coating mass deposited on the batch. Dissolution profiles are predicted
from the coating weight gain using a first-order permeation model.

The full modelling background — ODE derivations, all equations, the
inverse-modelling pipeline and nomenclature — is documented in
[MODELLING_BACKGROUND_README.md](MODELLING_BACKGROUND_README.md).

The twin can be used interactively either through the Jupyter notebooks
(see below) or through a **Streamlit web app** (locally: `streamlit run app.py` or cloud deployed: https://huggingface.co/spaces/SamdGuizani/fluid-bed-coating-twin).

<img width="1269" height="614" alt="image" src="https://github.com/user-attachments/assets/0490885e-4f53-4e23-9163-a353ae02547d" />


Two empirical correlations, fitted on the 19-run DoE, replace fixed
literature values for the coating attrition rates:

| Parameter | Model | R² | LOO-CV ratio |
|---|---|---|---|
| `r_spraying` | Spray rate + coating conc + DM ratio + interaction | 0.844 | 1.11 |
| `r_drying` | Batch size + DM ratio + SSA + humidity + interaction | 0.851 | 1.36 |

Both models were selected by two-stage AICc exhaustive subset search.

---

## Repository structure

```
.
├── app.py                           # Streamlit web app (replicates notebook 05b)
│
├── notebooks/
│   ├── 01_preheating.ipynb          # Single-stage pre-heating exploration
│   ├── 02_spraying.ipynb            # Single-stage spraying exploration
│   ├── 03_drying.ipynb              # Single-stage drying exploration
│   ├── 04_dissolution.ipynb         # First-order dissolution model
│   ├── 05a_FB_Coating_Digital_Twin_Demo.ipynb   # Full twin — manual r sliders
│   ├── 05b_FB_Coating_Digital_Twin.ipynb        # Full twin — empirical correlations
│   └── 06_FB_Twin_Validation.ipynb  # Validation against 20 DoE runs
│
├── src/
│   ├── fluid_bed/                   # Installable simulation package
│   │   ├── parameters.py            # ProcessParameters dataclass
│   │   ├── kinetics.py              # Drying/evaporation kinetics
│   │   ├── transfer.py              # Heat & mass transfer coefficients
│   │   ├── drying.py                # Drying rate calculation
│   │   ├── data_loader.py           # Bosch CSV / dissolution / .mat parsers
│   │   └── models/
│   │       ├── preheating.py        # Pre-heating ODE solver
│   │       ├── spraying.py          # Spraying ODE solver
│   │       ├── drying_stage.py      # Drying ODE solver + legacy r_drying correlation
│   │       ├── dissolution.py       # Dissolution model
│   │       └── coating_correlations.py  # DoE-fitted r_spraying & r_drying functions
│   │
│   ├── extract_doe_inputs.py        # Extract process parameters from raw DoE CSVs
│   └── Coating_WG_coef_optimization/
│       ├── 01_fit_dissolution_k.py          # Fit k [1/s] from observed dissolution
│       ├── 02_invert_dissolution_model.py   # Back-calculate WG from k
│       ├── 03_invert_spraying_ode.py        # Analytical inversion → r_spraying
│       ├── 04_invert_drying_ode.py          # Analytical inversion → r_drying
│       ├── 05_validate_inferred_params.py   # RMSE vs observed dissolution
│       ├── 06a_fit_r_spraying.py            # AICc regression for r_spraying
│       └── 06b_fit_r_drying.py             # AICc regression for r_drying
│
├── MATLAB_Legacy/                   # Original MATLAB source (reference only)
│   └── Coater 1 stage/
│       └── *.m, *.mlx
│
├── MODELLING_BACKGROUND_README.md   # Full modelling background (equations, derivations)
├── environment.yml
└── pyproject.toml
```

> **Data note:** the `data/` folder (raw DoE CSVs, dissolution profiles,
> fitted outputs) is excluded from version control via `.gitignore`.
> Contact the author for information.

---

## Installation

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate FB_twin

# 2. Install the fluid_bed package in editable mode
pip install -e .
```

Requires Python 3.11.

---

## Notebooks

### Stage-by-stage exploration (01–04)
Walk through each process stage in isolation: parameter sensitivity,
temperature and solvent profiles, and dissolution curve shape.

### Interactive digital twin — Demo (05a)
Full PH → SP → DR simulation with sliders for every process parameter,
including **manual sliders for r_spray (0–40 ×10⁻⁶) and r_dry
(0–10 ×10⁻³)**. Intended for exploring the sensitivity of coating
weight gain and dissolution to the loss-rate coefficients.
A dotted line shows the theoretical no-loss WG for reference.

### Interactive digital twin — Empirical correlations (05b)
Same interface but **r_spray and r_dry are computed automatically** from
the process parameters using the DoE-fitted correlations. The humidity
slider operates in real units (g/kg) to feed the r_drying model correctly.

### DoE validation (06)
Run-by-run comparison against observed temperature profiles and dissolution
data for all 20 DoE runs. A slider selects the run; both figures update
automatically.

---

## Streamlit web app

A browser-based version of the digital twin (same model as notebook 05b:
`r_spray` and `r_dry` computed from the DoE-fitted empirical correlations):

```bash
conda activate FB_twin
streamlit run app.py
```

The app is also deployed at https://huggingface.co/spaces/SamdGuizani/fluid-bed-coating-twin.

The app opens in the browser with process-parameter controls in a sidebar
and the simulated temperature/solvent/coating profiles and predicted
dissolution curve in the main panel — no Jupyter required, convenient for
sharing the twin with non-programmers.

---

## Coefficient optimisation pipeline

Run the scripts in `src/Coating_WG_coef_optimization/` in order to
reproduce the fitted correlations from scratch:

```bash
python src/extract_doe_inputs.py          # populate DoE_recipe_and_inputs.csv

python src/Coating_WG_coef_optimization/01_fit_dissolution_k.py
python src/Coating_WG_coef_optimization/02_invert_dissolution_model.py
python src/Coating_WG_coef_optimization/03_invert_spraying_ode.py
python src/Coating_WG_coef_optimization/04_invert_drying_ode.py
python src/Coating_WG_coef_optimization/05_validate_inferred_params.py
python src/Coating_WG_coef_optimization/06a_fit_r_spraying.py
python src/Coating_WG_coef_optimization/06b_fit_r_drying.py
```

Each script prints a summary and saves results to `data/`.
Run 20 is excluded from steps 02–06 (anomalous experiment results, leading to
a physically impossible implied weight gain).

---

## Key modelling assumptions

- Particle temperature is uniform within the bed (lumped model)
- Gas temperature is quasi-steady (effectiveness-NTU approximation)
- Coating attrition during spraying is order-0: `dMc/dt = −r_spray`
- Coating attrition during drying is order-1: `dMc/dt = −r_dry · Mc/batch`
- Dissolution follows a first-order permeation model:
  `F(t) = 100 · (1 − exp(−k·t))` with `k ∝ SSA² / coating_mass`
- Inlet air moisture is set to zero for all simulations
  (humidity enters only through the empirical r_drying correlation)

See [MODELLING_BACKGROUND_README.md](MODELLING_BACKGROUND_README.md) for the
complete list of assumptions and limitations, the full ODE systems and the
parameter-estimation methodology.

---

## Reference

Dataset: DoE, December 2018 - 20-run fluid bed coating experiment
(internal; not redistributed in this repository).
