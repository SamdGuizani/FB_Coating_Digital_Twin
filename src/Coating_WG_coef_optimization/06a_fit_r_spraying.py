"""
Step 6a - Correlation for r_spraying using AICc-driven model selection.

Strategy (two-stage AICc exhaustive search):

  Stage 1 — Main effects
    Candidate predictors (9 variables from DoE_recipe_and_inputs.csv):
      spray_rate_g_min    : measured spray rate [g/min]
      MCP                 : microclimate pressure factor [0.2 / 0.33 / 0.45]
      atom_pressure_bar   : atomisation air pressure [bar]
      batch_size_kg       : batch mass [kg]
      coating_conc_pct    : coating solution concentration [%]
      SSA_cm2g            : specific surface area [cm^2/g]
      humidity_g_kg       : measured absolute inlet humidity [g/kg]
      sp_inlet_T_C        : inlet air temperature during spraying [degC]
      dm_ratio_g_kg       : dry coating mass target / batch mass * 1000 [g/kg]
                            (engineered: Qty_sol * coating_conc / batch * 1000)
    Search: all 2^9 = 512 subsets -> AICc selects 3 main effects.

  Stage 2 — Pairwise interactions
    Fix the 3 selected main effects; search all 2^3 = 8 subsets of their
    pairwise interactions (main effects centred to reduce collinearity).
    AICc selects one interaction term.

Selected model (4 terms + intercept):
    spray_rate  +  coating_conc  +  DM_ratio  +  coating_conc * DM_ratio

Outputs
-------
- data/r_spraying_model_metrics.csv    -- R2, Adj-R2, AICc, RMSE, LOO-CV RMSE
- data/r_spraying_model_coeff.csv      -- coefficients, std errors, t, p
- data/r_spraying_parity.png           -- parity + residuals for selected model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import combinations

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA    = os.path.join(ROOT, "data")
RSP_CSV = os.path.join(DATA, "r_spraying_inferred.csv")
DOE_CSV = os.path.join(DATA, "DoE_recipe_and_inputs.csv")
OUT_MET = os.path.join(DATA, "r_spraying_model_metrics.csv")
OUT_COE = os.path.join(DATA, "r_spraying_model_coeff.csv")
OUT_PNG = os.path.join(DATA, "r_spraying_parity.png")

# ── Data preparation ──────────────────────────────────────────────────────────
rsp = pd.read_csv(RSP_CSV); rsp["Run"] = rsp["Run"].str.strip()
doe = pd.read_csv(DOE_CSV); doe["Run"] = doe["Run"].str.strip()

df = rsp[["Run", "r_spraying_kgs"]].merge(
    doe[["Run", "MCP", "Qty_solution", "Batch_size",
         "Coating_Solution_Concentration", "SSA",
         "ph_humidity_g_kg", "sp_spray_rate_g_min",
         "sp_atom_pressure_bar", "sp_inlet_T_C"]],
    on="Run")
df = df[df["Run"] != "Run_20"].copy()

df["dm_ratio_g_kg"] = (df["Qty_solution"]
                       * df["Coating_Solution_Concentration"] / 100.0
                       / df["Batch_size"] * 1000.0)
df["r_sp_1e6"] = df["r_spraying_kgs"] * 1e6

y = df["r_sp_1e6"].values
n = len(y)

# ── AICc helper ───────────────────────────────────────────────────────────────
def aicc(m, k):
    return m.aic + 2.0 * k * (k + 1) / max(n - k - 1, 1e-9)

def fit_ols(df_local, col_list):
    X = sm.add_constant(df_local[col_list], prepend=True)
    m = sm.OLS(y, X).fit()
    return m, aicc(m, len(m.params))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Main-effect AICc search (512 subsets)
# ═══════════════════════════════════════════════════════════════════════════════
MAIN_COLS = {
    "sp_spray_rate_g_min"          : "Spray rate (g/min)",
    "MCP"                          : "MCP",
    "sp_atom_pressure_bar"         : "Atomisation pressure (bar)",
    "Batch_size"                   : "Batch size (kg)",
    "Coating_Solution_Concentration": "Coating concentration (%)",
    "SSA"                          : "SSA (cm2/g)",
    "ph_humidity_g_kg"             : "Inlet humidity (g/kg)",
    "sp_inlet_T_C"                 : "Inlet T spraying (degC)",
    "dm_ratio_g_kg"                : "DM ratio (g/kg)",
}
all_main = list(MAIN_COLS.keys())

stage1 = []
for size in range(0, len(all_main) + 1):
    for subset in combinations(all_main, size):
        cols = list(subset)
        m, ac = fit_ols(df, cols)
        stage1.append({"features": cols, "k": len(m.params),
                        "R2": m.rsquared, "AICc": ac})

s1 = pd.DataFrame(stage1).sort_values("AICc").reset_index(drop=True)
best_main = s1.iloc[0]["features"]
best_main_aicc = s1.iloc[0]["AICc"]

print("=" * 70)
print("Stage 1 — Main-effect AICc search  (2^9 = 512 subsets)")
print(f"Best model: k={int(s1.iloc[0]['k'])}  "
      f"R2={s1.iloc[0]['R2']:.4f}  AICc={best_main_aicc:.2f}")
print(f"Selected features: {[MAIN_COLS[f] for f in best_main]}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Interaction search on centred selected main effects (8 subsets)
# ═══════════════════════════════════════════════════════════════════════════════
# Centre selected main effects
for col in best_main:
    df[col + "_c"] = df[col] - df[col].mean()
core_c = [col + "_c" for col in best_main]

# Short labels for the centred core features
CORE_LABELS = {col + "_c": MAIN_COLS[col] + " (c)" for col in best_main}

# Build all pairwise interactions between centred core features
inter_pairs = list(combinations(core_c, 2))
inter_names = {}
for a, b in inter_pairs:
    key = a + "_X_" + b
    df[key] = df[a] * df[b]
    la = MAIN_COLS[a.replace("_c", "")].replace(" (%)", "").replace(" (g/min)", "")
    lb = MAIN_COLS[b.replace("_c", "")].replace(" (%)", "").replace(" (g/min)", "")
    inter_names[key] = f"{la} x {lb}"
inter_keys = list(inter_names.keys())

stage2 = []
stage2_models = []
for size in range(0, len(inter_keys) + 1):
    for subset in combinations(inter_keys, size):
        cols = core_c + list(subset)
        m, ac = fit_ols(df, cols)
        stage2.append({"interactions": list(subset), "k": len(m.params),
                        "R2": m.rsquared, "Adj_R2": m.rsquared_adj, "AICc": ac})
        stage2_models.append(m)

s2 = pd.DataFrame(stage2).sort_values("AICc").reset_index(drop=True)
sort_idx = pd.DataFrame(stage2).sort_values("AICc").index.tolist()
s2_sorted_models = [stage2_models[i] for i in sort_idx]

best_aicc2 = s2.iloc[0]["AICc"]
s2["dAICc"]  = s2["AICc"] - best_aicc2
rel = np.exp(-0.5 * s2["dAICc"]); s2["weight"] = rel / rel.sum()

print()
print("=" * 80)
print("Stage 2 — Interaction AICc search  (2^3 = 8 subsets, main effects fixed)")
print(f"{'Rank':<5} {'k':>3} {'R2':>7} {'AdjR2':>7} {'AICc':>9} "
      f"{'dAICc':>7} {'weight':>8}  Interactions")
print("-" * 80)
for rank, (_, row) in enumerate(s2.iterrows(), 1):
    inter_str = (", ".join(inter_names[i] for i in row["interactions"])
                 if row["interactions"] else "(none)")
    print(f"{rank:<5} {int(row['k']):>3} {row['R2']:>7.4f} {row['Adj_R2']:>7.4f} "
          f"{row['AICc']:>9.2f} {row['dAICc']:>7.2f} {row['weight']:>8.4f}  {inter_str}")
print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# Selected model — full statistics
#
# Rank-1 and rank-2 are within dAICc = 0.87 (essentially equivalent support).
# Rank-1 adds Spray x Coating concentration (p = 0.063, borderline).
# Rank-2 keeps only Coating concentration x DM ratio (p = 0.012, significant).
# We select rank-2 for parsimony: all four terms are significant, and the
# single interaction term is more interpretable.
# ═══════════════════════════════════════════════════════════════════════════════
# Find the rank-2 model: exactly the single interaction "Coating conc x DM ratio"
target_inter = next(k for k in inter_keys
                    if "Coating_Solution_Concentration_c" in k
                    and "dm_ratio_g_kg_c" in k)
rank2_rec   = next((r, m) for (_, r), m in zip(s2.iterrows(), s2_sorted_models)
                   if list(r["interactions"]) == [target_inter])
best_inter  = [target_inter]
best_model  = rank2_rec[1]
final_cols  = core_c + list(best_inter)

FINAL_LABELS = {**CORE_LABELS, **{k: inter_names[k] for k in inter_names}}
FINAL_LABELS["const"] = "Intercept"

# LOO-CV RMSE (no external dependency)
X_arr = sm.add_constant(df[final_cols], prepend=True).values
preds = np.zeros(n)
for i in range(n):
    mask = np.ones(n, dtype=bool); mask[i] = False
    beta = np.linalg.lstsq(X_arr[mask], y[mask], rcond=None)[0]
    preds[i] = float(X_arr[i] @ beta)

insamp_rmse = float(np.sqrt(best_model.mse_resid))
loocv_rmse  = float(np.sqrt(np.mean((y - preds) ** 2)))

print()
print("=" * 70)
print("Selected model")
print(f"  Main effects : {[MAIN_COLS[f] for f in best_main]}")
print(f"  Interaction  : {[inter_names[i] for i in best_inter]}")
selected_aicc = aicc(best_model, len(best_model.params))
print(f"  R2           : {best_model.rsquared:.4f}")
print(f"  Adj-R2       : {best_model.rsquared_adj:.4f}")
print(f"  AICc         : {selected_aicc:.2f}  (rank-1 AICc={best_aicc2:.2f}, dAICc={selected_aicc-best_aicc2:.2f})")
print(f"  In-sample RMSE  : {insamp_rmse:.2f} x10^-6 kg/s")
print(f"  LOO-CV RMSE     : {loocv_rmse:.2f} x10^-6 kg/s")
print(f"  CV/in-sample    : {loocv_rmse/insamp_rmse:.2f}")
print("=" * 70)
print()
print(f"{'Parameter':<35} {'Coeff(1e-6)':>13} {'Std err':>9} "
      f"{'t':>7} {'p':>8}  sig")
print("-" * 70)
coeff_rows = []
for pname in best_model.params.index:
    c  = best_model.params[pname]
    se = best_model.bse[pname]
    t  = best_model.tvalues[pname]
    p  = best_model.pvalues[pname]
    sig = ("***" if p < 0.001 else "**" if p < 0.01 else
           "*"   if p < 0.05  else "."  if p < 0.10 else "")
    lbl = FINAL_LABELS.get(pname, pname)
    print(f"{lbl:<35} {c:>13.4f} {se:>9.4f} {t:>7.2f} {p:>8.4f}  {sig}")
    coeff_rows.append({"Parameter": lbl, "Coefficient_1e6_kgs": c,
                        "Std_err": se, "t_stat": t, "p_value": p, "Signif": sig})
print("=" * 70)
print("Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")

# ── Save outputs ──────────────────────────────────────────────────────────────
metrics = pd.DataFrame([{
    "Model"       : "Spray rate + Coating conc + DM ratio + Coating conc x DM ratio",
    "n"           : n,
    "k"           : len(best_model.params),
    "R2"          : round(best_model.rsquared, 4),
    "Adj_R2"      : round(best_model.rsquared_adj, 4),
    "AICc"        : round(selected_aicc, 2),
    "RMSE_insamp" : round(insamp_rmse, 3),
    "RMSE_LOOCV"  : round(loocv_rmse, 3),
    "CV_ratio"    : round(loocv_rmse / insamp_rmse, 2),
}])
metrics.to_csv(OUT_MET, index=False)

coeff_df = pd.DataFrame(coeff_rows)
coeff_df.to_csv(OUT_COE, index=False)

print(f"\nMetrics saved  : {OUT_MET}")
print(f"Coefficients   : {OUT_COE}")

# ── Parity + residual plot ────────────────────────────────────────────────────
y_fit  = best_model.fittedvalues
resid  = best_model.resid

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Step 6a -- r_spraying selected model\n"
    "Spray rate + Coating conc + DM ratio + Coating conc x DM ratio",
    fontsize=11, fontweight="bold")

ax = axes[0]
lim = max(y.max(), y_fit.max()) * 1.08
ax.scatter(y, y_fit, color="steelblue", s=60, zorder=3)
ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
for xi, yi, run in zip(y, y_fit, df["Run"]):
    ax.annotate(run.replace("Run_", ""), (xi, yi),
                textcoords="offset points", xytext=(4, 2), fontsize=7)
ax.set_xlabel("Inferred r_spraying (x10^-6 kg/s)")
ax.set_ylabel("Predicted r_spraying (x10^-6 kg/s)")
ax.set_title(f"Parity  R2={best_model.rsquared:.3f}  Adj-R2={best_model.rsquared_adj:.3f}")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(y_fit, resid, color="darkorange", s=60, zorder=3)
ax.axhline(0, color="k", lw=1, ls="--")
for xi, ri, run in zip(y_fit, resid, df["Run"]):
    ax.annotate(run.replace("Run_", ""), (xi, ri),
                textcoords="offset points", xytext=(4, 2), fontsize=7)
ax.set_xlabel("Fitted r_spraying (x10^-6 kg/s)")
ax.set_ylabel("Residual (x10^-6 kg/s)")
ax.set_title(f"Residuals   RMSE={insamp_rmse:.2f}  LOO-CV={loocv_rmse:.2f}")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=120)
plt.close(fig)
print(f"Parity plot    : {OUT_PNG}")
