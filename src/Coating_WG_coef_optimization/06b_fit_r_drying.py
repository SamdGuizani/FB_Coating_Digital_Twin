"""
Step 6b - Correlation for r_drying using AICc-driven model selection.

Strategy (two-stage AICc exhaustive search, mirroring Step 6a):

  Stage 1 — Main effects
    Candidate predictors (9 variables):
      spray_rate_g_min    : measured spray rate during coating [g/min]
      MCP                 : microclimate pressure factor [0.2 / 0.33 / 0.45]
      atom_pressure_bar   : atomisation air pressure [bar]
      batch_size_kg       : batch mass [kg]
      coating_conc_pct    : coating solution concentration [%]
      dm_ratio_g_kg       : dry coating mass target / batch * 1000 [g/kg]
      SSA_cm2g            : specific surface area [cm^2/g]
      humidity_g_kg       : measured absolute inlet humidity [g/kg]
      drying_dur_min      : drying phase duration [min]
    Search: all 2^9 = 512 subsets -> AICc selects best main-effect subset.

  Stage 2 — Pairwise interactions
    Fix the selected main effects; search all 2^m subsets of their pairwise
    interactions (main effects centred). AICc selects the best interaction model.
    Parsimony rule: if rank-1 and rank-2 are within dAICc < 2 and rank-2 has
    all terms significant, prefer the simpler model.

Outputs
-------
- data/r_drying_model_metrics.csv   -- R2, Adj-R2, AICc, RMSE, LOO-CV RMSE
- data/r_drying_model_coeff.csv     -- coefficients, std errors, t, p
- data/r_drying_parity.png          -- parity + residuals for selected model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import combinations

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA    = os.path.join(ROOT, "data")
RDR_CSV = os.path.join(DATA, "r_drying_inferred.csv")
DOE_CSV = os.path.join(DATA, "DoE_recipe_and_inputs.csv")
OUT_MET = os.path.join(DATA, "r_drying_model_metrics.csv")
OUT_COE = os.path.join(DATA, "r_drying_model_coeff.csv")
OUT_PNG = os.path.join(DATA, "r_drying_parity.png")

# ── Data preparation ──────────────────────────────────────────────────────────
rdr = pd.read_csv(RDR_CSV); rdr["Run"] = rdr["Run"].str.strip()
doe = pd.read_csv(DOE_CSV); doe["Run"] = doe["Run"].str.strip()

df = rdr[["Run", "r_dry_inferred"]].merge(
    doe[["Run", "MCP", "Qty_solution", "Batch_size",
         "Coating_Solution_Concentration", "SSA",
         "ph_humidity_g_kg", "sp_spray_rate_g_min",
         "sp_atom_pressure_bar", "dr_duration_min"]],
    on="Run")
df = df[df["Run"] != "Run_20"].copy()

df["dm_ratio_g_kg"] = (df["Qty_solution"]
                       * df["Coating_Solution_Concentration"] / 100.0
                       / df["Batch_size"] * 1000.0)

# Scale response to 1e-3 kg/s for readable coefficients
df["r_dr_1e3"] = df["r_dry_inferred"] * 1e3

y = df["r_dr_1e3"].values
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
    "dm_ratio_g_kg"                : "DM ratio (g/kg)",
    "SSA"                          : "SSA (cm2/g)",
    "ph_humidity_g_kg"             : "Inlet humidity (g/kg)",
    "dr_duration_min"              : "Drying duration (min)",
}
all_main = list(MAIN_COLS.keys())

stage1 = []
for size in range(0, len(all_main) + 1):
    for subset in combinations(all_main, size):
        cols = list(subset)
        m, ac = fit_ols(df, cols)
        stage1.append({"features": cols, "k": len(m.params),
                        "R2": m.rsquared, "Adj_R2": m.rsquared_adj, "AICc": ac})

s1 = pd.DataFrame(stage1).sort_values("AICc").reset_index(drop=True)
best_main      = s1.iloc[0]["features"]
best_main_aicc = s1.iloc[0]["AICc"]

# Show top-10 for transparency
s1_best  = s1.iloc[0]
best_aicc_s1 = s1_best["AICc"]
s1["dAICc"] = s1["AICc"] - best_aicc_s1
rel = np.exp(-0.5 * s1["dAICc"]); s1["weight"] = rel / rel.sum()

print("=" * 80)
print("Stage 1 — Main-effect AICc search  (2^9 = 512 subsets)")
print(f"{'Rank':<5} {'k':>3} {'R2':>7} {'AdjR2':>7} {'AICc':>9} "
      f"{'dAICc':>7} {'weight':>8}  Features selected")
print("-" * 80)
for rank, (_, row) in enumerate(s1.head(10).iterrows(), 1):
    feat_str = ", ".join(MAIN_COLS[f][:20] for f in row["features"]) \
               if row["features"] else "(intercept only)"
    print(f"{rank:<5} {int(row['k']):>3} {row['R2']:>7.4f} {row['Adj_R2']:>7.4f} "
          f"{row['AICc']:>9.2f} {row['dAICc']:>7.2f} {row['weight']:>8.4f}  {feat_str}")
print("=" * 80)
print(f"\nSelected main effects: {[MAIN_COLS[f] for f in best_main]}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Interaction search on centred selected main effects
# ═══════════════════════════════════════════════════════════════════════════════
for col in best_main:
    df[col + "_c"] = df[col] - df[col].mean()
core_c = [col + "_c" for col in best_main]

CORE_LABELS = {col + "_c": MAIN_COLS[col] + " (c)" for col in best_main}

# Build all pairwise interaction terms
inter_pairs = list(combinations(core_c, 2))
inter_names = {}
for a, b in inter_pairs:
    key = a + "_X_" + b
    df[key] = df[a] * df[b]
    la = MAIN_COLS[a.replace("_c", "")].replace(" (%)", "").replace(" (g/min)", "") \
                                        .replace(" (kg)", "").replace(" (bar)", "") \
                                        .replace(" (cm2/g)", "").replace(" (min)", "") \
                                        .replace(" (g/kg)", "")
    lb = MAIN_COLS[b.replace("_c", "")].replace(" (%)", "").replace(" (g/min)", "") \
                                        .replace(" (kg)", "").replace(" (bar)", "") \
                                        .replace(" (cm2/g)", "").replace(" (min)", "") \
                                        .replace(" (g/kg)", "")
    inter_names[key] = f"{la} x {lb}"
inter_keys = list(inter_names.keys())

n_inter = len(inter_keys)
print(f"Stage 2 — Interaction AICc search  "
      f"({len(best_main)} main effects -> {n_inter} pairs -> 2^{n_inter} = {2**n_inter} subsets)")

stage2 = []
stage2_models = []
for size in range(0, n_inter + 1):
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
s2["dAICc"] = s2["AICc"] - best_aicc2
rel = np.exp(-0.5 * s2["dAICc"]); s2["weight"] = rel / rel.sum()

print()
print("=" * 85)
print(f"{'Rank':<5} {'k':>3} {'R2':>7} {'AdjR2':>7} {'AICc':>9} "
      f"{'dAICc':>7} {'weight':>8}  Interactions")
print("-" * 85)
for rank, (_, row) in enumerate(s2.head(12).iterrows(), 1):
    inter_str = (", ".join(inter_names[i] for i in row["interactions"])
                 if row["interactions"] else "(none — main effects only)")
    print(f"{rank:<5} {int(row['k']):>3} {row['R2']:>7.4f} {row['Adj_R2']:>7.4f} "
          f"{row['AICc']:>9.2f} {row['dAICc']:>7.2f} {row['weight']:>8.4f}  {inter_str}")
print("=" * 85)


# ═══════════════════════════════════════════════════════════════════════════════
# Model selection: AICc rank-1 unless rank-1 and rank-2 within dAICc < 2 AND
# rank-2 has simpler structure (fewer terms) with all terms significant.
# ═══════════════════════════════════════════════════════════════════════════════
def all_significant(ols_model, alpha=0.05):
    return bool((ols_model.pvalues.drop("const", errors="ignore") < alpha).all())

rank1_m = s2_sorted_models[0]
rank1_inter = list(s2.iloc[0]["interactions"])
rank2_m = s2_sorted_models[1]
rank2_inter = list(s2.iloc[1]["interactions"])

delta12 = s2.iloc[1]["dAICc"]
rank2_simpler = len(rank2_inter) < len(rank1_inter)
rank2_all_sig = all_significant(rank2_m)

if delta12 < 2.0 and rank2_simpler and rank2_all_sig:
    chosen_inter = rank2_inter
    chosen_model = rank2_m
    chosen_aicc  = s2.iloc[1]["AICc"]
    reason = (f"rank-2 preferred (dAICc={delta12:.2f}<2, simpler, all terms p<0.05)")
else:
    chosen_inter = rank1_inter
    chosen_model = rank1_m
    chosen_aicc  = best_aicc2
    reason = "rank-1 selected (rank-2 not clearly simpler/all-significant)"

final_cols = core_c + chosen_inter
FINAL_LABELS = {**CORE_LABELS, **{k: inter_names[k] for k in inter_names}}
FINAL_LABELS["const"] = "Intercept"

# LOO-CV RMSE
X_arr  = sm.add_constant(df[final_cols], prepend=True).values
preds  = np.zeros(n)
for i in range(n):
    mask = np.ones(n, dtype=bool); mask[i] = False
    beta = np.linalg.lstsq(X_arr[mask], y[mask], rcond=None)[0]
    preds[i] = float(X_arr[i] @ beta)

insamp_rmse = float(np.sqrt(chosen_model.mse_resid))
loocv_rmse  = float(np.sqrt(np.mean((y - preds) ** 2)))

print()
print("=" * 70)
print(f"Selected model  ({reason})")
print(f"  Main effects : {[MAIN_COLS[f] for f in best_main]}")
print(f"  Interaction  : {[inter_names[i] for i in chosen_inter] if chosen_inter else ['none']}")
print(f"  R2           : {chosen_model.rsquared:.4f}")
print(f"  Adj-R2       : {chosen_model.rsquared_adj:.4f}")
print(f"  AICc         : {chosen_aicc:.2f}")
print(f"  In-sample RMSE  : {insamp_rmse:.3f} x10^-3 kg/s")
print(f"  LOO-CV RMSE     : {loocv_rmse:.3f} x10^-3 kg/s")
print(f"  CV/in-sample    : {loocv_rmse/insamp_rmse:.2f}")
print("=" * 70)
print()
print(f"{'Parameter':<40} {'Coeff(1e-3)':>12} {'Std err':>9} "
      f"{'t':>7} {'p':>8}  sig")
print("-" * 80)
coeff_rows = []
for pname in chosen_model.params.index:
    c   = chosen_model.params[pname]
    se  = chosen_model.bse[pname]
    t   = chosen_model.tvalues[pname]
    p   = chosen_model.pvalues[pname]
    sig = ("***" if p < 0.001 else "**" if p < 0.01 else
           "*"   if p < 0.05  else "."  if p < 0.10 else "")
    lbl = FINAL_LABELS.get(pname, pname)
    print(f"{lbl:<40} {c:>12.4f} {se:>9.4f} {t:>7.2f} {p:>8.4f}  {sig}")
    coeff_rows.append({"Parameter": lbl, "Coefficient_1e3_kgs": c,
                        "Std_err": se, "t_stat": t, "p_value": p, "Signif": sig})
print("=" * 80)
print("Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")

# ── Save outputs ──────────────────────────────────────────────────────────────
model_name = (" + ".join([MAIN_COLS[f] for f in best_main])
              + (" + " + " + ".join(inter_names[i] for i in chosen_inter)
                 if chosen_inter else ""))
metrics = pd.DataFrame([{
    "Model"       : model_name,
    "n"           : n,
    "k"           : len(chosen_model.params),
    "R2"          : round(chosen_model.rsquared, 4),
    "Adj_R2"      : round(chosen_model.rsquared_adj, 4),
    "AICc"        : round(chosen_aicc, 2),
    "RMSE_insamp" : round(insamp_rmse, 4),
    "RMSE_LOOCV"  : round(loocv_rmse, 4),
    "CV_ratio"    : round(loocv_rmse / insamp_rmse, 2),
}])
metrics.to_csv(OUT_MET, index=False)
pd.DataFrame(coeff_rows).to_csv(OUT_COE, index=False)

print(f"\nMetrics saved  : {OUT_MET}")
print(f"Coefficients   : {OUT_COE}")

# ── Parity + residual plot ────────────────────────────────────────────────────
y_fit = chosen_model.fittedvalues
resid = chosen_model.resid

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
inter_title = (", ".join(inter_names[i] for i in chosen_inter)
               if chosen_inter else "no interactions")
fig.suptitle(
    f"Step 6b -- r_drying selected model\n"
    f"Main: {', '.join(MAIN_COLS[f] for f in best_main)}\n"
    f"Interaction: {inter_title}",
    fontsize=10, fontweight="bold")

ax = axes[0]
lim = max(y.max(), y_fit.max()) * 1.08
ax.scatter(y, y_fit, color="steelblue", s=60, zorder=3)
ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
for xi, yi, run in zip(y, y_fit, df["Run"]):
    ax.annotate(run.replace("Run_", ""), (xi, yi),
                textcoords="offset points", xytext=(4, 2), fontsize=7)
ax.set_xlabel("Inferred r_drying (x10^-3 kg/s)")
ax.set_ylabel("Predicted r_drying (x10^-3 kg/s)")
ax.set_title(f"Parity  R2={chosen_model.rsquared:.3f}  "
             f"Adj-R2={chosen_model.rsquared_adj:.3f}")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(y_fit, resid, color="darkorange", s=60, zorder=3)
ax.axhline(0, color="k", lw=1, ls="--")
for xi, ri, run in zip(y_fit, resid, df["Run"]):
    ax.annotate(run.replace("Run_", ""), (xi, ri),
                textcoords="offset points", xytext=(4, 2), fontsize=7)
ax.set_xlabel("Fitted r_drying (x10^-3 kg/s)")
ax.set_ylabel("Residual (x10^-3 kg/s)")
ax.set_title(f"Residuals   RMSE={insamp_rmse:.3f}  LOO-CV={loocv_rmse:.3f}")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=120)
plt.close(fig)
print(f"Parity plot    : {OUT_PNG}")
