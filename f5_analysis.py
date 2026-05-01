#!/usr/bin/env python3
"""F5 mean2D logistic regression — pre-registered analysis (2026-05-01).
Loads cv7 (pass) + cv9 (cancel_m2d) from 15-scene sweep, joins positionally
at matching iters, fits Model B (interaction) + Model C (stratified),
classifies into scenario 1/1.5/2/3 per pre-reg.

Usage: f5_analysis.py <root_dir>  (root contains f5_scan{N}/{cancellation_v9.npz, densify_log_v7.npz})
"""
import sys, os, glob, json
import numpy as np

if len(sys.argv) < 2:
    print("usage: f5_analysis.py <root>"); sys.exit(2)
root = sys.argv[1]

# ---- 1. Load + join ----
SCENES = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
rows = []  # list of (scene, iter, gauss_id, pass, cancel_m2d, log_scale, log_nrays, log_opacity)
loaded, missing = [], []
for s in SCENES:
    d = os.path.join(root, f"f5_scan{s}")
    p9 = os.path.join(d, "cancellation_v9.npz")
    p7 = os.path.join(d, "densify_log_v7.npz")
    if not (os.path.exists(p9) and os.path.exists(p7)):
        missing.append(s); continue
    z9 = np.load(p9); z7 = np.load(p7, allow_pickle=True)
    iter9 = z9["iter"]
    iter7 = z7["iter"]
    common = sorted(set(iter9.tolist()) & set(iter7.tolist()))
    for it in common:
        i9 = int(np.where(iter9 == it)[0][0])
        i7 = int(np.where(iter7 == it)[0][0])
        idx = z9["idx"][i9]
        cm = z9["cancel_m2d"][i9]
        ms = z9["max_scale"][i9]
        nr = z9["n_rays"][i9]
        # opacity: prefer w_sum-based proxy from cv9 (has per-Gauss); cv7 stores per-iter opacity array
        op_arr = z7["opacity"][i7]
        pass_arr = z7["pt"][i7]
        N7 = len(pass_arr)
        # Skip indices out of bounds (densify may have happened between cv9 and cv7 at same iter)
        valid = idx < N7
        if valid.sum() == 0: continue
        for j in np.where(valid)[0]:
            gi = int(idx[j])
            ps = int(pass_arr[gi])
            ms_v = float(ms[j]); cm_v = float(cm[j]); nr_v = float(nr[j])
            op_v = float(op_arr[gi])
            if not (np.isfinite(cm_v) and np.isfinite(ms_v) and ms_v > 0 and op_v > 0 and nr_v > 0):
                continue
            if not (0 <= cm_v <= 1.0 + 1e-5):
                continue
            rows.append((s, it, gi, ps, cm_v, np.log(ms_v), np.log(nr_v), np.log(op_v)))
    loaded.append(s)

print(f"Loaded scenes: {loaded}")
print(f"Missing scenes: {missing}")
print(f"Total observations: {len(rows)}")
if len(rows) < 1000:
    print("ERROR: too few observations"); sys.exit(1)

import pandas as pd
df = pd.DataFrame(rows, columns=["scene", "iter", "gid", "pass", "cancel_m2d", "log_scale", "log_nrays", "log_opacity"])

# ---- 2. Sample size pre-check ----
print("\n=== Sample size by (size_quantile × pass) ===")
qbins = np.percentile(df["log_scale"], [0, 20, 40, 60, 80, 100])
df["q"] = np.clip(np.digitize(df["log_scale"], qbins[1:-1]), 0, 4)
sz = df.groupby(["q", "pass"]).size().unstack(fill_value=0)
print(sz)
under = (sz < 50).sum().sum()
if under > 0:
    print(f"WARN: {under} cells underpowered (n<50)")

# ---- 3. Model B (interaction, scene fixed effects) ----
import statsmodels.formula.api as smf
print("\n=== Model B: interaction with scene fixed effect ===")
try:
    mB = smf.logit(
        "Q('pass') ~ log_scale * cancel_m2d + log_nrays + log_opacity + C(scene)",
        data=df
    ).fit(disp=0, maxiter=200)
    coef = mB.params; se = mB.bse; pv = mB.pvalues
    for k in ["cancel_m2d", "log_scale", "log_scale:cancel_m2d", "log_nrays", "log_opacity"]:
        if k in coef.index:
            print(f"  {k:30s} β={coef[k]:+.4f}  SE={se[k]:.4f}  p={pv[k]:.2e}  OR={np.exp(coef[k]):.3f}")
except Exception as e:
    print(f"Model B fail: {e}")
    mB = None

# ---- 4. Model C (stratified per quantile) ----
print("\n=== Model C: stratified by size quantile ===")
results_C = {}
for q in range(5):
    sub = df[df["q"] == q]
    if len(sub) < 100 or sub["pass"].nunique() < 2:
        print(f"  q{q}: n={len(sub)} skipped (insufficient)"); continue
    try:
        mC = smf.logit(
            "Q('pass') ~ cancel_m2d + log_nrays + log_opacity + C(scene)",
            data=sub
        ).fit(disp=0, maxiter=200)
        b = mC.params["cancel_m2d"]; s = mC.bse["cancel_m2d"]; p = mC.pvalues["cancel_m2d"]
        OR = np.exp(b); ci = (np.exp(b - 1.96 * s), np.exp(b + 1.96 * s))
        results_C[q] = {"OR": OR, "CI": ci, "p": p, "n": len(sub)}
        print(f"  q{q}: n={len(sub):>6}  OR={OR:.3f}  CI=({ci[0]:.3f}, {ci[1]:.3f})  p={p:.2e}")
    except Exception as e:
        print(f"  q{q} fail: {e}")

# ---- 5. Pre-registered scenario classification (q3-q4 mean OR) ----
print("\n=== Scenario classification (pre-registered) ===")
ors = [results_C[q]["OR"] for q in (3, 4) if q in results_C]
ps = [results_C[q]["p"] for q in (3, 4) if q in results_C]
if not ors:
    print("ERROR: q3/q4 missing — cannot classify"); sys.exit(1)
or_mean = float(np.mean(ors))
p_max = float(np.max(ps))  # conservative: worst p among q3, q4
print(f"  Mean OR (q3, q4) = {or_mean:.3f}")
print(f"  Max p-value (q3, q4) = {p_max:.2e}")

if or_mean < 0.5 and p_max < 0.001:
    sc = "1"
    msg = "STRONG: Conditional mechanism survives mean2D, OR < 0.5 with p<0.001"
elif 0.5 <= or_mean <= 0.85 and p_max < 0.05:
    sc = "1.5"
    msg = "ATTENUATED: Mechanism weakened but survives, 0.5 ≤ OR ≤ 0.85 with p<0.05"
elif 0.85 < or_mean < 1.15:
    sc = "2"
    msg = "INVALID: Mechanism statistically null, 0.85 < OR < 1.15"
elif or_mean >= 1.15 and p_max < 0.05:
    sc = "3"
    msg = "REVERSED: Cancel→pass odds INCREASE, OR > 1.15 with p<0.05"
else:
    sc = "ambiguous"
    msg = "AMBIGUOUS: did not match any pre-registered scenario cleanly"

print(f"\n  >>> SCENARIO {sc}: {msg}")
print(f"  Compare to prior transMat-based OR=0.30 (q3-q5)")

# ---- 6. Free analyses (F1, 가설 1) ----
print("\n=== F1: Size monotonic (median cancel_m2d by quantile) ===")
for q in range(5):
    sub = df[df["q"] == q]
    if len(sub) > 0:
        print(f"  q{q}: n={len(sub):>6}  median cancel_m2d = {sub['cancel_m2d'].median():.3f}")

print("\n=== 가설 1: Pearson r(log_scale, cancel_m2d) ===")
r = np.corrcoef(df["log_scale"], df["cancel_m2d"])[0, 1]
print(f"  r = {r:.4f}  (n={len(df)})")

# ---- 7. Save summary ----
summary = {
    "n_obs": len(df),
    "scenes": loaded,
    "missing": missing,
    "model_C": {q: {"OR": float(v["OR"]), "CI_low": float(v["CI"][0]), "CI_high": float(v["CI"][1]), "p": float(v["p"]), "n": int(v["n"])} for q, v in results_C.items()},
    "or_mean_q34": or_mean,
    "p_max_q34": p_max,
    "scenario": sc,
    "scenario_msg": msg,
    "F1_median_by_q": {q: float(df[df["q"] == q]["cancel_m2d"].median()) for q in range(5) if (df["q"] == q).sum() > 0},
    "h1_pearson_r": float(r),
}
out = os.path.join(root, "f5_summary.json")
json.dump(summary, open(out, "w"), indent=2)
print(f"\n=== Summary saved: {out} ===")
