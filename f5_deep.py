#!/usr/bin/env python3
"""F5 deep diagnostic — saturation, per-scene heterogeneity, n_rays stratification,
iter dynamics, confounder check (var_d, opacity gradient).
Output: f5_deep_summary.json + console report.
"""
import sys, os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

if len(sys.argv) < 2:
    print("usage: f5_deep.py <root>"); sys.exit(2)
root = sys.argv[1]

SCENES = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
rows = []
for s in SCENES:
    d = os.path.join(root, f"f5_scan{s}")
    p9 = os.path.join(d, "cancellation_v9.npz")
    p7 = os.path.join(d, "densify_log_v7.npz")
    if not (os.path.exists(p9) and os.path.exists(p7)):
        continue
    z9 = np.load(p9); z7 = np.load(p7, allow_pickle=True)
    iter9 = z9["iter"]; iter7 = z7["iter"]
    common = sorted(set(iter9.tolist()) & set(iter7.tolist()))
    for it in common:
        i9 = int(np.where(iter9 == it)[0][0])
        i7 = int(np.where(iter7 == it)[0][0])
        idx = z9["idx"][i9]; cm = z9["cancel_m2d"][i9]
        ms = z9["max_scale"][i9]; nr = z9["n_rays"][i9]
        vd = z9["var_d"][i9] if "var_d" in z9.files else np.zeros_like(cm)
        op_arr = z7["opacity"][i7]; pass_arr = z7["pt"][i7]
        N7 = len(pass_arr); valid = idx < N7
        for j in np.where(valid)[0]:
            gi = int(idx[j])
            ps = int(pass_arr[gi])
            ms_v = float(ms[j]); cm_v = float(cm[j]); nr_v = float(nr[j]); vd_v = float(vd[j])
            op_v = float(op_arr[gi])
            if not (np.isfinite(cm_v) and np.isfinite(ms_v) and ms_v > 0 and op_v > 0 and nr_v > 0):
                continue
            if not (0 <= cm_v <= 1.0 + 1e-5):
                continue
            rows.append((s, it, gi, ps, cm_v, np.log(ms_v), np.log(nr_v), np.log(op_v), vd_v))

df = pd.DataFrame(rows, columns=["scene","iter","gid","pass","cancel_m2d","log_scale","log_nrays","log_opacity","var_d"])
qbins = np.percentile(df["log_scale"], [0,20,40,60,80,100])
df["q"] = np.clip(np.digitize(df["log_scale"], qbins[1:-1]), 0, 4)
print(f"Total n={len(df)} across {df['scene'].nunique()} scenes")

summary = {"n_obs": len(df), "n_scenes": int(df["scene"].nunique())}

# === 1. Saturation diagnostic ===
print("\n=== 1. Saturation diagnostic (cancel=1.0 ratio per quantile) ===")
sat_table = {}
for q in range(5):
    sub = df[df["q"] == q]
    if len(sub) == 0: continue
    sat = (sub["cancel_m2d"] > 0.999).sum() / len(sub)
    nrays_med = sub["n_rays"].median() if False else np.exp(sub["log_nrays"].median())
    nrays_med_sat = np.exp(sub.loc[sub["cancel_m2d"] > 0.999, "log_nrays"].median()) if (sub["cancel_m2d"] > 0.999).sum() > 0 else float("nan")
    nrays_med_unsat = np.exp(sub.loc[sub["cancel_m2d"] <= 0.999, "log_nrays"].median()) if (sub["cancel_m2d"] <= 0.999).sum() > 0 else float("nan")
    print(f"  q{q}: sat={sat*100:5.1f}%  median(n_rays)_all={nrays_med:.1f}  median(n_rays|sat)={nrays_med_sat:.1f}  median(n_rays|unsat)={nrays_med_unsat:.1f}")
    sat_table[q] = {"sat_ratio": float(sat), "nrays_med": float(nrays_med), "nrays_sat": float(nrays_med_sat), "nrays_unsat": float(nrays_med_unsat)}
summary["saturation"] = sat_table

# === 2. Per-scene heterogeneity (q3-q4 OR per scene) ===
print("\n=== 2. Per-scene heterogeneity (Model C q4 OR per scene) ===")
per_scene = {}
for s in SCENES:
    sub = df[(df["scene"] == s) & (df["q"] == 4)]
    if len(sub) < 100 or sub["pass"].nunique() < 2: continue
    try:
        m = smf.logit("Q('pass') ~ cancel_m2d + log_nrays + log_opacity", data=sub).fit(disp=0, maxiter=200)
        b = m.params["cancel_m2d"]; se = m.bse["cancel_m2d"]; p = m.pvalues["cancel_m2d"]
        OR = np.exp(b)
        print(f"  scan{s}: n={len(sub):>5} q4 OR={OR:.3f}  p={p:.2e}")
        per_scene[s] = {"OR": float(OR), "p": float(p), "n": len(sub)}
    except Exception as e:
        print(f"  scan{s}: failed: {e}")
summary["per_scene_q4"] = per_scene
ors_per = [v["OR"] for v in per_scene.values()]
if ors_per:
    print(f"  mean={np.mean(ors_per):.3f}  median={np.median(ors_per):.3f}  IQR=[{np.percentile(ors_per,25):.3f}, {np.percentile(ors_per,75):.3f}]  range=[{min(ors_per):.3f}, {max(ors_per):.3f}]")
    print(f"  Scenes with OR<0.3: {sum(1 for o in ors_per if o<0.3)}/{len(ors_per)}")
    summary["per_scene_q4_stats"] = {"mean": float(np.mean(ors_per)), "median": float(np.median(ors_per)), "min": float(min(ors_per)), "max": float(max(ors_per))}

# === 3. n_rays-stratified rerun (high n_rays only) ===
print("\n=== 3. n_rays-stratified (high n_rays subset) ===")
nrays_med = df["log_nrays"].median()
hi = df[df["log_nrays"] > nrays_med]
print(f"  filter: log_nrays > {nrays_med:.2f} (n_rays > {np.exp(nrays_med):.1f}); subset n={len(hi)}")
hi_results = {}
for q in (3, 4):
    sub = hi[hi["q"] == q]
    if len(sub) < 100: continue
    try:
        m = smf.logit("Q('pass') ~ cancel_m2d + log_nrays + log_opacity + C(scene)", data=sub).fit(disp=0, maxiter=200)
        OR = np.exp(m.params["cancel_m2d"]); p = m.pvalues["cancel_m2d"]
        print(f"  q{q} (high n_rays): n={len(sub)}  OR={OR:.3f}  p={p:.2e}")
        hi_results[q] = {"OR": float(OR), "p": float(p), "n": len(sub)}
    except Exception as e:
        print(f"  q{q} fail: {e}")
summary["high_nrays_q34"] = hi_results

# === 4. Iter dynamics ===
print("\n=== 4. Iter dynamics (early <7000 vs mid 7000-12000 vs late >=12000) ===")
df["phase"] = pd.cut(df["iter"], bins=[0, 7000, 12000, 100000], labels=["early","mid","late"])
iter_results = {}
for ph in ["early","mid","late"]:
    sub = df[(df["phase"] == ph) & (df["q"] >= 3)]
    if len(sub) < 100: continue
    try:
        m = smf.logit("Q('pass') ~ cancel_m2d + log_scale + log_nrays + log_opacity + C(scene)", data=sub).fit(disp=0, maxiter=200)
        OR = np.exp(m.params["cancel_m2d"]); p = m.pvalues["cancel_m2d"]
        n_pass = sub["pass"].sum()
        print(f"  {ph} (q3+q4): n={len(sub):>6}  pass={n_pass:>5}  OR={OR:.3f}  p={p:.2e}")
        iter_results[ph] = {"OR": float(OR), "p": float(p), "n": len(sub), "n_pass": int(n_pass)}
    except Exception as e:
        print(f"  {ph} fail: {e}")
summary["iter_phase_q34"] = iter_results

# === 5. Confounder: var_d correlation ===
print("\n=== 5. Confounder check (var_d as multi-surface proxy) ===")
print(f"  Pearson r(cancel_m2d, var_d) = {np.corrcoef(df['cancel_m2d'], df['var_d'])[0,1]:.4f}")
# Add var_d to logistic regression — does cancel survive?
print("  Logistic regression with var_d added (q4):")
sub = df[df["q"] == 4]
try:
    m = smf.logit("Q('pass') ~ cancel_m2d + var_d + log_nrays + log_opacity + C(scene)", data=sub).fit(disp=0, maxiter=200)
    OR_c = np.exp(m.params["cancel_m2d"]); p_c = m.pvalues["cancel_m2d"]
    OR_v = np.exp(m.params["var_d"]); p_v = m.pvalues["var_d"]
    print(f"    cancel_m2d OR={OR_c:.3f} p={p_c:.2e}")
    print(f"    var_d      OR={OR_v:.3f} p={p_v:.2e}")
    summary["confound_var_d"] = {"r": float(np.corrcoef(df['cancel_m2d'], df['var_d'])[0,1]),
                                 "cancel_OR_with_vard": float(OR_c), "cancel_p_with_vard": float(p_c),
                                 "vard_OR": float(OR_v), "vard_p": float(p_v)}
except Exception as e:
    print(f"    fail: {e}")

# === Save ===
out = os.path.join(root, "f5_deep_summary.json")
json.dump(summary, open(out, "w"), indent=2, default=str)
print(f"\n=== Deep summary saved: {out} ===")
