#!/usr/bin/env python3
"""Smoke analysis for cv9 mean2D cancellation. Pre-registered safeguard.
Pass = range valid, no NaN, triangle inequality holds, transMat correlation classified."""
import sys, numpy as np
if len(sys.argv) < 2:
    print("usage: analyze_smoke_cv9.py <cancellation_v9.npz>"); sys.exit(2)
z = np.load(sys.argv[1])
print(f"file: {sys.argv[1]}")
print(f"keys: {list(z.keys())}")
print(f"shape iter={z['iter'].shape} cancel_m2d={z['cancel_m2d'].shape}")

c = z["cancel_m2d"].flatten()
sx, sy, sm = z["sx_m"].flatten(), z["sy_m"].flatten(), z["sum_m"].flatten()
sg, ct = z["signed_m"].flatten(), z["cancel_t"].flatten()
ms = z["max_scale"].flatten()
nr = z["n_rays"].flatten() if "n_rays" in z.files else None

# 1. NaN/Inf
nbad = (~np.isfinite(c)).sum()
print(f"\n[1] NaN/Inf in cancel_m2d: {nbad} ({100*nbad/c.size:.2f}%)")

# 2. Range
mask = np.isfinite(c)
in_range = ((c[mask] >= -1e-5) & (c[mask] <= 1.0+1e-5)).sum()
print(f"[2] Range [0,1]: {in_range}/{mask.sum()} = {100*in_range/mask.sum():.2f}%")
print(f"    min={c[mask].min():.4f} max={c[mask].max():.4f}")

# 3. Triangle inequality: signed_m <= sum_m (i.e. cancel_m2d >= 0)
m2 = mask & (sm > 1e-12)
violation = (sg[m2] > sm[m2] + 1e-5).sum()
print(f"[3] Triangle (signed<=norm): {m2.sum()-violation}/{m2.sum()} = {100*(m2.sum()-violation)/m2.sum():.2f}% pass")

# 4. Distribution
active = m2 & (nr > 0) if nr is not None else m2
ca = c[active]
print(f"\n[4] Distribution (active, n_rays>0, n={active.sum()}):")
print(f"    cancel_m2d  p10={np.percentile(ca,10):.3f} median={np.median(ca):.3f} p90={np.percentile(ca,90):.3f}")
frozen_one = (ca > 0.999).sum() / max(ca.size,1)
frozen_zero = (ca < 0.001).sum() / max(ca.size,1)
print(f"    frozen@1 (cancel>0.999): {100*frozen_one:.2f}%")
print(f"    frozen@0 (cancel<0.001): {100*frozen_zero:.2f}%")
print(f"    expected: median 0.4-0.8, frozen ratios low")

# 5. Correlation with transMat cancel (pre-registered (a)/(b)/(c))
m3 = active & np.isfinite(ct) & (ct >= -1e-5) & (ct <= 1.0+1e-5)
if m3.sum() > 100:
    r = np.corrcoef(c[m3], ct[m3])[0,1]
    print(f"\n[5] cancel_m2d vs cancel_t correlation (n={m3.sum()}): r = {r:.4f}")
    if r > 0.9:
        cls = "(a) HIGH - transMat is good proxy, prior results robust"
    elif r >= 0.5:
        cls = "(b) MODERATE - proxy partial effect, redo logistic regression w/ mean2D"
    else:
        cls = "(c) LOW - reinterpret all transMat results"
    print(f"    Pre-registered classification: {cls}")
else:
    print(f"\n[5] insufficient data (n={m3.sum()}) for correlation")

# 6. Size dependence
try:
    bins = np.percentile(ms[active], [0,20,40,60,80,100])
    print(f"\n[6] Size-binned cancel_m2d (q0-q4 by max_scale):")
    for i in range(5):
        bm = active & (ms >= bins[i]) & (ms <= bins[i+1])
        if bm.sum() > 10:
            print(f"    q{i}: scale=[{bins[i]:.4g},{bins[i+1]:.4g}] n={bm.sum()} median_cancel={np.median(c[bm]):.3f}")
except Exception as e:
    print(f"\n[6] size binning failed: {e}")

ok = (nbad == 0) and (in_range == mask.sum()) and (violation == 0)
print(f"\n=== VERDICT: {'PASS' if ok else 'FAIL'} (sanity asserts) ===")
sys.exit(0 if ok else 1)
