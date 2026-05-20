"""
Aggregate 9-scene × 2-view × 5-channel corrected cm from lgc6_corr/*.npz.

Output: channel × scene table of cm mean (active mask: sm > median of nonzero).
Compare to prior cv9 v5 raw-form claims for each channel.
"""
import os, sys, numpy as np

REPO = '/data/hizibu7/repos/Seraph/2d-gaussian-splatting'
DIR = f'{REPO}/paper_figures_v2/lgc6_corr'
SCENES = [3, 8, 28, 35, 40, 50, 82, 97, 105]
VIEWS = [0, 16]
CHANNELS = ['op', 'tm', 'color', 'm2d', 'n3d']

GROUPS = {'Fur': [105, 82, 50], 'Detail': [97, 3, 28], 'Simple': [40, 8, 35]}

def active_mean(cm, sm):
    nz = sm > 1e-9
    if nz.sum() == 0: return None
    med = np.median(sm[nz])
    act = sm > max(med, 1e-9)
    if act.sum() == 0: return None
    return float(cm[act].mean()), float(np.median(cm[act])), int(act.sum())

def main():
    # per (scene, view, channel)
    table = {}
    for S in SCENES:
        for V in VIEWS:
            f = f'{DIR}/lgc6_s{S}_v{V}_corr.npz'
            if not os.path.exists(f):
                print(f'MISSING {f}')
                continue
            d = np.load(f)
            for ch in CHANNELS:
                cm = d[f'cm_{ch}']
                sm = d[f'{ch}_sm']
                r = active_mean(cm, sm)
                if r is None:
                    continue
                table[(S, V, ch)] = r

    # Per-channel cross-scene summary (averaged across 18 cells)
    print("=" * 78)
    print("Per-channel cm_corrected: cross 9-scene × 2-view summary")
    print("=" * 78)
    print(f"{'ch':>8} {'cells':>6} {'cm_med':>10} {'cm_mean':>10} {'cm_p25':>10} {'cm_p75':>10}")
    for ch in CHANNELS:
        vals = [v[0] for (S, V, c), v in table.items() if c == ch]
        if not vals:
            continue
        a = np.array(vals)
        print(f"{ch:>8} {len(a):>6d} {np.median(a):>10.4f} {a.mean():>10.4f} {np.percentile(a,25):>10.4f} {np.percentile(a,75):>10.4f}")

    # Per-channel × per-group
    print()
    print("=" * 78)
    print("Per-channel × group (Fur / Detail / Simple), mean of cm_mean over scenes & views")
    print("=" * 78)
    print(f"{'ch':>8} {'Fur':>10} {'Detail':>10} {'Simple':>10}  {'F>D>S?':>12}")
    for ch in CHANNELS:
        grp_means = {}
        for gname, slist in GROUPS.items():
            vs = [v[0] for (S, V, c), v in table.items() if c == ch and S in slist]
            grp_means[gname] = np.mean(vs) if vs else float('nan')
        f, d, s = grp_means['Fur'], grp_means['Detail'], grp_means['Simple']
        ord_str = 'YES' if (f > d > s) else 'NO'
        print(f"{ch:>8} {f:>10.4f} {d:>10.4f} {s:>10.4f}  {ord_str:>12}")

    # Per-scene table (one channel per row, mean over 2 views)
    print()
    print("=" * 78)
    print("Per-scene cm_corrected (mean over v0 and v16)")
    print("=" * 78)
    print(f"{'ch':>8} " + " ".join([f"{f's{S}':>8}" for S in SCENES]))
    for ch in CHANNELS:
        row = []
        for S in SCENES:
            vs = [table.get((S, V, ch), (np.nan, 0, 0))[0] for V in VIEWS]
            vs = [v for v in vs if not np.isnan(v)]
            row.append(np.mean(vs) if vs else float('nan'))
        print(f"{ch:>8} " + " ".join([f"{v:>8.4f}" if not np.isnan(v) else f"{'-':>8}" for v in row]))

    # Verdict for Stage 1 channel selection
    print()
    print("=" * 78)
    print("STAGE 1 CHANNEL SELECTION")
    print("=" * 78)
    for ch in CHANNELS:
        vals = [v[0] for (S, V, c), v in table.items() if c == ch]
        if not vals: continue
        m = np.mean(vals)
        if m > 0.5:
            verdict = 'STRONG signal - primary candidate'
        elif m > 0.3:
            verdict = 'moderate signal - secondary'
        elif m > 0.15:
            verdict = 'weak - marginal'
        else:
            verdict = 'noise floor - drop'
        print(f"  {ch:>8}: mean cm = {m:.4f}  → {verdict}")

if __name__ == '__main__':
    main()

