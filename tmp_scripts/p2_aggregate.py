"""
Aggregate Phase 2 split-direction results across 4 methods × 3 seeds.

Sources:
- Train PSNR: events.out.tfevents.* in eval/p2_<METHOD>_seed<S>_scan3/
  (tag: 'train/loss_viewpoint - psnr', averaged over 5 fixed train cameras at iter 30K)
- DTU CD: eval/p2_<METHOD>_seed<S>_scan3/dtu_eval/results.json
  (keys: 'mean_d2s', 'mean_s2d', 'overall', plus F-scores)
- n_gauss: from point_cloud.ply header

Output:
- Per-run table (CSV + console)
- Per-method aggregate (mean ± std)
- Pre-registered decision table evaluation
- Verdict (PASS / REJECT)
"""
import os, glob, json
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

METHODS = ['V', 'cancel', 'random_dir', 'orthogonal_dir', 'cancel_in_plane', 'cancel_bias_vanilla']
SEEDS = [0, 1, 2]
BASE = 'eval'
PSNR_TAG = 'train/loss_viewpoint - psnr'

# Pre-registered thresholds (committed in memory before any new run completes)
# See: project_cancel_split_direction.md
THRESH = {
    'psnr_min_diff':  0.30,  # dB (Cohen's d=0.5, ~1 std)
    'psnr_strong':    0.59,  # dB (t>2 detection)
    'cd_min_diff':    0.05,  # relative (5%)
    'cd_strong':      0.10,  # relative (10%)
}


def get_psnr(run_dir):
    # Primary: tfevents iter 30K
    ev = glob.glob(f'{run_dir}/events*')
    if ev:
        ea = EventAccumulator(ev[0], size_guidance={'scalars': 0})
        ea.Reload()
        if PSNR_TAG in ea.Tags()['scalars']:
            evs = ea.Scalars(PSNR_TAG)
            last_30k = [e.value for e in evs if e.step == 30000]
            if last_30k:
                return float(last_30k[-1])
    # Fallback: parse from training log (tfevents write bug seen on cbv s1)
    import re, os, glob as g
    log_dir = 'logs'
    method_seed = os.path.basename(run_dir).replace('p2_', '').replace('_scan3', '')
    # Find log file by checking each p2main log for the matching OUT path
    for log_path in g.glob(f'{log_dir}/p2main_*.log'):
        try:
            with open(log_path, 'rb') as f:
                f.seek(0, 2); end = f.tell()
                f.seek(max(0, end - 5000))  # last 5KB
                tail = f.read().decode('latin-1', errors='ignore')
        except Exception:
            continue
        if run_dir not in tail and method_seed not in tail:
            continue
        m = re.search(r'\[ITER 30000\] Evaluating train: L1 [\d.]+ PSNR ([\d.]+)', tail)
        if m:
            return float(m.group(1))
    return None


def get_n_gauss(run_dir):
    ply = os.path.join(run_dir, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
    if not os.path.exists(ply):
        return None
    with open(ply, 'rb') as f:
        head = f.read(2000).decode('latin-1', errors='ignore')
    for line in head.split('\n'):
        if line.startswith('element vertex'):
            return int(line.split()[2])
    return None


def get_cd(run_dir):
    res = os.path.join(run_dir, 'dtu_eval', 'results.json')
    if not os.path.exists(res):
        return None
    with open(res) as f:
        d = json.load(f)
    return {
        'mean_d2s': d.get('mean_d2s'),
        'mean_s2d': d.get('mean_s2d'),
        'overall':  d.get('overall'),
    }


def collect():
    rows = []
    for m in METHODS:
        for s in SEEDS:
            rd = f'{BASE}/p2_{m}_seed{s}_scan3'
            rows.append({
                'method': m,
                'seed':   s,
                'exists': os.path.isdir(rd),
                'psnr':   get_psnr(rd),
                'n_gauss': get_n_gauss(rd),
                'cd':     get_cd(rd),
            })
    return rows


def per_method_stats(rows):
    stats = {}
    for m in METHODS:
        mrows = [r for r in rows if r['method'] == m and r['psnr'] is not None]
        psnrs = [r['psnr'] for r in mrows]
        ngs = [r['n_gauss'] for r in mrows if r['n_gauss'] is not None]
        cds = [r['cd']['overall'] for r in mrows if r['cd'] and r['cd']['overall'] is not None]
        stats[m] = {
            'n':       len(mrows),
            'psnr':    (np.mean(psnrs), np.std(psnrs, ddof=1) if len(psnrs) > 1 else 0, psnrs) if psnrs else None,
            'n_gauss': (np.mean(ngs),   np.std(ngs, ddof=1) if len(ngs) > 1 else 0)            if ngs   else None,
            'cd':      (np.mean(cds),   np.std(cds, ddof=1) if len(cds) > 1 else 0, cds)       if cds   else None,
        }
    return stats


def print_per_run(rows):
    print("\n=== PER-RUN TABLE ===")
    print(f"{'method':<16} {'seed':<5} {'psnr':<8} {'n_gauss':<10} {'CD_d2s':<8} {'CD_s2d':<8} {'CD_overall':<10}")
    for r in rows:
        psnr = f"{r['psnr']:.3f}" if r['psnr'] is not None else '-'
        ng = f"{r['n_gauss']}" if r['n_gauss'] else '-'
        if r['cd']:
            d2s = f"{r['cd']['mean_d2s']:.3f}" if r['cd']['mean_d2s'] is not None else '-'
            s2d = f"{r['cd']['mean_s2d']:.3f}" if r['cd']['mean_s2d'] is not None else '-'
            ov  = f"{r['cd']['overall']:.3f}"  if r['cd']['overall']  is not None else '-'
        else:
            d2s = s2d = ov = '-'
        print(f"{r['method']:<16} {r['seed']:<5} {psnr:<8} {ng:<10} {d2s:<8} {s2d:<8} {ov:<10}")


def print_aggregate(stats):
    print("\n=== PER-METHOD AGGREGATE ===")
    print(f"{'method':<16} {'n':<3} {'PSNR mean±std':<18} {'n_gauss mean±std':<22} {'CD mean±std':<20}")
    for m, s in stats.items():
        psnr_str = f"{s['psnr'][0]:.3f}±{s['psnr'][1]:.3f}" if s['psnr'] else '-'
        ng_str   = f"{s['n_gauss'][0]:.0f}±{s['n_gauss'][1]:.0f}" if s['n_gauss'] else '-'
        cd_str   = f"{s['cd'][0]:.3f}±{s['cd'][1]:.3f}" if s['cd'] else '-'
        print(f"{m:<16} {s['n']:<3} {psnr_str:<18} {ng_str:<22} {cd_str:<20}")


def evaluate_decision(stats):
    print("\n=== PRE-REGISTERED DECISION TABLE ===")
    cells = []

    def pair_diff_psnr(a, b):
        if not stats[a]['psnr'] or not stats[b]['psnr']:
            return None
        return stats[a]['psnr'][0] - stats[b]['psnr'][0]

    def pair_rel_cd(a, b):
        if not stats[a]['cd'] or not stats[b]['cd']:
            return None
        if stats[b]['cd'][0] == 0:
            return None
        return (stats[b]['cd'][0] - stats[a]['cd'][0]) / stats[b]['cd'][0]  # +ve = cancel better (smaller CD)

    d_psnr_cancel_random = pair_diff_psnr('cancel', 'random_dir')
    d_psnr_cancel_orth   = pair_diff_psnr('cancel', 'orthogonal_dir')
    d_psnr_random_orth   = pair_diff_psnr('random_dir', 'orthogonal_dir')
    d_psnr_cancel_V      = pair_diff_psnr('cancel', 'V')

    d_cd_cancel_random = pair_rel_cd('cancel', 'random_dir')
    d_cd_cancel_orth   = pair_rel_cd('cancel', 'orthogonal_dir')

    cells.append(('cancel vs random PSNR (Δ dB)',     d_psnr_cancel_random,
                  f">= {THRESH['psnr_min_diff']:.2f} (medium), >= {THRESH['psnr_strong']:.2f} (strong)"))
    cells.append(('cancel vs orth PSNR (Δ dB)',       d_psnr_cancel_orth,
                  f">= {THRESH['psnr_min_diff']:.2f} (medium), >= {THRESH['psnr_strong']:.2f} (strong)"))
    cells.append(('random vs orth PSNR sanity (Δ dB)', d_psnr_random_orth,
                  "|Δ| <= 0.30 expected (both null wrt cancel)"))
    cells.append(('cancel vs V reference PSNR (Δ dB)', d_psnr_cancel_V,
                  "reference; V uses different sampling (anisotropic Gaussian)"))
    cells.append(('cancel vs random CD (relative)',   d_cd_cancel_random,
                  f">= {THRESH['cd_min_diff']:.2%} (medium), >= {THRESH['cd_strong']:.2%} (strong)"))
    cells.append(('cancel vs orth CD (relative)',     d_cd_cancel_orth,
                  f">= {THRESH['cd_min_diff']:.2%} (medium), >= {THRESH['cd_strong']:.2%} (strong)"))

    print(f"{'comparison':<45} {'achieved':<14} {'threshold':<55}")
    for name, val, thr in cells:
        v = f"{val:+.3f}" if val is not None else 'TBD'
        print(f"{name:<45} {v:<14} {thr:<55}")

    print("\n=== VERDICT ===")
    if any(v is None for _, v, _ in cells):
        print("INCOMPLETE — not all 12 runs done. Wait for queue to drain.")
        return

    psnr_pass_random = d_psnr_cancel_random >= THRESH['psnr_min_diff']
    psnr_pass_orth   = d_psnr_cancel_orth   >= THRESH['psnr_min_diff']
    psnr_strong      = d_psnr_cancel_random >= THRESH['psnr_strong']
    cd_pass_random   = d_cd_cancel_random   >= THRESH['cd_min_diff']
    cd_pass_orth     = d_cd_cancel_orth     >= THRESH['cd_min_diff']
    random_orth_null = abs(d_psnr_random_orth) <= THRESH['psnr_min_diff']

    if psnr_strong and psnr_pass_orth and cd_pass_random:
        print("STRONG PASS — cancel direction informative on both PSNR (large effect) and CD")
    elif (psnr_pass_random and psnr_pass_orth) or (cd_pass_random and cd_pass_orth):
        print("PASS — cancel beats both random and orthogonal on at least one primary metric")
    elif psnr_pass_random and not psnr_pass_orth:
        print("PASS with caveat — cancel beats random but orth-control inconclusive (small effect)")
    elif abs(d_psnr_cancel_random) <= THRESH['psnr_min_diff']:
        print("REJECT — cancel ≈ random (within noise). Direction is not informative (Jaccard-pattern recurrence).")
    elif d_psnr_cancel_random < -THRESH['psnr_min_diff']:
        print("STRONG REJECT — cancel direction worse than random. Direction is harmful.")
    else:
        print("AMBIGUOUS — partial signal but criteria not met. See per-metric details.")

    if not random_orth_null:
        print(f"  NOTE: random_dir vs orthogonal_dir diff = {d_psnr_random_orth:+.3f} dB exceeds null band — check for confounds.")


def save_csv(rows):
    out = 'eval/p2_results.csv'
    with open(out, 'w') as f:
        f.write('method,seed,psnr,n_gauss,cd_mean_d2s,cd_mean_s2d,cd_overall\n')
        for r in rows:
            psnr = r['psnr'] if r['psnr'] is not None else ''
            ng = r['n_gauss'] if r['n_gauss'] is not None else ''
            d2s = r['cd']['mean_d2s'] if r['cd'] else ''
            s2d = r['cd']['mean_s2d'] if r['cd'] else ''
            ov  = r['cd']['overall']  if r['cd'] else ''
            f.write(f"{r['method']},{r['seed']},{psnr},{ng},{d2s},{s2d},{ov}\n")
    print(f"\nCSV saved: {out}")




def evaluate_decision_in_plane(stats):
    """Pre-registered rule for cancel_in_plane fix attempt (committed before any cancel_in_plane data)."""
    print("\n=== CANCEL_IN_PLANE FIX VERDICT (separate pre-registered rule) ===")
    if not stats.get('cancel_in_plane') or not stats['cancel_in_plane']['psnr']:
        print("INCOMPLETE — cancel_in_plane runs not done.")
        return
    if not stats.get('V') or not stats['V']['psnr']:
        print("INCOMPLETE — V baseline missing.")
        return
    psnr_cip = stats['cancel_in_plane']['psnr'][0]
    psnr_V   = stats['V']['psnr'][0]
    print(f"  cancel_in_plane PSNR: {psnr_cip:.3f}  (V baseline: {psnr_V:.3f})")
    if psnr_cip > psnr_V:
        margin = psnr_cip - psnr_V
        print(f"  STRONG PASS — beats V baseline by +{margin:.3f} dB. Cancel signal salvageable. Multi-scene expansion recommended.")
    elif psnr_cip > 27.7:
        print(f"  MARGINAL — between 27.7 and V ({psnr_V:.3f}). Off-disk fix helped but cancel signal still not informative enough. Further investigation needed.")
    else:
        print(f"  REJECT — below 27.7 floor. Cancel direction confirmed archive. Off-disk fix did NOT salvage.")
    # CD comparison if available
    if stats['cancel_in_plane'].get('cd') and stats['V'].get('cd'):
        cd_cip = stats['cancel_in_plane']['cd'][0]
        cd_V   = stats['V']['cd'][0]
        rel    = (cd_V - cd_cip) / cd_V if cd_V else 0
        print(f"  CD overall: cancel_in_plane={cd_cip:.3f}, V={cd_V:.3f}, relative diff={rel:+.2%} (positive = cancel_in_plane better)")



def evaluate_decision_bias_vanilla(stats):
    """Pre-registered rule for cancel_bias_vanilla (committed before any cbv data)."""
    print("\n=== CANCEL_BIAS_VANILLA VERDICT (separate pre-registered rule, lambda=0.5) ===")
    if not stats.get('cancel_bias_vanilla') or not stats['cancel_bias_vanilla']['psnr']:
        print("INCOMPLETE — cancel_bias_vanilla runs not done.")
        return
    if not stats.get('V') or not stats['V']['psnr']:
        print("INCOMPLETE — V baseline missing.")
        return
    psnr_cbv = stats['cancel_bias_vanilla']['psnr'][0]
    psnr_V   = stats['V']['psnr'][0]
    print(f"  cancel_bias_vanilla PSNR: {psnr_cbv:.3f}  (V baseline: {psnr_V:.3f})")
    if psnr_cbv > psnr_V:
        margin = psnr_cbv - psnr_V
        print(f"  STRONG PASS — beats V baseline by +{margin:.3f} dB. Cancel signal informative when applied as soft bias on vanilla mechanism. Multi-scene + lambda sweep recommended.")
    elif psnr_cbv > 27.7:
        print(f"  MARGINAL — between 27.7 and V ({psnr_V:.3f}). Minimal-intervention design still does not surpass baseline. Cancel signal likely uninformative at this lambda.")
    else:
        print(f"  REJECT — below 27.7 floor. Even minimal-intervention biased vanilla fails. Cancel signal confirmed not actionable for split direction. Archive direction final.")
    # Mechanism ladder comparison
    cip = stats.get('cancel_in_plane')
    cv  = stats.get('cancel_bias_vanilla')
    cancel = stats.get('cancel')
    if cip and cv and cancel and all(x['psnr'] for x in [cip, cv, cancel]):
        print("  Mechanism ladder (PSNR):")
        print(f"    cancel              = {cancel['psnr'][0]:.3f}  (off-plane, deterministic mirror)")
        print(f"    cancel_in_plane     = {cip['psnr'][0]:.3f}  (in-plane, deterministic mirror)")
        print(f"    cancel_bias_vanilla = {cv['psnr'][0]:.3f}   (in-plane, stochastic biased mean)")
        delta_cip_cancel = cip['psnr'][0] - cancel['psnr'][0]
        delta_cv_cip     = cv['psnr'][0] - cip['psnr'][0]
        print(f"    Δ(in_plane − cancel)        = {delta_cip_cancel:+.3f} dB  (off-disk fix contribution)")
        print(f"    Δ(bias_vanilla − in_plane)  = {delta_cv_cip:+.3f} dB  (mechanism conservatism contribution)")

if __name__ == '__main__':
    rows = collect()
    print_per_run(rows)
    stats = per_method_stats(rows)
    print_aggregate(stats)
    evaluate_decision(stats)
    evaluate_decision_in_plane(stats)
    evaluate_decision_bias_vanilla(stats)
    save_csv(rows)

