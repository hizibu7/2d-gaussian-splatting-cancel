"""
Pre-flight Sanity (S1 + S2) for Phase 1 design.

S1: Pearson r(|∇μ|, cm_op_corrected) across active gaussians
S2: Candidate pool sizes
  OR_cancel: |∇μ| < thr_v AND cm rank top 30%  (vanilla 못 잡는 영역)
  AND_cancel: |∇μ| > thr_v AND cm rank top 30%  (vanilla 중 cancel 강한 것)

Vanilla 30K ckpt s1b_lam00_scan3 활용, view 0.
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
import diff_surfel_rasterization as dsr

THR_V = 0.0002

def main():
    p = ArgumentParser()
    mp = ModelParams(p, sentinel=True); pp = PipelineParams(p)
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--view_idx', type=int, default=0)
    args = get_combined_args(p); args.eval = False
    pipe = pp.extract(args)
    g = GaussianModel(args.sh_degree if args.sh_degree else 3)
    sc = Scene(args, g, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([0,0,0], dtype=torch.float32, device='cuda')
    cam = sc.getTrainCameras()[args.view_idx]
    for p_ in [g._xyz, g._rotation, g._scaling, g._opacity, g._features_dc, g._features_rest]:
        p_.requires_grad_(True)

    # Average mean2D grad across all views (proxy for xyz_gradient_accum used in densify)
    # For speed: use 5 views and average
    P = g._xyz.shape[0]
    dev = g._xyz.device
    grad_accum = torch.zeros(P, device=dev)
    cm_accum = torch.zeros(P, device=dev)
    n_views = 5
    cams = sc.getTrainCameras()[:n_views]
    for cam_i in cams:
        dsr.set_cancel_buffers(P, dev)
        for p_ in [g._xyz, g._rotation, g._scaling, g._opacity, g._features_dc, g._features_rest]:
            if p_.grad is not None: p_.grad.zero_()
        pkg = render(cam_i, g, pipe, bg)
        gt = cam_i.original_image.cuda()
        loss = 0.8*l1_loss(pkg['render'], gt) + 0.2*(1.0 - ssim(pkg['render'], gt))
        loss.backward()
        # mean2D magnitude (densify signal proxy)
        b = dsr.get_cancel_buffers()
        m2d_sm = b.get('mean2D', torch.zeros(P, device=dev))
        m2d_sx = b.get('mean2D_sx', torch.zeros(P, device=dev))
        m2d_sy = b.get('mean2D_sy', torch.zeros(P, device=dev))
        grad_norm = torch.sqrt(m2d_sx*m2d_sx + m2d_sy*m2d_sy)
        grad_accum += grad_norm
        # cm_op corrected
        sm_op = b.get('opacity', torch.zeros(P, device=dev))
        alpha = torch.sigmoid(g._opacity.squeeze())
        sig_prime = alpha * (1.0 - alpha)
        sg_op_corr = (g._opacity.grad.squeeze() / (sig_prime + 1e-12)).abs()
        cm = 1.0 - sg_op_corr / (sm_op + 1e-12)
        cm_accum += cm.detach()
    grad_avg = (grad_accum / n_views).detach().cpu().numpy()
    cm_avg = (cm_accum / n_views).detach().cpu().numpy()

    # Active mask: m2d signal active
    active = grad_avg > 1e-9
    print(f"[INFO] Total gauss: {P}, active (m2d>0): {active.sum()}")
    g_a = grad_avg[active]
    c_a = cm_avg[active]

    # S1: Pearson correlation
    from scipy.stats import pearsonr
    r, pval = pearsonr(g_a, c_a)
    print(f"\n=== S1: Pearson r(|∇μ|, cm_op) ===")
    print(f"  r = {r:.4f}, p = {pval:.2e}")
    print(f"  Interpretation: ", end='')
    if abs(r) > 0.7:
        print(f"r>0.7 — cm REDUNDANT with magnitude → effect 약함 expected")
    elif abs(r) < 0.3:
        print(f"r<0.3 — cm INFORMATIVE → effect 큼 expected")
    else:
        print(f"0.3≤r≤0.7 — middle ground, 실험 가치 있음")

    # S2a: OR_cancel candidate pool (|∇μ| < thr_v AND cm rank top 30%)
    cm_p70 = np.percentile(c_a, 70)
    mag_below = g_a < THR_V
    cm_top = c_a >= cm_p70
    or_pool = mag_below & cm_top
    or_pct = 100 * or_pool.sum() / len(g_a)
    print(f"\n=== S2a: OR_cancel additional candidates ===")
    print(f"  (|∇μ| < {THR_V}) AND (cm > p70 = {cm_p70:.4f})")
    print(f"  Count: {or_pool.sum()} / {len(g_a)} ({or_pct:.2f}%)")
    print(f"  Verdict: ", end='')
    if or_pct >= 5:
        print(f">5% — VIABLE, OR effect expected")
    elif or_pct >= 1:
        print(f"1-5% — marginal effect")
    else:
        print(f"<1% — OR effect 미미 expected")

    # S2b: AND_cancel refined pool
    mag_above = g_a >= THR_V
    vanilla_count = mag_above.sum()
    and_pool = mag_above & cm_top
    and_pct = 100 * and_pool.sum() / max(vanilla_count, 1)
    print(f"\n=== S2b: AND_cancel refined candidates ===")
    print(f"  (|∇μ| > {THR_V}) AND (cm > p70 = {cm_p70:.4f})")
    print(f"  Count: {and_pool.sum()} / {vanilla_count} of vanilla cand ({and_pct:.2f}%)")
    print(f"  Verdict: ", end='')
    if and_pct >= 50:
        print(f">50% — VIABLE, AND filtering possible")
    elif and_pct >= 20:
        print(f"20-50% — partial filtering")
    else:
        print(f"<20% — AND too restrictive, likely under-densify")

    # Distribution summary
    print(f"\n=== Distribution summary (active gauss) ===")
    print(f"  |∇μ|: median={np.median(g_a):.6e}, p25={np.percentile(g_a,25):.6e}, p75={np.percentile(g_a,75):.6e}")
    print(f"  cm_op: median={np.median(c_a):.4f}, p25={np.percentile(c_a,25):.4f}, p75={np.percentile(c_a,75):.4f}")
    print(f"  vanilla candidates (|∇μ|>thr_v): {vanilla_count} ({100*vanilla_count/len(g_a):.1f}%)")
    print(f"  cm distribution percentiles: p30={np.percentile(c_a,30):.4f}, p50={np.percentile(c_a,50):.4f}, p70={np.percentile(c_a,70):.4f}, p90={np.percentile(c_a,90):.4f}")

if __name__ == '__main__':
    main()

