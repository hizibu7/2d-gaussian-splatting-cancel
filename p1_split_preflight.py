"""
Pre-flight for cancel-as-split-direction.

Question: Vanilla split candidates 중 high-cm 비율은?
- r = -0.25 (vanilla candidates = high mag → low cm avg)
- high-cm vanilla candidates 비율 낮으면 split-direction 영향 small

Output: 다양한 mag rank cutoff + cm threshold에서 비율.

Decision:
- ratio > 40% (high-cm vanilla candidates 많음): split direction effect 클 가능성
- ratio 20-40%: marginal
- ratio < 20%: split direction effect 미미
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
import diff_surfel_rasterization as dsr

def main():
    p = ArgumentParser()
    mp = ModelParams(p, sentinel=True); pp = PipelineParams(p)
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--n_views', type=int, default=50)
    args = get_combined_args(p); args.eval = False
    pipe = pp.extract(args)
    g = GaussianModel(args.sh_degree if args.sh_degree else 3)
    sc = Scene(args, g, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([0,0,0], dtype=torch.float32, device='cuda')
    for p_ in [g._xyz, g._rotation, g._scaling, g._opacity, g._features_dc, g._features_rest]:
        p_.requires_grad_(True)

    P = g._xyz.shape[0]
    dev = g._xyz.device
    grad_accum = torch.zeros(P, device=dev)
    cm_accum = torch.zeros(P, device=dev)
    cams = sc.getTrainCameras()[:args.n_views]
    for cam in cams:
        dsr.set_cancel_buffers(P, dev)
        for p_ in [g._xyz, g._rotation, g._scaling, g._opacity, g._features_dc, g._features_rest]:
            if p_.grad is not None: p_.grad.zero_()
        pkg = render(cam, g, pipe, bg)
        gt = cam.original_image.cuda()
        loss = 0.8*l1_loss(pkg['render'], gt) + 0.2*(1.0 - ssim(pkg['render'], gt))
        loss.backward()
        b = dsr.get_cancel_buffers()
        m2d_sx = b.get('mean2D_sx', torch.zeros(P, device=dev))
        m2d_sy = b.get('mean2D_sy', torch.zeros(P, device=dev))
        grad_accum += torch.sqrt(m2d_sx*m2d_sx + m2d_sy*m2d_sy)
        sm_op = b.get('opacity', torch.zeros(P, device=dev))
        alpha = torch.sigmoid(g._opacity.squeeze())
        sig_prime = alpha * (1.0 - alpha)
        sg_op = (g._opacity.grad.squeeze() / (sig_prime + 1e-12)).abs()
        cm = 1.0 - sg_op / (sm_op + 1e-12)
        cm_accum += cm.detach()
    grad_avg = (grad_accum / args.n_views).cpu().numpy()
    cm_avg = (cm_accum / args.n_views).cpu().numpy()
    active = grad_avg > 1e-12
    g_a = grad_avg[active]; c_a = cm_avg[active]
    print(f"[INFO] active={len(g_a)}, cm distribution: median={np.median(c_a):.3f}, p70={np.percentile(c_a, 70):.3f}, p90={np.percentile(c_a, 90):.3f}")

    # Various vanilla cutoffs (percentile of magnitude)
    print(f"\n{'mag_top_X%':>12s} {'n_vanilla':>10s} {'cm>0.3 ratio':>15s} {'cm>0.5 ratio':>15s} {'cm>0.7 ratio':>15s} {'cm>0.9 ratio':>15s}")
    for mag_X in [5, 10, 15, 25, 50]:
        mag_threshold = np.percentile(g_a, 100 - mag_X)
        vanilla_pool = g_a >= mag_threshold
        n_vanilla = vanilla_pool.sum()
        ratios = []
        for cm_th in [0.3, 0.5, 0.7, 0.9]:
            high_cm_in_vanilla = (vanilla_pool & (c_a > cm_th)).sum()
            ratio = high_cm_in_vanilla / max(n_vanilla, 1) * 100
            ratios.append(f'{ratio:.1f}%')
        print(f"{mag_X:>11d}% {n_vanilla:>10d} " + " ".join([f'{r:>15s}' for r in ratios]))

    # Overall verdict
    print(f"\n=== Split direction viability verdict ===")
    print(f"Assuming vanilla densify triggers ~25% of active gauss:")
    mag_th_25 = np.percentile(g_a, 75)
    v_pool_25 = g_a >= mag_th_25
    n_v_25 = v_pool_25.sum()
    for cm_th in [0.3, 0.5, 0.7]:
        high_in_v = (v_pool_25 & (c_a > cm_th)).sum()
        ratio = high_in_v / max(n_v_25, 1) * 100
        if cm_th == 0.5:
            if ratio > 40: verdict = "STRONG — split direction effect likely"
            elif ratio > 20: verdict = "MODERATE — marginal effect possible"
            else: verdict = "WEAK — split direction effect 미미"
            print(f"  cm>{cm_th}: {ratio:.1f}% of vanilla candidates → {verdict}")
        else:
            print(f"  cm>{cm_th}: {ratio:.1f}% of vanilla candidates")

if __name__ == '__main__':
    main()

