"""
v7: extends v6 (channel-corrected cancel measurement) to dist + normal losses.
photo loss is already measured in lgc6_corr/. v7 handles dist + normal.

Per call = 1 view × 1 loss × 5 channel-corrected cm.
Output: {out_prefix}_{loss}_corr.npz with (m2d/op/tm/color/n3d) × (sg, sm) + cm_each + nr.
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
import diff_surfel_rasterization as dsr

SH_C0 = 0.28209479177387814

def measure_one_loss(g, cam, pipe, bg, loss_kind, lam=0.2):
    P = g._xyz.shape[0]; dev = g._xyz.device
    dsr.set_cancel_buffers(P, dev)
    for p in [g._xyz, g._rotation, g._scaling, g._opacity, g._features_dc, g._features_rest]:
        if p.grad is not None: p.grad.zero_()
    pkg = render(cam, g, pipe, bg)
    gt = cam.original_image.cuda()

    if loss_kind == 'photo':
        Ll1 = l1_loss(pkg['render'], gt)
        loss = (1.0 - lam) * Ll1 + lam * (1.0 - ssim(pkg['render'], gt))
    elif loss_kind == 'dist':
        loss = pkg['rend_dist'].mean()
    elif loss_kind == 'normal':
        rend_n = pkg['rend_normal']; surf_n = pkg['surf_normal']
        loss = (1.0 - (rend_n * surf_n).sum(dim=0))[None].mean()
    else:
        raise ValueError(f'unknown loss_kind {loss_kind}')

    loss.backward()

    b = dsr.get_cancel_buffers()
    lg = dsr.get_last_grads()
    z = torch.zeros(P, device=dev)

    sx = b.get("mean2D_sx", z); sy = b.get("mean2D_sy", z)
    sm_m2d = b.get("mean2D", z)
    sg_m2d = torch.sqrt(sx*sx + sy*sy)

    sm_op = b.get("opacity", z)
    op_g = g._opacity.grad
    if op_g is not None:
        alpha = torch.sigmoid(g._opacity.squeeze())
        sig_prime = alpha * (1.0 - alpha)
        sg_op_corr = (op_g.squeeze() / (sig_prime + 1e-12)).abs()
    else:
        sg_op_corr = z

    sm_tm = b.get("transMat", z)
    grad_tm = lg.get("transMat", None)
    if grad_tm is not None:
        grad_tm_flat = grad_tm.reshape(P, -1)
        sg_vec_tm = grad_tm_flat.norm(dim=1)
    else:
        sg_vec_tm = z

    sm_color = b.get("color", z)
    fdc_g = g._features_dc.grad
    if fdc_g is not None:
        sg_color_pre = fdc_g[:,0,:].norm(dim=1)
        sg_color_corr = sg_color_pre / SH_C0
    else:
        sg_color_corr = z

    nx = b.get("normal3D_sx", z); ny = b.get("normal3D_sy", z); nz = b.get("normal3D_sz", z)
    sm_n3d = b.get("normal3D", z)
    sg_vec_n3d = torch.sqrt(nx*nx + ny*ny + nz*nz)

    nr = b.get("n_rays", z)

    def cm(sg, sm):
        return (1.0 - sg / (sm + 1e-12)).detach().cpu().numpy()

    return {
        'm2d_sg': sg_m2d.detach().cpu().numpy(),
        'm2d_sm': sm_m2d.detach().cpu().numpy(),
        'op_sg':  sg_op_corr.detach().cpu().numpy(),
        'op_sm':  sm_op.detach().cpu().numpy(),
        'tm_sg':  sg_vec_tm.detach().cpu().numpy(),
        'tm_sm':  sm_tm.detach().cpu().numpy(),
        'color_sg': sg_color_corr.detach().cpu().numpy(),
        'color_sm': sm_color.detach().cpu().numpy(),
        'n3d_sg': sg_vec_n3d.detach().cpu().numpy(),
        'n3d_sm': sm_n3d.detach().cpu().numpy(),
        'nr':     nr.detach().cpu().numpy(),
        'cm_m2d':   cm(sg_m2d, sm_m2d),
        'cm_op':    cm(sg_op_corr, sm_op),
        'cm_tm':    cm(sg_vec_tm, sm_tm),
        'cm_color': cm(sg_color_corr, sm_color),
        'cm_n3d':   cm(sg_vec_n3d, sm_n3d),
    }

def main():
    p = ArgumentParser()
    mp = ModelParams(p, sentinel=True); pp = PipelineParams(p)
    p.add_argument('--iteration', type=int, default=17000)
    p.add_argument('--view_idx', type=int, default=0)
    p.add_argument('--loss_kind', type=str, required=True, choices=['photo', 'dist', 'normal'])
    p.add_argument('--out_prefix', type=str, required=True)
    args = get_combined_args(p); args.eval = False
    pipe = pp.extract(args)
    g = GaussianModel(args.sh_degree if args.sh_degree else 3)
    sc = Scene(args, g, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([0,0,0], dtype=torch.float32, device='cuda')
    cam = sc.getTrainCameras()[args.view_idx]
    for p_ in [g._xyz, g._rotation, g._scaling, g._opacity, g._features_dc, g._features_rest]:
        p_.requires_grad_(True)
    r = measure_one_loss(g, cam, pipe, bg, args.loss_kind)
    out_path = f'{args.out_prefix}_{args.loss_kind}_corr.npz'
    np.savez_compressed(out_path, **r)
    print(f"saved {out_path}", flush=True)
    for ch in ['op', 'tm', 'color', 'm2d', 'n3d']:
        sm = r[f'{ch}_sm']; cm_v = r[f'cm_{ch}']
        active = sm > 1e-9
        if active.sum() == 0:
            print(f"  {ch}: no active"); continue
        ca = cm_v[active]
        print(f"  {ch}: n={int(active.sum())} cm_med={np.median(ca):.3f} cm_mean={ca.mean():.3f}", flush=True)

if __name__ == '__main__':
    main()

