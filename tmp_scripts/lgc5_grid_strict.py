"""
9-scene × 3-view grid (single mode only), all 3 loss × 5 grad-attr combos.
Strict thresholds: cm_corr >= 0.7, nr >= 30.

Reads paper_figures_v2/lgc5_grp9/*.npz (v5 corrected, hidden factor uncorrected).
Output: paper_figures_v2/lgc5_strict/grid_{loss}_{attr}.png (15 PNG)
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, '.')
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

GROUPS = {
    'Fur':    {'scans': [105, 82, 50], 'color': '#e34a33'},
    'Detail': {'scans': [97, 3, 28],   'color': '#fdae61'},
    'Simple': {'scans': [8, 35, 40],   'color': '#2c7fb8'},
}
SCAN_ORDER = [105, 82, 50, 97, 3, 28, 8, 35, 40]
VIEWS = [0, 16, 32]
DATA_DIR = 'paper_figures_v2/lgc5_grp9'
OUT_DIR = 'paper_figures_v2/lgc5_strict'
LOSSES = ['photo', 'dist', 'normal']
GRADS = ['m2d', 'op', 'tm', 'color', 'n3d']

def quat_to_R(q):
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2*(qy*qy + qz*qz); R[:, 0, 1] = 2*(qx*qy - qw*qz); R[:, 0, 2] = 2*(qx*qz + qw*qy)
    R[:, 1, 0] = 2*(qx*qy + qw*qz); R[:, 1, 1] = 1 - 2*(qx*qx + qz*qz); R[:, 1, 2] = 2*(qy*qz - qw*qx)
    R[:, 2, 0] = 2*(qx*qz - qw*qy); R[:, 2, 1] = 2*(qy*qz + qw*qx); R[:, 2, 2] = 1 - 2*(qx*qx + qy*qy)
    return R

def render_image(g, cam, pipe, bg):
    with torch.no_grad():
        pkg = render(cam, g, pipe, bg)
    return pkg['render'].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

def compute_mask_lgc5(npz, loss, G, g_abs_mean_min, nr_min, cm_thr):
    """lgc5_grp9 schema: {loss}_{G}_sm/sg/cm_corr, {loss}_nr."""
    sm_key = f'{loss}_{G}_sm'
    cm_key = f'{loss}_{G}_cm_corr'
    nr_key = f'{loss}_nr'
    if sm_key not in npz or cm_key not in npz or nr_key not in npz:
        return None
    sm = npz[sm_key]; cm = npz[cm_key]; nr = npz[nr_key]
    nr_safe = np.maximum(nr, 1.0)
    valid = np.isfinite(cm) & np.isfinite(sm) & np.isfinite(nr)
    return valid & (sm/nr_safe >= g_abs_mean_min) & (nr >= nr_min) & (cm >= cm_thr)

def overlay(ax, img, sxyz, sR, ss, proj, sel):
    H, W = img.shape[:2]
    ax.imshow(img)
    if len(sel) > 0:
        u = sR[sel, :, 0] * ss[sel, 0:1]; v = sR[sel, :, 1] * ss[sel, 1:2]
        Nseg = 24; th = np.linspace(0, 2*np.pi, Nseg+1)
        ct = np.cos(th)[None, :, None]; st = np.sin(th)[None, :, None]
        bd = sxyz[sel, None, :] + ct*u[:, None, :] + st*v[:, None, :]
        bd_h = np.concatenate([bd, np.ones((bd.shape[0], bd.shape[1], 1))], axis=-1)
        p2 = bd_h @ proj
        p2 = p2[..., :3] / p2[..., 3:4].clip(min=1e-6)
        px = (p2[..., 0]+1)/2*W; py = (p2[..., 1]+1)/2*H
        segs = np.stack([px, py], axis=-1)
        lc = LineCollection(segs, colors='red', linewidths=0.4, alpha=0.35)
        ax.add_collection(lc)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

def grp(scan):
    for gn, info in GROUPS.items():
        if scan in info['scans']: return gn
    return ''

def main():
    p = ArgumentParser()
    mp = ModelParams(p, sentinel=True); pp = PipelineParams(p)
    p.add_argument('--ckpt_prefix', type=str, default='eval/ck17_scan')
    p.add_argument('--data_root', type=str, default='data/DTU_Full/scan')
    p.add_argument('--data_root_alt', type=str, default='data/DTU/scan')
    p.add_argument('--iteration', type=int, default=17000)
    p.add_argument('--cm_corr_min', type=float, default=0.7)
    p.add_argument('--g_abs_mean_min', type=float, default=1e-7)
    p.add_argument('--nr_min', type=int, default=30)
    args = p.parse_args(sys.argv[1:]); args.eval = False
    os.makedirs(OUT_DIR, exist_ok=True)
    bg = torch.tensor([0,0,0], dtype=torch.float32, device='cuda')

    cache = {}
    for scan in SCAN_ORDER:
        src = args.data_root + str(scan); alt = args.data_root_alt + str(scan)
        if os.path.isdir(alt): src = alt
        args.source_path = src
        args.model_path = args.ckpt_prefix + str(scan)
        args.resolution = 2
        g = GaussianModel(args.sh_degree if args.sh_degree else 3)
        sc = Scene(args, g, load_iteration=args.iteration, shuffle=False)
        pipe = pp.extract(args)
        cams = sc.getTrainCameras()
        xyz = g.get_xyz.detach().cpu().numpy()
        rot_q = g._rotation.detach().cpu().numpy()
        rot_q = rot_q / np.linalg.norm(rot_q, axis=1, keepdims=True).clip(min=1e-8)
        sca = g.get_scaling.detach().cpu().numpy()
        R = quat_to_R(rot_q)
        npzs = {}
        for V in VIEWS:
            npzp = f'{DATA_DIR}/lgc5_s{scan}_v{V}_arrays.npz'
            npzs[V] = dict(np.load(npzp)) if os.path.exists(npzp) else None
        per_view = {}
        for V in VIEWS:
            cam = cams[V % len(cams)]
            img = render_image(g, cam, pipe, bg)
            proj = cam.full_proj_transform.detach().cpu().numpy()
            per_view[V] = {'img': img, 'proj': proj}
        cache[scan] = {'xyz': xyz, 'R': R, 'sca': sca, 'views': per_view, 'npzs': npzs}
        del g, sc; torch.cuda.empty_cache()
        print(f'cached s{scan}', flush=True)

    for loss in LOSSES:
        for G in GRADS:
            fig, axs = plt.subplots(9, 3, figsize=(18, 42))
            any_data = False
            for ridx, scan in enumerate(SCAN_ORDER):
                c = cache[scan]
                for cidx, V in enumerate(VIEWS):
                    ax = axs[ridx, cidx]
                    nz = c['npzs'][V]
                    if nz is None:
                        ax.text(0.5, 0.5, f's{scan} v{V} MISSING', ha='center', va='center')
                        ax.axis('off'); continue
                    mask = compute_mask_lgc5(nz, loss, G, args.g_abs_mean_min, args.nr_min, args.cm_corr_min)
                    if mask is None:
                        ax.text(0.5, 0.5, f'{loss}/{G} key missing', ha='center', va='center')
                        ax.axis('off'); continue
                    any_data = True
                    sel = np.where(mask)[0]
                    vimg = c['views'][V]
                    overlay(ax, vimg['img'], c['xyz'], c['R'], c['sca'], vimg['proj'], sel)
                    ax.set_title(f's{scan} [{grp(scan)}] v{V} n={len(sel)}', fontsize=9,
                                 color=GROUPS[grp(scan)]['color'] if grp(scan) else 'black')
            if not any_data:
                plt.close(); print(f'skip {loss}/{G} (no data)'); continue
            plt.suptitle(f'{loss} | {G} | v5 cm_corr >= {args.cm_corr_min}, nr >= {args.nr_min}', fontsize=14)
            plt.tight_layout()
            out_p = f'{OUT_DIR}/grid_{loss}_{G}.png'
            plt.savefig(out_p, dpi=100, bbox_inches='tight')
            plt.close()
            print(f'wrote {out_p}', flush=True)

if __name__ == '__main__':
    main()

