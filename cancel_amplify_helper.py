"""F intervention helper: measure v5 cancel mask & amplify grads.
- measure_cancel_mask: forward+backward photo loss, return per-gauss mask
  for cancel-high gauss. Destroys current .grad (caller must save first).
- amplify_grads_inplace: scale .grad[mask] by gamma for all gauss params.
"""
import torch, numpy as np
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
import diff_surfel_rasterization as dsr

SH0 = 0.28209479177387814

def _zero_grads(g):
    for p in [g._xyz, g._rotation, g._scaling, g._opacity, g._features_dc, g._features_rest]:
        if p.grad is not None: p.grad.zero_()

def measure_cancel_mask(g, scene, pipe, bg, cm_thr=0.7, g_abs_mean_min=1e-7, nr_min=10,
                       channel='op', n_views=3):
    """Returns boolean torch tensor on cuda, length=P.
       channel in {'op','tm','color','m2d','n3d'}.
       Aggregates over n_views random train cameras.
    """
    cams = scene.getTrainCameras()
    P = g._xyz.shape[0]; dev = g._xyz.device
    n = min(n_views, len(cams))
    idxs = np.linspace(0, len(cams)-1, n).astype(int)
    mask_acc = torch.zeros(P, dtype=torch.bool, device=dev)
    for vi in idxs:
        cam = cams[int(vi)]
        _zero_grads(g)
        dsr.set_cancel_buffers(P, dev)
        pkg = render(cam, g, pipe, bg)
        gt = cam.original_image.cuda()
        loss = 0.8 * l1_loss(pkg['render'], gt) + 0.2 * (1.0 - ssim(pkg['render'], gt))
        loss.backward()
        b = dsr.get_cancel_buffers()
        z = torch.zeros(P, device=dev)
        nr = b.get("n_rays", z).clone()
        if channel == 'op':
            sm = b.get("opacity", z).clone()
            og = g._opacity.grad
            sg = og.abs().squeeze() if og is not None else z
        elif channel == 'tm':
            sm = b.get("transMat", z).clone()
            rg = g._rotation.grad; sg_ = g._scaling.grad
            rn = rg.norm(dim=1) if rg is not None else z
            sn = sg_.norm(dim=1) if sg_ is not None else z
            sg = torch.sqrt(rn*rn + sn*sn)
        elif channel == 'color':
            sm = b.get("color", z).clone()
            fdc = g._features_dc.grad
            sg = (fdc[:,0,:].norm(dim=1) / SH0) if fdc is not None else z
        elif channel == 'm2d':
            sx = b.get("mean2D_sx", z).clone(); sy = b.get("mean2D_sy", z).clone()
            sm = b.get("mean2D", z).clone()
            sg = torch.sqrt(sx*sx + sy*sy)
        elif channel == 'n3d':
            nx = b.get("normal3D_sx", z).clone(); ny = b.get("normal3D_sy", z).clone(); nz = b.get("normal3D_sz", z).clone()
            sm = b.get("normal3D", z).clone()
            sg = torch.sqrt(nx*nx + ny*ny + nz*nz)
        else:
            raise ValueError(f'unknown channel {channel}')
        nr_safe = nr.clamp(min=1.0)
        cm_raw = 1.0 - sg / (sm + 1e-12)
        cm_null = 1.0 - 1.0 / torch.sqrt(nr_safe)
        denom = (1.0 - cm_null).clamp(min=1e-6)
        cm_corr = (cm_raw - cm_null) / denom
        cm_corr = cm_corr.clamp(max=1.0)
        mean_abs = sm / nr_safe
        view_mask = (mean_abs >= g_abs_mean_min) & (nr >= nr_min) & (cm_corr >= cm_thr)
        mask_acc = mask_acc | view_mask
    _zero_grads(g)
    return mask_acc

def amplify_grads_inplace(g, mask, gamma):
    """scale .grad[mask] by gamma for all gauss params (in place)."""
    if mask is None or gamma == 0.0: return
    factor = 1.0 + float(gamma)
    for p in [g._xyz, g._rotation, g._scaling, g._opacity, g._features_dc, g._features_rest]:
        if p.grad is not None:
            p.grad[mask] *= factor

