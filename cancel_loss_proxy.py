"""
Stage 1b: Disagreement surrogate (Σ|r_p|-|Σ r_p|), no ratio degenerate.

Differences vs Stage 1 (ratio form):
- L_g = Σ_p |r_p * mask_p| - |Σ_p r_p * mask_p|  (disagreement form, no division)
- Disk mask within projected radius (proper footprint approx)
- High-N filter (radii > radii_min)
- K samples per iter (default 2000)
- Activation deferred to iter > cancel_loss_start (default 15000, post-densify)

Properties:
- r_p → 0 (photo convergence) ⇒ L_g → 0 (no conflict with photo loss)
- All r_p same sign ⇒ Σ|r_p| = |Σ r_p| ⇒ L_g = 0 (no cancel, no penalty)
- Half +,半 - ⇒ Σ|r_p| = N|r|, |Σ r_p| = 0 ⇒ L_g = N|r| (max cancel, max penalty)
- Always ≥ 0 by triangle inequality
"""
import torch

def L_cancel_proxy(rendered, gt, means3D, full_proj_transform, radii, K=2000, radii_min=5, eps=1e-12):
    H, W = rendered.shape[1:]
    r = (rendered - gt).sum(dim=0)  # (H, W) scalar residual (sum over RGB)

    with torch.no_grad():
        homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=1)
        p_clip = homo @ full_proj_transform
        denom = p_clip[:, 3:4].clamp(min=1e-6)
        p_ndc = p_clip[:, :3] / denom
        px = (p_ndc[:, 0] + 1.0) * W * 0.5
        py = (p_ndc[:, 1] + 1.0) * H * 0.5

        active_ids = torch.where(radii > radii_min)[0]
        if len(active_ids) == 0:
            return torch.tensor(0.0, device=rendered.device, requires_grad=True)
        if len(active_ids) > K:
            perm = torch.randperm(len(active_ids), device=active_ids.device)[:K]
            sample_ids = active_ids[perm]
        else:
            sample_ids = active_ids

    L_terms = []
    dev = rendered.device
    for gid in sample_ids.tolist():
        cx = float(px[gid].item())
        cy = float(py[gid].item())
        rad = float(radii[gid].item())
        x0 = max(0, int(cx - rad))
        x1 = min(W, int(cx + rad) + 1)
        y0 = max(0, int(cy - rad))
        y1 = min(H, int(cy + rad) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        patch_r = r[y0:y1, x0:x1]
        ys = torch.arange(y0, y1, device=dev, dtype=torch.float32)
        xs = torch.arange(x0, x1, device=dev, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        dx = xx - cx; dy = yy - cy
        mask = ((dx*dx + dy*dy) <= rad*rad).to(patch_r.dtype)
        masked_r = patch_r * mask
        abs_sum = masked_r.abs().sum()
        signed_sum_abs = masked_r.sum().abs()
        # Disagreement: Σ|r·m| − |Σ r·m|  (always ≥ 0, triangle inequality)
        L_g = abs_sum - signed_sum_abs
        L_terms.append(L_g)

    if not L_terms:
        return torch.tensor(0.0, device=rendered.device, requires_grad=True)
    return torch.stack(L_terms).mean()

