"""
Compute split directions for densify candidates based on cancel signal.

For each candidate Gauss:
- Project mean3D to image
- Footprint disk pixels (within projected radii)
- Per-pixel residual sign decomposition
- (+) centroid vs (−) centroid in image space
- 2D direction = (+) − (−), unit
- Unproject to 3D world space at gauss depth

Used for split_method ∈ {cancel, orthogonal_dir, cancel_in_plane, cancel_bias_vanilla,
                         cancel_argmin, cancel_argmin_dir_random}.
"""
import math
import torch

def compute_cancel_directions(means3D, radii, full_proj_transform, world_view_transform, tanfovx, tanfovy, residual_map, candidate_mask=None):
    """
    Returns (N, 3) world-space unit directions. Zero for invalid candidates.
    """
    H, W = residual_map.shape
    N = means3D.shape[0]
    dev = means3D.device
    directions = torch.zeros(N, 3, device=dev)

    # Project all means to image
    homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=1)  # (N, 4)
    p_clip = homo @ full_proj_transform
    p_ndc = p_clip[:, :3] / p_clip[:, 3:4].clamp(min=1e-6)
    px = (p_ndc[:, 0] + 1.0) * W * 0.5
    py = (p_ndc[:, 1] + 1.0) * H * 0.5

    # Depth in view space
    p_view = homo @ world_view_transform  # (N, 4)
    depth_view = p_view[:, 2]

    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)

    # Inverse camera rotation (world from camera)
    R_w2c = world_view_transform[:3, :3]  # (3, 3)
    R_c2w = R_w2c.T

    candidates = candidate_mask if candidate_mask is not None else torch.ones(N, dtype=torch.bool, device=dev)

    for gid in range(N):
        if not candidates[gid]: continue
        rad = float(radii[gid].item())
        if rad <= 1: continue
        cx = float(px[gid].item()); cy = float(py[gid].item())
        x0 = max(0, int(cx - rad)); x1 = min(W, int(cx + rad) + 1)
        y0 = max(0, int(cy - rad)); y1 = min(H, int(cy + rad) + 1)
        if x1 <= x0 or y1 <= y0: continue
        patch = residual_map[y0:y1, x0:x1]
        ys = torch.arange(y0, y1, device=dev, dtype=torch.float32)
        xs = torch.arange(x0, x1, device=dev, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        dx = xx - cx; dy = yy - cy
        disk = (dx*dx + dy*dy) <= rad*rad
        patch_in = patch[disk]
        pos_in = patch_in > 1e-6
        neg_in = patch_in < -1e-6
        if pos_in.sum() < 3 or neg_in.sum() < 3: continue
        coords_x_d = xx[disk]; coords_y_d = yy[disk]
        c_pos_x = coords_x_d[pos_in].mean()
        c_pos_y = coords_y_d[pos_in].mean()
        c_neg_x = coords_x_d[neg_in].mean()
        c_neg_y = coords_y_d[neg_in].mean()
        dpx = c_pos_x - c_neg_x
        dpy = c_pos_y - c_neg_y
        d2_norm = torch.sqrt(dpx*dpx + dpy*dpy) + 1e-9
        dpx = dpx / d2_norm; dpy = dpy / d2_norm
        z = float(depth_view[gid].item())
        if abs(z) < 1e-6: continue
        cam_dx = (dpx * z / fx).item()
        cam_dy = (dpy * z / fy).item()
        cam_off = torch.tensor([cam_dx, cam_dy, 0.0], device=dev)
        world_dir = R_c2w @ cam_off
        wn = world_dir.norm() + 1e-9
        directions[gid] = world_dir / wn
    return directions


def compute_cancel_argmin_offsets(means3D, radii, full_proj_transform, world_view_transform,
                                  tanfovx, tanfovy, residual_map, candidate_mask=None):
    """
    Returns (N, 3) world-space RAW offset vectors (NOT unit normalized).
    Each offset = unproject((c+ - c-)/2 image-plane) -> world.
    For each child, child mean ← parent ± offset_world after tangent projection + clamp.
    Zero for invalid candidates.
    """
    H, W = residual_map.shape
    N = means3D.shape[0]
    dev = means3D.device
    offsets = torch.zeros(N, 3, device=dev)

    homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=1)
    p_clip = homo @ full_proj_transform
    p_ndc = p_clip[:, :3] / p_clip[:, 3:4].clamp(min=1e-6)
    px = (p_ndc[:, 0] + 1.0) * W * 0.5
    py = (p_ndc[:, 1] + 1.0) * H * 0.5
    p_view = homo @ world_view_transform
    depth_view = p_view[:, 2]
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)
    R_w2c = world_view_transform[:3, :3]
    R_c2w = R_w2c.T

    candidates = candidate_mask if candidate_mask is not None else torch.ones(N, dtype=torch.bool, device=dev)

    for gid in range(N):
        if not candidates[gid]: continue
        rad = float(radii[gid].item())
        if rad <= 1: continue
        cx = float(px[gid].item()); cy = float(py[gid].item())
        x0 = max(0, int(cx - rad)); x1 = min(W, int(cx + rad) + 1)
        y0 = max(0, int(cy - rad)); y1 = min(H, int(cy + rad) + 1)
        if x1 <= x0 or y1 <= y0: continue
        patch = residual_map[y0:y1, x0:x1]
        ys = torch.arange(y0, y1, device=dev, dtype=torch.float32)
        xs = torch.arange(x0, x1, device=dev, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        dx = xx - cx; dy = yy - cy
        disk = (dx*dx + dy*dy) <= rad*rad
        patch_in = patch[disk]
        pos_in = patch_in > 1e-6
        neg_in = patch_in < -1e-6
        if pos_in.sum() < 3 or neg_in.sum() < 3: continue
        coords_x_d = xx[disk]; coords_y_d = yy[disk]
        c_pos_x = coords_x_d[pos_in].mean(); c_pos_y = coords_y_d[pos_in].mean()
        c_neg_x = coords_x_d[neg_in].mean(); c_neg_y = coords_y_d[neg_in].mean()
        # Raw half-offset (closed-form argmin half-distance between (+/-) centroids)
        dpx_raw = (c_pos_x - c_neg_x) * 0.5
        dpy_raw = (c_pos_y - c_neg_y) * 0.5
        z = float(depth_view[gid].item())
        if abs(z) < 1e-6: continue
        cam_dx = (dpx_raw * z / fx).item()
        cam_dy = (dpy_raw * z / fy).item()
        cam_off = torch.tensor([cam_dx, cam_dy, 0.0], device=dev)
        world_off = R_c2w @ cam_off
        offsets[gid] = world_off
    return offsets


def sample_tangent_random_directions(N, candidate_mask, rotations_quat, device, seed=0):
    """
    Per candidate: uniform 2D direction in surfel tangent plane (span of R[:,0], R[:,1]).
    Returns (N, 3) world-space UNIT directions. Zero for non-candidates.
    """
    from utils.general_utils import build_rotation
    directions = torch.zeros(N, 3, device=device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    cand_idx = torch.where(candidate_mask)[0]
    n_cand = cand_idx.numel()
    if n_cand == 0: return directions
    R = build_rotation(rotations_quat[cand_idx])  # (n_cand, 3, 3)
    theta = torch.rand(n_cand, generator=gen, device=device) * (2 * math.pi)
    e0 = R[:, :, 0]  # tangent basis vector 1
    e1 = R[:, :, 1]  # tangent basis vector 2
    d = torch.cos(theta).unsqueeze(-1) * e0 + torch.sin(theta).unsqueeze(-1) * e1
    d = d / (d.norm(dim=-1, keepdim=True) + 1e-9)
    directions[cand_idx] = d
    return directions


def sample_random_unit_directions(N, candidate_mask, device, seed=0):
    """Random unit 3D directions for each candidate (zero for non-candidates). LEGACY (random_dir)."""
    directions = torch.zeros(N, 3, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    n_cand = candidate_mask.sum().item()
    if n_cand == 0: return directions
    raw = torch.randn(n_cand, 3, generator=gen, device=device)
    raw = raw / (raw.norm(dim=1, keepdim=True) + 1e-9)
    directions[candidate_mask] = raw
    return directions


def orthogonalize_directions(cancel_dirs, candidate_mask, seed=0):
    """Random direction perpendicular to cancel_dirs per Gauss (zero for non-candidates)."""
    dev = cancel_dirs.device
    N = cancel_dirs.shape[0]
    out = torch.zeros_like(cancel_dirs)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for gid in range(N):
        if not candidate_mask[gid]: continue
        d = cancel_dirs[gid]
        if d.norm() < 1e-9: continue
        r = torch.randn(3, generator=gen, device=dev)
        r = r - (r @ d) * d
        rn = r.norm() + 1e-9
        out[gid] = r / rn
    return out


def project_to_tangent_plane(directions, rotations_quat):
    """
    Project (N, 3) world-space directions onto each Gauss's tangent plane.
    Surfel disk lies in span(R[:,0], R[:,1]); normal = R[:,2].
    Zero-vector inputs (non-candidates) stay zero.
    """
    from utils.general_utils import build_rotation
    R = build_rotation(rotations_quat)            # (N, 3, 3)
    normal = R[:, :, 2]                           # (N, 3)
    dn = (directions * normal).sum(dim=-1, keepdim=True)  # (N, 1)
    dip = directions - dn * normal                # remove normal component
    n = dip.norm(dim=-1, keepdim=True)
    keep = (n > 1e-6).float()
    return dip / (n + 1e-9) * keep


def project_to_tangent_plane_keep_magnitude(offsets, rotations_quat):
    """
    Project (N, 3) world-space OFFSETS (not unit) onto tangent plane, PRESERVING magnitude.
    Output: offsets - (offsets·normal) * normal. NO normalization.
    """
    from utils.general_utils import build_rotation
    R = build_rotation(rotations_quat)
    normal = R[:, :, 2]
    dn = (offsets * normal).sum(dim=-1, keepdim=True)
    return offsets - dn * normal


def clamp_offset_to_parent_footprint(offsets, parent_scales, clamp_factor=1.5):
    """
    Clamp offset magnitude to clamp_factor × (2 × max_inplane_scale) = clamp_factor × parent_radius.
    Hard clamp (rescales offset vectors exceeding max_norm to max_norm).
    parent_scales: (N, 2) for 2DGS surfel — use max for in-plane radius.
    """
    # parent_radius proxy: 2 × max in-plane scale
    max_in = parent_scales.max(dim=-1, keepdim=True).values  # (N, 1)
    max_norm = clamp_factor * 2.0 * max_in.squeeze(-1)        # (N,)
    cur_norm = offsets.norm(dim=-1)                           # (N,)
    over = cur_norm > max_norm
    if over.any():
        scale = (max_norm[over] / (cur_norm[over] + 1e-9)).unsqueeze(-1)
        offsets = offsets.clone()
        offsets[over] = offsets[over] * scale
    return offsets

