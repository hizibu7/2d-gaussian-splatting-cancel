#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from cancel_loss_proxy import L_cancel_proxy
from cancel_split_direction import (compute_cancel_directions, sample_random_unit_directions, orthogonalize_directions, project_to_tangent_plane, compute_cancel_argmin_offsets, sample_tangent_random_directions, project_to_tangent_plane_keep_magnitude, clamp_offset_to_parent_footprint)
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.v10_utils import compute_pathological_mask
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import phase1_log
import traj_log
import traj_log_v2
import traj_log_cancel
import traj_log_cancel_v3
import traj_log_cancel_v4
import traj_log_cancel_v5
import traj_log_cancel_v6
import traj_log_cancel_v7
import traj_log_cancel_v8
import traj_log_cancel_v9
import traj_log_cancel_v10
import traj_log_cancel_v11
import traj_log_cancel_perloss
import cancel_prune
import traj_log_perview
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

@torch.no_grad()
def compute_normal_consistency(gaussians, scene, pipe, background, num_views=8, occlusion_th=0.05, return_consensus=False):
    """Per-Gaussian normal consistency across multiple views with occlusion check.

    If return_consensus=True, also returns per-Gaussian consensus normal (mean of visible normals).
    """
    train_cams = scene.getTrainCameras()
    indices = torch.randperm(len(train_cams))[:num_views]

    xyz = gaussians.get_xyz  # [N, 3]
    N = xyz.shape[0]
    device = xyz.device

    # Visibility mask [K, N] and sampled normals [K, N, 3]
    visibility_mask = torch.zeros(num_views, N, dtype=torch.bool, device=device)
    sampled_normals = torch.zeros(num_views, N, 3, device=device)

    xyz_hom = torch.cat([xyz, torch.ones(N, 1, device=device)], dim=1)  # [N, 4]

    for view_idx, cam_idx in enumerate(indices):
        cam = train_cams[cam_idx]
        pkg = render(cam, gaussians, pipe, background)
        normal_map = pkg["surf_normal"]    # [3, H, W]
        depth_map = pkg["surf_depth"]      # [1, H, W] — camera-space z-depth

        # Project Gaussian centers: NDC for pixel coords, camera-space for depth
        proj = cam.full_proj_transform  # [4, 4]
        xyz_proj = xyz_hom @ proj  # [N, 4]
        w = xyz_proj[:, 3:4] + 1e-8
        xy_ndc = xyz_proj[:, :2] / w  # [N, 2] in [-1, 1]

        # Camera-space z-depth (same space as surf_depth)
        W_mat = cam.world_view_transform  # [4, 4]
        xyz_cam = xyz_hom @ W_mat  # [N, 4]
        gaussian_depth_cam = xyz_cam[:, 2]  # [N] camera-space z

        # Frustum check
        in_frustum = (xy_ndc[:, 0].abs() < 1.0) & (xy_ndc[:, 1].abs() < 1.0) & (gaussian_depth_cam > 0)

        # Sample rendered depth for occlusion check
        grid = xy_ndc.view(1, N, 1, 2)
        rendered_depth_sampled = F.grid_sample(
            depth_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, 0, :, 0]  # [N]

        # Occlusion check: both in camera-space z now
        not_occluded = (gaussian_depth_cam - rendered_depth_sampled).abs() < occlusion_th

        visible = in_frustum & not_occluded
        visibility_mask[view_idx] = visible

        # Sample normals
        normal_sampled = F.grid_sample(
            normal_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, :, :, 0].T  # [N, 3]

        sampled_normals[view_idx] = normal_sampled

    # Per-Gaussian pairwise cosine similarity (only visible pairs)
    consistency = torch.zeros(N, device=device)
    pair_counts = torch.zeros(N, device=device)

    for i in range(num_views):
        for j in range(i + 1, num_views):
            both_visible = visibility_mask[i] & visibility_mask[j]
            if both_visible.any():
                cos_sim = (sampled_normals[i] * sampled_normals[j]).sum(dim=1).clamp(-1, 1)
                consistency += torch.where(both_visible, cos_sim, torch.zeros_like(cos_sim))
                pair_counts += both_visible.float()

    # Normalize
    has_pairs = pair_counts > 0
    consistency = torch.where(has_pairs, consistency / (pair_counts + 1e-8),
                              torch.ones_like(consistency) * 0.5)

    # Visibility diagnostics
    valid_views_per_gaussian = visibility_mask.float().sum(dim=0)
    print(f"[Consistency] visible views: mean={valid_views_per_gaussian.mean():.1f}, "
          f"min={valid_views_per_gaussian.min():.0f}, "
          f"median={valid_views_per_gaussian.median():.0f}, "
          f"gaussians with <2 views: {(valid_views_per_gaussian < 2).sum()}/{N}", flush=True)

    if return_consensus:
        # Compute per-Gaussian consensus normal (weighted mean of visible normals)
        weighted_normals = torch.zeros(N, 3, device=device)
        for k in range(num_views):
            mask = visibility_mask[k].unsqueeze(1).float()  # [N, 1]
            weighted_normals += mask * sampled_normals[k]
        normal_counts = visibility_mask.float().sum(dim=0).unsqueeze(1)  # [N, 1]
        consensus_normals = torch.where(
            normal_counts > 0,
            weighted_normals / (normal_counts + 1e-8),
            torch.zeros_like(weighted_normals)
        )
        consensus_normals = F.normalize(consensus_normals, dim=1, eps=1e-6)
        # Zero out Gaussians with < 2 visible views (unreliable consensus)
        consensus_normals[normal_counts.squeeze() < 2] = 0
        print(f"[Consensus] valid normals: {(normal_counts.squeeze() >= 2).sum()}/{N}", flush=True)
        return consistency.clamp(0, 1), consensus_normals  # [N], [N, 3]

    return consistency.clamp(0, 1)  # [N]




@torch.no_grad()
def compute_cross_view_depth_variance(gaussians, scene, pipe, background,
                                       num_views=8, occlusion_th=0.05,
                                       loop_threshold=0.5, normalize=True):
    """Compute per-Gaussian cross-view depth variance.
    
    For each Gaussian, project it into multiple views, sample the rendered depth,
    and compute the variance. High variance = depth inconsistency.
    
    Also computes a loop mask based on scale_conf.
    
    Returns:
        depth_var: [N] per-Gaussian depth variance (normalized if enabled)
        loop_weight: [N] weight for the loss (higher for loop Gaussians)
    """
    train_cams = scene.getTrainCameras()
    indices = torch.randperm(len(train_cams))[:num_views]
    
    xyz = gaussians.get_xyz  # [N, 3]
    N = xyz.shape[0]
    device = xyz.device
    
    xyz_hom = torch.cat([xyz, torch.ones(N, 1, device=device)], dim=1)
    
    visibility_mask = torch.zeros(num_views, N, dtype=torch.bool, device=device)
    sampled_depths = torch.zeros(num_views, N, device=device)
    
    for view_idx, cam_idx in enumerate(indices):
        cam = train_cams[cam_idx]
        pkg = render(cam, gaussians, pipe, background)
        depth_map = pkg["surf_depth"]  # [1, H, W]
        
        proj = cam.full_proj_transform
        xyz_proj = xyz_hom @ proj
        xy_ndc = xyz_proj[:, :2] / (xyz_proj[:, 3:4] + 1e-8)
        
        W_mat = cam.world_view_transform
        xyz_cam = xyz_hom @ W_mat
        gaussian_depth_cam = xyz_cam[:, 2]
        
        in_frustum = (xy_ndc[:, 0].abs() < 1.0) & (xy_ndc[:, 1].abs() < 1.0) & (gaussian_depth_cam > 0)
        
        grid = xy_ndc.view(1, N, 1, 2)
        rendered_depth = F.grid_sample(
            depth_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, 0, :, 0]
        
        not_occluded = (gaussian_depth_cam - rendered_depth).abs() < occlusion_th
        visible = in_frustum & not_occluded
        visibility_mask[view_idx] = visible
        sampled_depths[view_idx] = torch.where(visible, rendered_depth, torch.zeros_like(rendered_depth))
    
    # Compute per-Gaussian depth variance
    vis_count = visibility_mask.float().sum(dim=0)  # [N]
    depth_mean = torch.zeros(N, device=device)
    for k in range(num_views):
        depth_mean += torch.where(visibility_mask[k], sampled_depths[k], torch.zeros_like(depth_mean))
    depth_mean = torch.where(vis_count > 0, depth_mean / (vis_count + 1e-8), torch.zeros_like(depth_mean))
    
    depth_var = torch.zeros(N, device=device)
    for k in range(num_views):
        diff = torch.where(visibility_mask[k], sampled_depths[k] - depth_mean, torch.zeros_like(depth_mean))
        depth_var += diff ** 2
    depth_var = torch.where(vis_count > 1, depth_var / (vis_count - 1 + 1e-8), torch.zeros_like(depth_var))
    
    # Normalize by mean depth squared (coefficient of variation squared)
    if normalize:
        depth_var = depth_var / (depth_mean ** 2 + 1e-8)
    
    # Only valid for Gaussians seen in >= 2 views
    valid = vis_count >= 2
    depth_var = torch.where(valid, depth_var, torch.zeros_like(depth_var))
    
    # Loop weight: based on scale_conf (low scale_conf = loop candidate = high weight)
    scales = gaussians.get_scaling.clamp(min=1e-7)
    geo_mean = torch.sqrt(scales[:, 0] * scales[:, 1])
    scale_median = geo_mean.median()
    log_ratio = (torch.log(geo_mean) - torch.log(scale_median)).abs()
    scale_conf = (1.0 - log_ratio / 2.0).clamp(0, 1)
    scale_conf = torch.where(scale_conf.isnan(), torch.ones_like(scale_conf) * 0.5, scale_conf)
    
    # Soft weight: low scale_conf → high weight
    loop_weight = (1.0 - scale_conf).clamp(0, 1)
    # Hard threshold: zero out confident Gaussians
    loop_weight = torch.where(scale_conf < loop_threshold, loop_weight, torch.zeros_like(loop_weight))
    
    print(f"[CVD] depth_var: mean={depth_var.mean():.6f}, max={depth_var.max():.6f}, "
          f"loop_candidates={(scale_conf < loop_threshold).sum()}/{N}, "
          f"weighted_var={(loop_weight * depth_var).mean():.6f}", flush=True)
    
    return depth_var, loop_weight

@torch.no_grad()
def compute_multi_signal_confidence(gaussians, scene, pipe, background,
                                     num_views=8, occlusion_th=0.05,
                                     signal_weights=None):
    """Compute per-Gaussian geometric confidence from ALL available Gaussian properties.

    Signals extracted:
      [Multi-view rendering] (requires K render passes)
        1. normal_consistency: pairwise cosine similarity of rendered normals
        2. depth_consistency:  1 - normalized depth variance across views
        3. color_consistency:  1 - color variance across views (specular detector)

      [Gaussian attributes] (zero-cost, from existing parameters)
        4. opacity_conf:   high opacity → confident
        5. scale_conf:     scale near median → confident, extreme → uncertain
        6. aspect_conf:    aspect ratio near 1 → confident, elongated → uncertain
        7. sh_conf:        low SH rest magnitude → Lambertian → confident (specular filter)

      [Training dynamics] (from training state)
        8. grad_conf:      low late-training gradient → converged → confident

    Returns: per-Gaussian confidence [N] in [0, 1]
    """
    train_cams = scene.getTrainCameras()
    indices = torch.randperm(len(train_cams))[:num_views]

    xyz = gaussians.get_xyz  # [N, 3]
    N = xyz.shape[0]
    device = xyz.device

    # Default weights for each signal (can be overridden)
    default_weights = {
        'normal_consistency': 1.0,
        'depth_consistency': 0.5,
        'color_consistency': 0.5,
        'opacity': 0.3,
        'scale': 0.3,
        'aspect_ratio': 0.3,
        'sh_viewdep': 0.5,
        'gradient': 0.3,
    }
    w = signal_weights if signal_weights else default_weights

    # ================================================================
    # Part 1: Multi-view rendering signals
    # ================================================================
    visibility_mask = torch.zeros(num_views, N, dtype=torch.bool, device=device)
    sampled_normals = torch.zeros(num_views, N, 3, device=device)
    sampled_depths = torch.zeros(num_views, N, device=device)
    sampled_colors = torch.zeros(num_views, N, 3, device=device)

    xyz_hom = torch.cat([xyz, torch.ones(N, 1, device=device)], dim=1)

    for view_idx, cam_idx in enumerate(indices):
        cam = train_cams[cam_idx]
        pkg = render(cam, gaussians, pipe, background)
        normal_map = pkg["surf_normal"]    # [3, H, W]
        depth_map = pkg["surf_depth"]      # [1, H, W]
        color_map = pkg["render"]          # [3, H, W]

        proj = cam.full_proj_transform
        xyz_proj = xyz_hom @ proj
        xy_ndc = xyz_proj[:, :2] / (xyz_proj[:, 3:4] + 1e-8)

        W_mat = cam.world_view_transform
        xyz_cam = xyz_hom @ W_mat
        gaussian_depth_cam = xyz_cam[:, 2]

        in_frustum = (xy_ndc[:, 0].abs() < 1.0) & (xy_ndc[:, 1].abs() < 1.0) & (gaussian_depth_cam > 0)

        grid = xy_ndc.view(1, N, 1, 2)
        rendered_depth = F.grid_sample(
            depth_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, 0, :, 0]

        not_occluded = (gaussian_depth_cam - rendered_depth).abs() < occlusion_th
        visible = in_frustum & not_occluded
        visibility_mask[view_idx] = visible

        # Sample normals
        sampled_normals[view_idx] = F.grid_sample(
            normal_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, :, :, 0].T  # [N, 3]

        # Sample depths (camera-space)
        sampled_depths[view_idx] = rendered_depth

        # Sample colors
        sampled_colors[view_idx] = F.grid_sample(
            color_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, :, :, 0].T  # [N, 3]

    # Signal 1: Normal consistency (existing method)
    normal_cons = torch.zeros(N, device=device)
    pair_counts = torch.zeros(N, device=device)
    for i in range(num_views):
        for j in range(i + 1, num_views):
            both = visibility_mask[i] & visibility_mask[j]
            if both.any():
                cos_sim = (sampled_normals[i] * sampled_normals[j]).sum(dim=1).clamp(-1, 1)
                normal_cons += torch.where(both, cos_sim, torch.zeros_like(cos_sim))
                pair_counts += both.float()
    has_pairs = pair_counts > 0
    normal_cons = torch.where(has_pairs, normal_cons / (pair_counts + 1e-8),
                              torch.ones_like(normal_cons) * 0.5).clamp(0, 1)

    # Signal 2: Depth consistency (low variance = confident)
    vis_count = visibility_mask.float().sum(dim=0)  # [N]
    depth_mean = torch.zeros(N, device=device)
    depth_var = torch.zeros(N, device=device)
    for k in range(num_views):
        depth_mean += torch.where(visibility_mask[k], sampled_depths[k], torch.zeros_like(depth_mean))
    depth_mean = torch.where(vis_count > 0, depth_mean / (vis_count + 1e-8), torch.zeros_like(depth_mean))
    for k in range(num_views):
        diff = torch.where(visibility_mask[k], sampled_depths[k] - depth_mean, torch.zeros_like(depth_mean))
        depth_var += diff ** 2
    depth_var = torch.where(vis_count > 1, depth_var / (vis_count - 1 + 1e-8), torch.zeros_like(depth_var))
    # Normalize: coefficient of variation (std/mean)
    depth_cv = torch.sqrt(depth_var + 1e-8) / (depth_mean.abs() + 1e-4)
    depth_cons = (1.0 - depth_cv.clamp(0, 1)).clamp(0, 1)
    depth_cons = torch.where(vis_count >= 2, depth_cons, torch.ones_like(depth_cons) * 0.5)

    # Signal 3: Color consistency (low variance = Lambertian = confident)
    color_mean = torch.zeros(N, 3, device=device)
    for k in range(num_views):
        color_mean += torch.where(visibility_mask[k].unsqueeze(1), sampled_colors[k], torch.zeros_like(color_mean))
    color_mean = torch.where(vis_count.unsqueeze(1) > 0, color_mean / (vis_count.unsqueeze(1) + 1e-8), torch.zeros_like(color_mean))
    color_var = torch.zeros(N, device=device)
    for k in range(num_views):
        diff = torch.where(visibility_mask[k].unsqueeze(1), sampled_colors[k] - color_mean, torch.zeros_like(color_mean))
        color_var += (diff ** 2).sum(dim=1)
    color_var = torch.where(vis_count > 1, color_var / (vis_count - 1 + 1e-8), torch.zeros_like(color_var))
    # High color variance → specular/view-dependent → low confidence
    color_std = torch.sqrt(color_var + 1e-8)
    color_cons = (1.0 - (color_std / (color_std.median() + 1e-6)).clamp(0, 2) / 2.0).clamp(0, 1)
    color_cons = torch.where(vis_count >= 2, color_cons, torch.ones_like(color_cons) * 0.5)

    # ================================================================
    # Part 2: Gaussian attribute signals (zero-cost)
    # ================================================================

    # Signal 4: Opacity confidence
    opacity = gaussians.get_opacity.squeeze()  # [N]
    opacity_conf = opacity.clamp(0, 1)  # high opacity = confident

    # Signal 5: Scale confidence (near median = confident)
    scales = gaussians.get_scaling  # [N, 2]
    scales_safe = scales.clamp(min=1e-7)  # prevent NaN from exp(-inf)
    geo_mean = torch.sqrt(scales_safe[:, 0] * scales_safe[:, 1])
    scale_median = geo_mean.median()
    log_ratio = (torch.log(geo_mean) - torch.log(scale_median)).abs()
    scale_conf = (1.0 - log_ratio / 2.0).clamp(0, 1)  # ±e² from median → 0
    scale_conf = torch.where(scale_conf.isnan(), torch.ones_like(scale_conf) * 0.5, scale_conf)

    # Signal 6: Aspect ratio confidence (near 1 = confident)
    s_max = scales_safe.max(dim=1).values
    s_min = scales_safe.min(dim=1).values
    aspect = s_max / s_min
    aspect_conf = (1.0 - (aspect - 1.0).clamp(0, 10) / 10.0).clamp(0, 1)
    aspect_conf = torch.where(aspect_conf.isnan(), torch.ones_like(aspect_conf) * 0.5, aspect_conf)

    # Signal 7: SH view-dependence (low = Lambertian = confident)
    sh_dc = gaussians._features_dc.squeeze(1)  # [N, 3]
    sh_rest = gaussians._features_rest          # [N, 15, 3]
    sh_dc_norm = sh_dc.norm(dim=1) + 1e-8      # [N]
    sh_rest_norm = sh_rest.norm(dim=(1, 2))     # [N]
    sh_ratio = sh_rest_norm / sh_dc_norm        # high = view-dependent
    sh_conf = (1.0 / (1.0 + sh_ratio)).clamp(0, 1)  # sigmoid-like

    # ================================================================
    # Part 3: Training dynamics signals
    # ================================================================

    # Signal 8: Gradient stability (low gradient = converged = confident)
    if (hasattr(gaussians, 'xyz_gradient_accum') and gaussians.denom is not None
            and gaussians.xyz_gradient_accum.numel() > 0 and gaussians.denom.numel() > 0):
        grad_mag = (gaussians.xyz_gradient_accum / (gaussians.denom + 1e-8)).squeeze()
        grad_mag = grad_mag[:N]  # safety
        if grad_mag.shape[0] == N and (grad_mag > 0).any():
            grad_median = grad_mag[grad_mag > 0].median()
            grad_conf = (1.0 / (1.0 + grad_mag / (grad_median + 1e-8))).clamp(0, 1)
        else:
            grad_conf = torch.ones(N, device=device) * 0.5
    else:
        grad_conf = torch.ones(N, device=device) * 0.5

    # ================================================================
    # Combine: weighted average
    # ================================================================
    signals = {
        'normal_consistency': normal_cons,
        'depth_consistency': depth_cons,
        'color_consistency': color_cons,
        'opacity': opacity_conf,
        'scale': scale_conf,
        'aspect_ratio': aspect_conf,
        'sh_viewdep': sh_conf,
        'gradient': grad_conf,
    }

    weighted_sum = torch.zeros(N, device=device)
    total_weight = 0.0
    log_parts = []
    for name, signal in signals.items():
        sw = w.get(name, 0.0)
        if sw > 0:
            if signal.shape[0] != N:
                print(f"[MultiSignal] WARNING: {name} size {signal.shape[0]} != N={N}, skipping", flush=True)
                continue
            weighted_sum += sw * signal
            total_weight += sw
            log_parts.append(f"{name}={signal.mean():.3f}")

    confidence = (weighted_sum / (total_weight + 1e-8)).clamp(0, 1)

    # Diagnostics
    print(f"[MultiSignal] {' | '.join(log_parts)}", flush=True)
    print(f"[MultiSignal] combined: mean={confidence.mean():.3f}, "
          f"low(<0.3)={( confidence < 0.3).sum()}/{N}, "
          f"high(>0.7)={(confidence > 0.7).sum()}/{N}", flush=True)

    return confidence, signals  # return both combined and individual for analysis



@torch.no_grad()
def compute_split_signal_confidence(gaussians, scene, pipe, background,
                                     num_views=8, occlusion_th=0.05):
    """Compute per-Gaussian confidence split into two independent groups.

    Group A (Loop diagnosis): scale, opacity, aspect_ratio, gradient
      → Used for dist loss relaxation. Low loop_conf = loop Gaussian = relax dist.

    Group B (Rendering diagnosis): normal_consistency, depth_consistency, color_consistency, sh_viewdep
      → Used for densification. Low render_conf = rendering unstable = needs more Gaussians.

    The two groups are empirically independent (Pearson r < 0.1 cross-group).
    """
    train_cams = scene.getTrainCameras()
    indices = torch.randperm(len(train_cams))[:num_views]

    xyz = gaussians.get_xyz
    N = xyz.shape[0]
    device = xyz.device

    # ================================================================
    # Multi-view rendering signals (Group B components)
    # ================================================================
    visibility_mask = torch.zeros(num_views, N, dtype=torch.bool, device=device)
    sampled_normals = torch.zeros(num_views, N, 3, device=device)
    sampled_depths = torch.zeros(num_views, N, device=device)
    sampled_colors = torch.zeros(num_views, N, 3, device=device)

    xyz_hom = torch.cat([xyz, torch.ones(N, 1, device=device)], dim=1)

    for view_idx, cam_idx in enumerate(indices):
        cam = train_cams[cam_idx]
        pkg = render(cam, gaussians, pipe, background)
        normal_map = pkg["surf_normal"]
        depth_map = pkg["surf_depth"]
        color_map = pkg["render"]

        proj = cam.full_proj_transform
        xyz_proj = xyz_hom @ proj
        xy_ndc = xyz_proj[:, :2] / (xyz_proj[:, 3:4] + 1e-8)

        W_mat = cam.world_view_transform
        xyz_cam = xyz_hom @ W_mat
        gaussian_depth_cam = xyz_cam[:, 2]

        in_frustum = (xy_ndc[:, 0].abs() < 1.0) & (xy_ndc[:, 1].abs() < 1.0) & (gaussian_depth_cam > 0)

        grid = xy_ndc.view(1, N, 1, 2)
        rendered_depth = F.grid_sample(
            depth_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, 0, :, 0]

        not_occluded = (gaussian_depth_cam - rendered_depth).abs() < occlusion_th
        visible = in_frustum & not_occluded
        visibility_mask[view_idx] = visible

        sampled_normals[view_idx] = F.grid_sample(
            normal_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, :, :, 0].T

        sampled_depths[view_idx] = rendered_depth

        sampled_colors[view_idx] = F.grid_sample(
            color_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, :, :, 0].T

    vis_count = visibility_mask.float().sum(dim=0)

    # --- Normal consistency ---
    normal_cons = torch.zeros(N, device=device)
    pair_counts = torch.zeros(N, device=device)
    for i in range(num_views):
        for j in range(i + 1, num_views):
            both = visibility_mask[i] & visibility_mask[j]
            if both.any():
                cos_sim = (sampled_normals[i] * sampled_normals[j]).sum(dim=1).clamp(-1, 1)
                normal_cons += torch.where(both, cos_sim, torch.zeros_like(cos_sim))
                pair_counts += both.float()
    has_pairs = pair_counts > 0
    normal_cons = torch.where(has_pairs, normal_cons / (pair_counts + 1e-8),
                              torch.ones_like(normal_cons) * 0.5).clamp(0, 1)

    # --- Depth consistency ---
    depth_mean = torch.zeros(N, device=device)
    depth_var = torch.zeros(N, device=device)
    for k in range(num_views):
        depth_mean += torch.where(visibility_mask[k], sampled_depths[k], torch.zeros_like(depth_mean))
    depth_mean = torch.where(vis_count > 0, depth_mean / (vis_count + 1e-8), torch.zeros_like(depth_mean))
    for k in range(num_views):
        diff = torch.where(visibility_mask[k], sampled_depths[k] - depth_mean, torch.zeros_like(depth_mean))
        depth_var += diff ** 2
    depth_var = torch.where(vis_count > 1, depth_var / (vis_count - 1 + 1e-8), torch.zeros_like(depth_var))
    depth_cv = torch.sqrt(depth_var + 1e-8) / (depth_mean.abs() + 1e-4)
    depth_cons = (1.0 - depth_cv.clamp(0, 1)).clamp(0, 1)
    depth_cons = torch.where(vis_count >= 2, depth_cons, torch.ones_like(depth_cons) * 0.5)

    # --- Color consistency ---
    color_mean = torch.zeros(N, 3, device=device)
    for k in range(num_views):
        color_mean += torch.where(visibility_mask[k].unsqueeze(1), sampled_colors[k], torch.zeros_like(color_mean))
    color_mean = torch.where(vis_count.unsqueeze(1) > 0, color_mean / (vis_count.unsqueeze(1) + 1e-8), torch.zeros_like(color_mean))
    color_var = torch.zeros(N, device=device)
    for k in range(num_views):
        diff = torch.where(visibility_mask[k].unsqueeze(1), sampled_colors[k] - color_mean, torch.zeros_like(color_mean))
        color_var += (diff ** 2).sum(dim=1)
    color_var = torch.where(vis_count > 1, color_var / (vis_count - 1 + 1e-8), torch.zeros_like(color_var))
    color_std = torch.sqrt(color_var + 1e-8)
    color_cons = (1.0 - (color_std / (color_std.median() + 1e-6)).clamp(0, 2) / 2.0).clamp(0, 1)
    color_cons = torch.where(vis_count >= 2, color_cons, torch.ones_like(color_cons) * 0.5)

    # --- SH view-dependency ---
    sh_dc = gaussians._features_dc.squeeze(1)
    sh_rest = gaussians._features_rest
    sh_dc_norm = sh_dc.norm(dim=1) + 1e-8
    sh_rest_norm = sh_rest.norm(dim=(1, 2))
    sh_ratio = sh_rest_norm / sh_dc_norm
    sh_conf = (1.0 / (1.0 + sh_ratio)).clamp(0, 1)

    # ================================================================
    # Gaussian attribute signals (Group A components)
    # ================================================================
    opacity = gaussians.get_opacity.squeeze().clamp(0, 1)

    scales = gaussians.get_scaling.clamp(min=1e-7)
    geo_mean = torch.sqrt(scales[:, 0] * scales[:, 1])
    scale_median = geo_mean.median()
    log_ratio = (torch.log(geo_mean) - torch.log(scale_median)).abs()
    scale_conf = (1.0 - log_ratio / 2.0).clamp(0, 1)

    s_max = scales.max(dim=1).values
    s_min = scales.min(dim=1).values
    aspect_conf = (1.0 - (s_max / s_min - 1.0).clamp(0, 10) / 10.0).clamp(0, 1)

    if (hasattr(gaussians, 'xyz_gradient_accum') and gaussians.denom is not None
            and gaussians.xyz_gradient_accum.numel() > 0 and gaussians.denom.numel() > 0):
        grad_mag = (gaussians.xyz_gradient_accum / (gaussians.denom + 1e-8)).squeeze()[:N]
        if grad_mag.shape[0] == N and (grad_mag > 0).any():
            grad_median = grad_mag[grad_mag > 0].median()
            grad_conf = (1.0 / (1.0 + grad_mag / (grad_median + 1e-8))).clamp(0, 1)
        else:
            grad_conf = torch.ones(N, device=device) * 0.5
    else:
        grad_conf = torch.ones(N, device=device) * 0.5

    # ================================================================
    # Group A: Loop confidence (high = healthy, low = loop Gaussian)
    # ================================================================
    loop_conf = (
        0.3 * scale_conf +
        0.3 * opacity +
        0.3 * aspect_conf +
        0.3 * grad_conf
    ) / 1.2
    loop_conf = loop_conf.clamp(0, 1)

    # ================================================================
    # Group B: Render confidence (high = stable, low = needs densification)
    # ================================================================
    render_conf = (
        1.0 * normal_cons +
        0.5 * depth_cons +
        0.5 * color_cons +
        0.5 * sh_conf
    ) / 2.5
    render_conf = render_conf.clamp(0, 1)

    # Diagnostics
    print(f"[SplitSignal] loop_conf: mean={loop_conf.mean():.3f}, "
          f"low(<0.3)={(loop_conf < 0.3).sum()}/{N}, "
          f"high(>0.7)={(loop_conf > 0.7).sum()}/{N}", flush=True)
    print(f"[SplitSignal] render_conf: mean={render_conf.mean():.3f}, "
          f"low(<0.3)={(render_conf < 0.3).sum()}/{N}, "
          f"high(>0.7)={(render_conf > 0.7).sum()}/{N}", flush=True)
    print(f"[SplitSignal] cross-corr: {torch.corrcoef(torch.stack([loop_conf, render_conf]))[0,1]:.3f}", flush=True)

    return loop_conf, render_conf



@torch.no_grad()
def compute_purpose_signal(gaussians, scene, pipe, background,
                           num_views=8, occlusion_th=0.05):
    """Purpose-driven signal: each signal has a specific role, no weighted average.

    Returns:
        scale_conf [N]: low = large scale = loop Gaussian
        depth_conf [N]: low = depth inconsistent across views
        sh_conf [N]: low = view-dependent (specular)
        color_conf [N]: low = color varies across views (specular)
    """
    train_cams = scene.getTrainCameras()
    indices = torch.randperm(len(train_cams))[:num_views]

    xyz = gaussians.get_xyz
    N = xyz.shape[0]
    device = xyz.device

    # Multi-view rendering for depth and color consistency
    visibility_mask = torch.zeros(num_views, N, dtype=torch.bool, device=device)
    sampled_depths = torch.zeros(num_views, N, device=device)
    sampled_colors = torch.zeros(num_views, N, 3, device=device)

    xyz_hom = torch.cat([xyz, torch.ones(N, 1, device=device)], dim=1)

    for view_idx, cam_idx in enumerate(indices):
        cam = train_cams[cam_idx]
        pkg = render(cam, gaussians, pipe, background)
        depth_map = pkg["surf_depth"]
        color_map = pkg["render"]

        proj = cam.full_proj_transform
        xyz_proj = xyz_hom @ proj
        xy_ndc = xyz_proj[:, :2] / (xyz_proj[:, 3:4] + 1e-8)

        W_mat = cam.world_view_transform
        xyz_cam = xyz_hom @ W_mat
        gaussian_depth_cam = xyz_cam[:, 2]

        in_frustum = (xy_ndc[:, 0].abs() < 1.0) & (xy_ndc[:, 1].abs() < 1.0) & (gaussian_depth_cam > 0)

        grid = xy_ndc.view(1, N, 1, 2)
        rendered_depth = F.grid_sample(
            depth_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, 0, :, 0]

        not_occluded = (gaussian_depth_cam - rendered_depth).abs() < occlusion_th
        visible = in_frustum & not_occluded
        visibility_mask[view_idx] = visible
        sampled_depths[view_idx] = rendered_depth
        sampled_colors[view_idx] = F.grid_sample(
            color_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, :, :, 0].T

    vis_count = visibility_mask.float().sum(dim=0)

    # --- Depth consistency ---
    depth_mean = torch.zeros(N, device=device)
    depth_var = torch.zeros(N, device=device)
    for k in range(num_views):
        depth_mean += torch.where(visibility_mask[k], sampled_depths[k], torch.zeros_like(depth_mean))
    depth_mean = torch.where(vis_count > 0, depth_mean / (vis_count + 1e-8), torch.zeros_like(depth_mean))
    for k in range(num_views):
        diff = torch.where(visibility_mask[k], sampled_depths[k] - depth_mean, torch.zeros_like(depth_mean))
        depth_var += diff ** 2
    depth_var = torch.where(vis_count > 1, depth_var / (vis_count - 1 + 1e-8), torch.zeros_like(depth_var))
    depth_cv = torch.sqrt(depth_var + 1e-8) / (depth_mean.abs() + 1e-4)
    depth_conf = (1.0 - depth_cv.clamp(0, 1)).clamp(0, 1)
    depth_conf = torch.where(vis_count >= 2, depth_conf, torch.ones_like(depth_conf) * 0.5)

    # --- Color consistency ---
    color_mean = torch.zeros(N, 3, device=device)
    for k in range(num_views):
        color_mean += torch.where(visibility_mask[k].unsqueeze(1), sampled_colors[k], torch.zeros_like(color_mean))
    color_mean = torch.where(vis_count.unsqueeze(1) > 0, color_mean / (vis_count.unsqueeze(1) + 1e-8), torch.zeros_like(color_mean))
    color_var = torch.zeros(N, device=device)
    for k in range(num_views):
        diff = torch.where(visibility_mask[k].unsqueeze(1), sampled_colors[k] - color_mean, torch.zeros_like(color_mean))
        color_var += (diff ** 2).sum(dim=1)
    color_var = torch.where(vis_count > 1, color_var / (vis_count - 1 + 1e-8), torch.zeros_like(color_var))
    color_std = torch.sqrt(color_var + 1e-8)
    color_conf = (1.0 - (color_std / (color_std.median() + 1e-6)).clamp(0, 2) / 2.0).clamp(0, 1)
    color_conf = torch.where(vis_count >= 2, color_conf, torch.ones_like(color_conf) * 0.5)

    # --- Scale confidence ---
    scales = gaussians.get_scaling.clamp(min=1e-7)
    geo_mean = torch.sqrt(scales[:, 0] * scales[:, 1])
    scale_median = geo_mean.median()
    log_ratio = (torch.log(geo_mean) - torch.log(scale_median)).abs()
    scale_conf = (1.0 - log_ratio / 2.0).clamp(0, 1)

    # --- SH view-dependency ---
    sh_dc = gaussians._features_dc.squeeze(1)
    sh_rest = gaussians._features_rest
    sh_dc_norm = sh_dc.norm(dim=1) + 1e-8
    sh_rest_norm = sh_rest.norm(dim=(1, 2))
    sh_ratio = sh_rest_norm / sh_dc_norm
    sh_conf = (1.0 / (1.0 + sh_ratio)).clamp(0, 1)

    # --- Normal consistency ---
    normal_cons = torch.zeros(N, device=device)
    pair_counts = torch.zeros(N, device=device)
    # Need to render normals - reuse views
    sampled_normals = torch.zeros(num_views, N, 3, device=device)
    for view_idx, cam_idx in enumerate(indices):
        cam = train_cams[cam_idx]
        pkg = render(cam, gaussians, pipe, background)
        normal_map = pkg["surf_normal"]
        proj = cam.full_proj_transform
        xyz_proj = xyz_hom @ proj
        xy_ndc = xyz_proj[:, :2] / (xyz_proj[:, 3:4] + 1e-8)
        grid = xy_ndc.view(1, N, 1, 2)
        sampled_normals[view_idx] = F.grid_sample(
            normal_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )[0, :, :, 0].T
    for i in range(num_views):
        for j in range(i + 1, num_views):
            both = visibility_mask[i] & visibility_mask[j]
            if both.any():
                cos_sim = (sampled_normals[i] * sampled_normals[j]).sum(dim=1).clamp(-1, 1)
                normal_cons += torch.where(both, cos_sim, torch.zeros_like(cos_sim))
                pair_counts += both.float()
    has_pairs = pair_counts > 0
    normal_cons = torch.where(has_pairs, normal_cons / (pair_counts + 1e-8),
                              torch.ones_like(normal_cons) * 0.5).clamp(0, 1)

    # Diagnostics
    is_specular = (sh_conf < 0.3) | (color_conf < 0.3)
    loop_gate = scale_conf * depth_conf
    print(f"[PurposeSignal] scale_conf={scale_conf.mean():.3f}, depth_conf={depth_conf.mean():.3f}, "
          f"normal_cons={normal_cons.mean():.3f}, sh_conf={sh_conf.mean():.3f}, color_conf={color_conf.mean():.3f}", flush=True)
    print(f"[PurposeSignal] specular={is_specular.sum()}/{N} ({100*is_specular.float().mean():.1f}%), "
          f"loop_gate: mean={loop_gate.mean():.3f}, low(<0.3)={(loop_gate<0.3).sum()}/{N}", flush=True)

    return scale_conf, depth_conf, sh_conf, color_conf, normal_cons



def compute_dist_gradient_conflict(gaussians, scene, pipe, background, num_views=4):
    """Measure per-Gaussian dist gradient conflict across views.

    For each sampled view, computes gradient of dist loss w.r.t. Gaussian scaling.
    High cosine dissimilarity across views = gradient conflict = dist loss is harmful.

    Key fix: only compute conflict for Gaussians with significant gradient magnitude.
    Gaussians with negligible gradients contribute nothing to dist loss → no conflict.

    Returns: conflict score [N] in [0, 1], where 1 = max conflict.
    """
    train_cams = scene.getTrainCameras()
    indices = torch.randperm(len(train_cams))[:num_views]

    N = gaussians._scaling.shape[0]
    device = gaussians._scaling.device
    view_grads = []

    for cam_idx in indices:
        cam = train_cams[cam_idx]
        render_pkg = render(cam, gaussians, pipe, background)
        rend_dist = render_pkg["rend_dist"]
        dist_loss = rend_dist.mean()

        try:
            scaling_grad = torch.autograd.grad(
                dist_loss, gaussians._scaling,
                retain_graph=False, create_graph=False
            )[0]
            view_grads.append(scaling_grad.detach())
        except RuntimeError:
            continue

    if len(view_grads) < 2:
        return torch.zeros(N, device=device)  # no conflict info → apply dist normally

    view_grads = torch.stack(view_grads)  # [K, N, 2]

    # Per-Gaussian max gradient norm across views: identifies which Gaussians
    # actually participate in dist loss
    grad_norms = view_grads.norm(dim=2)  # [K, N]
    max_grad_norm = grad_norms.max(dim=0).values  # [N]

    # Only compute conflict for Gaussians with significant gradient
    # Use 10th percentile of non-zero norms as threshold
    nonzero_norms = max_grad_norm[max_grad_norm > 1e-10]
    if len(nonzero_norms) < 100:
        return torch.zeros(N, device=device)
    grad_threshold = nonzero_norms.quantile(0.1)
    significant = max_grad_norm > grad_threshold  # [N] bool

    # Per-Gaussian pairwise cosine similarity (only for significant Gaussians)
    consistency = torch.zeros(N, device=device)
    pair_count = 0

    for i in range(len(view_grads)):
        for j in range(i + 1, len(view_grads)):
            g_i = view_grads[i]  # [N, 2]
            g_j = view_grads[j]  # [N, 2]
            dot = (g_i * g_j).sum(dim=1)
            norm_i = g_i.norm(dim=1)
            norm_j = g_j.norm(dim=1)
            # Safe cosine: only where both norms are significant
            both_sig = (norm_i > 1e-10) & (norm_j > 1e-10)
            cos_sim = torch.where(
                both_sig,
                dot / (norm_i * norm_j + 1e-8),
                torch.ones_like(dot)  # no conflict if either grad is ~0
            )
            consistency += cos_sim.clamp(-1, 1)
            pair_count += 1

    if pair_count > 0:
        consistency = consistency / pair_count

    # Convert: consistency +1 (agree) → conflict 0, -1 (oppose) → conflict 1
    conflict = ((1.0 - consistency) / 2.0).clamp(0, 1)

    # Zero out conflict for insignificant Gaussians (they don't need gating)
    conflict = torch.where(significant, conflict, torch.zeros_like(conflict))

    n_sig = significant.sum()
    sig_conflict = conflict[significant]
    print(f"[GradConflict] significant={n_sig}/{N} "
          f"({100*n_sig/N:.1f}%), conflict: mean={sig_conflict.mean():.3f}, "
          f"median={sig_conflict.median():.3f}, high(>0.5)={( sig_conflict > 0.5).sum()}/{n_sig}",
          flush=True)

    return conflict


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    _p1_log_dir = None
    _p1_snap_iters = set()
    _conf_xyz_logit = None
    _conf_rot_logit = None
    _conf_optimizer = None
    if getattr(opt, 'phase1_logging', ''):
        _p1_snap_iters = set(phase1_log.parse_snapshot_iters(opt.snapshot_iters))
        _p1_log_dir = phase1_log.init_log_dir(opt.phase1_logging, dataset.source_path, sorted(_p1_snap_iters))
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, scale_init_cap=opt.scale_init_cap)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Store initial scale for loop-preventive densification
    if opt.loop_densify:
        gaussians.set_initial_scale()
        print(f"[LoopDensify] initial_scale stored for {len(gaussians.get_xyz)} Gaussians", flush=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Load geometric prior (VGGT/DUSt3R)
    geo_prior_pts = None
    geo_prior_tree = None
    if hasattr(args, 'geo_prior') and args.geo_prior and os.path.exists(args.geo_prior):
        import numpy as np_geo
        from scipy.spatial import KDTree as _KDTree
        prior_data = np_geo.load(args.geo_prior)
        geo_prior_pts_np = prior_data["points"].astype(np.float32) if 'np' in dir() else prior_data["points"].astype(__import__('numpy').float32)
        geo_prior_conf_np = prior_data["confidence"]
        geo_prior_pts = torch.tensor(geo_prior_pts_np, dtype=torch.float32, device="cuda")
        geo_prior_conf = torch.tensor(geo_prior_conf_np, dtype=torch.float32, device="cuda")
        # Normalize confidence to [0, 1]
        if geo_prior_conf.max() > 1.0:
            geo_prior_conf = (geo_prior_conf - geo_prior_conf.min()) / (geo_prior_conf.max() - geo_prior_conf.min() + 1e-8)
        # Build KDTree on CPU (avoids OOM from torch.cdist)
        geo_prior_tree = _KDTree(geo_prior_pts_np)
        print(f"[GeoPrior] Loaded {len(geo_prior_pts)} points, conf: {geo_prior_conf.mean():.3f}+-{geo_prior_conf.std():.3f}")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    v10_cache = {"mask": None, "n": 0}
    v12_cache = {"mask": None, "n": 0, "hist": None, "cams_sample": None, "hi_masks": None, "last_update": -1}
    ema_normal_for_log = 0.0
    ema_cons_for_log = 0.0

    # Multi-view normal consistency state
    normal_consistency = None
    multi_signals = None
    exp_b_risk = None  # per-Gaussian risk score for Exp B
    consensus_normals = None
    prev_num_gaussians = len(gaussians.get_xyz)

    # Cross-view depth consistency cache
    cvd_depth_var = None
    cvd_loop_weight = None
    prev_num_gaussians_cvd = len(gaussians.get_xyz)

    # ---- Sparse region Gaussian tracking ----
    sparse_tracker = None
    if opt.track_sparse:
        import json as _json
        # Identify sparse Gaussians at init: top 10% by NN distance
        with torch.no_grad():
            init_xyz = gaussians.get_xyz.detach()
            from pytorch3d.ops import knn_points
            try:
                knn = knn_points(init_xyz[None], init_xyz[None], K=2)
                nn_dist = knn.dists[0, :, 1].sqrt()  # distance to nearest neighbor
            except ImportError:
                # fallback: cdist (slower but works)
                dists = torch.cdist(init_xyz, init_xyz)
                dists.fill_diagonal_(float('inf'))
                nn_dist = dists.min(dim=1).values
            threshold = torch.quantile(nn_dist, 0.9)
            sparse_mask = nn_dist >= threshold
            sparse_indices = torch.where(sparse_mask)[0]
            print(f"[Track] Sparse Gaussians: {len(sparse_indices)}/{len(init_xyz)} "
                  f"(NN dist >= {threshold:.4f})")
        sparse_tracker = {
            'indices': sparse_indices,
            'init_positions': init_xyz[sparse_indices].clone(),
            'snapshots': [],
        }

    # ---- Depth CV gating state ----
    depth_ema = {}  # per-camera uid -> {mean: [1,H,W], var: [1,H,W], count: int}
    hinge_scale_cache = [None]  # cached scale_map for hinge dist

    # ---- Gradient conflict gating state ----
    grad_conflict_cache = None  # per-Gaussian conflict score [N]
    prev_num_gaussians_gc = 0

    # ---- Split-signal state ----
    loop_conf = None
    render_conf = None

    # ---- Purpose-signal state ----
    ps_scale_conf = None
    ps_depth_conf = None
    ps_sh_conf = None
    ps_color_conf = None
    ps_normal_conf = None

    if opt.log_cancel_v11:
        traj_log_cancel_v11.init(scene.model_path, opt.gt_depth_dir if opt.gt_depth_dir else None, scene.getTrainCameras(), render, pipe, background, k=opt.v11_k, n_samples=opt.v11_n_samples)
    if opt.log_perview:
        traj_log_perview.init(scene.model_path, scene.getTrainCameras(), render, pipe, background, n_sample=opt.perview_n_sample, anchors=tuple(int(x) for x in opt.perview_anchors.split(",")))
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    cancel_amp_mask = None
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Update multi-view normal consistency periodically
        # Determine earliest start: reg or densify
        consistency_earliest = opt.consistency_start_iter
        if opt.consistency_densify:
            consistency_earliest = min(consistency_earliest, opt.consistency_densify_start)

        use_consistency = (opt.consistency_normal or opt.consistency_only or opt.consistency_densify or opt.multi_signal or opt.relax_suppress or opt.consistency_loss or opt.split_signal or opt.purpose_signal or opt.diagnose_prescribe or opt.signal_role or opt.exp_b_risk_rot or opt.exp_c_risk_dist or opt.exp_a_risk_ms or opt.exp_a_risk_mesh)
        if use_consistency and iteration >= consistency_earliest:
            cur_N = len(gaussians.get_xyz)
            needs_update = (
                normal_consistency is None or
                cur_N != prev_num_gaussians or
                iteration % opt.consistency_update_interval == 0
            )
            if needs_update:
                if opt.purpose_signal:
                    ps_scale_conf, ps_depth_conf, ps_sh_conf, ps_color_conf, ps_normal_conf = compute_purpose_signal(
                        gaussians, scene, pipe, background, opt.consistency_views,
                        opt.consistency_occlusion_th)
                    # Set normal_consistency to dummy so the gating branch triggers
                    normal_consistency = ps_scale_conf
                elif opt.split_signal:
                    loop_conf, render_conf = compute_split_signal_confidence(
                        gaussians, scene, pipe, background, opt.consistency_views,
                        opt.consistency_occlusion_th)
                    normal_consistency = loop_conf  # reuse variable for gating
                elif opt.relax_suppress or opt.multi_signal or opt.diagnose_prescribe or opt.signal_role or opt.exp_b_risk_rot or opt.exp_c_risk_dist or opt.exp_a_risk_ms or opt.exp_a_risk_mesh:
                    ms_weights = {
                        'normal_consistency': opt.ms_w_normal if not opt.relax_suppress else 0.0,
                        'depth_consistency': opt.ms_w_depth if not opt.relax_suppress else 0.0,
                        'color_consistency': opt.ms_w_color if not opt.relax_suppress else 0.0,
                        'opacity': opt.ms_w_opacity if not opt.relax_suppress else 0.0,
                        'scale': opt.ms_w_scale,
                        'aspect_ratio': opt.ms_w_aspect,
                        'sh_viewdep': opt.ms_w_sh,
                        'gradient': opt.ms_w_grad,
                    }
                    normal_consistency, multi_signals = compute_multi_signal_confidence(
                        gaussians, scene, pipe, background, opt.consistency_views,
                        opt.consistency_occlusion_th, signal_weights=ms_weights)
                    # ==== Exp B: data-driven per-Gaussian risk score ====
                    if (opt.exp_b_risk_rot or opt.exp_c_risk_dist or opt.exp_a_risk_ms or opt.exp_a_risk_mesh) and multi_signals is not None:
                        _op = multi_signals['opacity'].clamp(0, 1)
                        _cc = multi_signals['color_consistency'].clamp(0, 1)
                        _nc = multi_signals['normal_consistency'].clamp(0, 1)
                        _sh = multi_signals['sh_viewdep'].clamp(0, 1)
                        _ar = multi_signals['aspect_ratio'].clamp(0, 1)
                        _scales = gaussians.get_scaling.clamp(min=1e-7)
                        _geo_mean = torch.sqrt(_scales[:, 0] * _scales[:, 1])

                        if opt.exp_a_risk_mesh:
                            # Exp A v3: mesh-error-based weights from Step 3 analysis
                            # Weights = |Spearman(signal, mesh_error)|, max-normalized
                            # Sign: good→high (neg corr) → (1-signal), bad→high (pos corr) → signal
                            _dc = multi_signals['depth_consistency'].clamp(0, 1)
                            _sc = multi_signals.get('scale', None)
                            if _sc is None:
                                _sc = (1.0 - ((torch.log(_geo_mean) - torch.log(_geo_mean.median())).abs() / 2.0).clamp(0, 1))
                            else:
                                _sc = _sc.clamp(0, 1)
                            W = 1.000 + 0.793 + 0.769 + 0.569 + 0.414 + 0.115 + 0.034
                            exp_b_risk = (
                                1.000 * (1.0 - _dc)       # depth_consistency: good→high → invert
                                + 0.793 * _cc              # color_consistency: bad→high → as-is
                                + 0.769 * (1.0 - _nc)     # normal_consistency: good→high → invert
                                + 0.569 * _sh              # sh_viewdep: bad→high → as-is
                                + 0.414 * (1.0 - _ar)     # aspect_ratio: good→high → invert
                                + 0.115 * (1.0 - _op)     # opacity: good→high → invert (low opacity = risky)
                                + 0.034 * (1.0 - _sc)     # scale: good→high → invert
                            ) / W
                        else:
                            # Exp A v2 / Exp B / Exp C: interference-based weights
                            _scale_large_risk = ((torch.log(_geo_mean) - torch.log(_geo_mean.median())).clamp(0, 2) / 2.0)
                            W = 1.000 + 0.390 + 0.305 + 0.183 + 0.054 + 0.053
                            exp_b_risk = (
                                1.000 * _op
                                + 0.390 * _cc
                                + 0.305 * (1.0 - _nc)
                                + 0.183 * _sh
                                + 0.054 * _scale_large_risk
                                + 0.053 * _ar
                            ) / W
                        exp_b_risk = exp_b_risk.clamp(0, 1).detach()
                        if iteration % 1000 == 0:
                            label = "v3-mesh" if opt.exp_a_risk_mesh else "ExpB"
                            print(f"[{label}@{iteration}] risk: mean={exp_b_risk.mean():.3f}, "
                                  f"p10={exp_b_risk.quantile(0.1):.3f}, p90={exp_b_risk.quantile(0.9):.3f}, "
                                  f"N={exp_b_risk.shape[0]}", flush=True)
                    # Extract render_conf from Group B signals for densification
                    if opt.split_signal_densify_threshold > 0 and multi_signals is not None:
                        _nc = multi_signals.get('normal_consistency', torch.ones(cur_N, device='cuda') * 0.5)
                        _dc = multi_signals.get('depth_consistency', torch.ones(cur_N, device='cuda') * 0.5)
                        _cc = multi_signals.get('color_consistency', torch.ones(cur_N, device='cuda') * 0.5)
                        _sh = multi_signals.get('sh_viewdep', torch.ones(cur_N, device='cuda') * 0.5)
                        render_conf = (1.0 * _nc + 0.5 * _dc + 0.5 * _cc + 0.5 * _sh) / 2.5
                        render_conf = render_conf.clamp(0, 1)
                        print(f"[MS+RenderDensify] render_conf: mean={render_conf.mean():.3f}, "
                              f"low(<0.3)={(render_conf < 0.3).sum()}/{cur_N}", flush=True)
                    # Also compute consensus normals if consistency_loss is enabled
                    if opt.consistency_loss:
                        _, consensus_normals = compute_normal_consistency(
                            gaussians, scene, pipe, background, opt.consistency_views,
                            opt.consistency_occlusion_th, return_consensus=True)
                elif opt.consistency_loss:
                    normal_consistency, consensus_normals = compute_normal_consistency(
                        gaussians, scene, pipe, background, opt.consistency_views,
                        opt.consistency_occlusion_th, return_consensus=True)
                else:
                    normal_consistency = compute_normal_consistency(
                        gaussians, scene, pipe, background, opt.consistency_views,
                        opt.consistency_occlusion_th)
                prev_num_gaussians = cur_N

        # Periodically compute gradient conflict scores
        if opt.grad_conflict_gate and iteration >= opt.grad_conflict_start:
            cur_N_gc = len(gaussians.get_xyz)
            needs_gc_update = (
                grad_conflict_cache is None or
                cur_N_gc != prev_num_gaussians_gc or
                iteration % opt.grad_conflict_interval == 0
            )
            if needs_gc_update:
                grad_conflict_cache = compute_dist_gradient_conflict(
                    gaussians, scene, pipe, background, opt.grad_conflict_views)
                prev_num_gaussians_gc = cur_N_gc
                if iteration % 1000 == 0:
                    print(f"[GradConflict@{iteration}] mean={grad_conflict_cache.mean():.3f}, "
                          f"high(>0.5)={(grad_conflict_cache > 0.5).sum()}/{cur_N_gc}", flush=True)

        # Cross-view depth consistency: compute/update cached variance
        if opt.cross_view_depth and iteration >= opt.cvd_start_iter:
            cur_N_cvd = len(gaussians.get_xyz)
            if (cvd_depth_var is None or
                cur_N_cvd != prev_num_gaussians_cvd or
                iteration % opt.cvd_update_interval == 0):
                cvd_depth_var, cvd_loop_weight = compute_cross_view_depth_variance(
                    gaussians, scene, pipe, background,
                    num_views=opt.cvd_num_views,
                    occlusion_th=opt.consistency_occlusion_th,
                    loop_threshold=opt.cvd_loop_threshold,
                    normalize=opt.cvd_normalize)
                prev_num_gaussians_cvd = cur_N_cvd

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Update per-camera depth EMA for depth_cv_gate
        if opt.depth_cv_gate and iteration >= opt.depth_cv_start_iter:
            with torch.no_grad():
                surf_depth = render_pkg["surf_depth"]  # [1, H, W]
                cam_uid = viewpoint_cam.uid
                alpha = opt.depth_cv_ema_alpha
                if cam_uid not in depth_ema:
                    depth_ema[cam_uid] = {
                        'mean': surf_depth.clone(),
                        'var': torch.zeros_like(surf_depth),
                        'count': 1,
                    }
                else:
                    ema = depth_ema[cam_uid]
                    delta = surf_depth - ema['mean']
                    ema['mean'] = (1 - alpha) * ema['mean'] + alpha * surf_depth
                    ema['var'] = (1 - alpha) * ema['var'] + alpha * delta ** 2
                    ema['count'] += 1

        # Candidate 5 / Phase 1: ensure cancel buffers sized for current gauss count
        if opt.cancel_composite_densify or opt.densify_method in ('OR_cancel', 'AND_cancel', 'AbsGS'):
            import diff_surfel_rasterization as _dsr_set
            _dsr_set.set_cancel_buffers(gaussians._xyz.shape[0], gaussians._xyz.device)
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        _Lrgb_part = (1.0 - opt.lambda_dssim) * Ll1
        _Lssim_part = opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = _Lrgb_part + _Lssim_part

        # Stage 1 Path γ: forward-tractable cancel-loss proxy (opacity intent)
        cancel_loss = torch.tensor(0.0, device=image.device)
        if opt.cancel_loss_lambda > 0 and iteration > opt.cancel_loss_start:
            cancel_loss = opt.cancel_loss_lambda * L_cancel_proxy(
                image, gt_image, gaussians.get_xyz,
                viewpoint_cam.full_proj_transform, render_pkg['radii'],
                K=opt.cancel_loss_K,
                radii_min=opt.cancel_loss_radii_min
            )
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]

        # Purpose-driven signal: each signal has a specific role
        if opt.purpose_signal and ps_scale_conf is not None and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                # Step 1: specular mask (per-Gaussian)
                is_specular = (ps_sh_conf < opt.ps_specular_th) | (ps_color_conf < opt.ps_specular_th)

                # Step 2: compute gate based on ps_mode
                if opt.ps_mode == 'scale_conservative':
                    # Scale only but clamped to [0.5, 1.0] — same range as MS
                    loop_gate = (0.5 + 0.5 * ps_scale_conf).clamp(0.5, 1.0)
                elif opt.ps_mode == 'scale_only':
                    loop_gate = ps_scale_conf ** opt.ps_gamma
                elif opt.ps_mode == 'depth_only':
                    loop_gate = ps_depth_conf ** opt.ps_beta
                elif opt.ps_mode == 'normal_only':
                    loop_gate = ps_normal_conf ** opt.ps_gamma if ps_normal_conf is not None else ps_depth_conf
                elif opt.ps_mode == 'scale_depth_sum':
                    loop_gate = (0.6 * ps_scale_conf + 0.4 * ps_depth_conf).clamp(0, 1)
                else:  # 'scale_depth' (default: product)
                    loop_gate = (ps_scale_conf ** opt.ps_gamma) * (ps_depth_conf ** opt.ps_beta)

                # Step 3: dist weight per-Gaussian
                # specular → 1.0 (keep dist), loop → loop_gate (relax dist)
                dist_weight_pg = torch.where(is_specular, torch.ones_like(loop_gate), loop_gate)
                dist_weight_pg = dist_weight_pg.clamp(0.1, 1.0)

                # Render to pixel space
                dw_color = dist_weight_pg.unsqueeze(1).expand(-1, 3)
                dw_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=dw_color)
                dist_weight_map = dw_pkg["render"][0:1]  # [1, H, W]
                dist_weight_map = dist_weight_map.clamp(0.1, 1.0)

            # Normal: vanilla (no relaxation - confirmed harmful)
            normal_loss = lambda_normal * (normal_error).mean()
            # Dist: purpose-driven weight
            dist_loss = lambda_dist * (dist_weight_map * rend_dist).mean()

        # Split-signal: loop_conf for dist relaxation only
        elif opt.split_signal and loop_conf is not None and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                # Render loop_conf as a map
                loop_color = loop_conf.unsqueeze(1).expand(-1, 3)
                loop_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=loop_color)
                loop_map = loop_pkg["render"][0:1]  # [1, H, W]

                # Gate: high loop_conf → healthy → keep dist, low → loop → relax
                c_low = opt.consistency_clamp_low
                loop_gate = ((loop_map - c_low) / (1.0 - c_low + 1e-6)).clamp(0.0, 1.0)
                adaptive_weight = (1.0 - 0.5 * loop_gate).clamp(0.1, 1.0)

            # Normal loss: no relaxation (keep vanilla)
            normal_loss = lambda_normal * (normal_error).mean()
            # Dist loss: relax where loop_conf is high (= healthy Gaussian gets dist)
            dist_loss = lambda_dist * (adaptive_weight * rend_dist).mean()

        # Relax-Suppress: split signals into relax (low→relax) and suppress (low→keep dist)
        # Diagnose-Prescribe: classify Gaussians by pathology, apply targeted treatment
        elif opt.diagnose_prescribe and normal_consistency is not None and multi_signals is not None and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                scale_conf = multi_signals.get('scale')
                depth_conf = multi_signals.get('depth_consistency')
                color_conf = multi_signals.get('color_consistency')
                opacity_val = gaussians.get_opacity.squeeze().clamp(0, 1)

                # Classification
                normal_conf = multi_signals.get('normal_consistency')
                is_loop = (scale_conf < opt.dp_loop_scale_th) & (normal_conf < opt.dp_loop_normal_th)
                is_artifact = (scale_conf >= opt.dp_artifact_scale_th) & (opacity_val < opt.dp_artifact_opacity_th) & (color_conf < opt.dp_artifact_color_th)
                is_healthy = ~is_loop & ~is_artifact

                # Prescription: per-Gaussian dist weight
                # Loop -> relax dist (weight 0.1), others -> keep dist (weight 1.0)
                dist_weight_per_g = torch.ones_like(scale_conf)
                dist_weight_per_g[is_loop] = 0.1

                # Render dist weight to pixel space
                dw_color = dist_weight_per_g.unsqueeze(1).expand(-1, 3)
                dw_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=dw_color)
                dist_weight_map = dw_pkg["render"][0:1]  # [1, H, W]

                # Prescription: per-Gaussian opacity suppression for artifacts
                artifact_penalty = torch.zeros_like(scale_conf)
                artifact_penalty[is_artifact] = opt.dp_artifact_suppress

                if iteration % 1000 == 0:
                    n_loop = is_loop.sum().item()
                    n_artifact = is_artifact.sum().item()
                    n_healthy = is_healthy.sum().item()
                    N_total = scale_conf.shape[0]
                    print(f"[DiagPrescribe@{iteration}] loop={n_loop}({n_loop/N_total*100:.1f}%) "
                          f"artifact={n_artifact}({n_artifact/N_total*100:.1f}%) "
                          f"healthy={n_healthy}({n_healthy/N_total*100:.1f}%)", flush=True)

            # Compute MS confidence map (continuous, like original multi_signal)
            conf_color = normal_consistency.unsqueeze(1).expand(-1, 3)
            conf_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=conf_color)
            confidence_map = conf_pkg["render"][0:1]
            c_low = opt.consistency_clamp_low
            ms_gate = ((confidence_map - c_low) / (1.0 - c_low + 1e-6)).clamp(0.0, 1.0)

            # Render artifact mask to pixel space
            artifact_flag = torch.zeros_like(scale_conf)
            artifact_flag[is_artifact] = 1.0
            af_color = artifact_flag.unsqueeze(1).expand(-1, 3)
            af_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=af_color)
            artifact_map = af_pkg["render"][0:1].clamp(0, 1)  # [1, H, W]

            # Combined gate: MS relaxation blocked where artifact density is high
            combined_gate = ms_gate * (1.0 - artifact_map)
            adaptive_weight = (1.0 - 0.5 * combined_gate).clamp(0.1, 1.0)

            # Normal loss: vanilla (don't touch normal reg)
            normal_loss = lambda_normal * (normal_error).mean()
            # Dist loss: MS-style continuous relaxation, artifact-protected
            dist_loss = lambda_dist * (adaptive_weight * rend_dist).mean()

        # Signal-Role: geometry vs appearance instability
        # Theory: dist relaxation helps only when instability is caused by dist (geometry)
        # not when instability is appearance-related (artifacts)
        elif opt.signal_role and multi_signals is not None and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                scale_conf = multi_signals.get('scale')
                depth_conf = multi_signals.get('depth_consistency')
                normal_conf = multi_signals.get('normal_consistency')
                opacity_val = gaussians.get_opacity.squeeze().clamp(0, 1)
                color_conf = multi_signals.get('color_consistency')
                sh_conf = multi_signals.get('sh_viewdep')

                # Geometry instability: dist is the cause
                geo_inst = 1.0 - (scale_conf + depth_conf + normal_conf) / 3.0

                # Appearance instability: dist is NOT the cause
                app_inst = 1.0 - (opacity_val + color_conf + sh_conf) / 3.0

                # dist_weight: relax only when geo unstable AND app stable
                dist_weight_pg = (1.0 - geo_inst * (1.0 - app_inst)).clamp(0.1, 1.0)

                # Render to pixel space
                dw_color = dist_weight_pg.unsqueeze(1).expand(-1, 3)
                dw_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=dw_color)
                dist_weight_map = dw_pkg["render"][0:1].clamp(0.1, 1.0)

                if iteration % 1000 == 0:
                    print(f'[SignalRole@{iteration}] geo_inst={geo_inst.mean():.3f} app_inst={app_inst.mean():.3f} '
                          f'dist_w={dist_weight_pg.mean():.3f}', flush=True)

            # Normal: vanilla (no relaxation)
            normal_loss = lambda_normal * (normal_error).mean()
            # Dist: geometry-aware weight
            dist_loss = lambda_dist * (dist_weight_map * rend_dist).mean()

        elif (opt.relax_suppress or opt.multi_signal) and normal_consistency is not None and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                if opt.relax_suppress and multi_signals is not None:
                    # Relax signals: low score → needs dist relaxation
                    relax_parts = []
                    relax_weights = []
                    for name, w_val in [('scale', opt.rs_relax_w_scale),
                                        ('depth_consistency', opt.rs_relax_w_depth),
                                        ('aspect_ratio', opt.rs_relax_w_aspect),
                                        ('gradient', opt.rs_relax_w_grad)]:
                        sig = multi_signals.get(name)
                        if sig is not None and w_val > 0:
                            relax_parts.append(w_val * sig)
                            relax_weights.append(w_val)
                    if relax_parts:
                        relax_score = sum(relax_parts) / (sum(relax_weights) + 1e-8)
                    else:
                        relax_score = torch.ones_like(normal_consistency) * 0.5

                    # Suppress signals: low score → specular → keep dist (block relaxation)
                    suppress_parts = []
                    suppress_weights = []
                    for name, w_val in [('sh_viewdep', opt.rs_suppress_w_sh),
                                        ('color_consistency', opt.rs_suppress_w_color)]:
                        sig = multi_signals.get(name)
                        if sig is not None and w_val > 0:
                            suppress_parts.append(w_val * sig)
                            suppress_weights.append(w_val)
                    if suppress_parts:
                        suppress_score = sum(suppress_parts) / (sum(suppress_weights) + 1e-8)
                    else:
                        suppress_score = torch.ones_like(normal_consistency) * 0.5

                    # dist_weight = (1 - relax_score) * suppress_score
                    # High when: relax_score low (needs relaxation) AND suppress_score high (not specular)
                    rs_gate_per_g = ((1.0 - relax_score) * suppress_score).clamp(0, 1)

                    # Render to pixel space
                    rs_color = rs_gate_per_g.unsqueeze(1).expand(-1, 3)
                    rs_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=rs_color)
                    rs_map = rs_pkg["render"][0:1]  # [1, H, W]

                    # Gate with threshold
                    c_low = opt.consistency_clamp_low
                    conf_gate = ((rs_map - c_low) / (1.0 - c_low + 1e-6)).clamp(0.0, 1.0)

                    if iteration % 1000 == 0:
                        print(f"[RelaxSuppress] relax={relax_score.mean():.3f} suppress={suppress_score.mean():.3f} "
                              f"gate={rs_gate_per_g.mean():.3f}", flush=True)
                else:
                    # Fallback: original multi_signal behavior
                    # MS + Artifact Guard: block relaxation for artifact Gaussians
                    if opt.ms_artifact_guard and multi_signals is not None:
                        scale_conf = multi_signals.get('scale')
                        color_conf = multi_signals.get('color_consistency')
                        opacity_val = gaussians.get_opacity.squeeze().clamp(0, 1)
                        is_artifact = (scale_conf >= opt.ms_ag_scale_th) & (opacity_val < opt.ms_ag_opacity_th) & (color_conf < opt.ms_ag_color_th)
                        # Loop Gaussians take priority: exclude from artifact mask
                        depth_conf = multi_signals.get('depth_consistency')
                        is_loop = (scale_conf < 0.4) & (depth_conf < 0.5)
                        is_artifact = is_artifact & ~is_loop
                        ms_conf = normal_consistency.clone()
                        ms_conf[is_artifact] = 0.0  # artifact → no relaxation
                        if iteration % 1000 == 0:
                            n_art = is_artifact.sum().item()
                            n_loop = is_loop.sum().item()
                            print(f'[MS+ArtifactGuard@{iteration}] artifacts={n_art}({n_art/scale_conf.shape[0]*100:.1f}%) loop_excl={n_loop}({n_loop/scale_conf.shape[0]*100:.1f}%)', flush=True)
                    else:
                        ms_conf = normal_consistency
                    conf_color = ms_conf.unsqueeze(1).expand(-1, 3)
                    conf_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=conf_color)
                    confidence_map = conf_pkg["render"][0:1]

                    c_low = opt.consistency_clamp_low
                    conf_gate = ((confidence_map - c_low) / (1.0 - c_low + 1e-6)).clamp(0.0, 1.0)

                    if opt.specular_guard and multi_signals is not None:
                        normal_cons_per_g = multi_signals.get('normal_consistency')
                        color_cons_per_g = multi_signals.get('color_consistency')
                        if normal_cons_per_g is not None and color_cons_per_g is not None:
                            specular_score = (normal_cons_per_g - color_cons_per_g).clamp(0, 1)
                            spec_color = specular_score.unsqueeze(1).expand(-1, 3)
                            spec_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=spec_color)
                            specular_map = spec_pkg["render"][0:1]
                            spec_suppress = (1.0 - specular_map.clamp(0, 1))
                            conf_gate = conf_gate * spec_suppress

                # Adaptive weight: gate=1 → relaxation, gate=0 → vanilla
                adaptive_weight = (1.0 - 0.5 * conf_gate).clamp(0.1, 1.0)

            # Normal: annealing support
            if opt.no_consistency_normal:
                normal_loss = lambda_normal * (normal_error).mean()
            elif opt.normal_anneal:
                # Vanilla until anneal_start, then linearly ramp up relaxation
                if iteration < opt.normal_anneal_start:
                    normal_loss = lambda_normal * (normal_error).mean()
                else:
                    ramp = min(1.0, (iteration - opt.normal_anneal_start) / max(opt.normal_anneal_ramp, 1))
                    normal_weight = 1.0 - ramp * (1.0 - adaptive_weight)
                    normal_loss = lambda_normal * (normal_weight * normal_error).mean()
            else:
                normal_loss = lambda_normal * (adaptive_weight * normal_error).mean()
            # Dist: always apply adaptive weight
            if opt.no_consistency_dist:
                dist_loss = lambda_dist * (rend_dist).mean()
            else:
                dist_loss = lambda_dist * (adaptive_weight * rend_dist).mean()
        # Ablation C: Uniform mild relaxation (no consistency, no scale)
        elif opt.uniform_mild and (lambda_normal > 0 or lambda_dist > 0):
            w = opt.uniform_mild_weight
            normal_loss = lambda_normal * (w * normal_error).mean()
            dist_loss = lambda_dist * (w * rend_dist).mean()
        # Ablation B: Consistency-only (no scale)
        elif opt.consistency_only and normal_consistency is not None and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                # Render consistency map only
                consistency_color = normal_consistency.unsqueeze(1).expand(-1, 3)  # [N, 3]
                consistency_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=consistency_color)
                consistency_map = consistency_pkg["render"][0:1]  # [1, H, W]

                # Gate: high consistency → open (relax), low → closed (vanilla)
                c_low = opt.consistency_clamp_low
                consistency_gate = ((consistency_map - c_low) / (1.0 - c_low + 1e-6)).clamp(0.0, 1.0)

                # Gate=1 → fixed 0.5 relaxation, Gate=0 → 1.0 (vanilla)
                adaptive_weight = (1.0 - 0.5 * consistency_gate).clamp(0.1, 1.0)

            normal_loss = lambda_normal * (adaptive_weight * normal_error).mean()
            dist_loss = lambda_dist * (adaptive_weight * rend_dist).mean()
        # Ablation A: Consistency × Scale (current method)
        elif opt.consistency_normal and normal_consistency is not None and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                scales = gaussians.get_scaling  # [N, 2]
                geo_mean_scale = torch.sqrt(scales[:, 0] * scales[:, 1])  # [N]
                # Pack scale (R) + consistency (G) into one override_color render
                combined_color = torch.stack([
                    geo_mean_scale,
                    normal_consistency,
                    torch.zeros_like(geo_mean_scale)
                ], dim=1)  # [N, 3]
                combined_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=combined_color)
                combined_map = combined_pkg["render"]  # [3, H, W]
                scale_map = combined_map[0:1]       # [1, H, W]
                consistency_map = combined_map[1:2]  # [1, H, W]

                # Scale weight: small scale → low weight
                scale_weight = (scale_map / (scale_map.median() + 1e-6)).clamp(0.1, 1.0)

                # Consistency gate: high consistency → gate=1 (allow relaxation), low → gate=0 (vanilla)
                c_low = opt.consistency_clamp_low
                consistency_gate = ((consistency_map - c_low) / (1.0 - c_low + 1e-6)).clamp(0.0, 1.0)

                # Combined: gate=1 → scale_weight, gate=0 → 1.0 (vanilla)
                adaptive_weight = consistency_gate * scale_weight + (1.0 - consistency_gate)
                adaptive_weight = adaptive_weight.clamp(0.1, 1.0)

            if opt.no_consistency_normal:
                normal_loss = lambda_normal * (normal_error).mean()
            else:
                normal_loss = lambda_normal * (adaptive_weight * normal_error).mean()
            if opt.no_consistency_dist:
                dist_loss = lambda_dist * (rend_dist).mean()
            else:
                dist_loss = lambda_dist * (adaptive_weight * rend_dist).mean()
        # Scale-proportional dist attenuation: dist_weight = 1 / (1 + scale/median)
        # Larger Gaussians get weaker dist loss, proportional to their scale
        elif opt.scale_proportional_dist and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                scales = gaussians.get_scaling  # [N, 2]
                geo_mean_scale = torch.sqrt(scales[:, 0] * scales[:, 1])  # [N]
                # Per-Gaussian weight: inversely proportional to scale
                # w = 1 / (1 + scale/median) → median-scale Gaussian gets w≈0.5, large gets w→0
                med_scale = geo_mean_scale.median().clamp(min=1e-7)
                per_g_weight = (1.0 / (1.0 + geo_mean_scale / med_scale)).clamp(0.05, 1.0)
                weight_color = per_g_weight.unsqueeze(1).expand(-1, 3)  # [N, 3]
                weight_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=weight_color)
                dist_weight = weight_pkg["render"][0:1].clamp(0.05, 1.0)  # [1, H, W]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (dist_weight * rend_dist).mean()
        # Scale-only gating (Experiment 2-A)
        elif opt.scale_gate and iteration >= opt.scale_gate_start_iter and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                scales = gaussians.get_scaling  # [N, 2]
                geo_mean_scale = torch.sqrt(scales[:, 0] * scales[:, 1])  # [N]
                scale_color = geo_mean_scale.unsqueeze(1).expand(-1, 3)  # [N, 3]
                scale_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=scale_color)
                scale_map = scale_pkg["render"][0:1]  # [1, H, W]

                if opt.scale_gate_invert:
                    # Large scale → relax (gap filler hypothesis)
                    # scale_map large → weight small
                    scale_weight = (1.0 - (scale_map / (scale_map.quantile(0.9) + 1e-6)).clamp(0, 1)).clamp(0.1, 1.0)
                else:
                    # Small scale → relax (detail hypothesis)
                    # scale_map small → weight small
                    scale_weight = (scale_map / (scale_map.median() + 1e-6)).clamp(0.1, 1.0)

            if opt.scale_gate_target in ('normal', 'both'):
                normal_loss = lambda_normal * (scale_weight * normal_error).mean()
            else:
                normal_loss = lambda_normal * (normal_error).mean()
            if opt.scale_gate_target in ('dist', 'both'):
                dist_loss = lambda_dist * (scale_weight * rend_dist).mean()
            else:
                dist_loss = lambda_dist * (rend_dist).mean()
        elif opt.hybrid_gate and iteration >= opt.hybrid_gate_start_iter and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                hy_q90 = torch.quantile(rend_dist.flatten(), 0.9) + 1e-8
                hy_norm = (rend_dist / hy_q90).clamp(0, 1)
                hybrid_weight = (1.0 - hy_norm).clamp(0.1, 1.0)
            hybrid_weight = hybrid_weight.detach()
            normal_loss = lambda_normal * (hybrid_weight * normal_error).mean()
            dist_loss = lambda_dist * (hybrid_weight * rend_dist).mean()
        elif opt.dist_freq_gate and iteration >= opt.dist_freq_start_iter and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                d_q90 = torch.quantile(rend_dist.flatten(), 0.9) + 1e-8
                d_norm = (rend_dist / d_q90).clamp(0, 1)
                dist_weight = (1.0 - d_norm).clamp(0.1, 1.0)
            dist_weight = dist_weight.detach()
            normal_loss = lambda_normal * (dist_weight * normal_error).mean()
            dist_loss = lambda_dist * (dist_weight * rend_dist).mean()
        elif opt.adaptive_normal and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                scales = gaussians.get_scaling  # [N, 2]
                geo_mean_scale = torch.sqrt(scales[:, 0] * scales[:, 1])  # [N]
                scale_color = geo_mean_scale.unsqueeze(1).expand(-1, 3)  # [N, 3]
                scale_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=scale_color)
                scale_map = scale_pkg["render"][0:1]  # [1, H, W]
                scale_weight = (scale_map / (scale_map.median() + 1e-6)).clamp(0.1, 1.0)
            normal_loss = lambda_normal * (scale_weight * normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
        elif opt.aniso_normal and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                scales = gaussians.get_scaling.clamp(min=1e-7)
                aniso = scales.max(dim=1).values / scales.min(dim=1).values
                aniso_color = aniso.unsqueeze(1).expand(-1, 3)
                aniso_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=aniso_color)
                aniso_map = aniso_pkg["render"][0:1]
                aniso_weight = (aniso_map / (aniso_map.median() + 1e-6)).clamp(0.1, 1.0)
            normal_loss = lambda_normal * (aniso_weight * normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
        elif opt.inverse_scale_normal and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                scales = gaussians.get_scaling
                geo_mean_scale = torch.sqrt(scales[:, 0] * scales[:, 1])
                scale_color = geo_mean_scale.unsqueeze(1).expand(-1, 3)
                scale_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=scale_color)
                scale_map = scale_pkg["render"][0:1]
                inv_weight = ((scale_map.median() + 1e-6) / (scale_map + 1e-6)).clamp(0.1, 1.0)
            normal_loss = lambda_normal * (inv_weight * normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
        elif opt.inv_aniso_normal and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                scales = gaussians.get_scaling.clamp(min=1e-7)
                aniso = scales.max(dim=1).values / scales.min(dim=1).values
                aniso_color = aniso.unsqueeze(1).expand(-1, 3)
                aniso_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=aniso_color)
                aniso_map = aniso_pkg["render"][0:1]
                inv_aniso_weight = ((aniso_map.median() + 1e-6) / (aniso_map + 1e-6)).clamp(0.1, 1.0)
            normal_loss = lambda_normal * (inv_aniso_weight * normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
        # Gradient conflict-aware dist loss: relax dist where gradient conflicts across views
        elif opt.grad_conflict_gate and grad_conflict_cache is not None and iteration >= opt.grad_conflict_start and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                # Render conflict score as a map
                conflict_color = grad_conflict_cache.unsqueeze(1).expand(-1, 3)  # [N, 3]
                conflict_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=conflict_color)
                conflict_map = conflict_pkg["render"][0:1]  # [1, H, W]

                # High conflict -> low weight (relax dist loss)
                gamma = opt.grad_conflict_gamma
                adaptive_weight = ((1.0 - conflict_map.clamp(0, 1)) ** gamma).clamp(0.1, 1.0)

            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (adaptive_weight * rend_dist).mean()

        # Depth CV gating: relax dist where depth is inconsistent across views
        elif opt.depth_cv_gate and iteration >= opt.depth_cv_start_iter and (lambda_normal > 0 or lambda_dist > 0):
            cam_uid = viewpoint_cam.uid
            if cam_uid in depth_ema and depth_ema[cam_uid]['count'] >= 5:
                with torch.no_grad():
                    ema = depth_ema[cam_uid]
                    depth_std = torch.sqrt(ema['var'] + 1e-8)
                    depth_cv_map = depth_std / (ema['mean'].abs() + 1e-4)  # [1, H, W]
                    # High CV → high conflict → low weight (relax dist)
                    gamma = opt.depth_cv_gamma
                    adaptive_weight = (1.0 - depth_cv_map.clamp(0, 1)) ** gamma
                    adaptive_weight = adaptive_weight.clamp(0.1, 1.0)
                normal_loss = lambda_normal * (normal_error).mean()
                dist_loss = lambda_dist * (adaptive_weight * rend_dist).mean()
            else:
                normal_loss = lambda_normal * (normal_error).mean()
                dist_loss = lambda_dist * (rend_dist).mean()
        # ==== Exp C: data-driven risk → pixel-space gate → attenuate dist loss ====
        elif opt.exp_c_risk_dist and exp_b_risk is not None and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                risk_color = exp_b_risk.unsqueeze(1).expand(-1, 3)  # [N, 3]
                risk_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=risk_color)
                risk_map = risk_pkg["render"][0:1]  # [1, H, W]
                risk_map = risk_map.clamp(0, 1)
                # High risk → attenuate dist. max_detach=0.5 → at risk=1, weight=0.5
                dist_weight = (1.0 - opt.exp_b_max_detach * risk_map).clamp(0.1, 1.0)
            normal_loss = lambda_normal * (normal_error).mean()  # normal full strength
            dist_loss = lambda_dist * (dist_weight * rend_dist).mean()
        # ==== Exp A: data-driven risk → pixel-space gate → attenuate BOTH dist & normal (MS-style) ====
        elif (opt.exp_a_risk_ms or opt.exp_a_risk_mesh) and exp_b_risk is not None and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                risk_color = exp_b_risk.unsqueeze(1).expand(-1, 3)  # [N, 3]
                risk_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=risk_color)
                risk_map = risk_pkg["render"][0:1]  # [1, H, W]
                risk_map = risk_map.clamp(0, 1)
                # Same attenuation formula as MS: high risk → relax both dist and normal
                adaptive_weight = (1.0 - opt.exp_b_max_detach * risk_map).clamp(0.1, 1.0)
            normal_loss = lambda_normal * (adaptive_weight * normal_error).mean()
            dist_loss = lambda_dist * (adaptive_weight * rend_dist).mean()
        elif opt.detail_aware_loss and (lambda_dist > 0 or lambda_normal > 0):
            with torch.no_grad():
                gt_gray = gt_image.mean(dim=0, keepdim=True).unsqueeze(0)
                sx = torch.tensor([[[[-1.,0,1],[-2,0,2],[-1,0,1]]]], device=gt_gray.device)
                sy = torch.tensor([[[[-1.,-2,-1],[0,0,0],[1,2,1]]]], device=gt_gray.device)
                import torch.nn.functional as F_
                gx = F_.conv2d(gt_gray, sx, padding=1)
                gy = F_.conv2d(gt_gray, sy, padding=1)
                gm = torch.sqrt(gx**2 + gy**2).squeeze(0)
                w_map = 1.0 / (1.0 + opt.detail_beta * gm)
            normal_loss = lambda_normal * (w_map * normal_error).mean()
            dist_loss = lambda_dist * (w_map * rend_dist).mean()
        elif opt.geom_detail_aware and (lambda_dist > 0 or lambda_normal > 0):
            with torch.no_grad():
                import torch.nn.functional as F_
                sn = surf_normal.unsqueeze(0)
                sx = torch.tensor([[[[-1.,0,1],[-2,0,2],[-1,0,1]]]], device=sn.device).expand(3,1,3,3).contiguous()
                sy = torch.tensor([[[[-1.,-2,-1],[0,0,0],[1,2,1]]]], device=sn.device).expand(3,1,3,3).contiguous()
                gx = F_.conv2d(sn, sx, padding=1, groups=3)
                gy = F_.conv2d(sn, sy, padding=1, groups=3)
                gm = torch.sqrt((gx**2 + gy**2).sum(dim=1))
                w_map = 1.0 / (1.0 + opt.geom_detail_beta * gm)
            normal_loss = lambda_normal * (w_map * normal_error).mean()
            dist_loss = lambda_dist * (w_map * rend_dist).mean()
        elif opt.tangent_normal_loss and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                import torch.nn.functional as F_
                sn = surf_normal.unsqueeze(0)
                k = int(2 * opt.tangent_blur_sigma * 2 + 1)
                kern = torch.ones(3, 1, k, k, device=sn.device) / (k*k)
                sm = F_.conv2d(sn, kern, padding=k//2, groups=3).squeeze(0)
                sm = sm / (sm.norm(dim=0, keepdim=True) + 1e-8)
            normal_error_t = (1 - (rend_normal * sm).sum(dim=0))[None]
            normal_loss = lambda_normal * normal_error_t.mean()
            dist_loss = lambda_dist * rend_dist.mean()
        elif opt.normal_coupled_dist and (lambda_normal > 0 or lambda_dist > 0):
            with torch.no_grad():
                agree = (rend_normal * surf_normal).sum(dim=0, keepdim=True).clamp(0, 1)
                w_dist = agree ** opt.normal_coupled_power
            normal_loss = lambda_normal * normal_error.mean()
            dist_loss = lambda_dist * (w_dist * rend_dist).mean()
        elif opt.v9_pathway_selective and (lambda_dist > 0 or lambda_normal > 0):
            with torch.no_grad():
                import torch.nn.functional as F_
                gt_gray = gt_image.mean(dim=0, keepdim=True).unsqueeze(0)
                sx = torch.tensor([[[[-1.,0,1],[-2,0,2],[-1,0,1]]]], device=gt_gray.device)
                sy = torch.tensor([[[[-1.,-2,-1],[0,0,0],[1,2,1]]]], device=gt_gray.device)
                gx = F_.conv2d(gt_gray, sx, padding=1)
                gy = F_.conv2d(gt_gray, sy, padding=1)
                gm = torch.sqrt(gx**2 + gy**2).squeeze(0)
                w_pixel = 1.0 / (1.0 + opt.v9_beta * gm)
                v9_corr_weight = opt.v9_alpha * (1.0 - w_pixel)
            normal_loss = lambda_normal * normal_error.mean()
            dist_loss = lambda_dist * rend_dist.mean()
            v9_corr_dist = lambda_dist * (v9_corr_weight * rend_dist).mean()
            v9_corr_normal = lambda_normal * (v9_corr_weight * normal_error).mean()
        elif opt.v10_context and (lambda_dist > 0 or lambda_normal > 0):
            normal_loss = lambda_normal * normal_error.mean()
            dist_loss = lambda_dist * rend_dist.mean()
        else:
            normal_loss = lambda_normal * (normal_error).mean()
            if opt.scale_aware_dist and lambda_dist > 0:
                # Anisotropic dist loss: weight by ray-normal alignment
                # Head-on rays (cos≈1): depth spread should be 0 → penalize strongly
                # Grazing rays (cos≈0): depth spread is natural → penalize weakly
                with torch.no_grad():
                    # Camera view direction (optical axis in world space)
                    # world_view_transform rows = [R|t], camera z-axis = 3rd row of R
                    w2c = viewpoint_cam.world_view_transform  # [4,4] row-major
                    view_dir = w2c[2, :3]  # camera z-axis in world coords
                    view_dir = view_dir / (view_dir.norm() + 1e-8)
                    
                    # rend_normal: [3, H, W] Gaussian disk normals (alpha-weighted)
                    # Normalize per-pixel (already roughly normalized but ensure)
                    rn = torch.nn.functional.normalize(rend_normal, dim=0, eps=1e-6)
                    
                    # cos angle = |dot(view_dir, normal)| per pixel
                    cos_map = torch.abs(
                        view_dir[0] * rn[0] + view_dir[1] * rn[1] + view_dir[2] * rn[2]
                    ).unsqueeze(0)  # [1, H, W]
                    
                dist_loss = lambda_dist * (cos_map * rend_dist).mean()
            elif opt.center_anchored_dist and lambda_dist > 0:
                # Hinge Dist Loss: allow depth spread proportional to Gaussian scale
                # dist_excess = max(0, rend_dist - tau * scale_map)
                # Small Gaussians: tolerance ≈ 0 → same as vanilla
                # Large Gaussians: tolerance > 0 → reduced penalty
                # Render scale map every 10 iters (memory-safe)
                if iteration % 10 == 0:
                    with torch.no_grad():
                        scales = gaussians.get_scaling
                        max_scale = scales.max(dim=1).values
                        scale_color = max_scale.unsqueeze(1).expand(-1, 3)
                        scale_pkg = render(viewpoint_cam, gaussians, pipe, background,
                                           override_color=scale_color)
                        hinge_scale_cache[0] = scale_pkg["render"][0:1].detach()
                
                if hinge_scale_cache[0] is not None:
                    tau = 1.0
                    tolerance = tau * hinge_scale_cache[0]
                    dist_excess = torch.nn.functional.relu(rend_dist - tolerance)
                    dist_loss = lambda_dist * dist_excess.mean()
                else:
                    dist_loss = lambda_dist * (rend_dist).mean()
            else:
                dist_loss = lambda_dist * (rend_dist).mean()

        # Consensus normal loss: push rend_normal toward multi-view consensus at mid-consistency regions
        consensus_loss = torch.tensor(0.0, device="cuda")
        if opt.consistency_loss and consensus_normals is not None and iteration >= opt.consistency_start_iter:
            with torch.no_grad():
                # Render consensus normal map (world space, fixed target)
                cons_normal_pkg = render(viewpoint_cam, gaussians, pipe, background,
                                         override_color=consensus_normals)
                consensus_normal_map = cons_normal_pkg["render"]  # [3, H, W]
                consensus_normal_map = F.normalize(consensus_normal_map, dim=0, eps=1e-6)

                # Render consistency weight map
                cons_w_color = normal_consistency.unsqueeze(1).expand(-1, 3)  # [N, 3]
                cons_w_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=cons_w_color)
                consistency_map = cons_w_pkg["render"][0:1]  # [1, H, W]

                # Mid-range mask: geometry edges where consensus is meaningful
                # Low (<clamp_low): noise/specular → skip
                # High (>loss_high): already good → skip
                # Mid: push toward consensus
                mid_mask = ((consistency_map > opt.consistency_clamp_low) &
                            (consistency_map < opt.consistency_loss_high)).float()

            # rend_normal has gradients → loss will adjust surfel orientations
            cons_agreement = (rend_normal * consensus_normal_map).sum(dim=0, keepdim=True)  # [1, H, W]
            consensus_loss = opt.lambda_consistency * (mid_mask * (1 - cons_agreement)).mean()

        # SfM density-aware scale regularization
        sfm_scale_loss = torch.tensor(0.0, device="cuda")
        if opt.sfm_scale_reg and iteration >= opt.sfm_scale_reg_start and gaussians.sfm_nn_distance.numel() > 0:
            scales = gaussians.get_scaling.clamp(min=1e-7)  # [N, 2]
            geo_mean_scale = torch.sqrt(scales[:, 0] * scales[:, 1])  # [N]
            nn_dist = gaussians.sfm_nn_distance.clamp(min=1e-5)  # [N]
            # Penalize scale growing beyond c * nn_distance
            ratio = (geo_mean_scale / nn_dist).clamp(max=100.0)
            excess = (ratio - opt.sfm_scale_reg_c).clamp(min=0, max=10.0)
            sfm_scale_loss = opt.lambda_sfm_scale * (excess ** 2).mean()

        # Cross-view depth consistency loss
        cvd_loss = torch.tensor(0.0, device="cuda")
        if opt.cross_view_depth and cvd_depth_var is not None and iteration >= opt.cvd_start_iter:
            # Render per-Gaussian depth_var * loop_weight as a pixel map
            cvd_val = (cvd_depth_var * cvd_loop_weight).clamp(0, 10)  # [N]
            cvd_color = cvd_val.unsqueeze(1).expand(-1, 3)  # [N, 3]
            cvd_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=cvd_color)
            cvd_map = cvd_pkg["render"][0:1]  # [1, H, W]
            cvd_loss = opt.lambda_cvd * cvd_map.mean()
            if iteration % 1000 == 0:
                print(f"[CVD Loss@{iteration}] cvd_loss={cvd_loss.item():.6f}", flush=True)

        # Geometric prior loss (KDTree-based, no OOM)
        geo_loss = torch.tensor(0.0, device="cuda")
        if geo_prior_pts is not None and geo_prior_tree is not None and iteration > args.geo_start_iter:
            lambda_geo = args.lambda_geo
            if args.geo_curriculum:
                progress = (iteration - args.geo_start_iter) / (opt.iterations - args.geo_start_iter)
                lambda_geo = args.lambda_geo * max(0.1, 1.0 - 0.8 * progress)
            
            xyz = gaussians.get_xyz  # [N, 3]
            with torch.no_grad():
                if xyz.shape[0] > 10000:
                    sample_idx = torch.randperm(xyz.shape[0], device=xyz.device)[:10000]
                else:
                    sample_idx = torch.arange(xyz.shape[0], device=xyz.device)
                
                # KDTree query on CPU — O(N log M), no memory explosion
                xyz_np = xyz[sample_idx].detach().cpu().numpy()
                min_dists_np, min_idx_np = geo_prior_tree.query(xyz_np)
                min_dists = torch.tensor(min_dists_np, dtype=torch.float32, device=xyz.device)
                min_idx = torch.tensor(min_idx_np, dtype=torch.long, device=xyz.device)
                
                median_dist = min_dists.median()
                valid_mask = min_dists < median_dist * 3
                
                # Confidence weighting
                nearest_conf = geo_prior_conf[min_idx[valid_mask]]
            
            if valid_mask.sum() > 0:
                nearest_prior = geo_prior_pts[min_idx[valid_mask]].detach()
                per_point_loss = ((xyz[sample_idx[valid_mask]] - nearest_prior) ** 2).sum(dim=1)
                geo_loss = lambda_geo * (nearest_conf * per_point_loss).mean()

        _conf_reg = None
        if opt.learnable_confidence and iteration >= opt.conf_warmup_iter:
            if _conf_xyz_logit is None:
                N_curr = gaussians._xyz.shape[0]
                _conf_xyz_logit = torch.nn.Parameter(torch.full((N_curr, 1), opt.conf_init_logit, device='cuda'))
                _conf_rot_logit = torch.nn.Parameter(torch.full((N_curr, 1), opt.conf_init_logit, device='cuda'))
                _conf_optimizer = torch.optim.Adam([_conf_xyz_logit, _conf_rot_logit], lr=opt.conf_lr)
                print(f'[LearnConf] init at iter {iteration}: N={N_curr} logit={opt.conf_init_logit}', flush=True)
            _conf_xyz_val = opt.conf_floor + (1.0 - opt.conf_floor) * torch.sigmoid(_conf_xyz_logit)
            _conf_rot_val = opt.conf_floor + (1.0 - opt.conf_floor) * torch.sigmoid(_conf_rot_logit)
            _conf_color = torch.cat([_conf_xyz_val, _conf_rot_val, _conf_xyz_val], dim=1)
            _conf_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=_conf_color)
            _conf_map = _conf_pkg["render"]
            _conf_xyz_map = _conf_map[0:1]
            _conf_rot_map = _conf_map[1:2]
            dist_loss = lambda_dist * (_conf_xyz_map * rend_dist).mean()
            normal_loss = lambda_normal * (_conf_rot_map * normal_error).mean()
            _conf_reg = ((1.0 - _conf_xyz_val) ** 2).mean() + ((1.0 - _conf_rot_val) ** 2).mean()
        # loss
        total_loss = loss + dist_loss + normal_loss + consensus_loss + sfm_scale_loss + cvd_loss + geo_loss + cancel_loss
        if _conf_reg is not None:
            total_loss = total_loss + opt.lambda_conf_reg * _conf_reg
        if _p1_log_dir is not None:
            phase1_log.maybe_snapshot(_p1_log_dir, iteration, _p1_snap_iters, gaussians, loss, dist_loss, normal_loss)

        if opt.log_trajectory and iteration % opt.log_trajectory_every == 0:
            traj_log.step(iteration, gaussians, loss, dist_loss, normal_loss, scene.model_path)
        if opt.log_per_gauss_v2 and iteration % opt.log_trajectory_every == 0:
            traj_log_v2.step(iteration, gaussians, loss, dist_loss, normal_loss, scene.model_path)
        if opt.log_cancellation and iteration % opt.log_trajectory_every == 0:
            traj_log_cancel.step(iteration, gaussians, loss, image, viewpoint_cam, scene.model_path)
        if opt.log_cancel_v3 and iteration % opt.log_trajectory_every == 0:
            traj_log_cancel_v3.step(iteration, gaussians, loss, viewpoint_cam, scene.model_path)
        if opt.log_cancel_v4 and iteration % opt.log_trajectory_every == 0:
            traj_log_cancel_v4.step(iteration, gaussians, loss, viewpoint_cam, scene.model_path)
        if opt.log_cancel_v6 and iteration % opt.log_trajectory_every == 0:
            traj_log_cancel_v6.step(iteration, gaussians, loss, viewpoint_cam, scene.model_path)
        if opt.log_cancel_v9 and iteration % opt.log_trajectory_every == 0:
            traj_log_cancel_v9.step(iteration, gaussians, loss, viewpoint_cam, scene.model_path)
        if iteration == opt.prune_cancel_iter and opt.prune_cancel_iter > 0:
            cancel_prune.prune_at(iteration, gaussians, loss, scene.model_path)
        if opt.log_cancel_perloss and iteration % opt.log_trajectory_every == 0:
            traj_log_cancel_perloss.step(iteration, gaussians, _Lrgb_part, _Lssim_part, dist_loss if ("dist_loss" in dir() and dist_loss is not None) else None, normal_loss if ("normal_loss" in dir() and normal_loss is not None) else None, viewpoint_cam, scene.model_path)
        if opt.log_perview and traj_log_perview.is_anchor(iteration):
            traj_log_perview.snapshot(iteration, gaussians)
        if opt.log_cancel_v8 and iteration % opt.log_trajectory_every == 0:
            traj_log_cancel_v8.step(iteration, gaussians, loss, viewpoint_cam, scene.model_path)
        if opt.log_cancel_v5 and iteration % opt.log_trajectory_every == 0:
            traj_log_cancel_v5.step(iteration, gaussians, loss, dist_loss, normal_loss, total_loss, viewpoint_cam, scene.model_path)

        if opt.v12_ratio_detect and (dist_loss.item() > 0 or normal_loss.item() > 0):
            cur_n = gaussians._xyz.shape[0]
            rot_active = iteration >= opt.v12_rot_start_iter
            mask_active = iteration >= opt.v12_warmup_iter
            update_due = mask_active and (
                v12_cache["mask"] is None or
                v12_cache["n"] != cur_n or
                (iteration - v12_cache["last_update"]) >= opt.v12_update_every
            )
            if update_due:
                if v12_cache["cams_sample"] is None:
                    import random as _rnd
                    _rnd.seed(42)
                    _all = scene.getTrainCameras()
                    _k = min(opt.v12_n_views, len(_all))
                    v12_cache["cams_sample"] = _rnd.sample(list(_all), _k)
                    from utils.v12_utils import compute_high_freq_mask as _chfm
                    v12_cache["hi_masks"] = [_chfm(c.original_image.cuda(), opt.v12_freq_pct) for c in v12_cache["cams_sample"]]
                from utils.v12_utils import compute_ratio_score
                ratio, det_any = compute_ratio_score(gaussians, v12_cache["cams_sample"], render, pipe, background, v12_cache["hi_masks"])
                valid = det_any > torch.quantile(det_any, 0.1)
                new_detail = (ratio >= opt.v12_ratio_thr) & valid
                if v12_cache["hist"] is None or v12_cache["hist"].shape[0] != cur_n:
                    v12_cache["hist"] = torch.zeros(cur_n, dtype=torch.long, device=gaussians._xyz.device)
                v12_cache["hist"] = torch.where(new_detail,
                    torch.full_like(v12_cache["hist"], opt.v12_hysteresis),
                    (v12_cache["hist"] - 1).clamp(min=0))
                v12_cache["mask"] = v12_cache["hist"] > 0
                v12_cache["n"] = cur_n
                v12_cache["last_update"] = iteration
                if iteration % 5000 == 0 or iteration == opt.v12_warmup_iter:
                    n_det = int(v12_cache["mask"].sum().item())
                    print(f"\n[V12@{iteration}] detail={n_det}/{cur_n} ({100*n_det/max(cur_n,1):.2f}%) thr={opt.v12_ratio_thr}", flush=True)
            if mask_active and v12_cache["mask"] is not None and v12_cache["n"] == cur_n:
                detail_f = v12_cache["mask"].float().unsqueeze(-1)
            else:
                detail_f = torch.zeros(cur_n, 1, device=gaussians._xyz.device)
            total_loss.backward(retain_graph=True)
            params = [gaussians._xyz, gaussians._rotation, gaussians._scaling, gaussians._opacity]
            dg = torch.autograd.grad(dist_loss, params, retain_graph=True, allow_unused=True) if dist_loss.item() > 0 else [None]*4
            ng = torch.autograd.grad(normal_loss, params, retain_graph=False, allow_unused=True) if normal_loss.item() > 0 else [None]*4
            def _z(g, p): return torch.zeros_like(p) if g is None else g
            if rot_active:
                params[1].grad = params[1].grad - _z(dg[1], params[1])
            if opt.v12_light_mode:
                params[0].grad = params[0].grad - detail_f * _z(dg[0], params[0])
                params[1].grad = params[1].grad - detail_f * _z(ng[1], params[1])
            else:
                params[0].grad = params[0].grad - detail_f * (_z(dg[0], params[0]) + _z(ng[0], params[0]))
                params[1].grad = params[1].grad - detail_f * _z(ng[1], params[1])
                params[2].grad = params[2].grad - detail_f * (_z(dg[2], params[2]) + _z(ng[2], params[2]))
                params[3].grad = params[3].grad - detail_f * (_z(dg[3], params[3]) + _z(ng[3], params[3]))
        elif (opt.rot_detach_dist or opt.opacity_detach_dist or opt.xyz_detach_dist or opt.scaling_detach_dist or opt.exp_b_risk_rot or opt.scale_detach_channel or opt.aniso_rot_detach) and dist_loss.item() > 0:

            # 2-pass backward: isolate dist contribution to specific parameters
            # Pass 1: everything except dist
            non_dist_loss = loss + normal_loss + consensus_loss + sfm_scale_loss + cvd_loss + geo_loss
            non_dist_loss.backward(retain_graph=True)
            # Save grads before dist contribution
            grad_saves = {}
            if opt.rot_detach_dist or opt.exp_b_risk_rot or opt.scale_detach_channel or opt.aniso_rot_detach:
                grad_saves["rotation"] = gaussians._rotation.grad.clone()
            if opt.scale_detach_channel:
                grad_saves["xyz"] = gaussians._xyz.grad.clone()
                grad_saves["scaling"] = gaussians._scaling.grad.clone()
                grad_saves["opacity"] = gaussians._opacity.grad.clone()
            if opt.opacity_detach_dist:
                grad_saves["opacity"] = gaussians._opacity.grad.clone()
            if opt.xyz_detach_dist:
                grad_saves["xyz"] = gaussians._xyz.grad.clone()
            if opt.scaling_detach_dist:
                grad_saves["scaling"] = gaussians._scaling.grad.clone()
            # Pass 2: dist_loss (accumulates dist contribution onto saved grads)
            dist_loss.backward()
            # Restore/attenuate grads
            if opt.scale_detach_channel:
                # Scale-proportional attenuation of a specific channel's dist gradient
                with torch.no_grad():
                    _scales = gaussians.get_scaling.clamp(min=1e-7)
                    _geo_mean = torch.sqrt(_scales[:, 0] * _scales[:, 1])
                    _med = _geo_mean.median().clamp(min=1e-7)
                    alpha_perG = (1.0 / (1.0 + _geo_mean / _med)).clamp(0.05, 1.0)
                ch = opt.scale_detach_channel
                param_map = {"xyz": gaussians._xyz, "rotation": gaussians._rotation,
                             "scaling": gaussians._scaling, "opacity": gaussians._opacity}
                for pname, param in param_map.items():
                    if pname == ch:
                        dist_grad = param.grad - grad_saves[pname]
                        expand_alpha = alpha_perG.unsqueeze(1) if param.dim() > 1 else alpha_perG
                        param.grad.copy_(grad_saves[pname] + expand_alpha * dist_grad)
                    # Other channels: keep full dist gradient (don't touch)
            elif opt.exp_b_risk_rot:
                dist_rot_grad = gaussians._rotation.grad - grad_saves["rotation"]
                N = gaussians._xyz.shape[0]
                if exp_b_risk is not None and exp_b_risk.shape[0] == N:
                    risk = exp_b_risk
                else:
                    risk = torch.zeros(N, device=dist_rot_grad.device)
                alpha_perG = (1.0 - opt.exp_b_max_detach * risk).clamp(0.0, 1.0).unsqueeze(1)
                gaussians._rotation.grad.copy_(grad_saves["rotation"] + alpha_perG * dist_rot_grad)
            elif opt.rot_detach_dist and opt.rot_detach_from_iter <= iteration <= opt.rot_detach_until_iter:
                alpha = opt.rot_detach_alpha
                dist_rot_grad = gaussians._rotation.grad - grad_saves["rotation"]
                gaussians._rotation.grad.copy_(grad_saves["rotation"] + alpha * dist_rot_grad)
            elif opt.aniso_rot_detach and opt.aniso_rot_detach_from_iter <= iteration <= opt.aniso_rot_detach_until_iter:
                with torch.no_grad():
                    _sc = gaussians.get_scaling.clamp(min=1e-7)
                    _aniso = _sc.max(dim=1).values / _sc.min(dim=1).values
                    _alpha = 1.0 / (1.0 + (_aniso / opt.aniso_rot_detach_scale) ** opt.aniso_rot_detach_power)
                    _alpha = _alpha.clamp(0.0, 1.0).unsqueeze(1)
                dist_rot_grad = gaussians._rotation.grad - grad_saves["rotation"]
                gaussians._rotation.grad.copy_(grad_saves["rotation"] + _alpha * dist_rot_grad)
            if opt.opacity_detach_dist and not opt.scale_detach_channel:
                gaussians._opacity.grad.copy_(grad_saves["opacity"])
            if opt.xyz_detach_dist and not opt.scale_detach_channel:
                gaussians._xyz.grad.copy_(grad_saves["xyz"])
            if opt.scaling_detach_dist and not opt.scale_detach_channel:
                gaussians._scaling.grad.copy_(grad_saves["scaling"])
        elif opt.v9_pathway_selective and (dist_loss.item() > 0 or normal_loss.item() > 0):
            total_loss.backward(retain_graph=True)
            if dist_loss.item() > 0:
                corr_xyz = torch.autograd.grad(v9_corr_dist, gaussians._xyz, retain_graph=True)[0]
                gaussians._xyz.grad -= corr_xyz
            if normal_loss.item() > 0:
                corr_rot = torch.autograd.grad(v9_corr_normal, gaussians._rotation)[0]
                gaussians._rotation.grad -= corr_rot
        elif opt.v10_context and (dist_loss.item() > 0 or normal_loss.item() > 0):
            cur_n = gaussians._xyz.shape[0]
            need = v10_cache["mask"] is None or v10_cache["n"] != cur_n or (iteration % opt.v10_update_every == 0)
            if need:
                if iteration >= opt.v10_warmup_iter:
                    m, _, _ = compute_pathological_mask(gaussians, opt.v10_k, opt.v10_aniso_thr, opt.v10_reg_thr)
                else:
                    m = torch.zeros(cur_n, dtype=torch.bool, device=gaussians._xyz.device)
                v10_cache["mask"] = m; v10_cache["n"] = cur_n
            mask_f = v10_cache["mask"].float().unsqueeze(1)
            total_loss.backward(retain_graph=True)
            if dist_loss.item() > 0:
                dxyz = torch.autograd.grad(dist_loss, gaussians._xyz, retain_graph=True)[0]
                gaussians._xyz.grad -= opt.v10_alpha * mask_f * dxyz
            if normal_loss.item() > 0:
                nrot = torch.autograd.grad(normal_loss, gaussians._rotation)[0]
                gaussians._rotation.grad -= opt.v10_alpha * mask_f * nrot
        elif opt.spa_v1 and (dist_loss.item() > 0 or normal_loss.item() > 0):
            # small Gaussian -> w=0.1 -> normal->rot & dist->xyz attenuated; large -> w=1.0 -> full flow
            with torch.no_grad():
                _sc = gaussians.get_scaling.clamp(min=1e-7)
                _geo = torch.sqrt(_sc[:, 0] * _sc[:, 1])
                _w = (_geo / _geo.median().clamp(min=1e-7)).clamp(0.1, 1.0)
            total_loss.backward(retain_graph=True)
            if normal_loss.item() > 0:
                n_rot = torch.autograd.grad(normal_loss, gaussians._rotation, retain_graph=True)[0]
                gaussians._rotation.grad -= (1.0 - _w).unsqueeze(1) * n_rot
            if dist_loss.item() > 0:
                d_xyz = torch.autograd.grad(dist_loss, gaussians._xyz)[0]
                gaussians._xyz.grad -= (1.0 - _w).unsqueeze(1) * d_xyz
        elif opt.hybrid_gate and (dist_loss.item() > 0 or normal_loss.item() > 0):
            total_loss.backward(retain_graph=True)
            params = [gaussians._xyz, gaussians._rotation, gaussians._scaling, gaussians._opacity]
            dg = torch.autograd.grad(dist_loss, params, retain_graph=True, allow_unused=True) if dist_loss.item() > 0 else [None]*4
            ng = torch.autograd.grad(normal_loss, params, retain_graph=False, allow_unused=True) if normal_loss.item() > 0 else [None]*4
            def _z(g, p): return torch.zeros_like(p) if g is None else g
            _beta = opt.hybrid_gate_beta
            params[0].grad = params[0].grad - _beta * _z(ng[0], params[0])
            params[1].grad = params[1].grad - _beta * _z(dg[1], params[1])
            params[2].grad = params[2].grad - _beta * (_z(dg[2], params[2]) + _z(ng[2], params[2]))
            params[3].grad = params[3].grad - _beta * (_z(dg[3], params[3]) + _z(ng[3], params[3]))
        elif opt.loss_purify and (dist_loss.item() > 0 or normal_loss.item() > 0):
            total_loss.backward(retain_graph=True)
            params = [gaussians._xyz, gaussians._rotation, gaussians._scaling, gaussians._opacity]
            dg = torch.autograd.grad(dist_loss, params, retain_graph=True, allow_unused=True) if dist_loss.item() > 0 else [None]*4
            ng = torch.autograd.grad(normal_loss, params, retain_graph=False, allow_unused=True) if normal_loss.item() > 0 else [None]*4
            def _z(g, p): return torch.zeros_like(p) if g is None else g
            params[0].grad = params[0].grad - _z(ng[0], params[0])
            params[1].grad = params[1].grad - _z(dg[1], params[1])
            params[2].grad = params[2].grad - _z(dg[2], params[2]) - _z(ng[2], params[2])
            params[3].grad = params[3].grad - _z(dg[3], params[3]) - _z(ng[3], params[3])
        elif opt.combo_ndxyz_rd and (normal_loss.item() > 0 or dist_loss.item() > 0):
            subs = []
            if normal_loss.item() > 0:
                g = torch.autograd.grad(normal_loss, gaussians._xyz, retain_graph=True, allow_unused=True)[0]
                if g is not None: subs.append((gaussians._xyz, g))
            if dist_loss.item() > 0:
                g = torch.autograd.grad(dist_loss, gaussians._rotation, retain_graph=True, allow_unused=True)[0]
                if g is not None: subs.append((gaussians._rotation, g))
            total_loss.backward()
            for param, g in subs:
                if param.grad is not None: param.grad -= g
        elif (opt.normal_detach_xyz or opt.normal_detach_scaling or opt.normal_detach_opacity) and normal_loss.item() > 0:
            targets = []
            if opt.normal_detach_xyz: targets.append(gaussians._xyz)
            if opt.normal_detach_scaling: targets.append(gaussians._scaling)
            if opt.normal_detach_opacity: targets.append(gaussians._opacity)
            pre = [torch.autograd.grad(normal_loss, t, retain_graph=True, allow_unused=True)[0] for t in targets]
            total_loss.backward()
            for t, pg in zip(targets, pre):
                if pg is not None and t.grad is not None:
                    t.grad -= pg
        elif opt.photo_detach_rotation and loss.item() > 0:
            pg_rot = torch.autograd.grad(loss, gaussians._rotation, retain_graph=True, allow_unused=True)[0]
            total_loss.backward()
            if pg_rot is not None and gaussians._rotation.grad is not None:
                gaussians._rotation.grad -= pg_rot
        else:
            total_loss.backward()

        if _conf_optimizer is not None:
            _conf_optimizer.step()
            _conf_optimizer.zero_grad(set_to_none=True)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if _p1_log_dir is not None and iteration % 100 == 0:
                phase1_log.append_scalar(_p1_log_dir, iteration, loss.item(), dist_loss.item(), normal_loss.item(), len(gaussians._xyz))
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            if opt.sfm_scale_reg and iteration % 1000 == 0 and sfm_scale_loss.item() > 0:
                print(f"[SfmScaleReg@{iteration}] loss={sfm_scale_loss.item():.6f}", flush=True)
            if opt.consistency_loss:
                ema_cons_for_log = 0.4 * consensus_loss.item() + 0.6 * ema_cons_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                if opt.consistency_loss:
                    loss_dict["cons"] = f"{ema_cons_for_log:.{5}f}"
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                if opt.mid_intervene_mode != "none":
                    import numpy as np
                    fp = os.path.join(scene.model_path, "final_state.npz")
                    np.savez(fp, gauss_id=gaussians.gauss_id.cpu().numpy(), opacity_logit=gaussians._opacity.detach().cpu().numpy().squeeze(), xyz=gaussians.get_xyz.detach().cpu().numpy(), scale=gaussians.get_scaling.detach().cpu().numpy())
                    print(f"saved final_state to {fp}", flush=True)
                if opt.log_per_gauss_v2: traj_log_v2.finalize()
                if opt.log_cancellation: traj_log_cancel.finalize()
                if opt.log_cancel_v3: traj_log_cancel_v3.finalize()
                if opt.log_cancel_v4: traj_log_cancel_v4.finalize()
                if opt.log_cancel_v5: traj_log_cancel_v5.finalize()
                if opt.log_cancel_v6: traj_log_cancel_v6.finalize()
                if opt.log_cancel_v7: traj_log_cancel_v7.finalize()
                if opt.log_cancel_v8: traj_log_cancel_v8.finalize()
                if opt.log_cancel_v9: traj_log_cancel_v9.finalize()
                if opt.log_cancel_v10: traj_log_cancel_v10.finalize()
                if opt.log_cancel_v11: traj_log_cancel_v11.finalize()
                if opt.log_cancel_perloss: traj_log_cancel_perloss.finalize()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if _conf_xyz_logit is not None:
                    import numpy as _np, os as _os
                    _cx = (opt.conf_floor + (1.0 - opt.conf_floor) * torch.sigmoid(_conf_xyz_logit)).detach().cpu().numpy()
                    _cr = (opt.conf_floor + (1.0 - opt.conf_floor) * torch.sigmoid(_conf_rot_logit)).detach().cpu().numpy()
                    _outd = _os.path.join(scene.model_path, 'point_cloud', 'iteration_' + str(iteration))
                    _np.savez(_os.path.join(_outd, 'confidence.npz'), conf_xyz=_cx, conf_rot=_cr)
                    print('[LearnConf] saved conf.npz xyz_med=%.3f rot_med=%.3f' % (float(_np.median(_cx)), float(_np.median(_cr))), flush=True)


            # ---- Sparse region tracking snapshot ----
            if sparse_tracker is not None and iteration % 1000 == 0:
                idx = sparse_tracker['indices']
                # Filter: some may have been pruned (index out of range)
                N_cur = len(gaussians.get_xyz)
                valid = idx < N_cur
                valid_idx = idx[valid]
                if len(valid_idx) > 0:
                    cur_xyz = gaussians.get_xyz[valid_idx].detach()
                    cur_opacity = gaussians.get_opacity[valid_idx].detach().squeeze()
                    cur_scales = gaussians.get_scaling[valid_idx].detach()
                    cur_geo_mean = torch.sqrt(cur_scales[:, 0] * cur_scales[:, 1])
                    init_pos = sparse_tracker['init_positions'][valid]
                    pos_delta = (cur_xyz - init_pos).norm(dim=1)
                    snapshot = {
                        'iter': iteration,
                        'n_tracked': int(len(valid_idx)),
                        'n_pruned': int((~valid).sum().item()),
                        'opacity_mean': float(cur_opacity.mean()),
                        'opacity_median': float(cur_opacity.median()),
                        'opacity_low_pct': float((cur_opacity < 0.1).float().mean() * 100),
                        'scale_mean': float(cur_geo_mean.mean()),
                        'scale_median': float(cur_geo_mean.median()),
                        'pos_delta_mean': float(pos_delta.mean()),
                        'pos_delta_median': float(pos_delta.median()),
                    }
                    sparse_tracker['snapshots'].append(snapshot)
                    if iteration % 5000 == 0:
                        print(f"\n[Track@{iteration}] sparse: n={snapshot['n_tracked']}, "
                              f"opacity={snapshot['opacity_mean']:.3f}, "
                              f"scale={snapshot['scale_mean']:.4f}, "
                              f"pos_delta={snapshot['pos_delta_mean']:.4f}")

            # Loop-preventive densification: detect and split rapidly growing Gaussians
            if (opt.loop_densify and iteration >= opt.loop_densify_start
                    and iteration < opt.densify_until_iter
                    and iteration % opt.loop_densify_interval == 0
                    and gaussians.initial_scale.numel() > 0):
                with torch.no_grad():
                    current_scale = gaussians.get_scaling  # [N, 2]
                    initial_scale = gaussians.initial_scale  # [N, 2]
                    N_g = min(current_scale.shape[0], initial_scale.shape[0])
                    
                    geo_current = torch.sqrt(current_scale[:N_g, 0] * current_scale[:N_g, 1]).clamp(min=1e-8)
                    geo_initial = torch.sqrt(initial_scale[:N_g, 0] * initial_scale[:N_g, 1]).clamp(min=1e-8)
                    
                    growth_rate = geo_current / geo_initial
                    loop_mask = growth_rate > opt.loop_densify_threshold
                    
                    # Cap the number of splits
                    if loop_mask.sum() > opt.loop_densify_max:
                        _, topk_idx = growth_rate.topk(opt.loop_densify_max)
                        new_mask = torch.zeros_like(loop_mask)
                        new_mask[topk_idx] = True
                        loop_mask = new_mask
                    
                    n_loop = loop_mask.sum().item()
                    if n_loop > 0:
                        # Pad mask to full size if needed
                        if loop_mask.shape[0] < current_scale.shape[0]:
                            loop_mask = torch.cat([loop_mask, 
                                torch.zeros(current_scale.shape[0] - loop_mask.shape[0], 
                                           dtype=torch.bool, device="cuda")])
                        
                        # Force split via densify_and_split
                        dummy_grads = torch.zeros(current_scale.shape[0], device="cuda")
                        gaussians.densify_and_split(dummy_grads, 1e10, scene.cameras_extent, 
                                                     N=2, force_split_mask=loop_mask)
                        print(f"[LoopDensify@{iteration}] split {n_loop} Gaussians "
                              f"(growth>{opt.loop_densify_threshold:.1f}, "
                              f"max_growth={growth_rate.max():.2f}, "
                              f"total={len(gaussians.get_xyz)})", flush=True)
                        # After split, Gaussian count changed — skip densification stats this iter
                        # to avoid shape mismatch with visibility_filter/radii from pre-split render
                        continue

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # Phase 1: AbsGS uses CUDA mean2D buffer (per-pixel L2 norm sum) for accum
                if opt.densify_method == 'AbsGS':
                    import diff_surfel_rasterization as _dsr_abs
                    _bab = _dsr_abs.get_cancel_buffers()
                    _abs_g = _bab.get('mean2D', None)
                    if _abs_g is not None:
                        gaussians.xyz_gradient_accum[visibility_filter] += _abs_g[visibility_filter].unsqueeze(-1)
                        gaussians.denom[visibility_filter] += 1
                    else:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # Candidate 5 / Phase 1 cancel methods: accumulate cm_op
                if opt.cancel_composite_densify or opt.densify_method in ('OR_cancel', 'AND_cancel'):
                    import diff_surfel_rasterization as _dsr
                    _b = _dsr.get_cancel_buffers()
                    _sm = _b.get('opacity', None)
                    if _sm is not None and gaussians._opacity.grad is not None:
                        _alpha = torch.sigmoid(gaussians._opacity.squeeze())
                        _sigp = _alpha * (1.0 - _alpha)
                        _sg = (gaussians._opacity.grad.squeeze() / (_sigp + 1e-12)).abs()
                        _cm = 1.0 - _sg / (_sm + 1e-12)
                        gaussians.accumulate_cm(_cm)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # Consistency-guided densification: pass consistency to filter
                    cons_mask = None
                    if opt.consistency_densify and normal_consistency is not None and iteration >= opt.consistency_densify_start:
                        cons_mask = normal_consistency >= opt.consistency_densify_threshold
                    # Render-conf densification: allow densify where rendering is unstable
                    if render_conf is not None and opt.split_signal_densify_threshold > 0:
                        n_rc = min(render_conf.shape[0], cons_mask.shape[0])
                        render_unstable = render_conf[:n_rc] < opt.split_signal_densify_threshold
                        cons_mask[:n_rc] = cons_mask[:n_rc] | render_unstable
                        if iteration % 1000 == 0:
                            print(f"[SplitDensify@{iteration}] render_unstable={render_unstable.sum()}/{n_rc}", flush=True)
                    # Confidence-gated densification: pass confidence scores
                    conf_scores = None
                    if opt.confidence_gated_densify and normal_consistency is not None:
                        conf_scores = normal_consistency
                    # Scale-confidence split: use multi-signal confidence if available
                    sc_conf = None
                    if opt.scale_confidence_split and normal_consistency is not None:
                        sc_conf = normal_consistency
                    if opt.log_cancel_v7:
                        traj_log_cancel_v7.log_at_densify(iteration, gaussians, opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, scene.model_path)
                    if opt.log_cancel_v10:
                        traj_log_cancel_v10.log_at_densify(iteration, gaussians, opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, scene.model_path)
                    if opt.log_cancel_v11:
                        traj_log_cancel_v11.log_at_densify(iteration, gaussians)
                    _cancel_factor = None
                    _cancel_thresh = opt.densify_grad_threshold
                    if opt.cancel_composite_densify and gaussians.cm_accum is not None and gaussians.cm_count > 0:
                        _cancel_factor = (gaussians.cm_accum / gaussians.cm_count).detach()
                        _cancel_thresh = opt.cancel_composite_grad_threshold
                    # Phase 1 cancel methods: pass cm_avg even without composite mode
                    if opt.densify_method in ('OR_cancel', 'AND_cancel') and gaussians.cm_accum is not None and gaussians.cm_count > 0:
                        _cancel_factor = (gaussians.cm_accum / gaussians.cm_count).detach()
                    # Cancel-as-split-direction (Phase 2)
                    _split_dirs = None
                    _split_bias_dirs = None
                    _split_argmin_offsets = None
                    if opt.split_method != 'V':
                        with torch.no_grad():
                            P_g = gaussians._xyz.shape[0]
                            grads_pre = (gaussians.xyz_gradient_accum / gaussians.denom).squeeze()
                            grads_pre[grads_pre != grads_pre] = 0
                            candidate_mask = grads_pre > opt.densify_grad_threshold
                            if candidate_mask.sum() > 0:
                                residual_map = (image - gt_image).sum(dim=0)
                                if opt.split_method == 'cancel':
                                    _split_dirs = compute_cancel_directions(
                                        gaussians.get_xyz, render_pkg['radii'],
                                        viewpoint_cam.full_proj_transform, viewpoint_cam.world_view_transform,
                                        __import__("math").tan(viewpoint_cam.FoVx*0.5), __import__("math").tan(viewpoint_cam.FoVy*0.5),
                                        residual_map, candidate_mask=candidate_mask)
                                elif opt.split_method == 'random_dir':
                                    _split_dirs = sample_random_unit_directions(P_g, candidate_mask, image.device, seed=opt.method_random_seed + iteration)
                                elif opt.split_method == 'orthogonal_dir':
                                    _c_dirs = compute_cancel_directions(
                                        gaussians.get_xyz, render_pkg['radii'],
                                        viewpoint_cam.full_proj_transform, viewpoint_cam.world_view_transform,
                                        __import__("math").tan(viewpoint_cam.FoVx*0.5), __import__("math").tan(viewpoint_cam.FoVy*0.5),
                                        residual_map, candidate_mask=candidate_mask)
                                    _split_dirs = orthogonalize_directions(_c_dirs, candidate_mask, seed=opt.method_random_seed + iteration)
                                elif opt.split_method == 'cancel_in_plane':
                                    _c_dirs = compute_cancel_directions(
                                        gaussians.get_xyz, render_pkg['radii'],
                                        viewpoint_cam.full_proj_transform, viewpoint_cam.world_view_transform,
                                        __import__("math").tan(viewpoint_cam.FoVx*0.5), __import__("math").tan(viewpoint_cam.FoVy*0.5),
                                        residual_map, candidate_mask=candidate_mask)
                                    _split_dirs = project_to_tangent_plane(_c_dirs, gaussians._rotation)
                                elif opt.split_method == 'cancel_bias_vanilla':
                                    _c_dirs = compute_cancel_directions(
                                        gaussians.get_xyz, render_pkg['radii'],
                                        viewpoint_cam.full_proj_transform, viewpoint_cam.world_view_transform,
                                        __import__("math").tan(viewpoint_cam.FoVx*0.5), __import__("math").tan(viewpoint_cam.FoVy*0.5),
                                        residual_map, candidate_mask=candidate_mask)
                                    _split_bias_dirs = project_to_tangent_plane(_c_dirs, gaussians._rotation)
                                    _split_dirs = None  # don't trigger mirror branch
                                elif opt.split_method == 'cancel_argmin':
                                    _raw = compute_cancel_argmin_offsets(
                                        gaussians.get_xyz, render_pkg['radii'],
                                        viewpoint_cam.full_proj_transform, viewpoint_cam.world_view_transform,
                                        __import__("math").tan(viewpoint_cam.FoVx*0.5), __import__("math").tan(viewpoint_cam.FoVy*0.5),
                                        residual_map, candidate_mask=candidate_mask)
                                    _tang = project_to_tangent_plane_keep_magnitude(_raw, gaussians._rotation)
                                    _split_argmin_offsets = clamp_offset_to_parent_footprint(
                                        _tang, gaussians.get_scaling, clamp_factor=opt.argmin_clamp_factor)
                                    _split_dirs = None
                                elif opt.split_method == 'cancel_argmin_dir_random':
                                    _raw = compute_cancel_argmin_offsets(
                                        gaussians.get_xyz, render_pkg['radii'],
                                        viewpoint_cam.full_proj_transform, viewpoint_cam.world_view_transform,
                                        __import__("math").tan(viewpoint_cam.FoVx*0.5), __import__("math").tan(viewpoint_cam.FoVy*0.5),
                                        residual_map, candidate_mask=candidate_mask)
                                    _tang = project_to_tangent_plane_keep_magnitude(_raw, gaussians._rotation)
                                    _argmin_clamped = clamp_offset_to_parent_footprint(
                                        _tang, gaussians.get_scaling, clamp_factor=opt.argmin_clamp_factor)
                                    _mag = _argmin_clamped.norm(dim=-1, keepdim=True)
                                    _rand_dir = sample_tangent_random_directions(
                                        P_g, candidate_mask, gaussians._rotation, image.device,
                                        seed=opt.method_random_seed + iteration)
                                    _split_argmin_offsets = _rand_dir * _mag
                                    _split_dirs = None
                    gaussians.densify_and_prune(_cancel_thresh, opt.opacity_cull, scene.cameras_extent, size_threshold, consistency_mask=cons_mask, scale_normalize=opt.scale_normalize, confidence_gated_densify=opt.confidence_gated_densify, confidence_scores=sc_conf if (opt.scale_confidence_split and sc_conf is not None) else conf_scores, scale_confidence_split=opt.scale_confidence_split, sc_scale_percentile=opt.sc_scale_percentile, sc_conf_threshold=opt.sc_conf_threshold, sc_max_count=opt.sc_max_count, cancel_composite_factor=_cancel_factor, densify_method=opt.densify_method, cancel_rank_threshold=opt.cancel_rank_threshold, method_random_seed=opt.method_random_seed, split_directions=_split_dirs, split_bias_dirs=_split_bias_dirs, bias_lambda=0.5, split_argmin_offsets=_split_argmin_offsets)
                    if opt.cancel_composite_densify or opt.densify_method in ('OR_cancel', 'AND_cancel'):
                        gaussians.reset_cm_accum()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                if opt.mid_intervene_mode != "none" and iteration >= opt.mid_intervene_start and iteration <= opt.mid_intervene_end and (iteration - opt.mid_intervene_start) % opt.mid_intervene_step == 0:
                    import importlib
                    mim_name = getattr(opt,'mid_intervene_module','mid_intervene_render') or 'mid_intervene_render'
                    mim = importlib.import_module(mim_name)
                    with torch.enable_grad():
                        mim.apply(iteration, gaussians, scene, pipe, background, opt.mid_intervene_mode, log_path=os.path.join(scene.model_path, "mid_intervene_log.json"))


            # F: cancel amplification (per-gauss grad scaling)
            if opt.cancel_amplify_gamma > 0 and iteration >= opt.cancel_amplify_start:
                if iteration % opt.cancel_amplify_interval == 0:
                    from cancel_amplify_helper import measure_cancel_mask
                    _saved = []
                    for _p in [gaussians._xyz, gaussians._rotation, gaussians._scaling,
                               gaussians._opacity, gaussians._features_dc, gaussians._features_rest]:
                        _saved.append(_p.grad.clone() if _p.grad is not None else None)
                    with torch.enable_grad():
                        cancel_amp_mask = measure_cancel_mask(
                            gaussians, scene, pipe, background,
                            cm_thr=opt.cancel_amplify_cm_thr,
                            g_abs_mean_min=opt.cancel_amplify_g_abs_mean_min,
                            nr_min=opt.cancel_amplify_nr_min)
                    for _p, _gv in zip([gaussians._xyz, gaussians._rotation, gaussians._scaling,
                                        gaussians._opacity, gaussians._features_dc, gaussians._features_rest], _saved):
                        if _gv is None:
                            _p.grad = None
                        else:
                            if _p.grad is None: _p.grad = _gv
                            else: _p.grad.copy_(_gv)
                    print(f'[CancelAmp@{iteration}] measured cancel mask: {int(cancel_amp_mask.sum().item())}/{cancel_amp_mask.shape[0]} gauss flagged', flush=True)
                if cancel_amp_mask is not None and cancel_amp_mask.shape[0] == gaussians._xyz.shape[0]:
                    from cancel_amplify_helper import amplify_grads_inplace
                    amplify_grads_inplace(gaussians, cancel_amp_mask, opt.cancel_amplify_gamma)
                elif cancel_amp_mask is not None:
                    # gauss count changed (densify), invalidate mask
                    cancel_amp_mask = None

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

    # ---- Save sparse tracking results ----
    if sparse_tracker is not None and sparse_tracker['snapshots']:
        import json as _json
        track_path = os.path.join(scene.model_path, 'sparse_tracking.json')
        with open(track_path, 'w') as f:
            _json.dump(sparse_tracker['snapshots'], f, indent=2)
        print(f"\n[Track] Saved {len(sparse_tracker['snapshots'])} snapshots to {track_path}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--geo_prior", type=str, default=None, help="Path to geometric prior .npz")
    parser.add_argument("--lambda_geo", type=float, default=0.1, help="Geometric loss weight")
    parser.add_argument("--geo_start_iter", type=int, default=3000, help="Start geometric loss after this iter")
    parser.add_argument("--geo_curriculum", action="store_true", help="Decay geo weight over time")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
