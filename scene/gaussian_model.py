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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        # Cancel-composite densification (Candidate 5) buffer
        self.cm_accum = None
        self.cm_count = 0
        self.denom = torch.empty(0)
        self.cumulative_view_count = torch.empty(0)  # Never reset, tracks total visibility
        self.sfm_nn_distance = torch.empty(0)
        self.initial_scale = torch.empty(0)  # [N, 2] for loop-preventive densification  # SfM NN distance prior (fixed at init)
        self.optimizer = None
        self.percent_dense = 0
        self.gauss_id = torch.empty(0, dtype=torch.long)
        self.next_id = 0  # Counter for assigning new IDs
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, scale_init_cap : float = 0.0):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales_raw = torch.sqrt(dist2)
        self.sfm_nn_distance = scales_raw.detach().clone()  # Fixed SfM geometric prior
        # SfM-aware initialization: cap outlier scales to prevent self-reinforcing loop
        if scale_init_cap > 0:
            cap = scales_raw.median() * scale_init_cap
            n_capped = (scales_raw > cap).sum().item()
            scales_raw = torch.clamp(scales_raw, max=cap.item())
            print(f"[ScaleInitCap] median={scales_raw.median():.6f}, cap={cap:.6f}, capped={n_capped}/{len(scales_raw)}")
        scales = torch.log(scales_raw)[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        N = self.get_xyz.shape[0]
        self.gauss_id = torch.arange(N, device="cuda", dtype=torch.long)
        self.next_id = N

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.cumulative_view_count = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.gauss_id.shape[0] > 0:
            self.gauss_id = self.gauss_id[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.cumulative_view_count.shape[0] > 0:
            self.cumulative_view_count = self.cumulative_view_count[valid_points_mask]
        if self.sfm_nn_distance.numel() > 0:
            self.sfm_nn_distance = self.sfm_nn_distance[valid_points_mask]
        if self.initial_scale.numel() > 0:
            self.initial_scale = self.initial_scale[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        n_existing = self.gauss_id.shape[0]
        n_new = self.get_xyz.shape[0] - n_existing
        if n_new > 0:
            new_ids = torch.arange(self.next_id, self.next_id + n_new, device="cuda", dtype=torch.long)
            self.gauss_id = torch.cat([self.gauss_id, new_ids])
            self.next_id += n_new
        # Extend cumulative_view_count for new Gaussians (0 initial), do NOT reset existing
        if self.cumulative_view_count.shape[0] > 0:
            n_new = self.get_xyz.shape[0] - self.cumulative_view_count.shape[0]
            if n_new > 0:
                self.cumulative_view_count = torch.cat([
                    self.cumulative_view_count,
                    torch.zeros((n_new, 1), device="cuda")
                ], dim=0)

        # Extend initial_scale: new Gaussians get current scale as their initial
        if self.initial_scale.numel() > 0:
            n_new = self.get_xyz.shape[0] - self.initial_scale.shape[0]
            if n_new > 0:
                # New Gaussians (from split/clone) start fresh with current scale
                new_init_scale = self.get_scaling[-n_new:].detach().clone()
                self.initial_scale = torch.cat([self.initial_scale, new_init_scale], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, force_split_mask=None, split_directions=None, split_bias_dirs=None, bias_lambda=0.5, split_argmin_offsets=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # Force-split: large scale + low confidence, regardless of gradient
        if force_split_mask is not None:
            padded_force = torch.zeros((n_init_points), dtype=torch.bool, device="cuda")
            padded_force[:force_split_mask.shape[0]] = force_split_mask
            selected_pts_mask = torch.logical_or(selected_pts_mask, padded_force)

        if split_directions is not None:
            # Directional split: children at parent ± direction × offset_magnitude
            n_init = self.get_xyz.shape[0]
            padded_dirs = torch.zeros((n_init, 3), device='cuda')
            padded_dirs[:split_directions.shape[0]] = split_directions
            sel_dirs = padded_dirs[selected_pts_mask]  # (n_sel, 3)
            sel_max_scale = self.get_scaling[selected_pts_mask].max(dim=1, keepdim=True).values  # (n_sel, 1)
            offset = sel_dirs * sel_max_scale  # (n_sel, 3)
            parent_xyz = self.get_xyz[selected_pts_mask]  # (n_sel, 3)
            new_xyz_chunks = []
            for i in range(N):
                sign = 1.0 if i == 0 else -1.0
                if N > 2:  # fallback: scale sign for >2 splits
                    sign = (2.0 * i / (N - 1)) - 1.0
                new_xyz_chunks.append(parent_xyz + sign * offset)
            new_xyz = torch.cat(new_xyz_chunks, dim=0)
        elif split_bias_dirs is not None:
            # Vanilla mechanism (anisotropic Gaussian, in-plane) with biased mean toward ±cancel direction
            n_init = self.get_xyz.shape[0]
            padded_bias = torch.zeros((n_init, 3), device='cuda')
            padded_bias[:split_bias_dirs.shape[0]] = split_bias_dirs  # tangent-plane world-coord unit dirs
            sel_bias_world = padded_bias[selected_pts_mask]  # (n_sel, 3)
            R_sel = build_rotation(self._rotation[selected_pts_mask])  # (n_sel, 3, 3)
            # World → local frame: bias_local = R.T @ bias_world. After tangent-plane projection, 3rd dim ≈ 0.
            bias_local = torch.bmm(R_sel.transpose(1, 2), sel_bias_world.unsqueeze(-1)).squeeze(-1)  # (n_sel, 3)
            max_inplane = self.get_scaling[selected_pts_mask].max(dim=1, keepdim=True).values  # (n_sel, 1)
            mean_bias_per_gauss = bias_local * max_inplane * bias_lambda  # (n_sel, 3)
            # ±mirror: child 0 gets +bias, child 1 gets -bias (still stochastic around each pole)
            mean_chunks = []
            for i in range(N):
                sign = 1.0 if i == 0 else -1.0
                if N > 2:
                    sign = (2.0 * i / (N - 1)) - 1.0
                mean_chunks.append(sign * mean_bias_per_gauss)
            means = torch.cat(mean_chunks, dim=0)  # (n_sel*N, 3)
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
            samples = torch.normal(mean=means, std=stds)  # local-frame samples around ±bias
            rots = R_sel.repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        elif split_argmin_offsets is not None:
            # cancel_argmin / cancel_argmin_dir_random: vanilla anisotropic stochastic noise (independent per child) + ±world-frame δ mirror.
            n_init = self.get_xyz.shape[0]
            padded_off = torch.zeros((n_init, 3), device='cuda')
            padded_off[:split_argmin_offsets.shape[0]] = split_argmin_offsets
            sel_off_world = padded_off[selected_pts_mask]  # (n_sel, 3), already tangent-projected + clamped
            stds_arg = self.get_scaling[selected_pts_mask].repeat(N, 1)
            stds_arg = torch.cat([stds_arg, 0 * torch.ones_like(stds_arg[:, :1])], dim=-1)
            means_zero = torch.zeros_like(stds_arg)
            samples_arg = torch.normal(mean=means_zero, std=stds_arg)  # element-wise independent
            rots_arg = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
            noise_world = torch.bmm(rots_arg, samples_arg.unsqueeze(-1)).squeeze(-1)  # (n_sel*N, 3)
            delta_chunks = []
            for i in range(N):
                sgn = 1.0 if i == 0 else -1.0
                if N > 2:
                    sgn = (2.0 * i / (N - 1)) - 1.0
                delta_chunks.append(sgn * sel_off_world)
            delta_world = torch.cat(delta_chunks, dim=0)  # (n_sel*N, 3)
            parent_world = self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_xyz = parent_world + noise_world + delta_world
        else:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
            means = torch.zeros_like(stds)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        _split_sfm_nn = self.sfm_nn_distance[selected_pts_mask].repeat(N) if self.sfm_nn_distance.numel() > 0 else None
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        if _split_sfm_nn is not None:
            self.sfm_nn_distance = torch.cat([self.sfm_nn_distance, _split_sfm_nn])

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def set_initial_scale(self):
        """Store current scale as initial reference for loop detection."""
        self.initial_scale = self.get_scaling.detach().clone()  # [N, 2]

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        _clone_sfm_nn = self.sfm_nn_distance[selected_pts_mask] if self.sfm_nn_distance.numel() > 0 else None
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        if _clone_sfm_nn is not None:
            self.sfm_nn_distance = torch.cat([self.sfm_nn_distance, _clone_sfm_nn])

    def accumulate_cm(self, cm_per_gauss):
        """Accumulate per-Gauss cancel cm (detached). Resets size if mismatch."""
        cm_d = cm_per_gauss.detach()
        if self.cm_accum is None or self.cm_accum.shape[0] != cm_d.shape[0]:
            self.cm_accum = cm_d.clone()
            self.cm_count = 1
        else:
            self.cm_accum += cm_d
            self.cm_count += 1

    def reset_cm_accum(self):
        self.cm_accum = None
        self.cm_count = 0

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, consistency_mask=None, scale_normalize=False, confidence_gated_densify=False, confidence_scores=None, scale_confidence_split=False, sc_scale_percentile=90, sc_conf_threshold=0.3, sc_max_count=500, cancel_composite_factor=None, densify_method='V', cancel_rank_threshold=0.7, method_random_seed=0, split_directions=None, split_bias_dirs=None, bias_lambda=0.5, split_argmin_offsets=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # Candidate 5: composite criterion grads * cm_avg  (LEGACY: V mode only with explicit cancel_composite_densify)
        if cancel_composite_factor is not None and densify_method == 'V':
            cm_avg = cancel_composite_factor
            n = min(grads.shape[0], cm_avg.shape[0])
            grads[:n] = grads[:n] * cm_avg[:n].unsqueeze(-1)

        # Scale-normalized gradient: compensate for gradient attenuation in large Gaussians
        if scale_normalize:
            max_scale = self.get_scaling.max(dim=1).values.unsqueeze(1)  # [N, 1]
            # Normalize: large scale → boost gradient, small scale → keep as-is
            # Use median as reference to avoid extreme values
            scale_median = max_scale.median()
            scale_factor = (max_scale / (scale_median + 1e-8)).clamp(1.0, 100.0)  # only boost, never reduce
            grads = grads * scale_factor

        # Confidence-gated scale normalization: only boost low-confidence (pathological) Gaussians
        if confidence_gated_densify and confidence_scores is not None:
            n = min(grads.shape[0], confidence_scores.shape[0])
            max_scale = self.get_scaling.max(dim=1).values.unsqueeze(1)[:n]
            scale_median = max_scale.median()
            scale_factor = (max_scale / (scale_median + 1e-8)).clamp(1.0, 50.0)
            # Only apply boost to low-confidence Gaussians (confidence < 0.5)
            gate = (confidence_scores[:n] < 0.5).float()
            gated_factor = 1.0 + (scale_factor - 1.0) * gate
            grads[:n] = grads[:n] * gated_factor

        # If consistency_mask provided, zero out gradients for low-consistency Gaussians
        # This prevents them from being densified (both clone and split check grad >= threshold)
        if consistency_mask is not None:
            n = min(grads.shape[0], consistency_mask.shape[0])
            blocked = ~consistency_mask[:n]
            grads[:n][blocked] = 0.0

        # Scale-Confidence forced split: split large + low-confidence Gaussians
        # regardless of gradient. This breaks the self-reinforcing loop directly.
        force_mask = None
        if scale_confidence_split and confidence_scores is not None:
            N = self._xyz.shape[0]
            max_scale = self.get_scaling.max(dim=1).values  # [N]
            scale_th = torch.quantile(max_scale, sc_scale_percentile / 100.0)
            large = max_scale > scale_th
            n = min(N, confidence_scores.shape[0])
            low_conf = torch.zeros(N, dtype=torch.bool, device="cuda")
            low_conf[:n] = confidence_scores[:n] < sc_conf_threshold
            force_mask = large & low_conf
            if force_mask.sum() > 0:
                print(f"[Scale-Conf Split] force split {force_mask.sum().item()} Gaussians (scale_th={scale_th:.4f}, conf_th={sc_conf_threshold})")
            # Cap to avoid memory explosion
            if force_mask.sum() > sc_max_count:
                # Keep the ones with highest scale
                candidates = torch.where(force_mask)[0]
                scales_cand = max_scale[candidates]
                _, topk_idx = scales_cand.topk(sc_max_count)
                new_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
                new_mask[candidates[topk_idx]] = True
                force_mask = new_mask

        # Phase 1 method dispatch: mask grads to control which Gauss are eligible to split
        if densify_method != 'V' and densify_method != 'AbsGS':
            n = grads.shape[0]
            dev = grads.device
            vanilla_mask = (grads.squeeze() > max_grad)
            cm_a = cancel_composite_factor if cancel_composite_factor is not None else torch.zeros(n, device=dev)
            if cm_a.shape[0] < n:
                cm_a = torch.cat([cm_a, torch.zeros(n - cm_a.shape[0], device=dev)])
            elif cm_a.shape[0] > n:
                cm_a = cm_a[:n]
            if densify_method == 'OR_cancel':
                nv_idx = torch.where(~vanilla_mask)[0]
                if len(nv_idx) > 0:
                    k = max(1, int(len(nv_idx) * (1 - cancel_rank_threshold)))
                    _, top = cm_a[nv_idx].topk(k)
                    add = torch.zeros(n, dtype=torch.bool, device=dev); add[nv_idx[top]] = True
                    eligible = vanilla_mask | add
                else: eligible = vanilla_mask
            elif densify_method == 'OR_random':
                nv_idx = torch.where(~vanilla_mask)[0]
                if len(nv_idx) > 0:
                    k = max(1, int(len(nv_idx) * (1 - cancel_rank_threshold)))
                    gen = torch.Generator(device=dev).manual_seed(method_random_seed)
                    perm = torch.randperm(len(nv_idx), generator=gen, device=dev)[:k]
                    add = torch.zeros(n, dtype=torch.bool, device=dev); add[nv_idx[perm]] = True
                    eligible = vanilla_mask | add
                else: eligible = vanilla_mask
            elif densify_method == 'AND_cancel':
                v_idx = torch.where(vanilla_mask)[0]
                if len(v_idx) > 0:
                    k = max(1, int(len(v_idx) * (1 - cancel_rank_threshold)))
                    _, top = cm_a[v_idx].topk(k)
                    keep = torch.zeros(n, dtype=torch.bool, device=dev); keep[v_idx[top]] = True
                    eligible = keep
                else: eligible = vanilla_mask
            elif densify_method == 'AND_random':
                v_idx = torch.where(vanilla_mask)[0]
                if len(v_idx) > 0:
                    k = max(1, int(len(v_idx) * (1 - cancel_rank_threshold)))
                    gen = torch.Generator(device=dev).manual_seed(method_random_seed)
                    perm = torch.randperm(len(v_idx), generator=gen, device=dev)[:k]
                    keep = torch.zeros(n, dtype=torch.bool, device=dev); keep[v_idx[perm]] = True
                    eligible = keep
                else: eligible = vanilla_mask
            else:
                eligible = vanilla_mask
            # Zero out grads for ineligible -> vanilla densify_and_clone/split won't trigger
            grads_phase1 = grads.clone()
            grads_phase1[~eligible] = 0.0
            grads = grads_phase1
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, force_split_mask=force_mask, split_directions=split_directions, split_bias_dirs=split_bias_dirs, bias_lambda=bias_lambda, split_argmin_offsets=split_argmin_offsets)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.cumulative_view_count.shape[0] > 0:
            self.cumulative_view_count[update_filter] += 1