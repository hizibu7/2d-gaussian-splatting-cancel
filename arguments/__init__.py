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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05
        self.cancel_loss_lambda = 0.0
        self.cancel_loss_K = 200
        self.cancel_loss_start = 7000
        self.cancel_loss_radii_min = 5
        self.cancel_composite_densify = False
        self.cancel_composite_grad_threshold = 0.0001
        # Phase 1 method dispatch
        self.densify_method = 'V'  # V | OR_cancel | OR_random | AND_cancel | AND_random | AbsGS
        self.cancel_rank_threshold = 0.7  # top (1 - thr) fraction of candidate pool
        self.method_random_seed = 0
        self.argmin_clamp_factor = 1.5
        # Cancel-as-split-direction (Phase 2)
        self.split_method = 'V'  # V | cancel | random_dir | orthogonal_dir
        self.split_offset_scale = 1.0  # offset magnitude as multiple of max scale

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.adaptive_normal = False
        self.consistency_normal = False
        self.consistency_update_interval = 1000
        self.consistency_views = 8
        self.consistency_start_iter = 15000
        self.consistency_clamp_low = 0.3
        self.consistency_occlusion_th = 0.05
        self.consistency_only = False
        self.uniform_mild = False
        self.uniform_mild_weight = 0.75
        self.no_consistency_dist = False
        self.no_consistency_normal = False
        self.consistency_densify = False
        self.consistency_densify_start = 3000
        self.consistency_densify_threshold = 0.5
        # Consensus normal loss
        self.consistency_loss = False
        self.lambda_consistency = 0.05
        self.consistency_loss_high = 0.7
        # Multi-signal confidence
        self.multi_signal = False
        self.ms_w_normal = 1.0
        self.ms_w_depth = 0.5
        self.ms_w_color = 0.5
        self.ms_w_opacity = 0.3
        self.ms_w_scale = 0.3
        self.ms_w_aspect = 0.3
        self.ms_w_sh = 0.5
        self.ms_w_grad = 0.3
        # Scale-only gating experiment (Exp 2-A)
        self.scale_gate = False          # enable scale-only gating
        self.scale_gate_invert = False   # False: small scale → relax, True: large scale → relax
        self.scale_gate_target = 'dist'  # 'dist', 'normal', 'both'
        self.scale_gate_start_iter = 7000
        self.rot_detach_dist = False  # Block dist gradient from flowing to rotation
        self.rot_detach_alpha = 0.0  # Partial detach: 0=full detach, 1=no detach
        self.rot_detach_from_iter = 0  # Start detaching from this iteration
        self.rot_detach_until_iter = 30000  # Stop detaching at this iteration
        self.normal_detach_xyz = False
        self.normal_detach_scaling = False
        self.normal_detach_opacity = False
        self.photo_detach_rotation = False
        self.combo_ndxyz_rd = False  # Combined: normal_detach_xyz + rot_detach_dist
        self.log_trajectory = False  # Per-iter gradient/loss trajectory logging
        self.log_trajectory_every = 100
        self.log_per_gauss_v2 = False
        self.log_cancellation = False
        self.log_cancel_v3 = False
        self.log_cancel_v4 = False
        self.log_cancel_v5 = False  # Log every N iterations
        self.log_cancel_v6 = False
        self.log_cancel_v7 = False
        self.log_cancel_v8 = False
        self.log_cancel_v9 = False
        self.log_cancel_v10 = False
        self.log_cancel_v11 = False
        self.log_cancel_perloss = False
        self.prune_cancel_iter = 0
        self.mid_intervene_mode = "none"
        self.mid_intervene_module = "mid_intervene_render"
        self.mid_intervene_start = 15001
        self.mid_intervene_end = 28001
        self.mid_intervene_step = 1000
        self.cancel_amplify_gamma = 0.0
        self.cancel_amplify_start = 5000
        self.cancel_amplify_interval = 500
        self.cancel_amplify_cm_thr = 0.7
        self.cancel_amplify_g_abs_mean_min = 1e-7
        self.cancel_amplify_nr_min = 10
        self.log_perview = False
        self.perview_anchors = "3000,7000,14000"
        self.perview_n_sample = 1000
        self.gt_depth_dir = ""
        self.v11_k = 3.0
        self.v11_n_samples = 256
        self.exp_b_risk_rot = False
        self.exp_b_max_detach = 0.5
        self.exp_c_risk_dist = False
        self.exp_a_risk_ms = False
        self.exp_a_risk_mesh = False  # Exp A v3: mesh-error-based signal weights
        self.scale_proportional_dist = False  # Scale-proportional dist attenuation
        self.scale_detach_channel = ""  # Per-channel scale-proportional dist attenuation: "rotation", "xyz", "scaling", "opacity"
        self.aniso_rot_detach = False
        self.aniso_rot_detach_scale = 6.0
        self.aniso_rot_detach_power = 2.0
        self.aniso_rot_detach_from_iter = 0
        self.aniso_rot_detach_until_iter = 30000
        self.detail_aware_loss = False  # GT-gradient weighted dist/normal
        self.detail_beta = 10.0  # Strength of down-weighting at high-grad pixels
        self.geom_detail_aware = False  # Surface-normal-variance weighted dist/normal
        self.geom_detail_beta = 10.0
        self.tangent_normal_loss = False
        self.tangent_blur_sigma = 2.0
        self.normal_coupled_dist = False
        self.normal_coupled_power = 2.0
        self.v9_pathway_selective = False
        self.v9_alpha = 1.0
        self.v9_beta = 10.0
        self.v10_context = False
        self.v10_alpha = 1.0
        self.v10_k = 16
        self.v10_aniso_thr = 6.0
        self.v10_reg_thr = 0.9
        self.v10_update_every = 500
        self.v10_warmup_iter = 7000
        self.loss_purify = False
        self.v12_ratio_detect = False
        self.v12_ratio_thr = 0.5
        self.v12_freq_pct = 30.0
        self.v12_update_every = 2000
        self.v12_warmup_iter = 15000
        self.v12_rot_start_iter = 7000
        self.v12_n_views = 8
        self.v12_hysteresis = 3
        self.v12_light_mode = False
        self.dist_freq_gate = False
        self.dist_freq_start_iter = 0
        self.hybrid_gate = False
        self.hybrid_gate_beta = 0.7
        self.hybrid_gate_start_iter = 0
        # SPA v1: per-Gaussian scale weight attenuates normal->rotation and dist->xyz for small Gaussians (detail)
        self.spa_v1 = False
        self.aniso_normal = False
        self.inverse_scale_normal = False
        self.inv_aniso_normal = False
        self.learnable_confidence = False
        self.lambda_conf_reg = 0.1
        self.conf_warmup_iter = 15000
        self.conf_init_logit = 2.0
        self.conf_lr = 0.01
        self.conf_floor = 0.1
        self.phase1_logging = ''  # output dir for trajectory log (empty = disabled)
        self.snapshot_iters = '5000,10000,15000,20000,30000'
        self.opacity_detach_dist = False  # Block dist gradient from flowing to opacity
        self.xyz_detach_dist = False  # Block dist gradient from flowing to xyz
        self.scaling_detach_dist = False  # Block dist gradient from flowing to scaling
        self.scale_normalize = False  # Scale-normalized densification gradient
        self.confidence_gated_densify = False  # Confidence-gated scale normalization
        self.scale_init_cap = 0.0  # SfM-aware init: cap scale to median*cap (0=disabled)
        self.specular_guard = False  # Block dist relaxation on specular surfaces
        self.ms_artifact_guard = False  # MS + artifact mask: block dist relaxation on artifacts
        self.ms_ag_scale_th = 0.5  # scale_conf >= this = scale-normal
        self.ms_ag_opacity_th = 0.3  # opacity < this = artifact candidate
        self.ms_ag_color_th = 0.4  # color_cons < this = artifact confirmed
        # Relax-Suppress: split signals into relax (low→relax dist) and suppress (low→keep dist)
        # Signal-Role: role-aware weighted average (no thresholds)
        # Relax signals: low → relax dist (scale, depth)
        # Suppress signals: low → keep dist (opacity, color)
        # All enter directly into weighted avg; weights set roles
        self.signal_role = False
        self.sr_w_scale = 1.0     # relax signal
        self.sr_w_depth = 1.0     # relax signal
        self.sr_w_opacity = 0.5   # suppress signal
        self.sr_w_color = 0.5     # suppress signal
        self.relax_suppress = False
        self.rs_relax_w_scale = 1.0
        self.rs_relax_w_depth = 1.0
        self.rs_relax_w_aspect = 0.5
        self.rs_relax_w_grad = 0.5
        self.rs_suppress_w_sh = 1.0
        self.rs_suppress_w_color = 1.0
        # Cross-view depth consistency loss
        self.cross_view_depth = False
        self.lambda_cvd = 0.1           # loss weight
        self.cvd_num_views = 8           # views to sample
        self.cvd_update_interval = 100   # recompute every N iters
        self.cvd_start_iter = 3000       # start after densification begins
        self.cvd_loop_threshold = 0.5    # scale_conf below this = loop candidate
        self.cvd_normalize = True        # normalize variance by mean depth
        # Loop-preventive densification
        self.loop_densify = False
        self.loop_densify_threshold = 2.5    # growth rate above this → force split
        self.loop_densify_interval = 500     # check every N iters
        self.loop_densify_start = 1000       # start checking from this iter
        self.loop_densify_max = 1000         # max Gaussians to split per check
        self.track_sparse = False  # Log sparse region Gaussian dynamics
        self.normal_anneal = False  # Anneal normal relaxation: vanilla until anneal_start, then ramp up
        self.normal_anneal_start = 15000  # Start normal relaxation after this iter
        self.normal_anneal_ramp = 5000  # Ramp up over this many iters (linear)
        self.scale_confidence_split = False  # Force-split large + low-confidence Gaussians
        self.sc_scale_percentile = 90  # Scale percentile threshold for force-split
        self.sc_conf_threshold = 0.3  # Confidence threshold (below = low confidence)
        self.sc_max_count = 500  # Max Gaussians to force-split per iteration
        # Depth CV gating: relax dist loss where depth is inconsistent across views
        self.depth_cv_gate = False
        self.depth_cv_ema_alpha = 0.01
        self.depth_cv_start_iter = 3000
        self.depth_cv_gamma = 1.0
        # Diagnose-Prescribe: classify Gaussians by pathology, apply targeted treatment
        self.diagnose_prescribe = False
        self.dp_loop_scale_th = 0.5      # scale_conf below this = loop candidate
        self.dp_loop_normal_th = 0.7     # normal_cons below this = loop confirmed
        self.dp_artifact_scale_th = 0.5  # scale_conf above this = scale-normal
        self.dp_artifact_opacity_th = 0.3  # opacity below this = artifact candidate
        self.dp_artifact_color_th = 0.4  # color_cons below this = artifact confirmed
        self.dp_artifact_suppress = 0.1  # opacity suppression strength for artifacts
        # Split-signal: loop_conf for dist, render_conf for densification
        self.split_signal = False
        self.split_signal_densify_threshold = 0.3  # render_conf below this → force densify
        # Purpose-driven signal: each signal has a specific role
        self.purpose_signal = False
        self.ps_gamma = 1.0          # scale_conf exponent
        self.ps_beta = 0.5           # depth_conf exponent
        self.ps_specular_th = 0.3    # SH/color threshold for specular mask
        self.ps_mode = 'scale_depth'  # scale_only, depth_only, normal_only, scale_depth, scale_depth_sum
        # Idea 1: SfM density-aware scale regularization
        self.sfm_scale_reg = False
        self.lambda_sfm_scale = 0.1
        self.sfm_scale_reg_c = 3.0       # allow scale up to c * nn_distance before penalty
        self.sfm_scale_reg_start = 3000
        # Idea 2: Gradient conflict-aware dist loss
        self.grad_conflict_gate = False
        self.grad_conflict_interval = 500  # recompute every N iterations
        self.grad_conflict_views = 4       # views to sample
        self.grad_conflict_start = 3000    # start gating after this iter
        self.grad_conflict_gamma = 1.0     # sharpness of conflict -> relaxation mapping
        # Scale-aware dist loss: normalize dist by Gaussian scale
        self.scale_aware_dist = False
        self.center_anchored_dist = False  # CA-dist: penalize intersection-center depth diff
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

