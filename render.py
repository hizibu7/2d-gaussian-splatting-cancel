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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--confidence_tsdf", action="store_true", help='Mesh: mask low-confidence pixels in TSDF')
    parser.add_argument("--conf_tsdf_threshold", default=0.3, type=float, help='Mesh: confidence threshold for TSDF masking')
    parser.add_argument("--dbc_alpha", default=0.0, type=float, help='Mesh: depth bias correction alpha (0=off)')
    parser.add_argument("--dwf_tau", default=0.0, type=float, help='Mesh: discrepancy-weighted fusion threshold (0=off)')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir)
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir)
    
    
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())

        # Compute per-view confidence maps if confidence_tsdf is enabled
        confidence_maps = None
        if args.confidence_tsdf:
            print("Computing confidence maps for TSDF masking ...")
            from train import compute_multi_signal_confidence
            from functools import partial as fpartial
            confidence, _ = compute_multi_signal_confidence(
                gaussians, scene, pipe, background, num_views=8, occlusion_th=0.05)
            # Render confidence map for each training view
            conf_color = confidence.unsqueeze(1).expand(-1, 3)  # [N, 3]
            confidence_maps = []
            for viewpoint_cam in tqdm(scene.getTrainCameras(), desc="Rendering confidence maps"):
                conf_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=conf_color)
                confidence_maps.append(conf_pkg["render"][0:1].cpu())  # [1, H, W]
            print(f"Confidence TSDF: threshold={args.conf_tsdf_threshold}")

        # Depth Bias Correction (DBC): correct shallow bias before TSDF
        if args.dbc_alpha > 0:
            print(f"Applying Depth Bias Correction (alpha={args.dbc_alpha})...")
            dbc_viewpoints = scene.getTrainCameras()
            for i, viewpoint_cam in tqdm(enumerate(dbc_viewpoints), desc="DBC", total=len(dbc_viewpoints)):
                with torch.no_grad():
                    # Compute per-Gaussian center depth in camera space
                    w2c = viewpoint_cam.world_view_transform
                    xyz = gaussians.get_xyz
                    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
                    xyz_cam = (xyz_homo @ w2c)[:, :3]
                    z_center = xyz_cam[:, 2:3]
                    z_center_color = z_center.expand(-1, 3)
                    
                    # Render center depth map
                    center_pkg = render(viewpoint_cam, gaussians, pipe, background,
                                        override_color=z_center_color)
                    rend_alpha = center_pkg.get("rend_alpha", None)
                    if rend_alpha is None:
                        # Render alpha separately
                        alpha_pkg = render(viewpoint_cam, gaussians, pipe, background)
                        rend_alpha = alpha_pkg["rend_alpha"]
                    center_depth = center_pkg["render"][0:1]
                    center_depth = center_depth / (rend_alpha + 1e-8)
                    center_depth = torch.nan_to_num(center_depth, 0, 0)
                    
                    # Current depth
                    orig_depth = gaussExtractor.depthmaps[i].cuda()
                    
                    # Bias = center - intersection (positive = shallow)
                    bias = center_depth - orig_depth
                    valid = (orig_depth > 0.01) & (rend_alpha > 0.5)
                    
                    # Correct
                    corrected = orig_depth.clone()
                    corrected[valid] = orig_depth[valid] + args.dbc_alpha * bias[valid]
                    corrected = corrected.clamp(min=0)
                    
                    gaussExtractor.depthmaps[i] = corrected.cpu()
            
            print(f"DBC applied to {len(dbc_viewpoints)} depth maps")


        # Discrepancy-Weighted Fusion (DWF): mask unreliable depth pixels
        if args.dwf_tau > 0:
            print(f"Applying Discrepancy-Weighted Fusion (tau={args.dwf_tau})...")
            dwf_viewpoints = scene.getTrainCameras()
            n_masked_total = 0
            n_pixels_total = 0
            for i, viewpoint_cam in tqdm(enumerate(dwf_viewpoints), desc="DWF", total=len(dwf_viewpoints)):
                with torch.no_grad():
                    # Compute per-Gaussian center depth in camera space
                    w2c = viewpoint_cam.world_view_transform
                    xyz = gaussians.get_xyz
                    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
                    xyz_cam = (xyz_homo @ w2c)[:, :3]
                    z_center = xyz_cam[:, 2:3]
                    z_center_color = z_center.expand(-1, 3)
                    
                    # Render center depth map
                    center_pkg = render(viewpoint_cam, gaussians, pipe, background,
                                        override_color=z_center_color)
                    alpha_pkg = render(viewpoint_cam, gaussians, pipe, background)
                    rend_alpha = alpha_pkg["rend_alpha"]
                    center_depth = center_pkg["render"][0:1]
                    center_depth = center_depth / (rend_alpha + 1e-8)
                    center_depth = torch.nan_to_num(center_depth, 0, 0)
                    
                    # Current depth
                    surf_depth = gaussExtractor.depthmaps[i].cuda()
                    
                    # Relative discrepancy
                    valid = (surf_depth > 0.01) & (rend_alpha > 0.5)
                    rel_disc = torch.zeros_like(surf_depth)
                    rel_disc[valid] = torch.abs(center_depth[valid] - surf_depth[valid]) / surf_depth[valid]
                    
                    # Mask high-discrepancy pixels
                    mask = (rel_disc > args.dwf_tau) & valid
                    masked_depth = surf_depth.clone()
                    masked_depth[mask] = 0
                    
                    n_masked_total += mask.sum().item()
                    n_pixels_total += valid.sum().item()
                    
                    gaussExtractor.depthmaps[i] = masked_depth.cpu()
            
            pct = 100 * n_masked_total / max(n_pixels_total, 1)
            print(f"DWF: masked {n_masked_total}/{n_pixels_total} pixels ({pct:.1f}%)")

        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc,
                                                        confidence_maps=confidence_maps, conf_threshold=args.conf_tsdf_threshold)
        
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))