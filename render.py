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
from scene import Scene, GeoRefineModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.thermal_correction_model import ThermalCorrectionModel
from train import compute_geo_fields, render_and_save_physical_fields
import imageio
import numpy as np
import time


def _is_temporal_thermal_render(gaussians, view=None):
    return bool(
        getattr(gaussians, "has_temporal_thermal_branch", False)
        and (view is None or getattr(view, "is_temporal_thermal", False))
    )


def _temporal_override_color(gaussians, view=None, fid=None):
    if not _is_temporal_thermal_render(gaussians, view):
        return None
    return gaussians.temporal_thermal_color(view, fid=fid)


def _use_static_temporal_geometry(gaussians, view=None, temporal_use_view_deform=True):
    return _is_temporal_thermal_render(gaussians, view) and not bool(temporal_use_view_deform)


def _temporal_tcm_weight(opt, iteration=None):
    if opt is None:
        return 0.0
    target = max(float(getattr(opt, "temporal_tcm_weight", 0.0)), 0.0)
    if iteration is None or target <= 0.0:
        return target
    start_iter = max(int(getattr(opt, "temporal_tcm_start_iter", 0)), 0)
    if int(iteration) < start_iter:
        return 0.0
    ramp_iters = max(int(getattr(opt, "temporal_tcm_ramp_iters", 0)), 0)
    if ramp_iters <= 0:
        return target
    progress = min(max((int(iteration) - start_iter + 1) / float(ramp_iters), 0.0), 1.0)
    return target * progress


def _apply_temporal_tcm(image, temporal_tcm_model, opt, iteration=None):
    weight = _temporal_tcm_weight(opt, iteration)
    if temporal_tcm_model is None or weight <= 0.0:
        return image
    return torch.clamp(image + weight * temporal_tcm_model.step(image), 0.0, 1.0)



def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background,
               geo_model, device, rgb_geo_model=None, temporal_use_view_deform=True, temporal_tcm_model=None,
               opt=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_rgb_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_rgb")
    gts_rgb_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_rgb")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if any(getattr(view, "is_rgbt", False) for view in views):
        makedirs(render_rgb_path, exist_ok=True)
        makedirs(gts_rgb_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    t_list = []
    time_interval = 1.0 / max(len(views), 1)
    no_eval_noise = lambda _: 0.0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        is_rgbt = getattr(view, "is_rgbt", False)
        if _use_static_temporal_geometry(gaussians, view, temporal_use_view_deform):
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            d_xyz, d_rotation, d_scaling = compute_geo_fields(
                view,
                gaussians,
                geo_model,
                pipeline,
                background,
                opt,
                iteration,
                no_eval_noise,
                time_interval,
                device,
                is_blender=True,
                feature_set="thermal" if is_rgbt else "rgb",
                eval_mode=True,
            )

        t_start = time.time()
        results = render(
            view,
            gaussians,
            pipeline,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
            device,
            is_6dof,
            override_color=_temporal_override_color(gaussians, view),
            feature_set="thermal" if is_rgbt else "rgb",
        )
        t_end = time.time()
        rendering = results["render"]
        if _is_temporal_thermal_render(gaussians, view):
            rendering = _apply_temporal_tcm(rendering, temporal_tcm_model, opt)
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)  # Normalize depth for visualization

        # Save rendered outputs
        gt = view.original_image[0:3, :, :]  # Ground truth image
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(depth, os.path.join(depth_path, f'{idx:05d}.png'))
        if is_rgbt and view.original_rgb_image is not None:
            rgb_geo = rgb_geo_model if rgb_geo_model is not None else geo_model
            rgb_d_xyz, rgb_d_rotation, rgb_d_scaling = compute_geo_fields(
                view,
                gaussians,
                rgb_geo,
                pipeline,
                background,
                opt,
                iteration,
                no_eval_noise,
                time_interval,
                device,
                is_blender=True,
                feature_set="rgb",
                eval_mode=True,
            )
            rgb_rendering = render(
                view,
                gaussians,
                pipeline,
                background,
                rgb_d_xyz,
                rgb_d_rotation,
                rgb_d_scaling,
                device,
                is_6dof,
                feature_set="rgb",
            )["render"]
            torchvision.utils.save_image(rgb_rendering, os.path.join(render_rgb_path, f'{idx:05d}.png'))
            torchvision.utils.save_image(view.original_rgb_image[0:3, :, :], os.path.join(gts_rgb_path, f'{idx:05d}.png'))

        torch.cuda.synchronize()
        t_list.append(t_end - t_start)

    # Calculate and print FPS
    t = np.array(t_list[5:])  # Skip initial warm-up renders
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {xyz.shape[0]}')


def interpolate_time(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, geo_model, device,
                     temporal_use_view_deform=True):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).to(device)
        
        if _use_static_temporal_geometry(gaussians, view, temporal_use_view_deform):
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            # Calculate view-dependent terms (same as render_set)
            R = view.R
            T = view.T
            R_torch = torch.from_numpy(R).float().to(device)
            T_torch = torch.from_numpy(T).float().to(device)
            Rt = torch.eye(4, device=device)
            Rt[:3, :3] = R_torch.t()
            Rt[:3, 3] = T_torch

            c2w = Rt
            cam_pos = c2w[:3, 3]
            view_dir = -c2w[:3, 2]

            xyz = gaussians.get_xyz
            N = xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            view_dir = view_dir.unsqueeze(0).expand(N, -1)
            cam_pos = view.camera_center.unsqueeze(0).expand(N, -1)

            d_xyz, d_rotation, d_scaling = geo_model.step(xyz.detach(), time_input, cam_pos, view_dir)
        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, device, is_6dof,
            override_color=_temporal_override_color(gaussians, view, fid=fid),
        )
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_view(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, geo_model, device):
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    render_poses = torch.stack(render_wander_path(view), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        # Calculate view-dependent terms (same as render_set)
        R_torch = torch.from_numpy(R).float().to(device)
        T_torch = torch.from_numpy(T).float().to(device)
        Rt = torch.eye(4, device=device)
        Rt[:3, :3] = R_torch.t()
        Rt[:3, 3] = T_torch
        
        c2w = Rt
        cam_pos = c2w[:3, 3]
        view_dir = -c2w[:3, 2]
        
        xyz = gaussians.get_xyz
        N = xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
        view_dir = view_dir.unsqueeze(0).expand(N, -1)
        cam_pos = view.camera_center.unsqueeze(0).expand(N, -1)
        
        d_xyz, d_rotation, d_scaling = geo_model.step(xyz.detach(), time_input, cam_pos, view_dir)
        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, device, is_6dof,
            override_color=_temporal_override_color(gaussians, view, fid=fid),
        )
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        # acc = results["acc"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, geo_model, device):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).to(device)

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        # Calculate view-dependent terms (same as render_set)
        R_torch = torch.from_numpy(R).float().to(device)
        T_torch = torch.from_numpy(T).float().to(device)
        Rt = torch.eye(4, device=device)
        Rt[:3, :3] = R_torch.t()
        Rt[:3, 3] = T_torch
        
        c2w = Rt
        cam_pos = c2w[:3, 3]
        view_dir = -c2w[:3, 2]
        
        xyz = gaussians.get_xyz
        N = xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
        view_dir = view_dir.unsqueeze(0).expand(N, -1)
        cam_pos = view.camera_center.unsqueeze(0).expand(N, -1)
        
        d_xyz, d_rotation, d_scaling = geo_model.step(xyz.detach(), time_input, cam_pos, view_dir)
        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, device, is_6dof,
            override_color=_temporal_override_color(gaussians, view, fid=fid),
        )
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_poses(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, geo_model, device):
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)
    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        # Calculate view-dependent terms (same as render_set)
        R_torch = torch.from_numpy(R_cur).float().to(device)
        T_torch = torch.from_numpy(T_cur).float().to(device)
        Rt = torch.eye(4, device=device)
        Rt[:3, :3] = R_torch.t()
        Rt[:3, 3] = T_torch
        
        c2w = Rt
        cam_pos = c2w[:3, 3]
        view_dir = -c2w[:3, 2]
        
        xyz = gaussians.get_xyz
        N = xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
        view_dir = view_dir.unsqueeze(0).expand(N, -1)
        cam_pos = view.camera_center.unsqueeze(0).expand(N, -1)
        
        d_xyz, d_rotation, d_scaling = geo_model.step(xyz.detach(), time_input, cam_pos, view_dir)

        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, device, is_6dof,
            override_color=_temporal_override_color(gaussians, view, fid=fid),
        )
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def interpolate_view_original(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background,
                              geo_model, device):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).to(device)

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        # Calculate view-dependent terms (same as render_set)
        R_torch = torch.from_numpy(R_cur).float().to(device)
        T_torch = torch.from_numpy(T_cur).float().to(device)
        Rt = torch.eye(4, device=device)
        Rt[:3, :3] = R_torch.t()
        Rt[:3, 3] = T_torch
        
        c2w = Rt
        cam_pos = c2w[:3, 3]
        view_dir = -c2w[:3, 2]
        
        xyz = gaussians.get_xyz
        N = xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
        view_dir = view_dir.unsqueeze(0).expand(N, -1)
        cam_pos = view.camera_center.unsqueeze(0).expand(N, -1)
        
        d_xyz, d_rotation, d_scaling = geo_model.step(xyz.detach(), time_input, cam_pos, view_dir)

        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, device, is_6dof,
            override_color=_temporal_override_color(gaussians, view, fid=fid),
        )
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str, device, opt=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, device)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        geo_model = GeoRefineModel(device, dataset.is_blender, dataset.is_6dof)
        geo_model.load_weights(dataset.model_path)
        rgb_geo_model = None
        if getattr(scene, "has_rgbt", False):
            rgb_geo_model = GeoRefineModel(device, dataset.is_blender, dataset.is_6dof)
            rgb_geo_model.load_weights(dataset.model_path, subdir="GeometryRGB")
        temporal_use_view_deform = bool(getattr(dataset, "temporal_use_view_deform", False))
        temporal_tcm_model = None
        if getattr(scene, "has_temporal_thermal", False) and _temporal_tcm_weight(opt) > 0.0:
            candidate_tcm = ThermalCorrectionModel(
                device, max_residual=float(getattr(opt, "temporal_tcm_max_residual", 0.1))
            )
            if candidate_tcm.load_weights(dataset.model_path, iteration=scene.loaded_iter if scene.loaded_iter else -1):
                temporal_tcm_model = candidate_tcm
                print("Loaded TemporalTCM correction for temporal thermal rendering.")
            else:
                print("TemporalTCM weights not found; rendering temporal thermal branch without correction.")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        if mode == "physical_fields":
            render_and_save_physical_fields(
                scene,
                gaussians,
                geo_model,
                pipeline,
                dataset.model_path,
                opt,
                scene.loaded_iter,
                device,
                dataset.is_6dof,
                dataset.load2gpu_on_the_fly,
            )
            return

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all

        if not skip_train:
            if mode == "render":
                render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                            scene.getTrainCameras(), gaussians, pipeline,
                            background, geo_model, device, rgb_geo_model=rgb_geo_model,
                            temporal_use_view_deform=temporal_use_view_deform,
                            temporal_tcm_model=temporal_tcm_model, opt=opt)
            else:
                render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                            scene.getTrainCameras(), gaussians, pipeline,
                            background, geo_model, device)

        if not skip_test:
            if mode == "render":
                render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                            scene.getTestCameras(), gaussians, pipeline,
                            background, geo_model, device, rgb_geo_model=rgb_geo_model,
                            temporal_use_view_deform=temporal_use_view_deform,
                            temporal_tcm_model=temporal_tcm_model, opt=opt)
            else:
                render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                            scene.getTestCameras(), gaussians, pipeline,
                            background, geo_model, device)


if __name__ == "__main__":
    # Use CUDA_VISIBLE_DEVICES environment variable to control GPU selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, device, sentinel=True)
    pipeline = PipelineParams(parser)
    optimization = OptimizationParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", default=True, action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original', 'physical_fields'])
    args = get_combined_args(parser, device)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, device)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.mode,
        device,
        optimization.extract(args),
    )
