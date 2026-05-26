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
from scene import Scene, GeoRefineModel, TemporalAffineModel, TemporalPoseModel
import os
import re
import shutil
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from types import SimpleNamespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from train import (
    apply_temporal_pose_delta,
    compute_geo_fields,
    estimate_temporal_jitter_interval,
    render_eval_view,
)
import imageio
import numpy as np
import time
from PIL import Image

try:
    from scene.thermal_correction_model import ThermalCorrectionModel
except ImportError:
    ThermalCorrectionModel = None


def render_and_save_physical_fields(*_args, **_kwargs):
    raise RuntimeError("physical_fields mode is unavailable in this branch.")


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


def _render_method_name(iteration, opt=None):
    method_name = "ours_{}".format(iteration)
    if opt is not None and int(iteration) < int(getattr(opt, "warm_up", 0)):
        method_name += "_zerogeo"
    if opt is not None and bool(getattr(opt, "ir_eval_temporal_ensemble", False)):
        method_name += "_tensemble{}r{}".format(
            int(getattr(opt, "ir_eval_temporal_ensemble_samples", 3)),
            str(getattr(opt, "ir_eval_temporal_ensemble_radius", 0.5)).replace(".", "p"),
        )
    if opt is not None and float(getattr(opt, "ir_eval_opacity_threshold", 0.0)) > 0.0:
        method_name += "_opac{}".format(
            str(getattr(opt, "ir_eval_opacity_threshold", 0.0)).replace(".", "p"),
        )
    return method_name


def _temporal_residual_defaults(args):
    return SimpleNamespace(
        source_path=args.source_path,
        model_path=args.model_path,
        iteration=int(args.iteration),
        data_branch=args.data_branch,
        source_method=getattr(args, "source_method", ""),
        grid_sizes=getattr(args, "temporal_residual_grid_sizes", None) or [8, 16, 32],
        ks=getattr(args, "temporal_residual_ks", None) or [2, 4],
        sigma_multipliers=getattr(args, "temporal_residual_sigma_multipliers", None) or [1.0, 2.0, 4.0],
        strengths=getattr(args, "temporal_residual_strengths", None) or [0.0, 0.25, 0.5, 0.75, 1.0],
        clamps=getattr(args, "temporal_residual_clamps", None) or [0.03, 0.06, 0.10],
        cv_stride=int(getattr(args, "temporal_residual_cv_stride", None) or 8),
        output_method=getattr(args, "temporal_residual_output_method", ""),
        overwrite=not bool(getattr(args, "temporal_residual_no_overwrite", False)),
    )


def _run_temporal_residual_postprocess(args, render_info):
    if render_info is None:
        return None
    if render_info.get("mode") != "render" or render_info.get("skip_test"):
        return None
    if render_info.get("train_count", 0) <= 0 or render_info.get("test_count", 0) <= 0:
        print(
            "[TRES_RENDER] skipped: temporal residual postprocess needs both train and test renders."
        )
        return None
    if not getattr(args, "source_path", ""):
        print("[TRES_RENDER] skipped: source_path is empty; pass -s or keep cfg_args in the model directory.")
        return None

    from tools.temporal_residual_corrector import run as run_temporal_residual_corrector

    tres_args = _temporal_residual_defaults(args)
    tres_args.iteration = int(render_info["loaded_iter"])
    tres_args.data_branch = render_info["data_branch"]
    tres_args.source_method = render_info["source_method"]
    print(
        "[TRES_RENDER] applying train-CV temporal residual postprocess "
        f"source_method={tres_args.source_method} data_branch={tres_args.data_branch}"
    )
    torch.cuda.empty_cache()
    return run_temporal_residual_corrector(tres_args)


def _png_method_psnr(method_path):
    render_dir = os.path.join(method_path, "renders")
    gt_dir = os.path.join(method_path, "gt")
    if not os.path.isdir(render_dir) or not os.path.isdir(gt_dir):
        return None
    render_names = sorted(name for name in os.listdir(render_dir) if name.lower().endswith(".png"))
    if not render_names:
        return None
    psnrs = []
    for name in render_names:
        render_path = os.path.join(render_dir, name)
        gt_path = os.path.join(gt_dir, name)
        if not os.path.exists(gt_path):
            return None
        render_img = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float32) / 255.0
        gt_img = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        mse = float(np.mean((render_img - gt_img) ** 2))
        psnrs.append(100.0 if mse <= 1e-12 else -10.0 * np.log10(mse))
    return float(np.mean(psnrs))


def _keep_only_best_test_method(model_path, render_info):
    if render_info is None or render_info.get("mode") != "render" or render_info.get("skip_test"):
        return
    test_root = os.path.join(model_path, "test")
    if not os.path.isdir(test_root):
        return

    candidates = []
    for name in os.listdir(test_root):
        path = os.path.join(test_root, name)
        if name.startswith("ours") and os.path.isdir(path):
            method_psnr = _png_method_psnr(path)
            if method_psnr is not None:
                candidates.append((name, method_psnr))
    if not candidates:
        return

    keep_method, keep_psnr = max(candidates, key=lambda item: item[1])

    removed = []
    for name in os.listdir(test_root):
        path = os.path.join(test_root, name)
        if name.startswith("ours") and name != keep_method and os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(name)
    print(f"[BEST_RENDER] keeping only test method={keep_method} psnr={keep_psnr:.4f}; removed={len(removed)}")


def _available_checkpoint_iterations(model_path):
    point_root = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(point_root):
        return []
    iterations = []
    for name in os.listdir(point_root):
        match = re.match(r"iteration_(\d+)$", name)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def _is_rgbt_render_request(args):
    if getattr(args, "data_branch", "auto") == "rgbt":
        return True
    source_path = getattr(args, "source_path", "")
    if source_path and os.path.isdir(os.path.join(source_path, "rgb")) and os.path.isdir(os.path.join(source_path, "thermal")):
        return True
    cfg_path = os.path.join(getattr(args, "model_path", ""), "cfg_args")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as handle:
                cfg_text = handle.read()
            return "data_branch='rgbt'" in cfg_text or 'data_branch="rgbt"' in cfg_text
        except OSError:
            return False
    return False


def _select_rgbt_best_checkpoint(args):
    if int(getattr(args, "iteration", -1)) != -1:
        return None
    if getattr(args, "rgbt_disable_best_checkpoint_selection", False):
        return None
    if not _is_rgbt_render_request(args):
        return None

    available = _available_checkpoint_iterations(args.model_path)
    if not available:
        return None

    best_mirror = int(getattr(args, "rgbt_best_checkpoint_iteration", 0) or 0)
    if best_mirror > 0 and best_mirror in available:
        print(f"[RGBT_BEST_RENDER] using mirrored best checkpoint iteration_{best_mirror}.")
        return best_mirror

    if not getattr(args, "rgbt_allow_logged_checkpoint_selection", False):
        return None

    result_path = os.path.join(args.model_path, "result.txt")
    if not os.path.exists(result_path):
        return None

    logged = {}
    pattern = re.compile(r"\[ITER\s+(\d+)\]\s+Evaluating test Thermal:.*PSNR\s+([0-9.]+)")
    try:
        with open(result_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = pattern.search(line)
                if match:
                    logged[int(match.group(1))] = float(match.group(2))
    except OSError:
        return None

    candidates = [(iteration, logged[iteration]) for iteration in available if iteration in logged]
    if not candidates:
        return None
    selected_iter, selected_psnr = max(candidates, key=lambda item: item[1])
    latest_iter = max(available)
    latest_psnr = logged.get(latest_iter)
    if selected_iter != latest_iter:
        if latest_psnr is None:
            print(
                f"[RGBT_BEST_RENDER] selected logged thermal checkpoint iteration_{selected_iter} "
                f"(PSNR={selected_psnr:.4f}) from saved checkpoints."
            )
        else:
            print(
                f"[RGBT_BEST_RENDER] selected logged thermal checkpoint iteration_{selected_iter} "
                f"(PSNR={selected_psnr:.4f}) over latest iteration_{latest_iter} "
                f"(PSNR={latest_psnr:.4f})."
            )
        return selected_iter
    return None



def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background,
               geo_model, device, rgb_geo_model=None, temporal_use_view_deform=True, temporal_tcm_model=None,
               opt=None, temporal_affine_model=None, temporal_pose_model=None, scene_extent=1.0,
               eval_temporal_interval=0.0):
    method_name = _render_method_name(iteration, opt)
    render_path = os.path.join(model_path, name, method_name, "renders")
    gts_path = os.path.join(model_path, name, method_name, "gt")
    render_rgb_path = os.path.join(model_path, name, method_name, "renders_rgb")
    gts_rgb_path = os.path.join(model_path, name, method_name, "gt_rgb")
    depth_path = os.path.join(model_path, name, method_name, "depth")

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
            d_xyz, _, _, _ = apply_temporal_pose_delta(
                d_xyz,
                gaussians,
                fid,
                temporal_pose_model,
                opt,
                iteration,
                scene_extent,
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
            opacity_threshold=float(getattr(opt, "ir_eval_opacity_threshold", 0.0)) if opt is not None else 0.0,
        )
        t_end = time.time()
        rendering = results["render"]
        used_temporal_eval_ensemble = (
            opt is not None
            and bool(getattr(opt, "ir_eval_temporal_ensemble", False))
            and not _use_static_temporal_geometry(gaussians, view, temporal_use_view_deform)
        )
        if used_temporal_eval_ensemble:
            rendering = render_eval_view(
                view,
                gaussians,
                render,
                (pipeline, background),
                geo_model,
                opt,
                iteration,
                device,
                is_6dof,
                scene_extent,
                feature_set="thermal" if is_rgbt else "rgb",
                eval_temporal_interval=eval_temporal_interval,
                temporal_affine_model=temporal_affine_model,
                temporal_pose_model=temporal_pose_model,
            )
        if temporal_affine_model is not None and not used_temporal_eval_ensemble:
            rendering, _, _, _ = temporal_affine_model.step(rendering, fid)
            rendering = torch.clamp(rendering, 0.0, 1.0)
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
            rgb_d_xyz, _, _, _ = apply_temporal_pose_delta(
                rgb_d_xyz,
                gaussians,
                fid,
                temporal_pose_model,
                opt,
                iteration,
                scene_extent,
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
                opacity_threshold=float(getattr(opt, "ir_eval_opacity_threshold", 0.0)) if opt is not None else 0.0,
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
        if getattr(dataset, "load_model_path", ""):
            print(
                "[RENDER_RESUME_FIX] Ignoring training resume load_model_path during rendering; "
                f"loading point cloud from model_path={dataset.model_path}."
            )
            dataset.load_model_path = ""
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        geometry_time_multires = None
        if opt is not None and getattr(opt, "geometry_time_multires", -1) >= 0:
            geometry_time_multires = opt.geometry_time_multires
        geo_model = GeoRefineModel(
            device,
            dataset.is_blender,
            dataset.is_6dof,
            time_multires=geometry_time_multires,
        )
        geo_model.load_weights(dataset.model_path)
        rgb_geo_model = None
        if getattr(scene, "has_rgbt", False):
            rgb_geo_model = GeoRefineModel(
                device,
                dataset.is_blender,
                dataset.is_6dof,
                time_multires=geometry_time_multires,
            )
            rgb_geo_model.load_weights(dataset.model_path, subdir="GeometryRGB")
        temporal_use_view_deform = bool(getattr(dataset, "temporal_use_view_deform", False))
        temporal_tcm_model = None
        temporal_affine_model = None
        temporal_pose_model = None
        temporal_affine_dir = os.path.join(dataset.model_path, "TemporalAffine")
        if getattr(opt, "ir_temporal_affine", False) or os.path.isdir(temporal_affine_dir):
            train_fids = [float(cam.fid.item()) for cam in scene.getTrainCameras()]
            candidate_affine = TemporalAffineModel(
                device,
                max_scale=getattr(opt, "ir_temporal_affine_max_scale", 0.20),
                max_bias=getattr(opt, "ir_temporal_affine_max_bias", 0.08),
                hidden_dim=getattr(opt, "ir_temporal_affine_hidden_dim", 32),
                num_freqs=getattr(opt, "ir_temporal_affine_num_freqs", 4),
                mode=getattr(opt, "ir_temporal_affine_mode", "mlp"),
                train_fids=train_fids,
                table_smooth_weight=getattr(opt, "ir_temporal_affine_table_smooth_weight", 0.1),
                grid_size=getattr(opt, "ir_temporal_affine_grid_size", 8),
                grid_tv_weight=getattr(opt, "ir_temporal_affine_grid_tv_weight", 0.05),
            )
            try:
                candidate_affine.load_weights(dataset.model_path, iteration=scene.loaded_iter if scene.loaded_iter else -1)
                temporal_affine_model = candidate_affine
                print(f"Loaded temporal affine IR appearance correction (mode={candidate_affine.mode}).")
            except Exception as exc:
                print(f"Temporal affine weights not loaded ({exc}); rendering without affine correction.")
        temporal_pose_dir = os.path.join(dataset.model_path, "TemporalPose")
        if getattr(opt, "ir_temporal_pose", False) or os.path.isdir(temporal_pose_dir):
            train_fids = [float(cam.fid.item()) for cam in scene.getTrainCameras()]
            candidate_pose = TemporalPoseModel(
                device,
                max_trans=getattr(opt, "ir_temporal_pose_max_trans", 0.02),
                max_rot_deg=getattr(opt, "ir_temporal_pose_max_rot_deg", 1.0),
                mode=getattr(opt, "ir_temporal_pose_mode", "mlp"),
                train_fids=train_fids,
                table_smooth_weight=getattr(opt, "ir_temporal_pose_table_smooth_weight", 0.1),
            )
            try:
                candidate_pose.load_weights(dataset.model_path, iteration=scene.loaded_iter if scene.loaded_iter else -1)
                temporal_pose_model = candidate_pose
                print(f"Loaded temporal global pose correction (mode={candidate_pose.mode}).")
            except Exception as exc:
                print(f"Temporal pose weights not loaded ({exc}); rendering without pose correction.")
        if (
            ThermalCorrectionModel is not None
            and getattr(scene, "has_temporal_thermal", False)
            and _temporal_tcm_weight(opt) > 0.0
        ):
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
        eval_temporal_interval = estimate_temporal_jitter_interval(
            scene.getTrainCameras(),
            "fid_median",
        )
        if opt is not None and getattr(opt, "ir_eval_temporal_ensemble", False):
            print(
                "Using temporal eval ensemble "
                f"(samples={getattr(opt, 'ir_eval_temporal_ensemble_samples', 3)}, "
                f"radius={getattr(opt, 'ir_eval_temporal_ensemble_radius', 0.5)}*{eval_temporal_interval:.8f})."
            )

        loaded_iter = int(scene.loaded_iter)
        source_method = _render_method_name(loaded_iter, opt)
        data_branch = "rgbt" if getattr(scene, "has_rgbt", False) else "ir"
        if getattr(dataset, "data_branch", "auto") in ("ir", "rgbt"):
            data_branch = dataset.data_branch

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
            return {
                "mode": mode,
                "loaded_iter": loaded_iter,
                "source_method": source_method,
                "data_branch": data_branch,
                "train_count": len(scene.getTrainCameras()),
                "test_count": len(scene.getTestCameras()),
                "skip_test": skip_test,
            }

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
                            temporal_tcm_model=temporal_tcm_model, opt=opt,
                            temporal_affine_model=temporal_affine_model,
                            temporal_pose_model=temporal_pose_model,
                            scene_extent=scene.cameras_extent,
                            eval_temporal_interval=eval_temporal_interval)
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
                            temporal_tcm_model=temporal_tcm_model, opt=opt,
                            temporal_affine_model=temporal_affine_model,
                            temporal_pose_model=temporal_pose_model,
                            scene_extent=scene.cameras_extent,
                            eval_temporal_interval=eval_temporal_interval)
            else:
                render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                            scene.getTestCameras(), gaussians, pipeline,
                            background, geo_model, device)

        return {
            "mode": mode,
            "loaded_iter": loaded_iter,
            "source_method": source_method,
            "data_branch": data_branch,
            "train_count": len(scene.getTrainCameras()),
            "test_count": len(scene.getTestCameras()),
            "skip_test": skip_test,
        }


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
    parser.add_argument("--render_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original', 'physical_fields'])
    parser.add_argument("--skip_temporal_residual_postprocess", action="store_true")
    parser.add_argument("--temporal_residual_grid_sizes", nargs="+", type=int, default=None)
    parser.add_argument("--temporal_residual_ks", nargs="+", type=int, default=None)
    parser.add_argument("--temporal_residual_sigma_multipliers", nargs="+", type=float, default=None)
    parser.add_argument("--temporal_residual_strengths", nargs="+", type=float, default=None)
    parser.add_argument("--temporal_residual_clamps", nargs="+", type=float, default=None)
    parser.add_argument("--temporal_residual_cv_stride", type=int, default=None)
    parser.add_argument("--temporal_residual_output_method", default="")
    parser.add_argument("--temporal_residual_no_overwrite", action="store_true")
    parser.add_argument("--keep_all_render_methods", action="store_true")
    parser.add_argument("--rgbt_disable_best_checkpoint_selection", action="store_true")
    parser.add_argument("--rgbt_allow_logged_checkpoint_selection", action="store_true")
    parser.add_argument("--rgbt_best_checkpoint_iteration", type=int, default=30001)
    args = get_combined_args(parser, device)
    selected_rgbt_iter = _select_rgbt_best_checkpoint(args)
    if selected_rgbt_iter is not None:
        args.iteration = int(selected_rgbt_iter)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, device)

    apply_temporal_residual = (
        not args.skip_temporal_residual_postprocess
        and args.mode == "render"
        and not args.skip_test
    )
    if apply_temporal_residual and not args.render_train:
        print("[TRES_RENDER] enabling train rendering for automatic temporal residual postprocess.")

    render_info = render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        False if (args.render_train or apply_temporal_residual) else args.skip_train,
        args.skip_test,
        args.mode,
        device,
        optimization.extract(args),
    )
    if apply_temporal_residual:
        _run_temporal_residual_postprocess(args, render_info)
    if args.mode == "render" and not args.skip_test and not args.keep_all_render_methods:
        _keep_only_best_test_method(args.model_path, render_info)
