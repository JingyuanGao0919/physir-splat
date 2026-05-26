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

from datetime import datetime
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from random import randint
import math

from utils.loss_utils import (
    compute_rt_consistency_loss,
    gradient_l1_loss,
    image_gradient_mean,
    l1_loss,
    residual_robust_l1_loss,
    ssim,
    trimmed_residual_loss,
)
from gaussian_renderer import render, render_depth_acc, GaussianRasterizationSettings, GaussianRasterizer
import sys
from scene import (
    Scene,
    GaussianModel,
    GeoRefineModel,
    TemporalAffineModel,
    TemporalPoseModel,
    AtmosModel,
    PlanckModel,
    RadiometricModel,
)
from utils.general_utils import safe_state, get_linear_noise_func
from tqdm import tqdm
from utils.image_utils import psnr
from utils.radiative_utils import (
    fit_and_log_physir_attributes,
    physir_render_blend,
    radiative_confidence_mse_loss,
    radiative_confidence_values,
    rt_guidance_loss,
    tir_auxiliary_gate,
)
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from itertools import islice, count

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def ramped_weight(max_weight, iteration, start_iter, ramp_iters):
    if max_weight <= 0.0 or iteration < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return float(max_weight)
    progress = min(1.0, (iteration - start_iter + 1) / float(ramp_iters))
    return float(max_weight) * progress


def temporal_jitter_factor(opt, iteration, smooth_term):
    base = float(smooth_term(iteration))
    floor = max(0.0, float(getattr(opt, "temporal_jitter_min_factor", 0.0)))
    if floor <= 0.0:
        return base
    return max(base, floor)


def parse_multiscale_scales(scales):
    if scales is None:
        return []
    if isinstance(scales, (list, tuple)):
        raw_scales = scales
    else:
        raw_scales = str(scales).replace(";", ",").split(",")
    parsed = []
    for scale in raw_scales:
        try:
            value = int(scale)
        except (TypeError, ValueError):
            continue
        if value > 1 and value not in parsed:
            parsed.append(value)
    return parsed


def multiscale_image_loss(image, gt_image, scales, mode="rmse", eps=1e-8):
    losses = []
    image4 = image.unsqueeze(0)
    gt4 = gt_image.unsqueeze(0)
    for scale in parse_multiscale_scales(scales):
        if image4.shape[-2] < scale or image4.shape[-1] < scale:
            continue
        pred_s = F.avg_pool2d(image4, kernel_size=scale, stride=scale)
        gt_s = F.avg_pool2d(gt4, kernel_size=scale, stride=scale)
        diff2 = (pred_s - gt_s).pow(2).mean()
        if str(mode).lower() in ("mse", "l2"):
            losses.append(diff2)
        else:
            losses.append(torch.sqrt(diff2.clamp_min(eps)))
    if not losses:
        return image.new_tensor(0.0)
    return torch.stack(losses).mean()


def downsample_sparse_depth(depth, scale_factor):
    scale_factor = int(scale_factor)
    if scale_factor <= 1:
        return depth
    height, width = depth.shape[-2], depth.shape[-1]
    target_h = max(1, height // scale_factor)
    target_w = max(1, width // scale_factor)
    cropped = depth[..., : target_h * scale_factor, : target_w * scale_factor]
    valid = cropped > 0.0
    large = torch.full_like(cropped, 1e8)
    depth_for_min = torch.where(valid, cropped, large)
    pooled = -F.max_pool2d(-depth_for_min.view(1, 1, target_h * scale_factor, target_w * scale_factor),
                           kernel_size=scale_factor, stride=scale_factor)[0, 0]
    pooled_valid = F.max_pool2d(valid.float().view(1, 1, target_h * scale_factor, target_w * scale_factor),
                                kernel_size=scale_factor, stride=scale_factor)[0, 0] > 0.0
    return torch.where(pooled_valid, pooled, torch.zeros_like(pooled))


def sparse_depth_supervision_loss(
    viewpoint_cam,
    gaussians,
    pipe,
    background,
    d_xyz_full,
    d_rotation_full,
    d_scaling_full,
    device,
    is_6dof,
    feature_set,
    opt,
):
    gt_sparse_depth = getattr(viewpoint_cam, "depth", None)
    if gt_sparse_depth is None:
        return image_like_zero(background), {"valid": 0, "total": 0, "acc_mean": 0.0, "acc_p90": 0.0, "acc_max": 0.0, "pred_median": 0.0, "gt_median": 0.0, "scale": 1, "oom": 0}

    gt_sparse_depth = gt_sparse_depth.to(device)
    depth_scale_factor = max(1, int(getattr(opt, "ir_sparse_depth_scale_factor", 1)))
    gt_sparse_depth = downsample_sparse_depth(gt_sparse_depth, depth_scale_factor)
    sparse_mask = gt_sparse_depth > 0.0
    sparse_total = int(sparse_mask.sum().item())
    if sparse_total == 0:
        return image_like_zero(background), {"valid": 0, "total": 0, "acc_mean": 0.0, "acc_p90": 0.0, "acc_max": 0.0, "pred_median": 0.0, "gt_median": 0.0, "scale": depth_scale_factor, "oom": 0}

    try:
        depth_pkg = render_depth_acc(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            d_xyz_full,
            d_rotation_full,
            d_scaling_full,
            device,
            is_6dof,
            feature_set=feature_set,
            scale_factor=depth_scale_factor,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return image_like_zero(background), {
            "valid": 0,
            "total": sparse_total,
            "acc_mean": 0.0,
            "acc_p90": 0.0,
            "acc_max": 0.0,
            "pred_median": 0.0,
            "gt_median": 0.0,
            "scale": depth_scale_factor,
            "oom": 1,
        }
    pred_depth = depth_pkg["depth"][0]
    pred_acc = depth_pkg["acc"][0].clamp_min(1e-6)
    if bool(getattr(opt, "ir_sparse_depth_use_acc_normalize", True)):
        pred_depth = pred_depth / pred_acc

    mask = sparse_mask & torch.isfinite(pred_depth) & (pred_acc >= float(opt.ir_sparse_depth_min_acc))
    valid = int(mask.sum().item())
    if valid == 0:
        sparse_acc = pred_acc[sparse_mask].detach()
        return image_like_zero(background), {
            "valid": 0,
            "total": sparse_total,
            "acc_mean": float(sparse_acc.mean().item()) if sparse_acc.numel() else 0.0,
            "acc_p90": float(torch.quantile(sparse_acc.float(), 0.90).item()) if sparse_acc.numel() else 0.0,
            "acc_max": float(sparse_acc.max().item()) if sparse_acc.numel() else 0.0,
            "pred_median": 0.0,
            "gt_median": 0.0,
            "scale": depth_scale_factor,
            "oom": 0,
        }

    gt_values = gt_sparse_depth[mask]
    pred_values = pred_depth[mask]
    scale = gt_values.detach().median().clamp_min(1e-3)
    depth_loss = F.smooth_l1_loss(
        pred_values / scale,
        gt_values / scale,
        beta=float(opt.ir_sparse_depth_huber_beta),
    )
    acc_values = pred_acc[mask].detach()
    return depth_loss, {
        "valid": valid,
        "total": sparse_total,
        "acc_mean": float(acc_values.mean().item()),
        "acc_p90": float(torch.quantile(acc_values.float(), 0.90).item()),
        "acc_max": float(acc_values.max().item()),
        "pred_median": float(pred_values.detach().median().item()),
        "gt_median": float(gt_values.detach().median().item()),
        "scale": depth_scale_factor,
        "oom": 0,
    }


def image_like_zero(tensor):
    return tensor.new_tensor(0.0)


def estimate_temporal_jitter_interval(cameras, mode="fid_median"):
    mode = str(mode).lower()
    if mode == "stack":
        return None
    if not cameras:
        return 1.0
    if mode == "train_count":
        return 1.0 / max(len(cameras), 1)

    fids = []
    for cam in cameras:
        fid = getattr(cam, "fid", None)
        if torch.is_tensor(fid):
            fids.append(float(fid.detach().reshape(-1)[0].cpu().item()))
        elif fid is not None:
            fids.append(float(fid))
    if len(fids) < 2:
        return 1.0 / max(len(cameras), 1)

    fid_tensor = torch.tensor(sorted(set(fids)), dtype=torch.float32)
    diffs = fid_tensor[1:] - fid_tensor[:-1]
    diffs = diffs[diffs > 1e-8]
    if diffs.numel() == 0:
        return 1.0 / max(len(cameras), 1)
    return float(torch.median(diffs).item())


def condition_geometry_inputs(cam_pos, view_dir, mode="view"):
    mode = str(mode or "view").lower()
    if mode in ("view", "full"):
        return cam_pos, view_dir
    zero_cam = torch.zeros_like(cam_pos)
    zero_dir = torch.zeros_like(view_dir)
    if mode in ("none", "time", "temporal", "xyz_time", "static"):
        return zero_cam, zero_dir
    if mode in ("dir", "view_dir", "viewdir"):
        return zero_cam, view_dir
    if mode in ("camera", "cam", "cam_pos", "position"):
        return cam_pos, zero_dir
    return cam_pos, view_dir


def compute_residual_densify_weight(image, gt_image, means3D, visibility_filter, viewpoint_cam, opt):
    power = float(getattr(opt, "densify_residual_weight_power", 0.0))
    if power <= 0.0 or visibility_filter is None or int(visibility_filter.sum().item()) == 0:
        return None, None, {
            "residual_weight_mean": 1.0,
            "residual_weight_p90": 1.0,
            "residual_weight_max": 1.0,
            "residual_sample_mean": 0.0,
            "residual_sample_p90": 0.0,
        }

    with torch.no_grad():
        residual = (image.detach() - gt_image.detach()).abs().mean(dim=0, keepdim=True).unsqueeze(0)
        blur = int(getattr(opt, "densify_residual_weight_blur", 1))
        if blur > 1:
            if blur % 2 == 0:
                blur += 1
            residual = F.avg_pool2d(residual, kernel_size=blur, stride=1, padding=blur // 2)

        visible_idx = visibility_filter.nonzero(as_tuple=True)[0]
        pts_h = torch.cat(
            [means3D.detach(), torch.ones_like(means3D.detach()[:, :1])],
            dim=1,
        )
        proj = pts_h @ viewpoint_cam.full_proj_transform
        w = proj[:, 3:4]
        safe_w = torch.where(w.abs() > 1e-7, w, torch.ones_like(w))
        ndc = proj[:, :3] / safe_w
        coords = ndc[visible_idx, :2]
        valid = (w[visible_idx, 0].abs() > 1e-7) & torch.isfinite(coords).all(dim=1)
        valid = valid & (coords[:, 0].abs() <= 1.15) & (coords[:, 1].abs() <= 1.15)

        sampled = torch.zeros((visible_idx.numel(), 1), device=image.device, dtype=image.dtype)
        if valid.any():
            grid = coords[valid].clamp(-1.0, 1.0).view(1, -1, 1, 2)
            sampled_valid = F.grid_sample(
                residual,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).view(-1, 1)
            sampled[valid] = sampled_valid

        positive = sampled[sampled > 0]
        if positive.numel() == 0:
            return None, None, {
                "residual_weight_mean": 1.0,
                "residual_weight_p90": 1.0,
                "residual_weight_max": 1.0,
                "residual_sample_mean": 0.0,
                "residual_sample_p90": 0.0,
            }
        q = float(getattr(opt, "densify_residual_weight_quantile", 0.75))
        q = min(max(q, 0.05), 0.95)
        scale = torch.quantile(positive.float(), q).clamp_min(1e-6).to(sampled.dtype)
        ratio = (sampled / scale).clamp_min(0.0).pow(power)
        mean_ratio = ratio[valid].mean().clamp_min(1e-6) if valid.any() else ratio.mean().clamp_min(1e-6)
        ratio = ratio / mean_ratio
        ratio = ratio.clamp(
            float(getattr(opt, "densify_residual_weight_min", 0.25)),
            float(getattr(opt, "densify_residual_weight_max", 4.0)),
        )

        finite_weight = ratio[torch.isfinite(ratio)]
        finite_sample = sampled[torch.isfinite(sampled)]
        stats = {
            "residual_weight_mean": float(finite_weight.mean().item()) if finite_weight.numel() else 1.0,
            "residual_weight_p90": (
                float(torch.quantile(finite_weight.float(), 0.90).item()) if finite_weight.numel() else 1.0
            ),
            "residual_weight_max": float(finite_weight.max().item()) if finite_weight.numel() else 1.0,
            "residual_sample_mean": float(finite_sample.mean().item()) if finite_sample.numel() else 0.0,
            "residual_sample_p90": (
                float(torch.quantile(finite_sample.float(), 0.90).item()) if finite_sample.numel() else 0.0
            ),
        }
        return visible_idx, ratio, stats


def compute_geo_fields(viewpoint_cam, gaussians, geo_model, pipe, background,
                       opt, iteration, smooth_term, time_interval, device, is_blender,
                       feature_set, eval_mode=False, respect_warmup=True):
    warm_up = getattr(opt, "warm_up", 3_000)
    if respect_warmup and iteration < warm_up:
        return 0.0, 0.0, 0.0

    R = viewpoint_cam.R
    T = viewpoint_cam.T
    R_torch = torch.from_numpy(R).float().to(device)
    T_torch = torch.from_numpy(T).float().to(device)

    Rt = torch.eye(4, device=device)
    Rt[:3, :3] = R_torch.t()
    if eval_mode:
        Rt[:3, 3] = T_torch
    else:
        Rt[:3, 3] = -R_torch.t() @ T_torch

    c2w = Rt
    view_dir = -c2w[:3, 2]

    with torch.no_grad():
        tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=viewpoint_cam.image_height,
            image_width=viewpoint_cam.image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=viewpoint_cam.world_view_transform,
            projmatrix=viewpoint_cam.full_proj_transform,
            sh_degree=gaussians.active_sh_degree,
            campos=viewpoint_cam.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, device=device)
        screenspace_points_densify = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, device=device)
        shs = gaussians.get_thermal_features if feature_set in ("thermal", "tir", "ir") else gaussians.get_features

        rasterizer = GaussianRasterizer(raster_settings)
        _, radii, _ = rasterizer(
            means3D=gaussians.get_xyz,
            means2D=screenspace_points,
            means2D_densify=screenspace_points_densify,
            shs=shs,
            colors_precomp=None,
            opacities=gaussians.get_opacity,
            scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation,
            cov3D_precomp=None
        )

        visibility_filter = radii > 0
        visible_indices = visibility_filter.nonzero(as_tuple=True)[0]
        xyz_visible = gaussians.get_xyz[visible_indices]

    if xyz_visible.shape[0] == 0:
        return torch.zeros_like(gaussians.get_xyz), torch.zeros_like(gaussians.get_rotation), torch.zeros_like(gaussians.get_scaling)

    time_input = viewpoint_cam.fid.unsqueeze(0).expand(xyz_visible.shape[0], -1)
    if eval_mode or is_blender:
        ast_noise = 0
    else:
        jitter_factor = temporal_jitter_factor(opt, iteration, smooth_term)
        ast_noise = torch.randn(1, 1, device=device).expand(xyz_visible.shape[0], -1) * time_interval * jitter_factor
    view_dir = view_dir.unsqueeze(0).expand(xyz_visible.shape[0], -1)
    cam_pos = viewpoint_cam.camera_center.unsqueeze(0).expand(xyz_visible.shape[0], -1)
    cam_pos, view_dir = condition_geometry_inputs(
        cam_pos,
        view_dir,
        getattr(opt, "geometry_conditioning", "view"),
    )

    d_xyz, d_rotation, d_scaling = geo_model.step(xyz_visible.detach(), time_input + ast_noise, cam_pos, view_dir)

    d_xyz_full = torch.zeros_like(gaussians.get_xyz)
    d_xyz_full[visible_indices] = d_xyz
    d_rotation_full = torch.zeros_like(gaussians.get_rotation)
    d_rotation_full[visible_indices] = d_rotation
    d_scaling_full = torch.zeros_like(gaussians.get_scaling)
    d_scaling_full[visible_indices] = d_scaling

    return d_xyz_full, d_rotation_full, d_scaling_full


def apply_temporal_pose_delta(d_xyz_full, gaussians, fid, temporal_pose_model, opt, iteration, scene_extent):
    if (
        temporal_pose_model is None
        or opt is None
        or iteration < getattr(opt, "ir_temporal_pose_start_iter", 0)
    ):
        return d_xyz_full, None, None, None
    pose_delta_xyz, pose_trans, pose_rotvec, pose_reg = temporal_pose_model.step(
        gaussians.get_xyz,
        fid,
        scene_extent,
    )
    if torch.is_tensor(d_xyz_full):
        if d_xyz_full.shape == pose_delta_xyz.shape:
            d_xyz_full = d_xyz_full + pose_delta_xyz
    else:
        d_xyz_full = pose_delta_xyz
    return d_xyz_full, pose_trans, pose_rotvec, pose_reg


def get_eval_temporal_offsets(opt, interval, device, dtype):
    if opt is None or not bool(getattr(opt, "ir_eval_temporal_ensemble", False)):
        return torch.zeros(1, device=device, dtype=dtype)
    samples = int(getattr(opt, "ir_eval_temporal_ensemble_samples", 3) or 1)
    if samples <= 1:
        return torch.zeros(1, device=device, dtype=dtype)
    radius = float(getattr(opt, "ir_eval_temporal_ensemble_radius", 0.5)) * float(interval)
    if radius <= 0.0:
        return torch.zeros(1, device=device, dtype=dtype)
    return torch.linspace(-radius, radius, samples, device=device, dtype=dtype)


def render_eval_view(
    viewpoint,
    gaussians,
    renderFunc,
    renderArgs,
    geo_model,
    opt,
    iteration,
    device,
    is_6dof,
    scene_extent,
    feature_set="rgb",
    eval_temporal_interval=0.0,
    temporal_affine_model=None,
    temporal_pose_model=None,
    radiative_color=None,
    radiative_blend=0.0,
):
    original_fid = viewpoint.fid
    offsets = get_eval_temporal_offsets(
        opt,
        eval_temporal_interval,
        original_fid.device,
        original_fid.dtype,
    )
    rendered = []
    try:
        for offset in offsets:
            sample_fid = (original_fid + offset.reshape_as(original_fid)).clamp(0.0, 1.0)
            viewpoint.fid = sample_fid
            d_xyz_full, d_rotation_full, d_scaling_full = compute_geo_fields(
                viewpoint,
                gaussians,
                geo_model,
                renderArgs[0],
                renderArgs[1],
                opt,
                iteration,
                lambda _: 0.0,
                max(float(eval_temporal_interval), 1e-8),
                device,
                is_blender=True,
                feature_set=feature_set,
                eval_mode=True,
                respect_warmup=True,
            )
            d_xyz_full, _, _, _ = apply_temporal_pose_delta(
                d_xyz_full,
                gaussians,
                sample_fid,
                temporal_pose_model,
                opt,
                iteration,
                scene_extent,
            )
            image = renderFunc(
                viewpoint,
                gaussians,
                *renderArgs,
                d_xyz_full,
                d_rotation_full,
                d_scaling_full,
                device,
                is_6dof,
                radiative_color=radiative_color,
                radiative_blend=radiative_blend,
                feature_set=feature_set,
                opacity_threshold=float(getattr(opt, "ir_eval_opacity_threshold", 0.0)) if opt is not None else 0.0,
            )["render"]
            if (
                temporal_affine_model is not None
                and opt is not None
                and iteration >= getattr(opt, "ir_temporal_affine_start_iter", 0)
            ):
                image, _, _, _ = temporal_affine_model.step(image, sample_fid)
            rendered.append(image)
    finally:
        viewpoint.fid = original_fid
    if len(rendered) == 1:
        return rendered[0]
    return torch.stack(rendered, dim=0).mean(dim=0)


def quantize_like_save_image(image):
    image = torch.clamp(image, 0.0, 1.0)
    return torch.clamp(torch.floor(image * 255.0 + 0.5), 0.0, 255.0) / 255.0


def training(dataset, opt, pipe, testing_iterations, saving_iterations, device):
    start_time = time.time()
    tb_writer, model_path = prepare_output_and_logger(dataset)
    with open(os.path.join(model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**{**vars(dataset), **vars(opt), **vars(pipe)})))
    gaussians = GaussianModel(dataset.sh_degree, device)

    atmos_model = AtmosModel(device)
    atmos_model.train_setting(opt)
    radiometric_model = RadiometricModel(device)
    radiometric_model.train_setting(opt)
    planck_model = PlanckModel(device)
    planck_model.train_setting(opt)
    geometry_time_multires = None if opt.geometry_time_multires < 0 else opt.geometry_time_multires
    geo_model = GeoRefineModel(
        device,
        dataset.is_blender,
        dataset.is_6dof,
        time_multires=geometry_time_multires,
    )
    geo_model.train_setting(opt)

    requested_load_iteration = int(getattr(dataset, "load_iteration", 0) or 0)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=requested_load_iteration if requested_load_iteration != 0 else None,
    )
    gaussians.training_setup(opt, device)
    start_iteration = int(scene.loaded_iter or 0)
    checkpoint_model_path = getattr(dataset, "load_model_path", "") or model_path
    if start_iteration > 0:
        print(
            "[RESUME] loaded_iteration={} continuing_to={} model_path={} checkpoint_path={}".format(
                start_iteration,
                opt.iterations,
                model_path,
                checkpoint_model_path,
            )
        )
        geo_model.load_weights(checkpoint_model_path, iteration=start_iteration)
        atmos_model.load_weights(checkpoint_model_path, iteration=start_iteration)
        planck_model.load_weights(checkpoint_model_path, iteration=start_iteration)
        radiometric_model.load_weights(checkpoint_model_path, iteration=start_iteration)
    has_rgbt = getattr(scene, "has_rgbt", False)
    rgb_geo_model = None
    if has_rgbt:
        rgb_geo_model = GeoRefineModel(
            device,
            dataset.is_blender,
            dataset.is_6dof,
            time_multires=geometry_time_multires,
        )
        rgb_geo_model.train_setting(opt)
        print("Using separate thermal/RGB view-dependent geometry refinement.")
    temporal_affine_model = None
    if opt.ir_temporal_affine:
        train_fids = [float(cam.fid.item()) for cam in scene.getTrainCameras()]
        temporal_affine_model = TemporalAffineModel(
            device,
            max_scale=opt.ir_temporal_affine_max_scale,
            max_bias=opt.ir_temporal_affine_max_bias,
            hidden_dim=opt.ir_temporal_affine_hidden_dim,
            num_freqs=opt.ir_temporal_affine_num_freqs,
            mode=opt.ir_temporal_affine_mode,
            train_fids=train_fids,
            table_smooth_weight=opt.ir_temporal_affine_table_smooth_weight,
            grid_size=opt.ir_temporal_affine_grid_size,
            grid_tv_weight=opt.ir_temporal_affine_grid_tv_weight,
        )
        temporal_affine_model.train_setting(opt)
        if start_iteration > 0:
            try:
                temporal_affine_model.load_weights(checkpoint_model_path, iteration=start_iteration)
            except FileNotFoundError:
                print(
                    "[RESUME] temporal_affine checkpoint missing at iteration {}; "
                    "initializing this optional module from scratch.".format(start_iteration)
                )
        print(
            "Using temporal affine IR appearance correction "
            f"(mode={temporal_affine_model.mode}, train_fids={len(train_fids)})."
        )
    temporal_pose_model = None
    if opt.ir_temporal_pose:
        train_fids = [float(cam.fid.item()) for cam in scene.getTrainCameras()]
        temporal_pose_model = TemporalPoseModel(
            device,
            max_trans=opt.ir_temporal_pose_max_trans,
            max_rot_deg=opt.ir_temporal_pose_max_rot_deg,
            mode=opt.ir_temporal_pose_mode,
            train_fids=train_fids,
            table_smooth_weight=opt.ir_temporal_pose_table_smooth_weight,
        )
        temporal_pose_model.train_setting(opt)
        if start_iteration > 0:
            try:
                temporal_pose_model.load_weights(checkpoint_model_path, iteration=start_iteration)
            except FileNotFoundError:
                print(
                    "[RESUME] temporal_pose checkpoint missing at iteration {}; "
                    "initializing this optional module from scratch.".format(start_iteration)
                )
        print(
            "Using temporal global pose correction "
            f"(mode={temporal_pose_model.mode}, train_fids={len(train_fids)}, "
            f"max_trans={opt.ir_temporal_pose_max_trans:.4f}*extent, "
            f"max_rot={opt.ir_temporal_pose_max_rot_deg:.3f}deg)."
        )
    print(
        "[HYPOTHESIS] detail_loss_weight={:.4f}, rgb_detail_loss_weight={:.4f}, "
        "robust_loss_weight={:.4f}, robust_beta={:.4f}, robust_min_weight={:.3f}, "
        "rmse_loss_weight={:.4f}, trimmed_loss_weight={:.4f}(q={:.3f},mode={},win={}), "
        "multiscale_loss_weight={:.4f}({}/{}), "
        "late_rt_weight={:.4f}@{}, late_ssim_weight={:.4f}, "
        "eval_temporal_ensemble={}({}x radius{}), eval_opacity_threshold={:.4f}, "
        "temporal_affine={}({}), temporal_affine_smooth={}, densify_threshold={}, densify_abs_threshold={}, dual_densify={}, "
        "direction_aware={} abs_split={}, dgc={}, needle_interval={}, screen_weight_power={}, residual_weight_power={}, "
        "residual_weight_q={}, residual_weight_max={}, residual_split_only={}, densify_until={}, "
        "densify_max_points={}, densify_max_growth={}, densify_max_new_points={}, split_first={}, "
        "budget_recycle_points={}, budget_recycle_start={}, budget_recycle_opacity={}, budget_recycle_grad_factor={}, "
        "geo_time_multires={}, geo_conditioning={}, geo_train_until={}, geo_grad_clip={}, temporal_jitter={}*{} floor={}, geo_motion_reg={}, geo_temporal_smooth={}, "
        "temporal_pose={}({}). "
        "Testing whether IR/RGBT weak scenes are limited by edge/detail underfitting "
        "and insufficient high-gradient point allocation; robust loss targets "
        "view-inconsistent reflective/transient residuals. Trimmed RMSE follows SpotLessSplats/RobustNeRF-style "
        "inlier masking to stop transient or badly interpolated pixels from dominating late PSNR optimization. "
        "RMSE and multiscale pyramid losses align the optimized "
        "photometric objective with PSNR/MSE when L1+SSIM under-penalizes broad radiometric or boundary errors. "
        "Direction-aware densification "
        "tests whether weak geometry is caused by conflicting view-space gradients; "
        "residual-weighted densification tests whether FastGS/Taming-style image-space error maps can "
        "move the point budget toward persistent held-out residual structures instead of globally easy regions; "
        "split-only residual weighting tests whether residual maps should guide large-Gaussian refinement "
        "without increasing clone pressure in training-frame-specific error blobs; "
        "split-first budget recycling tests whether capped scenes need to remove low-utility Gaussians "
        "and reserve growth for high-confidence large-Gaussian splits, following AbsGS/Taming/FastGS density-control findings; "
        "needle perturbation tests whether elongated frozen primitives block detail recovery; "
        "geometry regularization tests whether the dynamic refinement MLP is overfitting held-out views. "
        "Geometry conditioning ablates whether camera/view-dependent deformation is incorrectly absorbing IR radiometry or pose residuals; "
        "Lower geometry_time_multires tests whether held-out IR frames need a low-pass temporal deformation field "
        "instead of a high-frequency per-frame correction. "
        "Poly/grid temporal affine tests whether weak IR scenes contain low-frequency sensor/exposure fields "
        "that should be modeled as radiometry instead of being absorbed by geometry. "
        "Temporal affine smoothness tests whether held-out every-8-frame interpolation improves when "
        "nearby fids share exposure fields instead of learning frame-isolated corrections. "
        "Temporal eval ensembling tests whether held-out metrics are limited by high-frequency deformation "
        "interpolation noise rather than by the radiance field itself. "
        "Eval opacity thresholding tests whether low-contribution floaters are hurting held-out views, "
        "following utility-pruning observations in density-control work. "
        "Fixed fid-space temporal jitter tests whether the previous per-stack jitter injected non-physical "
        "time perturbations and hurt interpolation to held-out frames. "
        "A nonzero late jitter floor tests whether every-8 held-out frames fail because the deformation field "
        "is only optimized at exact training fids after the original noise schedule has decayed. "
        "Temporal pose correction tests whether global video/SfM pose drift should be modeled before "
        "local per-Gaussian deformation.".format(
            opt.ir_gradient_loss_weight,
            opt.rgbt_rgb_gradient_loss_weight,
            opt.ir_robust_loss_weight,
            opt.ir_robust_loss_beta,
            opt.ir_robust_loss_min_weight,
            opt.ir_rmse_loss_weight,
            opt.ir_trimmed_loss_weight,
            opt.ir_trimmed_loss_quantile,
            opt.ir_trimmed_loss_mode,
            opt.ir_trimmed_loss_smooth_window,
            opt.ir_multiscale_loss_weight,
            opt.ir_multiscale_loss_mode,
            opt.ir_multiscale_loss_scales,
            opt.ir_late_consistency_weight,
            opt.ir_late_consistency_start_iter,
            opt.ir_late_ssim_weight,
            opt.ir_eval_temporal_ensemble,
            opt.ir_eval_temporal_ensemble_samples,
            opt.ir_eval_temporal_ensemble_radius,
            opt.ir_eval_opacity_threshold,
            opt.ir_temporal_affine,
            "{};grid{};tv{}".format(
                opt.ir_temporal_affine_mode,
                opt.ir_temporal_affine_grid_size,
                opt.ir_temporal_affine_grid_tv_weight,
            ),
            opt.ir_temporal_affine_smooth_weight,
            opt.densify_grad_threshold,
            opt.densify_grad_abs_threshold,
            opt.use_dual_gradient_densification,
            opt.use_direction_aware_densification,
            opt.direction_aware_split_abs,
            opt.use_density_guided_clone,
            opt.needle_perturb_interval,
            opt.densify_screen_weight_power,
            opt.densify_residual_weight_power,
            opt.densify_residual_weight_quantile,
            opt.densify_residual_weight_max,
            opt.densify_residual_weight_split_only,
            opt.densify_until_iter,
            opt.densify_max_points,
            opt.densify_max_growth,
            opt.densify_max_new_points,
            opt.densify_split_first,
            opt.densify_budget_recycle_points,
            opt.densify_budget_recycle_start,
            opt.densify_budget_recycle_opacity,
            opt.densify_budget_recycle_grad_factor,
            geo_model.geometry.t_multires,
            opt.geometry_conditioning,
            opt.geometry_train_until_iter,
            opt.geometry_grad_clip_norm,
            opt.temporal_jitter_mode,
            opt.temporal_jitter_scale,
            opt.temporal_jitter_min_factor,
            opt.geometry_motion_reg_weight,
            opt.geometry_temporal_smooth_weight,
            opt.ir_temporal_pose,
            opt.ir_temporal_pose_mode,
        )
    )
    print(
        "[SPARSE_DEPTH_HYPOTHESIS] load_sparse_depth={}, weight={:.5f}@{}+{}, "
        "min_acc={:.3f}, huber_beta={:.4f}, acc_normalize={}, scale_factor={}. "
        "Testing a DepthRegularizedGS-style COLMAP sparse-depth anchor: weak IR scenes may lose held-out PSNR "
        "because texture-poor thermal frames let geometry drift or allocate opacity to view-specific floaters; "
        "sparse camera-space depth should constrain broad geometry without using test images.".format(
            getattr(dataset, "load_sparse_depth", False),
            opt.ir_sparse_depth_weight,
            opt.ir_sparse_depth_start_iter,
            opt.ir_sparse_depth_ramp_iters,
            opt.ir_sparse_depth_min_acc,
            opt.ir_sparse_depth_huber_beta,
            opt.ir_sparse_depth_use_acc_normalize,
            opt.ir_sparse_depth_scale_factor,
        )
    )

    def save_training_state(iteration_to_save):
        scene.save(iteration_to_save)
        geo_model.save_weights(model_path, iteration_to_save)
        if rgb_geo_model is not None:
            rgb_geo_model.save_weights(model_path, iteration_to_save, subdir="GeometryRGB")
        atmos_model.save_weights(model_path, iteration_to_save)
        planck_model.save_weights(model_path, iteration_to_save)
        radiometric_model.save_weights(model_path, iteration_to_save)
        if temporal_affine_model is not None:
            temporal_affine_model.save_weights(model_path, iteration_to_save)
        if temporal_pose_model is not None:
            temporal_pose_model.save_weights(model_path, iteration_to_save)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32).to(device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(
        total=max(opt.iterations - start_iteration, 0),
        desc="Training progress",
    )
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    prev_probe_point_count = gaussians.get_xyz.shape[0]
    densify_initial_points = int(getattr(gaussians, "og_number_points", 0) or gaussians.get_xyz.shape[0])
    train_cameras = scene.getTrainCameras()
    fixed_time_interval = estimate_temporal_jitter_interval(train_cameras, opt.temporal_jitter_mode)
    if fixed_time_interval is None:
        print(
            "[TIME-JITTER] mode=stack scale={:.4f}: using legacy per-stack remaining camera count.".format(
                opt.temporal_jitter_scale,
            )
        )
    else:
        print(
            "[TIME-JITTER] mode={} interval={:.8f} scale={:.4f}: using fixed fid-space temporal noise.".format(
                opt.temporal_jitter_mode,
                fixed_time_interval,
                opt.temporal_jitter_scale,
            )
        )
    # for iteration in range(1, opt.iterations + 1):
    for iteration in range(start_iteration + 1, opt.iterations + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        if fixed_time_interval is None:
            time_interval = 1 / max(total_frame, 1)
        else:
            time_interval = fixed_time_interval * opt.temporal_jitter_scale

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
        geo_motion = None
        geo_motion_reg = None
        geo_smooth_reg = None
        geo_scaling_abs = None
        geo_rotation_abs = None
        pose_trans = None
        pose_rotvec = None
        pose_reg = None
        if iteration < opt.warm_up:
            d_xyz_full, d_rotation_full, d_scaling_full = 0.0, 0.0, 0.0
        else:
            R = viewpoint_cam.R
            T = viewpoint_cam.T
            R_torch = torch.from_numpy(R).float().to(device)  
            T_torch = torch.from_numpy(T).float().to(device)

            Rt = torch.eye(4, device=device)
            Rt[:3, :3] = R_torch.t()  
            Rt[:3, 3] = -R_torch.t() @ T_torch       

            c2w = Rt
            view_dir = -c2w[:3, 2]       

            with torch.no_grad():
                # Compute visibility_filter using rasterizer directly
                tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
                tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)
                raster_settings = GaussianRasterizationSettings(
                    image_height=viewpoint_cam.image_height,
                    image_width=viewpoint_cam.image_width,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=background,
                    scale_modifier=1.0,
                    viewmatrix=viewpoint_cam.world_view_transform,
                    projmatrix=viewpoint_cam.full_proj_transform,
                    sh_degree=gaussians.active_sh_degree,
                    campos=viewpoint_cam.camera_center,
                    prefiltered=False,
                    debug=pipe.debug
                )
                screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, device=device)
                screenspace_points_densify = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, device=device)
                opacity = gaussians.get_opacity
                scales = gaussians.get_scaling
                rotations = gaussians.get_rotation
                shs = gaussians.get_thermal_features if has_rgbt else gaussians.get_features
            
                rasterizer = GaussianRasterizer(raster_settings)
                _, radii, _ = rasterizer(
                    means3D=gaussians.get_xyz,
                    means2D=screenspace_points,
                    means2D_densify=screenspace_points_densify,
                    shs=shs,
                    colors_precomp=None,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None
                )

                visibility_filter = radii > 0
                visible_indices = visibility_filter.nonzero(as_tuple=True)[0]
                xyz_visible = gaussians.get_xyz[visible_indices]
            time_input = fid.unsqueeze(0).expand(xyz_visible.shape[0], -1)
            jitter_factor = temporal_jitter_factor(opt, iteration, smooth_term)
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device=device).expand(xyz_visible.shape[0], -1) * time_interval * jitter_factor
            view_dir = view_dir.unsqueeze(0).expand(xyz_visible.shape[0], -1)  # -> (N, 3)
            cam_pos  = viewpoint_cam.camera_center.unsqueeze(0).expand(xyz_visible.shape[0], -1)   # -> (N, 3)
            cam_pos, view_dir = condition_geometry_inputs(
                cam_pos,
                view_dir,
                getattr(opt, "geometry_conditioning", "view"),
            )

            d_xyz, d_rotation, d_scaling = geo_model.step(xyz_visible.detach(), time_input + ast_noise, cam_pos, view_dir)
            geo_motion = torch.linalg.norm(d_xyz, dim=-1)
            geo_scaling_abs = d_scaling.abs().mean()
            geo_rotation_abs = d_rotation.abs().mean()

            d_xyz_full = torch.zeros_like(gaussians.get_xyz)  # shape (N, 3)
            d_xyz_full[visible_indices] = d_xyz
            d_rotation_full = torch.zeros_like(gaussians.get_rotation)  # shape (N, 3)
            d_rotation_full[visible_indices] = d_rotation
            d_scaling_full = torch.zeros_like(gaussians.get_scaling)  # shape (N, 3)
            d_scaling_full[visible_indices] = d_scaling

        d_xyz_full, pose_trans, pose_rotvec, pose_reg = apply_temporal_pose_delta(
            d_xyz_full,
            gaussians,
            fid,
            temporal_pose_model,
            opt,
            iteration,
            scene.cameras_extent,
        )

        # Render
        radiative_blend = (
            physir_render_blend(iteration, opt)
            if opt.phys_main_render and gaussians.has_radiative_transfer_attributes()
            else 0.0
        )
        radiative_color = gaussians.get_physir_response_rgb if radiative_blend > 0.0 else None
        render_pkg_re = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            d_xyz_full,
            d_rotation_full,
            d_scaling_full,
            device,
            dataset.is_6dof,
            radiative_color=radiative_color,
            radiative_blend=radiative_blend,
            feature_set="thermal" if has_rgbt else "rgb",
        )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        temporal_scale_delta = None
        temporal_bias = None
        temporal_reg = None
        temporal_affine_smooth_reg = None
        temporal_affine_smooth_weight = 0.0
        if temporal_affine_model is not None and iteration >= opt.ir_temporal_affine_start_iter:
            affine_input_image = image
            image, temporal_scale_delta, temporal_bias, temporal_reg = temporal_affine_model.step(
                affine_input_image,
                fid,
            )
            temporal_affine_smooth_weight = ramped_weight(
                getattr(opt, "ir_temporal_affine_smooth_weight", 0.0),
                iteration,
                getattr(opt, "ir_temporal_affine_smooth_start_iter", opt.ir_temporal_affine_start_iter),
                getattr(opt, "ir_temporal_affine_smooth_ramp_iters", 1000),
            )
            if temporal_affine_smooth_weight > 0.0:
                temporal_affine_smooth_reg = temporal_affine_model.smoothness_reg(
                    affine_input_image,
                    fid,
                    max(float(time_interval), 1e-8),
                )

        gt_image = viewpoint_cam.original_image.to(device)
        
        # Loss
        Ll1 = l1_loss(image, gt_image)
        robust_weight = ramped_weight(
            opt.ir_robust_loss_weight,
            iteration,
            opt.ir_robust_loss_start_iter,
            opt.ir_robust_loss_ramp_iters,
        )
        robust_l1 = None
        robust_weight_mean = None
        robust_low_fraction = None
        effective_l1 = Ll1
        if robust_weight > 0.0:
            robust_l1, robust_weight_mean, robust_low_fraction = residual_robust_l1_loss(
                image,
                gt_image,
                beta=opt.ir_robust_loss_beta,
                min_weight=opt.ir_robust_loss_min_weight,
            )
            effective_l1 = (1.0 - robust_weight) * Ll1 + robust_weight * robust_l1
        rmse_loss = torch.sqrt(((image - gt_image) ** 2).mean().clamp_min(opt.ir_rmse_loss_eps))
        rmse_weight = ramped_weight(
            opt.ir_rmse_loss_weight,
            iteration,
            opt.ir_rmse_loss_start_iter,
            opt.ir_rmse_loss_ramp_iters,
        )
        photo_loss = effective_l1
        if rmse_weight > 0.0:
            photo_loss = (1.0 - rmse_weight) * effective_l1 + rmse_weight * rmse_loss
        trimmed_loss = None
        trimmed_keep_fraction = None
        trimmed_threshold = None
        trimmed_weight = ramped_weight(
            opt.ir_trimmed_loss_weight,
            iteration,
            opt.ir_trimmed_loss_start_iter,
            opt.ir_trimmed_loss_ramp_iters,
        )
        if trimmed_weight > 0.0:
            trimmed_loss, trimmed_keep_fraction, trimmed_threshold = trimmed_residual_loss(
                image,
                gt_image,
                quantile=opt.ir_trimmed_loss_quantile,
                mode=opt.ir_trimmed_loss_mode,
                eps=opt.ir_rmse_loss_eps,
                smooth_window=opt.ir_trimmed_loss_smooth_window,
            )
            photo_loss = (1.0 - trimmed_weight) * photo_loss + trimmed_weight * trimmed_loss
        multiscale_loss = multiscale_image_loss(
            image,
            gt_image,
            opt.ir_multiscale_loss_scales,
            mode=opt.ir_multiscale_loss_mode,
            eps=opt.ir_rmse_loss_eps,
        )
        multiscale_weight = ramped_weight(
            opt.ir_multiscale_loss_weight,
            iteration,
            opt.ir_multiscale_loss_start_iter,
            opt.ir_multiscale_loss_ramp_iters,
        )
        tir_consistency_loss = image.new_tensor(0.0)
        late_consistency_weight = float(getattr(opt, "ir_late_consistency_weight", 0.2))
        late_ssim_weight = float(getattr(opt, "ir_late_ssim_weight", opt.lambda_dssim))
        late_start_iter = int(getattr(opt, "ir_late_consistency_start_iter", 20000))
        if iteration <= late_start_iter or late_consistency_weight <= 0.0:
            loss = (1.0 - opt.lambda_dssim) * photo_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            tir_consistency_loss = compute_rt_consistency_loss(
                image,
                gt_image,
                atmos_model,
                planck_model,
                radiometric_model,
                device,
            )

            photo_weight = max(0.0, 1.0 - late_ssim_weight - late_consistency_weight)
            loss = (
                photo_weight * photo_loss
                + late_ssim_weight * (1.0 - ssim(image, gt_image))
                + late_consistency_weight * tir_consistency_loss
            )
        if multiscale_weight > 0.0:
            loss = loss + multiscale_weight * multiscale_loss

        detail_weight = ramped_weight(
            opt.ir_gradient_loss_weight,
            iteration,
            opt.ir_gradient_loss_start_iter,
            opt.ir_gradient_loss_ramp_iters,
        )
        detail_loss = None
        if detail_weight > 0.0:
            detail_loss = gradient_l1_loss(image, gt_image)
            loss = loss + detail_weight * detail_loss

        sparse_depth_loss = image.new_tensor(0.0)
        sparse_depth_stats = {
            "valid": 0,
            "total": 0,
            "acc_mean": 0.0,
            "acc_p90": 0.0,
            "acc_max": 0.0,
            "pred_median": 0.0,
            "gt_median": 0.0,
            "scale": max(1, int(getattr(opt, "ir_sparse_depth_scale_factor", 1))),
            "oom": 0,
        }
        sparse_depth_weight = ramped_weight(
            opt.ir_sparse_depth_weight,
            iteration,
            opt.ir_sparse_depth_start_iter,
            opt.ir_sparse_depth_ramp_iters,
        )
        if sparse_depth_weight > 0.0:
            sparse_depth_loss, sparse_depth_stats = sparse_depth_supervision_loss(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                d_xyz_full,
                d_rotation_full,
                d_scaling_full,
                device,
                dataset.is_6dof,
                "thermal" if has_rgbt else "rgb",
                opt,
            )
            if sparse_depth_stats["valid"] > 0:
                loss = loss + sparse_depth_weight * sparse_depth_loss

        rgb_loss = None
        rgb_detail_loss = None
        rgb_render_pkg = None
        if has_rgbt and viewpoint_cam.original_rgb_image is not None and opt.rgbt_rgb_loss_weight > 0.0:
            rgb_d_xyz_full, rgb_d_rotation_full, rgb_d_scaling_full = compute_geo_fields(
                viewpoint_cam,
                gaussians,
                rgb_geo_model,
                pipe,
                background,
                opt,
                iteration,
                smooth_term,
                time_interval,
                device,
                dataset.is_blender,
                feature_set="rgb",
            )
            rgb_render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                rgb_d_xyz_full,
                rgb_d_rotation_full,
                rgb_d_scaling_full,
                device,
                dataset.is_6dof,
                feature_set="rgb",
                detach_geometry=bool(getattr(opt, "rgbt_rgb_detach_geometry", 0)),
            )
            rgb_image = rgb_render_pkg["render"]
            gt_rgb_image = viewpoint_cam.original_rgb_image.to(device)
            rgb_l1 = l1_loss(rgb_image, gt_rgb_image)
            rgb_loss = (1.0 - opt.lambda_dssim) * rgb_l1 + opt.lambda_dssim * (1.0 - ssim(rgb_image, gt_rgb_image))
            loss = loss + opt.rgbt_rgb_loss_weight * rgb_loss
            rgb_detail_weight = ramped_weight(
                opt.rgbt_rgb_gradient_loss_weight,
                iteration,
                opt.ir_gradient_loss_start_iter,
                opt.ir_gradient_loss_ramp_iters,
            )
            if rgb_detail_weight > 0.0:
                rgb_detail_loss = gradient_l1_loss(rgb_image, gt_rgb_image)
                loss = loss + rgb_detail_weight * rgb_detail_loss

        physics_loss, _ = rt_guidance_loss(gaussians, opt, iteration)
        loss = loss + physics_loss
        if temporal_reg is not None and opt.ir_temporal_affine_reg_weight > 0.0:
            loss = loss + opt.ir_temporal_affine_reg_weight * temporal_reg
        if temporal_affine_smooth_reg is not None and temporal_affine_smooth_weight > 0.0:
            loss = loss + temporal_affine_smooth_weight * temporal_affine_smooth_reg
        if pose_reg is not None and opt.ir_temporal_pose_reg_weight > 0.0:
            loss = loss + opt.ir_temporal_pose_reg_weight * pose_reg
        geometry_motion_weight = ramped_weight(
            opt.geometry_motion_reg_weight,
            iteration,
            opt.geometry_motion_reg_start_iter,
            opt.geometry_motion_reg_ramp_iters,
        )
        if geometry_motion_weight > 0.0 and geo_motion is not None:
            extent = max(float(scene.cameras_extent), 1e-6)
            geo_motion_reg = (
                (d_xyz / extent).pow(2).mean()
                + 0.01 * d_scaling.pow(2).mean()
                + 0.01 * d_rotation.pow(2).mean()
            )
            loss = loss + geometry_motion_weight * geo_motion_reg
        geometry_smooth_weight = ramped_weight(
            opt.geometry_temporal_smooth_weight,
            iteration,
            opt.geometry_temporal_smooth_start_iter,
            opt.geometry_temporal_smooth_ramp_iters,
        )
        if geometry_smooth_weight > 0.0 and geo_motion is not None:
            smooth_jitter = torch.randn_like(time_input) * time_interval
            d_xyz_s, d_rotation_s, d_scaling_s = geo_model.step(
                xyz_visible.detach(),
                (time_input + smooth_jitter).clamp(0.0, 1.0),
                cam_pos,
                view_dir,
            )
            geo_smooth_reg = (
                (d_xyz - d_xyz_s).pow(2).mean()
                + 0.01 * (d_scaling - d_scaling_s).pow(2).mean()
                + 0.01 * (d_rotation - d_rotation_s).pow(2).mean()
            )
            loss = loss + geometry_smooth_weight * geo_smooth_reg

        if tir_auxiliary_gate(iteration, opt) > 0.0 and gaussians.has_radiative_transfer_attributes():
            with torch.no_grad():
                confidence_color = radiative_confidence_values(gaussians, opt).repeat(1, 3).contiguous()
                confidence_image = torch.clamp(
                    render(
                        viewpoint_cam,
                        gaussians,
                        pipe,
                        background,
                        d_xyz_full,
                        d_rotation_full,
                        d_scaling_full,
                        device,
                        dataset.is_6dof,
                        override_color=confidence_color,
                        feature_set="thermal" if has_rgbt else "rgb",
                    )["render"],
                    0.0,
                    1.0,
                )
            physics_aux_loss, _ = radiative_confidence_mse_loss(
                image, gt_image, confidence_image, opt, iteration
            )
            loss = loss + physics_aux_loss
        
        loss.backward()

        iter_end.record()

        residual_densify_weight = None
        residual_weight_stats = {
            "residual_weight_mean": 1.0,
            "residual_weight_p90": 1.0,
            "residual_weight_max": 1.0,
            "residual_sample_mean": 0.0,
            "residual_sample_p90": 0.0,
        }
        if (
            iteration < opt.densify_until_iter
            and iteration >= getattr(opt, "densify_residual_weight_start_iter", 3000)
            and getattr(opt, "densify_residual_weight_power", 0.0) > 0.0
        ):
            means3D_for_residual = gaussians.get_xyz.detach()
            if torch.is_tensor(d_xyz_full) and d_xyz_full.shape == gaussians.get_xyz.shape:
                means3D_for_residual = means3D_for_residual + d_xyz_full.detach()
            _, residual_densify_weight, residual_weight_stats = compute_residual_densify_weight(
                image,
                gt_image,
                means3D_for_residual,
                visibility_filter,
                viewpoint_cam,
                opt,
            )

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if opt.diagnostic_log_every > 0 and iteration % opt.diagnostic_log_every == 0:
                with torch.no_grad():
                    pred_grad_mean = float(image_gradient_mean(image).item())
                    gt_grad_mean = float(image_gradient_mean(gt_image).item())
                    grad_ratio = pred_grad_mean / max(gt_grad_mean, 1e-8)
                    visible_points = int(visibility_filter.sum().item())
                    radii_mean = float(radii[visibility_filter].float().mean().item()) if visible_points > 0 else 0.0
                    detail_loss_value = float(detail_loss.detach().item()) if detail_loss is not None else 0.0
                    rgb_detail_loss_value = float(rgb_detail_loss.detach().item()) if rgb_detail_loss is not None else 0.0
                    robust_l1_value = float(robust_l1.detach().item()) if robust_l1 is not None else 0.0
                    robust_weight_mean_value = float(robust_weight_mean.detach().item()) if robust_weight_mean is not None else 1.0
                    robust_low_fraction_value = float(robust_low_fraction.detach().item()) if robust_low_fraction is not None else 0.0
                    trimmed_loss_value = float(trimmed_loss.detach().item()) if trimmed_loss is not None else 0.0
                    trimmed_keep_value = float(trimmed_keep_fraction.detach().item()) if trimmed_keep_fraction is not None else 1.0
                    trimmed_threshold_value = float(trimmed_threshold.detach().item()) if trimmed_threshold is not None else 0.0
                    temporal_scale_value = float(temporal_scale_delta.mean().item()) if temporal_scale_delta is not None else 0.0
                    temporal_bias_value = float(temporal_bias.mean().item()) if temporal_bias is not None else 0.0
                    temporal_affine_smooth_value = (
                        float(temporal_affine_smooth_reg.detach().item())
                        if temporal_affine_smooth_reg is not None
                        else 0.0
                    )
                    temporal_table_stats = (
                        temporal_affine_model.diagnostic_stats()
                        if temporal_affine_model is not None
                        else {}
                    )
                    pose_stats = (
                        temporal_pose_model.diagnostic_stats(scene.cameras_extent)
                        if temporal_pose_model is not None
                        else {}
                    )
                    pose_reg_value = float(pose_reg.detach().item()) if pose_reg is not None else 0.0
                    geo_mean_value = float(geo_motion.mean().item()) if geo_motion is not None else 0.0
                    geo_p95_value = (
                        float(torch.quantile(geo_motion.detach(), 0.95).item())
                        if geo_motion is not None and geo_motion.numel() > 0
                        else 0.0
                    )
                    geo_scale_value = float(geo_scaling_abs.detach().item()) if geo_scaling_abs is not None else 0.0
                    geo_rot_value = float(geo_rotation_abs.detach().item()) if geo_rotation_abs is not None else 0.0
                    geo_reg_value = float(geo_motion_reg.detach().item()) if geo_motion_reg is not None else 0.0
                    geo_smooth_value = float(geo_smooth_reg.detach().item()) if geo_smooth_reg is not None else 0.0
                    current_jitter_std = (
                        float(time_interval * temporal_jitter_factor(opt, iteration, smooth_term))
                        if not dataset.is_blender and iteration >= opt.warm_up
                        else 0.0
                    )
                    diag_msg = (
                        "\n[DIAG][iter {iter}] loss={loss:.6f} l1={l1:.6f} "
                        "rmse_w={rmse_w:.5f} rmse={rmse:.6f} "
                        "trim_w={trim_w:.5f} trim={trim:.6f} trim_keep={trim_keep:.3f} trim_thr={trim_thr:.6f} "
                        "ms_w={ms_w:.5f} ms={ms:.6f} "
                        "depth_w={depth_w:.5f} depth={depth:.6f} depth_scale={depth_scale} depth_oom={depth_oom} depth_valid={depth_valid}/{depth_total} "
                        "depth_acc={depth_acc:.6f} depth_acc_p90={depth_acc_p90:.6f} depth_acc_max={depth_acc_max:.6f} "
                        "depth_pred_med={depth_pred_med:.4f} depth_gt_med={depth_gt_med:.4f} "
                        "rt_w={rt_w:.5f} rt={rt:.6f} "
                        "robust_w={robust_w:.5f} robust_l1={robust_l1:.6f} "
                        "robust_mean={robust_mean:.3f} robust_low={robust_low:.3f} "
                        "temp_scale={temp_scale:.5f} temp_bias={temp_bias:.5f} "
                        "temp_smooth_w={temp_smooth_w:.5f} temp_smooth={temp_smooth:.6f} "
                        "temp_tbl_s_std={temp_tbl_s_std:.5f} temp_tbl_s_rng=[{temp_tbl_s_min:.5f},{temp_tbl_s_max:.5f}] "
                        "temp_tbl_b_std={temp_tbl_b_std:.5f} temp_tbl_b_rng=[{temp_tbl_b_min:.5f},{temp_tbl_b_max:.5f}] "
                        "temp_poly_s_std={temp_poly_s_std:.5f} temp_poly_s_rng=[{temp_poly_s_min:.5f},{temp_poly_s_max:.5f}] "
                        "temp_poly_b_std={temp_poly_b_std:.5f} temp_poly_b_rng=[{temp_poly_b_min:.5f},{temp_poly_b_max:.5f}] "
                        "temp_tblpoly_s_coeff_std={temp_tblpoly_s_coeff_std:.5f} temp_tblpoly_s_coeff_max={temp_tblpoly_s_coeff_max:.5f} "
                        "temp_tblpoly_b_coeff_std={temp_tblpoly_b_coeff_std:.5f} temp_tblpoly_b_coeff_max={temp_tblpoly_b_coeff_max:.5f} "
                        "temp_grid_s_std={temp_grid_s_std:.5f} temp_grid_s_rng=[{temp_grid_s_min:.5f},{temp_grid_s_max:.5f}] "
                        "temp_grid_b_std={temp_grid_b_std:.5f} temp_grid_b_rng=[{temp_grid_b_min:.5f},{temp_grid_b_max:.5f}] "
                        "temp_grid_tv={temp_grid_tv:.6f} "
                        "temp_tblgrid_s_std={temp_tblgrid_s_std:.5f} temp_tblgrid_s_max={temp_tblgrid_s_max:.5f} "
                        "temp_tblgrid_b_std={temp_tblgrid_b_std:.5f} temp_tblgrid_b_max={temp_tblgrid_b_max:.5f} "
                        "temp_tblgrid_dt={temp_tblgrid_dt:.6f} "
                        "detail_w={detail_w:.5f} detail={detail:.6f} "
                        "geo_mean={geo_mean:.6f} geo_p95={geo_p95:.6f} "
                        "geo_scale={geo_scale:.6f} geo_rot={geo_rot:.6f} "
                        "geo_reg={geo_reg:.6f} geo_smooth={geo_smooth:.6f} "
                        "pose_trans={pose_trans:.6f} pose_rot_deg={pose_rot:.6f} pose_reg={pose_reg:.6f} "
                        "pose_tbl_t_std={pose_tbl_t_std:.6f} pose_tbl_t_max={pose_tbl_t_max:.6f} "
                        "pose_tbl_r_std={pose_tbl_r_std:.6f} pose_tbl_r_max={pose_tbl_r_max:.6f} "
                        "time_interval={time_interval:.8f} jitter_std={jitter_std:.8f} "
                        "pred_grad={pred_grad:.6f} gt_grad={gt_grad:.6f} grad_ratio={ratio:.3f} "
                        "resid_w_mean={resid_w_mean:.3f} resid_w_p90={resid_w_p90:.3f} "
                        "resid_w_max={resid_w_max:.3f} resid_s_mean={resid_s_mean:.6f} resid_s_p90={resid_s_p90:.6f} "
                        "rgb_detail={rgb_detail:.6f} points={points} visible={visible} radii_mean={radii:.3f}"
                    ).format(
                        iter=iteration,
                        loss=float(loss.detach().item()),
                        l1=float(Ll1.detach().item()),
                        rmse_w=rmse_weight,
                        rmse=float(rmse_loss.detach().item()),
                        trim_w=trimmed_weight,
                        trim=trimmed_loss_value,
                        trim_keep=trimmed_keep_value,
                        trim_thr=trimmed_threshold_value,
                        ms_w=multiscale_weight,
                        ms=float(multiscale_loss.detach().item()),
                        depth_w=sparse_depth_weight,
                        depth=float(sparse_depth_loss.detach().item()),
                        depth_scale=sparse_depth_stats.get("scale", 1),
                        depth_oom=sparse_depth_stats.get("oom", 0),
                        depth_valid=sparse_depth_stats.get("valid", 0),
                        depth_total=sparse_depth_stats.get("total", 0),
                        depth_acc=sparse_depth_stats.get("acc_mean", 0.0),
                        depth_acc_p90=sparse_depth_stats.get("acc_p90", 0.0),
                        depth_acc_max=sparse_depth_stats.get("acc_max", 0.0),
                        depth_pred_med=sparse_depth_stats.get("pred_median", 0.0),
                        depth_gt_med=sparse_depth_stats.get("gt_median", 0.0),
                        rt_w=late_consistency_weight if iteration > late_start_iter else 0.0,
                        rt=float(tir_consistency_loss.detach().item()),
                        robust_w=robust_weight,
                        robust_l1=robust_l1_value,
                        robust_mean=robust_weight_mean_value,
                        robust_low=robust_low_fraction_value,
                        temp_scale=temporal_scale_value,
                        temp_bias=temporal_bias_value,
                        temp_smooth_w=temporal_affine_smooth_weight,
                        temp_smooth=temporal_affine_smooth_value,
                        temp_tbl_s_std=temporal_table_stats.get("table_scale_std", 0.0),
                        temp_tbl_s_min=temporal_table_stats.get("table_scale_min", 0.0),
                        temp_tbl_s_max=temporal_table_stats.get("table_scale_max", 0.0),
                        temp_tbl_b_std=temporal_table_stats.get("table_bias_std", 0.0),
                        temp_tbl_b_min=temporal_table_stats.get("table_bias_min", 0.0),
                        temp_tbl_b_max=temporal_table_stats.get("table_bias_max", 0.0),
                        temp_poly_s_std=temporal_table_stats.get("poly_scale_std", 0.0),
                        temp_poly_s_min=temporal_table_stats.get("poly_scale_min", 0.0),
                        temp_poly_s_max=temporal_table_stats.get("poly_scale_max", 0.0),
                        temp_poly_b_std=temporal_table_stats.get("poly_bias_std", 0.0),
                        temp_poly_b_min=temporal_table_stats.get("poly_bias_min", 0.0),
                        temp_poly_b_max=temporal_table_stats.get("poly_bias_max", 0.0),
                        temp_tblpoly_s_coeff_std=temporal_table_stats.get("table_poly_scale_coeff_std", 0.0),
                        temp_tblpoly_s_coeff_max=temporal_table_stats.get("table_poly_scale_coeff_max", 0.0),
                        temp_tblpoly_b_coeff_std=temporal_table_stats.get("table_poly_bias_coeff_std", 0.0),
                        temp_tblpoly_b_coeff_max=temporal_table_stats.get("table_poly_bias_coeff_max", 0.0),
                        temp_grid_s_std=temporal_table_stats.get("grid_scale_std", 0.0),
                        temp_grid_s_min=temporal_table_stats.get("grid_scale_min", 0.0),
                        temp_grid_s_max=temporal_table_stats.get("grid_scale_max", 0.0),
                        temp_grid_b_std=temporal_table_stats.get("grid_bias_std", 0.0),
                        temp_grid_b_min=temporal_table_stats.get("grid_bias_min", 0.0),
                        temp_grid_b_max=temporal_table_stats.get("grid_bias_max", 0.0),
                        temp_grid_tv=temporal_table_stats.get("grid_tv", 0.0),
                        temp_tblgrid_s_std=temporal_table_stats.get("table_grid_scale_std", 0.0),
                        temp_tblgrid_s_max=temporal_table_stats.get("table_grid_scale_max", 0.0),
                        temp_tblgrid_b_std=temporal_table_stats.get("table_grid_bias_std", 0.0),
                        temp_tblgrid_b_max=temporal_table_stats.get("table_grid_bias_max", 0.0),
                        temp_tblgrid_dt=temporal_table_stats.get("table_grid_temporal_delta", 0.0),
                        detail_w=detail_weight,
                        detail=detail_loss_value,
                        geo_mean=geo_mean_value,
                        geo_p95=geo_p95_value,
                        geo_scale=geo_scale_value,
                        geo_rot=geo_rot_value,
                        geo_reg=geo_reg_value,
                        geo_smooth=geo_smooth_value,
                        pose_trans=pose_stats.get("trans_norm", 0.0),
                        pose_rot=pose_stats.get("rot_deg", 0.0),
                        pose_reg=pose_reg_value,
                        pose_tbl_t_std=pose_stats.get("table_trans_std", 0.0),
                        pose_tbl_t_max=pose_stats.get("table_trans_max", 0.0),
                        pose_tbl_r_std=pose_stats.get("table_rot_std", 0.0),
                        pose_tbl_r_max=pose_stats.get("table_rot_max", 0.0),
                        time_interval=float(time_interval),
                        jitter_std=current_jitter_std,
                        pred_grad=pred_grad_mean,
                        gt_grad=gt_grad_mean,
                        ratio=grad_ratio,
                        resid_w_mean=residual_weight_stats.get("residual_weight_mean", 1.0),
                        resid_w_p90=residual_weight_stats.get("residual_weight_p90", 1.0),
                        resid_w_max=residual_weight_stats.get("residual_weight_max", 1.0),
                        resid_s_mean=residual_weight_stats.get("residual_sample_mean", 0.0),
                        resid_s_p90=residual_weight_stats.get("residual_sample_p90", 0.0),
                        rgb_detail=rgb_detail_loss_value,
                        points=int(gaussians.get_xyz.shape[0]),
                        visible=visible_points,
                        radii=radii_mean,
                    )
                    print(diag_msg)
                    with open(os.path.join(model_path, "result.txt"), "a") as result_file:
                        result_file.write(diag_msg + "\n")

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(dataset.source_path, model_path, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), geo_model,
                                       dataset.load2gpu_on_the_fly, device, dataset.is_6dof, start_time=start_time,
                                       opt=opt, smooth_term=smooth_term, rgb_geo_model=rgb_geo_model,
                                       temporal_affine_model=temporal_affine_model,
                                       temporal_pose_model=temporal_pose_model)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration
                    if has_rgbt and getattr(opt, "rgbt_save_best_eval_checkpoint", False):
                        best_mirror_iteration = int(opt.iterations) + 1
                        print(
                            "[RGBT_BEST] thermal eval improved to {:.4f} at iter {}; "
                            "mirroring checkpoint to iteration_{} for default RGBT rendering.".format(
                                float(best_psnr),
                                int(best_iteration),
                                best_mirror_iteration,
                            )
                        )
                        save_training_state(best_mirror_iteration)

            if opt.phys_probe_every > 0 and iteration % opt.phys_probe_every == 0:
                fit_and_log_physir_attributes(iteration, gaussians, scene, opt, model_path, prev_probe_point_count)
                prev_probe_point_count = gaussians.get_xyz.shape[0]
                # Detailed radiative guidance and auxiliary-loss diagnostics are
                # intentionally kept out of the console/result.txt logs. Full
                # PhysIR statistics are still stored in phys_probe/probe_log.jsonl.

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                save_training_state(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                densify_weight = 1.0
                densify_denom_weight = 1.0
                densify_abs_weight = None
                if opt.densify_screen_weight_power > 0.0:
                    densify_weight = radii[visibility_filter].float().clamp(
                        min=1.0,
                        max=opt.densify_screen_weight_max,
                    ).pow(opt.densify_screen_weight_power)
                    densify_denom_weight = densify_weight
                if residual_densify_weight is not None:
                    # Unlike screen-size weighting, this residual term is deliberately
                    # not put into the denominator. Persistent high-error projected
                    # regions should raise a Gaussian's average densification score.
                    if getattr(opt, "densify_residual_weight_split_only", False):
                        densify_abs_weight = densify_weight * residual_densify_weight
                    else:
                        densify_weight = densify_weight * residual_densify_weight
                gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    visibility_filter,
                    weight=densify_weight,
                    abs_viewspace_point_tensor=viewspace_point_tensor_densify,
                    denom_weight=densify_denom_weight,
                    abs_weight=densify_abs_weight,
                )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    densify_max_total_points = int(opt.densify_max_points) if opt.densify_max_points > 0 else 0
                    if opt.densify_max_growth > 0.0 and densify_initial_points > 0:
                        growth_cap = int(densify_initial_points * opt.densify_max_growth)
                        densify_max_total_points = (
                            min(densify_max_total_points, growth_cap)
                            if densify_max_total_points > 0 else growth_cap
                        )
                    densify_stats = gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        device,
                        max_grad_abs=opt.densify_grad_abs_threshold,
                        use_dual_gradient=opt.use_dual_gradient_densification,
                        use_direction_aware=opt.use_direction_aware_densification,
                        direction_aware_base=opt.direction_aware_base,
                        direction_aware_scale=opt.direction_aware_scale,
                        direction_aware_power=opt.direction_aware_power,
                        direction_aware_split_abs=opt.direction_aware_split_abs,
                        density_guided_clone=opt.use_density_guided_clone,
                        density_guided_clone_scale=opt.density_guided_clone_scale,
                        max_new_points=opt.densify_max_new_points,
                        max_total_points=densify_max_total_points,
                        split_first=opt.densify_split_first,
                        budget_recycle_points=opt.densify_budget_recycle_points,
                        budget_recycle_start=opt.densify_budget_recycle_start,
                        budget_recycle_opacity=opt.densify_budget_recycle_opacity,
                        budget_recycle_grad_factor=opt.densify_budget_recycle_grad_factor,
                    )
                    if opt.densify_log_every > 0 and iteration % opt.densify_log_every == 0:
                        densify_msg = (
                            "\n[DENSIFY][iter {iter}] mode={mode} before={before} cloned={cloned} "
                            "split_added={split_added} pruned={pruned} after={after} "
                            "clone_thr={thr:.7f} clone_over={over_thr} clone_mean={mean:.7f} "
                            "clone_p90={p90:.7f} clone_p99={p99:.7f} clone_max={gmax:.7f} "
                            "split_thr={abs_thr:.7f} split_over={abs_over} split_mean={abs_mean:.7f} "
                            "split_p90={abs_p90:.7f} split_p99={abs_p99:.7f} split_max={abs_max:.7f} "
                            "budget_new={budget_new} budget_total={budget_total} "
                            "budget_ratio={budget_ratio:.4f} budget_rem={budget_rem_before}->{budget_rem_after} "
                            "recycle_target={recycle_target} recycle_cand={recycle_cand} recycle_pruned={recycle_pruned} "
                            "split_first={split_first} "
                            "clone_cand={clone_cand} clone_sel={clone_sel} clone_clip={clone_clip} "
                            "split_cand={split_cand} split_sel={split_sel} split_clip={split_clip} "
                            "cons_mean={cons_mean:.4f} cons_p10={cons_p10:.4f} cons_p50={cons_p50:.4f} "
                            "dir_w_mean={dir_w_mean:.4f} dir_w_p90={dir_w_p90:.4f} dir_w_max={dir_w_max:.4f} "
                            "dgc={dgc}"
                        ).format(
                            iter=iteration,
                            mode=densify_stats["mode"],
                            before=densify_stats["before"],
                            cloned=densify_stats["cloned"],
                            split_added=densify_stats["split_added"],
                            pruned=densify_stats["pruned"],
                            after=densify_stats["after"],
                            thr=densify_stats["grad_threshold"],
                            over_thr=densify_stats["grad_over_threshold"],
                            mean=densify_stats["grad_mean"],
                            p90=densify_stats["grad_p90"],
                            p99=densify_stats["grad_p99"],
                            gmax=densify_stats["grad_max"],
                            abs_thr=densify_stats["grad_abs_threshold"],
                            abs_over=densify_stats["grad_abs_over_threshold"],
                            abs_mean=densify_stats["grad_abs_mean"],
                            abs_p90=densify_stats["grad_abs_p90"],
                            abs_p99=densify_stats["grad_abs_p99"],
                            abs_max=densify_stats["grad_abs_max"],
                            budget_new=densify_stats["max_new_points"],
                            budget_total=densify_stats["max_total_points"],
                            budget_ratio=densify_stats["budget_ratio"],
                            budget_rem_before=densify_stats["budget_remaining_before"],
                            budget_rem_after=densify_stats["budget_remaining_after"],
                            recycle_target=densify_stats["recycle_target"],
                            recycle_cand=densify_stats["recycle_candidates"],
                            recycle_pruned=densify_stats["recycle_pruned"],
                            split_first=densify_stats["split_first"],
                            clone_cand=densify_stats["clone_candidates"],
                            clone_sel=densify_stats["clone_selected"],
                            clone_clip=densify_stats["clone_clipped"],
                            split_cand=densify_stats["split_candidates"],
                            split_sel=densify_stats["split_selected"],
                            split_clip=densify_stats["split_clipped"],
                            cons_mean=densify_stats["consistency_mean"],
                            cons_p10=densify_stats["consistency_p10"],
                            cons_p50=densify_stats["consistency_p50"],
                            dir_w_mean=densify_stats["direction_weight_mean"],
                            dir_w_p90=densify_stats["direction_weight_p90"],
                            dir_w_max=densify_stats["direction_weight_max"],
                            dgc=densify_stats["density_guided_clone"],
                        )
                        print(densify_msg)
                        with open(os.path.join(model_path, "result.txt"), "a") as result_file:
                            result_file.write(densify_msg + "\n")

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if (
                    opt.needle_perturb_interval > 0
                    and iteration < opt.needle_perturb_until_iter
                    and iteration % opt.needle_perturb_interval == 0):
                needle_stats = gaussians.needle_shape_perturbance(
                    device,
                    ratio_min=opt.needle_perturb_ratio_min,
                    ratio_max=opt.needle_perturb_ratio_max,
                )
                needle_msg = (
                    "[NEEDLE][iter {iter}] selected={selected} "
                    "ratio_mean={ratio_mean:.4f} ratio_p99={ratio_p99:.4f} "
                    "ratio_range=({ratio_min:.3f},{ratio_max:.3f})"
                ).format(
                    iter=iteration,
                    selected=needle_stats["selected"],
                    ratio_mean=needle_stats["ratio_mean"],
                    ratio_p99=needle_stats["ratio_p99"],
                    ratio_min=opt.needle_perturb_ratio_min,
                    ratio_max=opt.needle_perturb_ratio_max,
                )
                print(needle_msg)
                with open(os.path.join(model_path, "result.txt"), "a") as result_file:
                    result_file.write(needle_msg + "\n")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                if iteration < opt.geometry_train_until_iter:
                    if opt.geometry_grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            geo_model.geometry.parameters(),
                            max_norm=opt.geometry_grad_clip_norm,
                        )
                    geo_model.optimizer.step()
                if rgb_geo_model is not None:
                    if iteration < opt.geometry_train_until_iter:
                        if opt.geometry_grad_clip_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                rgb_geo_model.geometry.parameters(),
                                max_norm=opt.geometry_grad_clip_norm,
                            )
                        rgb_geo_model.optimizer.step()
                if temporal_affine_model is not None:
                    temporal_affine_model.optimizer.step()
                if temporal_pose_model is not None:
                    temporal_pose_model.optimizer.step()
                atmos_model.optimizer.step()
                planck_model.optimizer.step()
                radiometric_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                geo_model.optimizer.zero_grad()
                if iteration < opt.geometry_train_until_iter:
                    geo_model.update_learning_rate(iteration)
                if rgb_geo_model is not None:
                    rgb_geo_model.optimizer.zero_grad()
                    if iteration < opt.geometry_train_until_iter:
                        rgb_geo_model.update_learning_rate(iteration)
                if temporal_affine_model is not None:
                    temporal_affine_model.optimizer.zero_grad()
                if temporal_pose_model is not None:
                    temporal_pose_model.optimizer.zero_grad()
                atmos_model.optimizer.zero_grad()
                planck_model.optimizer.zero_grad()
                radiometric_model.optimizer.zero_grad()
                atmos_model.update_learning_rate(iteration)
                planck_model.update_learning_rate(iteration)
                radiometric_model.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))

    final_num_gaussians = scene.gaussians.get_xyz.shape[0]
    print(f"\n\033[1;35mFinal number of Gaussians: {final_num_gaussians}\033[0m")

    result_file_path = os.path.join(model_path, "result.txt")
    with open(result_file_path, "a") as result_file:
        result_file.write(f"Final number of Gaussians: {final_num_gaussians}\n")
        result_file.write(f"Best PSNR = {best_psnr} in Iteration {best_iteration}\n")


def prepare_output_and_logger(args):
    dataset_path = Path(args.source_path)  
    parts = dataset_path.parts
    try:
        dataset_name = parts[parts.index('TI-NSD') + 1]
    except ValueError:
        dataset_name = dataset_path.name or "default_dataset"
        
    if not args.model_path:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.model_path = os.path.join("./output/", f"{dataset_name}_{current_time}/all")

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, args.model_path


def training_report(source_path, model_path, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, geo_model, load2gpu_on_the_fly, device, is_6dof=False, scale_factor=1, start_time=None,
                    opt=None, smooth_term=None, rgb_geo_model=None, temporal_affine_model=None,
                    temporal_pose_model=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        eval_temporal_interval = estimate_temporal_jitter_interval(
            scene.getTrainCameras(),
            "fid_median",
        )
        if opt is not None and getattr(opt, "ir_eval_temporal_ensemble", False):
            offsets = get_eval_temporal_offsets(
                opt,
                eval_temporal_interval,
                device,
                torch.float32,
            )
            eval_msg = (
                "[EVAL_ENSEMBLE][iter {}] samples={} interval={:.8f} offsets={}"
            ).format(
                iteration,
                int(offsets.numel()),
                float(eval_temporal_interval),
                ",".join("{:.8f}".format(float(v)) for v in offsets.detach().cpu()),
            )
            print("\n" + eval_msg)
            with open(os.path.join(model_path, "result.txt"), "a") as result_file:
                result_file.write(eval_msg + "\n")
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})
        if testing_iterations and iteration == testing_iterations[0]:
            eval_metric_msg = (
                "[EVAL_METRICS][iter {}] validation PSNR/L1 uses save_image-compatible "
                "8-bit quantized tensors, matching metrics.py on rendered PNGs."
            ).format(iteration)
            print("\n" + eval_metric_msg)
            with open(os.path.join(model_path, "result.txt"), "a") as result_file:
                result_file.write(eval_metric_msg + "\n")

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device=device)
                gts = torch.tensor([], device=device)
                rgb_images = torch.tensor([], device=device)
                rgb_gts = torch.tensor([], device=device)
                image_names = []
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid

                    radiative_blend = 0.0
                    radiative_color = None
                    if opt is not None and opt.phys_eval_with_render and scene.gaussians.has_radiative_transfer_attributes():
                        radiative_blend = physir_render_blend(iteration, opt)
                        radiative_color = scene.gaussians.get_physir_response_rgb if radiative_blend > 0.0 else None

                    image = torch.clamp(
                        render_eval_view(
                            viewpoint,
                            scene.gaussians,
                            renderFunc,
                            renderArgs,
                            geo_model,
                            opt,
                            iteration,
                            device,
                            is_6dof,
                            scene.cameras_extent,
                            radiative_color=radiative_color,
                            radiative_blend=radiative_blend,
                            feature_set="thermal" if getattr(scene, "has_rgbt", False) else "rgb",
                            eval_temporal_interval=eval_temporal_interval,
                            temporal_affine_model=temporal_affine_model,
                            temporal_pose_model=temporal_pose_model,
                        ),
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)
                    image_metric = quantize_like_save_image(image)
                    gt_metric = quantize_like_save_image(gt_image)
   
                    images = torch.cat((images, image_metric.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_metric.unsqueeze(0)), dim=0)
                    image_names.append(str(getattr(viewpoint, "image_name", idx)))

                    if getattr(scene, "has_rgbt", False) and viewpoint.original_rgb_image is not None:
                        rgb_image = torch.clamp(
                            render_eval_view(
                                viewpoint,
                                scene.gaussians,
                                renderFunc,
                                renderArgs,
                                rgb_geo_model if rgb_geo_model is not None else geo_model,
                                opt,
                                iteration,
                                device,
                                is_6dof,
                                scene.cameras_extent,
                                feature_set="rgb",
                                eval_temporal_interval=eval_temporal_interval,
                                temporal_pose_model=temporal_pose_model,
                            ),
                            0.0,
                            1.0,
                        )
                        gt_rgb = torch.clamp(viewpoint.original_rgb_image.to(device), 0.0, 1.0)
                        rgb_images = torch.cat((rgb_images, quantize_like_save_image(rgb_image).unsqueeze(0)), dim=0)
                        rgb_gts = torch.cat((rgb_gts, quantize_like_save_image(gt_rgb).unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image_metric[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_metric[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                per_view_psnr = psnr(images, gts).view(-1)
                psnr_test = per_view_psnr.mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                result_file_path = os.path.join(model_path, "result.txt")
                thermal_label = "{} Thermal".format(config['name']) if getattr(scene, "has_rgbt", False) else config['name']
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, thermal_label, l1_test, psnr_test))
                with open(result_file_path, "a") as result_file:
                    result_file.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, thermal_label, l1_test, psnr_test))

                if per_view_psnr.numel() > 0:
                    sorted_psnr, sorted_idx = torch.sort(per_view_psnr)
                    p10_idx = min(max(int(0.10 * (per_view_psnr.numel() - 1)), 0), per_view_psnr.numel() - 1)
                    p50_idx = min(max(int(0.50 * (per_view_psnr.numel() - 1)), 0), per_view_psnr.numel() - 1)
                    worst_count = min(5, per_view_psnr.numel())
                    worst_entries = []
                    for worst_rank in range(worst_count):
                        view_idx = int(sorted_idx[worst_rank].item())
                        worst_entries.append("{}:{:.3f}".format(image_names[view_idx], float(sorted_psnr[worst_rank].item())))
                    val_diag = (
                        "[VAL_DIAG][iter {}][{}] psnr_min={:.4f} psnr_p10={:.4f} "
                        "psnr_median={:.4f} psnr_max={:.4f} worst={}"
                    ).format(
                        iteration,
                        thermal_label,
                        float(sorted_psnr[0].item()),
                        float(sorted_psnr[p10_idx].item()),
                        float(sorted_psnr[p50_idx].item()),
                        float(sorted_psnr[-1].item()),
                        ",".join(worst_entries),
                    )
                    print("\n" + val_diag)
                    with open(result_file_path, "a") as result_file:
                        result_file.write(val_diag + "\n")

                if rgb_images.numel() > 0:
                    rgb_l1_test = l1_loss(rgb_images, rgb_gts)
                    rgb_psnr_test = psnr(rgb_images, rgb_gts).mean()
                    print("\n[ITER {}] Evaluating {} RGB: L1 {} PSNR {}".format(iteration, config['name'], rgb_l1_test, rgb_psnr_test))
                    with open(result_file_path, "a") as result_file:
                        result_file.write("[ITER {}] Evaluating {} RGB: L1 {} PSNR {}\n".format(iteration, config['name'], rgb_l1_test, rgb_psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if rgb_images.numel() > 0:
                        tb_writer.add_scalar(config['name'] + '/rgb_loss_viewpoint - l1_loss', rgb_l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/rgb_loss_viewpoint - psnr', rgb_psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    if start_time and iteration == testing_iterations[-1]:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTraining completed in: {elapsed_time:.2f} seconds")
        result_file_path = os.path.join(model_path, "result.txt")
        with open(result_file_path, "a") as result_file:
            result_file.write(f"Training completed in: {elapsed_time:.2f} seconds\n")
            
    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    # Use CUDA_VISIBLE_DEVICES environment variable to control GPU selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, device)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000, 8000, 9000] + list(range(10000, 30001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, device)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, device)

    # All done
    print("\nTraining complete.")
