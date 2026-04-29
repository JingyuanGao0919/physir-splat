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
from random import randint
import math

from utils.loss_utils import compute_rt_consistency_loss, l1_loss, ssim
from gaussian_renderer import render, GaussianRasterizationSettings, GaussianRasterizer
import sys
from scene import (
    Scene,
    GaussianModel,
    GeoRefineModel,
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


def compute_geo_fields(viewpoint_cam, gaussians, geo_model, pipe, background,
                       opt, iteration, smooth_term, time_interval, device, is_blender,
                       feature_set, eval_mode=False, respect_warmup=True):
    if respect_warmup and iteration < opt.warm_up:
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
        ast_noise = torch.randn(1, 1, device=device).expand(xyz_visible.shape[0], -1) * time_interval * smooth_term(iteration)
    view_dir = view_dir.unsqueeze(0).expand(xyz_visible.shape[0], -1)
    cam_pos = viewpoint_cam.camera_center.unsqueeze(0).expand(xyz_visible.shape[0], -1)

    d_xyz, d_rotation, d_scaling = geo_model.step(xyz_visible.detach(), time_input + ast_noise, cam_pos, view_dir)

    d_xyz_full = torch.zeros_like(gaussians.get_xyz)
    d_xyz_full[visible_indices] = d_xyz
    d_rotation_full = torch.zeros_like(gaussians.get_rotation)
    d_rotation_full[visible_indices] = d_rotation
    d_scaling_full = torch.zeros_like(gaussians.get_scaling)
    d_scaling_full[visible_indices] = d_scaling

    return d_xyz_full, d_rotation_full, d_scaling_full


def training(dataset, opt, pipe, testing_iterations, saving_iterations, device):
    start_time = time.time()
    tb_writer, model_path = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, device)

    atmos_model = AtmosModel(device)
    atmos_model.train_setting(opt)
    radiometric_model = RadiometricModel(device)
    radiometric_model.train_setting(opt)
    planck_model = PlanckModel(device)
    planck_model.train_setting(opt)
    geo_model = GeoRefineModel(device, dataset.is_blender, dataset.is_6dof)
    geo_model.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, device)
    has_rgbt = getattr(scene, "has_rgbt", False)
    rgb_geo_model = None
    if has_rgbt:
        rgb_geo_model = GeoRefineModel(device, dataset.is_blender, dataset.is_6dof)
        rgb_geo_model.train_setting(opt)
        print("Using separate thermal/RGB view-dependent geometry refinement.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32).to(device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(total=opt.iterations, desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    prev_probe_point_count = gaussians.get_xyz.shape[0]
    # for iteration in range(1, opt.iterations + 1):
    for iteration in islice(count(1), opt.iterations):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
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
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device=device).expand(xyz_visible.shape[0], -1) * time_interval * smooth_term(iteration)
            view_dir = view_dir.unsqueeze(0).expand(xyz_visible.shape[0], -1)  # -> (N, 3)
            cam_pos  = viewpoint_cam.camera_center.unsqueeze(0).expand(xyz_visible.shape[0], -1)   # -> (N, 3)

            d_xyz, d_rotation, d_scaling = geo_model.step(xyz_visible.detach(), time_input + ast_noise, cam_pos, view_dir)

            d_xyz_full = torch.zeros_like(gaussians.get_xyz)  # shape (N, 3)
            d_xyz_full[visible_indices] = d_xyz
            d_rotation_full = torch.zeros_like(gaussians.get_rotation)  # shape (N, 3)
            d_rotation_full[visible_indices] = d_rotation
            d_scaling_full = torch.zeros_like(gaussians.get_scaling)  # shape (N, 3)
            d_scaling_full[visible_indices] = d_scaling

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

        gt_image = viewpoint_cam.original_image.to(device)
        
        # Loss
        Ll1 = l1_loss(image, gt_image)
        if iteration <= 20000:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            tir_consistency_loss = compute_rt_consistency_loss(
                image,
                gt_image,
                atmos_model,
                planck_model,
                radiometric_model,
                device,
            )

            loss = (1.0 - opt.lambda_dssim - 0.2) * (Ll1) + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.2 * tir_consistency_loss

        rgb_loss = None
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

        physics_loss, _ = rt_guidance_loss(gaussians, opt, iteration)
        loss = loss + physics_loss

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

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(dataset.source_path, model_path, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), geo_model,
                                       dataset.load2gpu_on_the_fly, device, dataset.is_6dof, start_time=start_time,
                                       opt=opt, smooth_term=smooth_term, rgb_geo_model=rgb_geo_model)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if opt.phys_probe_every > 0 and iteration % opt.phys_probe_every == 0:
                fit_and_log_physir_attributes(iteration, gaussians, scene, opt, model_path, prev_probe_point_count)
                prev_probe_point_count = gaussians.get_xyz.shape[0]
                # Detailed radiative guidance and auxiliary-loss diagnostics are
                # intentionally kept out of the console/result.txt logs. Full
                # PhysIR statistics are still stored in phys_probe/probe_log.jsonl.

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                geo_model.save_weights(model_path, iteration)
                if rgb_geo_model is not None:
                    rgb_geo_model.save_weights(model_path, iteration, subdir="GeometryRGB")
                atmos_model.save_weights(model_path, iteration)
                planck_model.save_weights(model_path, iteration)
                radiometric_model.save_weights(model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, device)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                geo_model.optimizer.step()
                if rgb_geo_model is not None:
                    rgb_geo_model.optimizer.step()
                atmos_model.optimizer.step()
                planck_model.optimizer.step()
                radiometric_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                geo_model.optimizer.zero_grad()
                geo_model.update_learning_rate(iteration)
                if rgb_geo_model is not None:
                    rgb_geo_model.optimizer.zero_grad()
                    rgb_geo_model.update_learning_rate(iteration)
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
                    opt=None, smooth_term=None, rgb_geo_model=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device=device)
                gts = torch.tensor([], device=device)
                rgb_images = torch.tensor([], device=device)
                rgb_gts = torch.tensor([], device=device)
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid

                    R = viewpoint.R
                    T = viewpoint.T
                    R_torch = torch.from_numpy(R).float().to(device)
                    T_torch = torch.from_numpy(T).float().to(device)

                    Rt = torch.eye(4, device=device)
                    Rt[:3, :3] = R_torch.t()  
                    Rt[:3, 3] = T_torch       

                    c2w = Rt
                    view_dir = -c2w[:3, 2]       

                    with torch.no_grad():
                        render_pkg = render(
                            viewpoint,
                            scene.gaussians,
                            *renderArgs,
                            0,
                            0,
                            0,
                            device,
                            is_6dof,
                            feature_set="thermal" if getattr(scene, "has_rgbt", False) else "rgb",
                        )
                        visibility_filter = render_pkg["visibility_filter"]
                        visible_indices = visibility_filter.nonzero(as_tuple=True)[0]  
                        xyz_visible = scene.gaussians.get_xyz[visible_indices]
                    time_input = fid.unsqueeze(0).expand(xyz_visible.shape[0], -1)
                    view_dir = view_dir.unsqueeze(0).expand(xyz_visible.shape[0], -1)  
                    cam_pos  = viewpoint.camera_center.unsqueeze(0).expand(xyz_visible.shape[0], -1)   

                    d_xyz, d_rotation, d_scaling = geo_model.step(xyz_visible.detach(), time_input, cam_pos, view_dir)

                    d_xyz_full = torch.zeros_like(scene.gaussians.get_xyz)  
                    d_xyz_full[visible_indices] = d_xyz
                    d_rotation_full = torch.zeros_like(scene.gaussians.get_rotation)  
                    d_rotation_full[visible_indices] = d_rotation
                    d_scaling_full = torch.zeros_like(scene.gaussians.get_scaling)  
                    d_scaling_full[visible_indices] = d_scaling

                    radiative_blend = 0.0
                    radiative_color = None
                    if opt is not None and opt.phys_eval_with_render and scene.gaussians.has_radiative_transfer_attributes():
                        radiative_blend = physir_render_blend(iteration, opt)
                        radiative_color = scene.gaussians.get_physir_response_rgb if radiative_blend > 0.0 else None

                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            *renderArgs,
                            d_xyz_full,
                            d_rotation_full,
                            d_scaling_full,
                            device,
                            is_6dof,
                            radiative_color=radiative_color,
                            radiative_blend=radiative_blend,
                            feature_set="thermal" if getattr(scene, "has_rgbt", False) else "rgb",
                        )["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)
   
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if getattr(scene, "has_rgbt", False) and viewpoint.original_rgb_image is not None:
                        rgb_d_xyz_full, rgb_d_rotation_full, rgb_d_scaling_full = d_xyz_full, d_rotation_full, d_scaling_full
                        if rgb_geo_model is not None and opt is not None and smooth_term is not None:
                            rgb_d_xyz_full, rgb_d_rotation_full, rgb_d_scaling_full = compute_geo_fields(
                                viewpoint,
                                scene.gaussians,
                                rgb_geo_model,
                                renderArgs[0],
                                renderArgs[1],
                                opt,
                                iteration,
                                smooth_term,
                                1.0 / max(len(config['cameras']), 1),
                                device,
                                is_blender=True,
                                feature_set="rgb",
                                eval_mode=True,
                            )
                        rgb_image = torch.clamp(
                            renderFunc(
                                viewpoint,
                                scene.gaussians,
                                *renderArgs,
                                rgb_d_xyz_full,
                                rgb_d_rotation_full,
                                rgb_d_scaling_full,
                                device,
                                is_6dof,
                                feature_set="rgb",
                            )["render"],
                            0.0, 1.0)
                        gt_rgb = torch.clamp(viewpoint.original_rgb_image.to(device), 0.0, 1.0)
                        rgb_images = torch.cat((rgb_images, rgb_image.unsqueeze(0)), dim=0)
                        rgb_gts = torch.cat((rgb_gts, gt_rgb.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                result_file_path = os.path.join(model_path, "result.txt")
                thermal_label = "{} Thermal".format(config['name']) if getattr(scene, "has_rgbt", False) else config['name']
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, thermal_label, l1_test, psnr_test))
                with open(result_file_path, "a") as result_file:
                    result_file.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, thermal_label, l1_test, psnr_test))

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
