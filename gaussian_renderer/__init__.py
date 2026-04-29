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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB, eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous
import math


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, device, is_6dof=False,
           scaling_modifier=1.0, override_color=None, scale_factor=1, radiative_color=None, radiative_blend=0.0,
           physical_color=None, physical_blend=0.0, feature_set="rgb", detach_geometry=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    scaled_height = viewpoint_camera.image_height // scale_factor
    scaled_width = viewpoint_camera.image_width // scale_factor

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device) + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(scaled_height),
        image_width=int(scaled_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    base_xyz = pc.get_xyz.detach() if detach_geometry else pc.get_xyz

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = base_xyz
        else:
            d_xyz_render = d_xyz.detach() if detach_geometry else d_xyz
            means3D = from_homogenous(
                torch.bmm(d_xyz_render, to_homogenous(base_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        d_xyz_render = d_xyz.detach() if detach_geometry and torch.is_tensor(d_xyz) else d_xyz
        means3D = base_xyz + d_xyz_render
    opacity = pc.get_opacity.detach() if detach_geometry else pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        if detach_geometry:
            cov3D_precomp = cov3D_precomp.detach()
    else:
        d_scaling_render = d_scaling.detach() if detach_geometry and torch.is_tensor(d_scaling) else d_scaling
        d_rotation_render = d_rotation.detach() if detach_geometry and torch.is_tensor(d_rotation) else d_rotation
        scales = (pc.get_scaling.detach() if detach_geometry else pc.get_scaling) + d_scaling_render
        rotations = (pc.get_rotation.detach() if detach_geometry else pc.get_rotation) + d_rotation_render

    render_features = pc.get_thermal_features if feature_set in ("thermal", "tir", "ir") else pc.get_features
    render_features_dc = pc.get_thermal_features_dc if feature_set in ("thermal", "tir", "ir") else pc._features_dc

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if radiative_color is None and physical_color is not None:
        radiative_color = physical_color
    if radiative_blend == 0.0 and physical_blend != 0.0:
        radiative_blend = physical_blend
    blend = float(max(min(radiative_blend, 1.0), 0.0))
    if override_color is not None:
        colors_precomp = override_color
    else:
        need_python_color = pipe.convert_SHs_python or (radiative_color is not None and blend > 0.0)
        if need_python_color:
            shs_view = render_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            # Keep the canonical 3DGS SH view direction. Using the refined
            # means here changes the baseline render path before PhysIR terms start.
            dir_pp = (base_xyz - viewpoint_camera.camera_center.repeat(render_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            sh_rgb = torch.clamp_min(sh2rgb + 0.5, 0.0)
            if radiative_color is not None and blend > 0.0 and radiative_color.shape == sh_rgb.shape:
                dc_rgb = torch.clamp(SH2RGB(render_features_dc[:, 0, :]), 0.0, 1.0)
                dc_luma = dc_rgb.mean(dim=1, keepdim=True).clamp_min(1e-6)
                radiative_luma = radiative_color.detach().mean(dim=1, keepdim=True).clamp(0.0, 1.0)
                luma_ratio = (radiative_luma / dc_luma).clamp(0.5, 1.5)
                radiative_dc_rgb = dc_rgb * luma_ratio
                sh_residual = sh_rgb - dc_rgb
                blended_dc = torch.lerp(dc_rgb, radiative_dc_rgb, blend)
                colors_precomp = torch.clamp_min(sh_residual + blended_dc, 0.0)
            else:
                colors_precomp = sh_rgb
        else:
            shs = render_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        means2D_densify=screenspace_points_densify,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "viewspace_points_densify": screenspace_points_densify,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth}
