"""Isolated coordinate reparameterization of the native 0.2 center-depth cutoff."""
import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def rasterize_with_near(view, gaussians, colors, background, near_plane, opacity=None):
    factor = 0.2 / near_plane
    xyz = gaussians.get_xyz
    scales = gaussians.get_scaling
    world_view = view.world_view_transform
    projection = view.full_proj_transform
    camera = view.camera_center
    if factor != 1.:
        xyz = xyz * factor
        scales = scales * factor
        world_view = world_view.clone()
        world_view[3, :3] = world_view[3, :3] * factor
        # Scale both projective near/far distances, preserving normalized z.
        perspective = view.projection_matrix.clone()
        perspective[3, 2] = perspective[3, 2] * factor
        projection = world_view.unsqueeze(0).bmm(perspective.unsqueeze(0)).squeeze(0)
        camera = camera * factor
    screen = torch.zeros_like(gaussians.get_xyz, requires_grad=True) + 0
    if torch.is_grad_enabled():
        screen.retain_grad()
    settings = GaussianRasterizationSettings(
        image_height=view.image_height, image_width=view.image_width,
        tanfovx=math.tan(view.FoVx / 2), tanfovy=math.tan(view.FoVy / 2),
        bg=background, scale_modifier=1., viewmatrix=world_view,
        projmatrix=projection, sh_degree=gaussians.active_sh_degree,
        campos=camera, prefiltered=False, debug=False)
    image, radii, depth = GaussianRasterizer(settings)(
        means3D=xyz, means2D=screen, shs=None,
        colors_precomp=colors.contiguous(),
        opacities=gaussians.get_opacity if opacity is None else opacity,
        scales=scales, rotations=gaussians.get_rotation, cov3D_precomp=None)
    return dict(render=image, viewspace_points=screen, visibility_filter=radii > 0,
                radii=radii, depth=depth / factor)
