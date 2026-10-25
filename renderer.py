"""Radiance -> path transmission -> visibility -> sensor/display formation."""
import math
import bisect

import torch
from torch import nn
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from measurement import band_radiance, flir_blackbody_signal
from utils.general_utils import build_rotation
from utils.sh_utils import eval_sh
from utils.ATF_utils import ATFNetwork
from near_plane_rasterizer import rasterize_with_near
from view_deformation import ViewDeformation, ShiftedGaussians, SubsetGaussians
from rgb_response import PoseResponse
from deformation_pose import DeformationPose


class RadiativeField(nn.Module):
    def __init__(self, profile, extent, atmosphere=False, reflection=False, temporal_source=False,
                 temporal_frequencies=10):
        super().__init__()
        self.extent = extent
        self.atmosphere = atmosphere
        self.reflection = reflection
        self.calibrated = profile['domain'] in ('brightness_temperature', 'flir_counts')
        self.calibration = profile['calibration'] if profile['domain'] == 'flir_counts' else None
        self.temporal_source = temporal_source
        self.source_time_anchors = profile.get('source_time_anchors')
        self.rgb_response_anchors = None
        self.rgb_pose_response = (PoseResponse(profile['rgb_response_poses'], extent)
                                  if profile.get('rgb_response_poses') is not None else None)
        if profile.get('rgb_response_anchors') is not None:
            self.enable_rgb_response(profile['rgb_response_anchors'])
        if temporal_source:
            self.source_network = ATFNetwork(t_multires=temporal_frequencies)
            active_bands = profile.get('temporal_active_bands')
            if active_bands is not None:
                mask = torch.ones(1 + 2 * temporal_frequencies)
                mask[1 + 2 * active_bands:] = 0
                self.register_buffer('time_band_mask', mask)
                embed_time = self.source_network.embed_time_fn
                self.source_network.embed_time_fn = lambda t: embed_time(t) * self.time_band_mask
            # Reuse the native coordinate/frame backbone as a source predictor.
            # The retained head begins at a steady source; the reference frame
            # at t=0.5 always has unit gain. This is not atmospheric extinction.
            nn.init.zeros_(self.source_network.abs.weight)
            nn.init.zeros_(self.source_network.abs.bias)
        self.raw_extinction = nn.Parameter(torch.tensor(-6.9))
        self.raw_atmosphere = nn.Parameter(torch.tensor(-0.43275213))
        self.raw_background = nn.Parameter(torch.tensor(-6.9))
        self.environment = nn.Parameter(torch.tensor([-0.43275213, 0., 0., 0.]))
        self.register_buffer('wavelength', torch.linspace(*profile.get('band_um', [7.5, 13.5]), 32))
        self.view_deformation = None
        self.deformation_warmup = profile.get('deformation_warmup', 3000)
        if profile.get('thermal_view_deformation', False):
            self.enable_view_deformation()
        self.deformation_pose = (DeformationPose(profile['deformation_pose_anchors'], extent)
                                 if profile.get('deformation_pose_anchors') is not None else None)

    def displacement(self, view, xyz):
        if self.deformation_pose is not None:
            return self.deformation_pose.value(view, xyz, self.view_deformation)
        frame = view.fid.reshape(1, 1).expand(len(xyz), 1)
        center = view.camera_center[None].expand(len(xyz), 3)
        direction = -view.world_view_transform[:3, 2][None].expand(len(xyz), 3)
        return self.view_deformation(xyz, frame, center, direction)

    def enable_view_deformation(self):
        # Preserve the baseline sampling RNG before the zero-head warmup ends.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.view_deformation = ViewDeformation().to(self.raw_background.device)

    def enable_rgb_response(self, anchors):
        self.rgb_response_anchors = anchors
        self.rgb_log_gain = nn.Parameter(torch.zeros(len(anchors), device=self.raw_background.device if hasattr(self, 'raw_background') else None))

    def rgb_gain(self, view):
        values = self.rgb_log_gain - self.rgb_log_gain.mean()
        if self.rgb_pose_response is not None:
            return self.rgb_pose_response.value(view, values).exp()
        t = float(view.fid)
        anchors = self.rgb_response_anchors
        index = bisect.bisect_left(anchors, t)
        if index == 0:
            value = values[0]
        elif index == len(anchors):
            value = values[-1]
        else:
            weight = (t - anchors[index - 1]) / (anchors[index] - anchors[index - 1])
            value = torch.lerp(values[index - 1], values[index], weight)
        return value.exp()

    def emission(self, gaussians, direction=None):
        source = gaussians.source
        if gaussians.source_degree and not self.calibrated:
            coefficients = torch.cat((torch.zeros_like(source), gaussians.source_sh), dim=1)
            source = source + eval_sh(gaussians.active_source_degree,
                                      coefficients[:, None, :], direction)
        value = F.softplus(source)
        if self.calibrated:
            temperature = 1 + 100 * value
            if self.calibration is not None:
                return flir_blackbody_signal(temperature, self.calibration) / flir_blackbody_signal(
                    temperature.new_tensor(300.), self.calibration)
            return band_radiance(temperature, self.wavelength) / band_radiance(
                self.wavelength.new_tensor(300.), self.wavelength)
        return value

    def source_gain_at(self, gaussians, frame):
        delta, _, _ = self.source_network(gaussians.get_xyz.detach(), frame)
        gain = F.softplus(0.541324854612918 + (2 * frame - 1) * delta)
        if hasattr(gaussians, 'source_spline'):
            # Five fixed acquisition-coordinate knots; the middle log-gain is
            # exactly zero, preserving the existing reference at frame 0.5.
            values = gaussians.source_spline
            knots = torch.cat((values[:, :2], torch.zeros_like(values[:, :1]), values[:, 2:]), dim=1)
            coord = frame * 4
            index = coord.floor().long().clamp(0, 3)
            weight = coord - index
            log_gain = knots.gather(1, index) * (1 - weight) + knots.gather(1, index + 1) * weight
            if hasattr(gaussians, 'source_detail'):
                # One zero-initialized triangular basis per coarse interval.
                # Coarse knot locations are unchanged; add four midpoint knots.
                basis = 1 - (2 * weight - 1).abs()
                log_gain = log_gain + basis * gaussians.source_detail.gather(1, index)
            if hasattr(gaussians, 'source_fine'):
                # Refine each nine-knot interval; existing knot values stay fixed.
                fine_coord = frame * 8
                fine_index = fine_coord.floor().long().clamp(0, 7)
                fine_weight = fine_coord - fine_index
                fine_basis = 1 - (2 * fine_weight - 1).abs()
                log_gain = log_gain + fine_basis * gaussians.source_fine.gather(1, fine_index)
            if hasattr(gaussians, 'source_finer'):
                # Add sixteen midpoint knots while preserving all seventeen anchors.
                finer_coord = frame * 16
                finer_index = finer_coord.floor().long().clamp(0, 15)
                finer_weight = finer_coord - finer_index
                finer_basis = 1 - (2 * finer_weight - 1).abs()
                log_gain = log_gain + finer_basis * gaussians.source_finer.gather(1, finer_index)
            gain = gain * log_gain.exp()
        return gain

    def source_gain(self, view, gaussians):
        frame = view.fid.reshape(1, 1).expand(len(gaussians.get_xyz), 1)
        if self.source_time_anchors is None:
            return self.source_gain_at(gaussians, frame)
        t = float(view.fid)
        anchors = self.source_time_anchors
        index = bisect.bisect_left(anchors, t)
        # Training anchors take exactly the original path, including gradients.
        if index < len(anchors) and anchors[index] == t:
            return self.source_gain_at(gaussians, frame)
        if index == 0:
            return self.source_gain_at(gaussians, torch.full_like(frame, anchors[0]))
        if index == len(anchors):
            return self.source_gain_at(gaussians, torch.full_like(frame, anchors[-1]))
        lo, hi = anchors[index - 1], anchors[index]
        left = self.source_gain_at(gaussians, torch.full_like(frame, lo))
        right = self.source_gain_at(gaussians, torch.full_like(frame, hi))
        return torch.lerp(left, right, (t - lo) / (hi - lo))

    def incident(self, view, gaussians):
        direction = view.camera_center[None] - gaussians.get_xyz
        distance = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        direction = direction / distance.clamp_min(1e-8)
        source = self.emission(gaussians, direction)
        if self.temporal_source:
            source = source * self.source_gain(view, gaussians)
        if self.reflection:
            axes = build_rotation(gaussians.get_rotation)
            thin_axis = gaussians.get_scaling.argmin(-1)
            normal = axes[torch.arange(len(axes), device=axes.device), :, thin_axis]
            reflected = 2 * (direction * normal).sum(-1, keepdim=True) * normal - direction
            environment = F.softplus(self.environment[0] + (reflected * self.environment[1:]).sum(-1, keepdim=True))
            rho = gaussians.reflectivity.sigmoid()
            source = (1 - rho) * source + rho * environment
        if self.atmosphere:
            transmission = torch.exp(-F.softplus(self.raw_extinction) * distance / self.extent)
            source = transmission * source + (1 - transmission) * F.softplus(self.raw_atmosphere)
        return source

    def regularization(self, gaussians):
        loss = self.environment.sum() * 0
        if self.reflection:
            loss = loss + 1e-3 * (gaussians.reflectivity.sigmoid() - 0.05).square().mean()
            loss = loss + 1e-4 * self.environment[1:].square().mean()
        if self.atmosphere:
            loss = loss + 1e-4 * F.softplus(self.raw_extinction).square()
        return loss


def native_rasterize(view, gaussians, colors, background):
    screen = torch.zeros_like(gaussians.get_xyz, requires_grad=True) + 0
    if torch.is_grad_enabled():
        screen.retain_grad()
    settings = GaussianRasterizationSettings(
        image_height=view.image_height, image_width=view.image_width,
        tanfovx=math.tan(view.FoVx / 2), tanfovy=math.tan(view.FoVy / 2),
        bg=background, scale_modifier=1., viewmatrix=view.world_view_transform,
        projmatrix=view.full_proj_transform, sh_degree=gaussians.active_sh_degree,
        campos=view.camera_center, prefiltered=False, debug=False)
    image, radii, depth = GaussianRasterizer(settings)(
        means3D=gaussians.get_xyz, means2D=screen, shs=None,
        colors_precomp=colors.contiguous(), opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling, rotations=gaussians.get_rotation, cov3D_precomp=None)
    return dict(render=image, viewspace_points=screen, visibility_filter=radii > 0, radii=radii)


def rasterize(view, gaussians, colors, background, near_mode='native'):
    if near_mode == 'native':
        return native_rasterize(view, gaussians, colors, background)
    opacity = gaussians.get_opacity
    accepted = None
    if near_mode == 'rescaled-native':
        depth = (gaussians.get_xyz * view.world_view_transform[:3, 2]).sum(-1) + view.world_view_transform[3, 2]
        accepted = depth > .2
        opacity = opacity * accepted[:, None]
    result = rasterize_with_near(view, gaussians, colors, background, view.znear, opacity)
    if accepted is not None:
        result['radii'] = result['radii'] * accepted
        result['visibility_filter'] = result['visibility_filter'] & accepted
    return result


def render(view, gaussians, field, measurement, rgb=False, iteration=None):
    near_mode = measurement.profile.get('near_mode', 'native')
    background = F.softplus(field.raw_background).expand(3).contiguous()
    accepted = None
    if rgb:
        direction = F.normalize(gaussians.get_xyz - view.camera_center[None], dim=-1)
        sh = gaussians.get_features.transpose(1, 2)
        color = (eval_sh(gaussians.active_sh_degree, sh, direction) + 0.5).clamp_min(0)
        rgb_background = background.new_full((3,), measurement.profile.get('rgb_background', 0.))
        visible = rasterize(view, gaussians, color, rgb_background, near_mode)
        accepted = visible['visibility_filter']
    else:
        with torch.no_grad():
            canonical = rasterize(view, gaussians, torch.ones_like(gaussians.get_xyz),
                                  gaussians.get_xyz.new_zeros(3), near_mode)
            accepted = canonical['visibility_filter']
    thermal_gaussians = gaussians
    if field.view_deformation is not None and (iteration is None or iteration >= field.deformation_warmup):
        xyz = gaussians.get_xyz[accepted].detach()
        displacement = torch.zeros_like(gaussians.get_xyz)
        displacement[accepted] = field.displacement(view, xyz)
        thermal_gaussians = ShiftedGaussians(gaussians, displacement)
    # Culled canonical points have zero displacement, hence still cannot
    # contribute. Global penalties remain on the complete Gaussian set.
    arriving = gaussians.get_xyz.new_zeros((len(gaussians.get_xyz), 1))
    arriving[accepted] = field.incident(view, SubsetGaussians(thermal_gaussians, accepted))
    thermal = rasterize(view, thermal_gaussians, arriving.expand(-1, 3), background, near_mode)
    radiance = thermal['render'][:1]
    thermal['radiance'] = radiance
    thermal['render'] = measurement(radiance)
    packages = [{key: thermal[key] for key in ('viewspace_points', 'visibility_filter', 'radii')}]
    if rgb:
        thermal['rgb'] = visible['render']
        if field.rgb_response_anchors is not None:
            thermal['rgb'] = thermal['rgb'] * field.rgb_gain(view)
        packages.append({key: visible[key] for key in ('viewspace_points', 'visibility_filter', 'radii')})
    thermal['density_packages'] = packages
    return thermal
