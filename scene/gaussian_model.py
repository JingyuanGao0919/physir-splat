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
    def __init__(self, sh_degree: int, device):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation, device)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance, device)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._thermal_features_dc = torch.empty(0)
        self._thermal_features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.og_number_points = 0

        # Diagnostic-only physical quantities. They are fitted by the physics
        # probe and are not part of rendering or the main optimizer.
        self._phys_iteration = -1
        self._phys_q_target = torch.empty(0)
        self._phys_q_phys = torch.empty(0)
        self._phys_temperature = torch.empty(0)
        self._phys_emissivity = torch.empty(0)
        self._phys_emission = torch.empty(0)
        self._phys_ambient = torch.empty(0)
        self._phys_delta = torch.empty(0)
        self._phys_rest_ratio = torch.empty(0)
        self._phys_rgb_dc = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

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
    def has_thermal_branch(self):
        return (
            torch.is_tensor(self._thermal_features_dc)
            and torch.is_tensor(self._thermal_features_rest)
            and self._thermal_features_dc.numel() > 0
            and self._thermal_features_rest.numel() > 0
        )

    @property
    def get_thermal_features(self):
        if not self.has_thermal_branch:
            return self.get_features
        return torch.cat((self._thermal_features_dc, self._thermal_features_rest), dim=1)

    @property
    def get_thermal_features_dc(self):
        return self._thermal_features_dc if self.has_thermal_branch else self._features_dc

    def enable_thermal_branch_from_rgb(self, device):
        if self.has_thermal_branch:
            return
        self._thermal_features_dc = nn.Parameter(self._features_dc.detach().clone().to(device).requires_grad_(True))
        self._thermal_features_rest = nn.Parameter(torch.zeros_like(self._features_rest, device=device).requires_grad_(True))

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_physical_q(self):
        return self._phys_q_phys

    @property
    def get_physical_temperature(self):
        return self._phys_temperature

    @property
    def get_physical_emissivity(self):
        return self._phys_emissivity

    @property
    def get_physical_emission(self):
        return self._phys_emission

    @property
    def get_physical_ambient(self):
        return self._phys_ambient

    @property
    def get_physical_delta(self):
        return self._phys_delta

    @property
    def get_physical_rgb(self):
        if not self.has_physical_probe():
            return torch.empty(0, device=self.get_xyz.device)
        return self._phys_q_phys.clamp(0.0, 1.0).repeat(1, 3)

    @property
    def get_passband_radiance(self):
        return self.get_physical_q

    @property
    def get_thermal_temperature(self):
        return self.get_physical_temperature

    @property
    def get_thermal_emissivity(self):
        return self.get_physical_emissivity

    @property
    def get_self_emission(self):
        return self.get_physical_emission

    @property
    def get_environment_irradiance(self):
        return self.get_physical_ambient

    @property
    def get_radiometric_residual(self):
        return self.get_physical_delta

    @property
    def get_physir_response_rgb(self):
        return self.get_physical_rgb

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, device, enable_thermal_branch=False):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.og_number_points = int(fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(device)), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        if enable_thermal_branch:
            self.enable_thermal_branch_from_rgb(device)
        else:
            self._thermal_features_dc = torch.empty(0, device=device)
            self._thermal_features_rest = torch.empty(0, device=device)
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def training_setup(self, training_args, device):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)

        self.spatial_lr_scale = 5

        rgb_feature_lr = training_args.feature_lr
        if self.has_thermal_branch:
            rgb_feature_lr *= getattr(training_args, "rgbt_rgb_feature_lr_mult", 1.0)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': rgb_feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': rgb_feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.has_thermal_branch:
            thermal_lr = getattr(training_args, "thermal_feature_lr", training_args.feature_lr)
            l.insert(3, {'params': [self._thermal_features_dc], 'lr': thermal_lr, "name": "thermal_dc"})
            l.insert(4, {'params': [self._thermal_features_rest], 'lr': thermal_lr / 20.0, "name": "thermal_rest"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
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
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        if self.has_thermal_branch:
            for i in range(self._thermal_features_dc.shape[1] * self._thermal_features_dc.shape[2]):
                l.append('t_dc_{}'.format(i))
            for i in range(self._thermal_features_rest.shape[1] * self._thermal_features_rest.shape[2]):
                l.append('t_rest_{}'.format(i))
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
        attr_chunks = [xyz, normals, f_dc, f_rest]
        if self.has_thermal_branch:
            t_dc = self._thermal_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            t_rest = self._thermal_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            attr_chunks.extend([t_dc, t_rest])
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attr_chunks.extend([opacities, scale, rotation])
        attributes = np.concatenate(attr_chunks, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def has_physical_probe(self):
        n = self.get_xyz.shape[0]
        required = (
            self._phys_q_target,
            self._phys_q_phys,
            self._phys_temperature,
            self._phys_emissivity,
            self._phys_emission,
            self._phys_ambient,
            self._phys_delta,
        )
        return n > 0 and all(torch.is_tensor(t) and t.numel() > 0 and t.shape[0] == n for t in required)

    def has_radiative_transfer_attributes(self):
        return self.has_physical_probe()

    def clear_physical_probe(self):
        device = self.get_xyz.device if self.get_xyz.numel() > 0 else self._xyz.device
        self._phys_iteration = -1
        self._phys_q_target = torch.empty(0, device=device)
        self._phys_q_phys = torch.empty(0, device=device)
        self._phys_temperature = torch.empty(0, device=device)
        self._phys_emissivity = torch.empty(0, device=device)
        self._phys_emission = torch.empty(0, device=device)
        self._phys_ambient = torch.empty(0, device=device)
        self._phys_delta = torch.empty(0, device=device)
        self._phys_rest_ratio = torch.empty(0, device=device)
        self._phys_rgb_dc = torch.empty(0, device=device)

    def clear_radiative_transfer_attributes(self):
        self.clear_physical_probe()

    def update_physical_probe(self, iteration, **kwargs):
        self._phys_iteration = int(iteration)
        for name, value in kwargs.items():
            if torch.is_tensor(value):
                setattr(self, f"_phys_{name}", value.detach().clone())
            else:
                setattr(self, f"_phys_{name}", value)

    def update_radiative_transfer_attributes(self, iteration, **kwargs):
        self.update_physical_probe(iteration, **kwargs)

    def save_physical_probe(self, path):
        if not self.has_physical_probe():
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "iteration": self._phys_iteration,
            "q_target": self._phys_q_target.detach().cpu(),
            "q_phys": self._phys_q_phys.detach().cpu(),
            "temperature": self._phys_temperature.detach().cpu(),
            "emissivity": self._phys_emissivity.detach().cpu(),
            "emission": self._phys_emission.detach().cpu(),
            "ambient": self._phys_ambient.detach().cpu(),
            "delta": self._phys_delta.detach().cpu(),
            "rest_ratio": self._phys_rest_ratio.detach().cpu(),
            "rgb_dc": self._phys_rgb_dc.detach().cpu(),
        }
        torch.save(state, path)
        return True

    def save_radiative_transfer_attributes(self, path):
        return self.save_physical_probe(path)

    def load_physical_probe(self, path):
        if not os.path.exists(path):
            return False
        device = self.get_xyz.device
        state = torch.load(path, map_location=device)
        self._phys_iteration = int(state.get("iteration", -1))
        for name in (
            "q_target",
            "q_phys",
            "temperature",
            "emissivity",
            "emission",
            "ambient",
            "delta",
            "rest_ratio",
            "rgb_dc",
        ):
            value = state.get(name, torch.empty(0, device=device))
            if torch.is_tensor(value):
                setattr(self, f"_phys_{name}", value.to(device))
        if not self.has_physical_probe():
            self.clear_physical_probe()
            return False
        return True

    def load_radiative_transfer_attributes(self, path):
        return self.load_physical_probe(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, device, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        thermal_dc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("t_dc_")]
        thermal_rest_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("t_rest_")]
        has_saved_thermal = len(thermal_dc_names) == 3 and len(thermal_rest_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        if has_saved_thermal:
            thermal_dc = np.zeros((xyz.shape[0], 3, 1))
            for idx, attr_name in enumerate(thermal_dc_names):
                thermal_dc[:, idx, 0] = np.asarray(plydata.elements[0][attr_name])
            thermal_extra = np.zeros((xyz.shape[0], len(thermal_rest_names)))
            for idx, attr_name in enumerate(thermal_rest_names):
                thermal_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            thermal_extra = thermal_extra.reshape((thermal_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(
                True))
        if has_saved_thermal:
            self._thermal_features_dc = nn.Parameter(
                torch.tensor(thermal_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
            self._thermal_features_rest = nn.Parameter(
                torch.tensor(thermal_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        else:
            self._thermal_features_dc = torch.empty(0, device=device)
            self._thermal_features_rest = torch.empty(0, device=device)
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

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
        if "thermal_dc" in optimizable_tensors:
            self._thermal_features_dc = optimizable_tensors["thermal_dc"]
            self._thermal_features_rest = optimizable_tensors["thermal_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, device, new_thermal_features_dc=None, new_thermal_features_rest=None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}
        if self.has_thermal_branch:
            d["thermal_dc"] = new_thermal_features_dc
            d["thermal_rest"] = new_thermal_features_rest

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        if "thermal_dc" in optimizable_tensors:
            self._thermal_features_dc = optimizable_tensors["thermal_dc"]
            self._thermal_features_rest = optimizable_tensors["thermal_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def _cap_selected_mask(self, selected_pts_mask, scores, max_selected):
        if max_selected is None:
            return selected_pts_mask, 0
        selected_count = int(selected_pts_mask.sum().item())
        max_selected = int(max_selected)
        if max_selected <= 0:
            return torch.zeros_like(selected_pts_mask), selected_count
        if selected_count <= max_selected:
            return selected_pts_mask, 0
        selected_idx = torch.where(selected_pts_mask)[0]
        selected_scores = scores[selected_idx].detach().reshape(-1)
        selected_scores[~torch.isfinite(selected_scores)] = 0.0
        keep = torch.topk(selected_scores, k=int(max_selected), largest=True).indices
        capped_mask = torch.zeros_like(selected_pts_mask)
        capped_mask[selected_idx[keep]] = True
        return capped_mask, selected_count - int(max_selected)

    def densify_and_split(self, grads, grad_threshold, scene_extent, device, N=2,
                          max_new_points=0, max_total_points=0):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)
        candidate_count = int(selected_pts_mask.sum().item())
        max_selected = candidate_count
        if max_new_points and max_new_points > 0:
            max_selected = min(max_selected, int(max_new_points) // max(int(N), 1))
        if max_total_points and max_total_points > 0:
            net_per_selected = max(int(N) - 1, 1)
            remaining = max(int(max_total_points) - int(self.get_xyz.shape[0]), 0)
            max_selected = min(max_selected, remaining // net_per_selected)
        selected_pts_mask, clipped_count = self._cap_selected_mask(selected_pts_mask, padded_grad, max_selected)
        selected_count = int(selected_pts_mask.sum().item())
        if selected_count == 0:
            return {
                "candidates": candidate_count,
                "selected": 0,
                "clipped": clipped_count,
            }

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_thermal_features_dc = self._thermal_features_dc[selected_pts_mask].repeat(N, 1, 1) if self.has_thermal_branch else None
        new_thermal_features_rest = self._thermal_features_rest[selected_pts_mask].repeat(N, 1, 1) if self.has_thermal_branch else None
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   device, new_thermal_features_dc, new_thermal_features_rest)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=bool)))
        self.prune_points(prune_filter)
        return {
            "candidates": candidate_count,
            "selected": selected_count,
            "clipped": clipped_count,
        }

    def densify_and_clone(self, grads, grad_threshold, scene_extent, device,
                          density_guided_clone=False, density_guided_clone_scale=1.0,
                          max_new_points=0, max_total_points=0):
        # Extract points that satisfy the gradient condition
        grad_norm = torch.norm(grads, dim=-1)
        selected_pts_mask = torch.where(grad_norm >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        candidate_count = int(selected_pts_mask.sum().item())
        max_selected = candidate_count
        if max_new_points and max_new_points > 0:
            max_selected = min(max_selected, int(max_new_points))
        if max_total_points and max_total_points > 0:
            remaining = max(int(max_total_points) - int(self.get_xyz.shape[0]), 0)
            max_selected = min(max_selected, remaining)
        selected_pts_mask, clipped_count = self._cap_selected_mask(selected_pts_mask, grad_norm, max_selected)
        selected_count = int(selected_pts_mask.sum().item())
        if selected_count == 0:
            return {
                "candidates": candidate_count,
                "selected": 0,
                "clipped": clipped_count,
            }

        if density_guided_clone and selected_pts_mask.any():
            # ReAct-GS style density-guided clone: place cloned small Gaussians
            # near their local point spacing instead of exactly duplicating xyz.
            dist2 = torch.clamp_min(distCUDA2(self._xyz.detach()), 0.0000001)
            stds = torch.sqrt(dist2)[..., None].repeat(1, 3) * float(density_guided_clone_scale)
            new_xyz = torch.normal(mean=self._xyz[selected_pts_mask], std=stds[selected_pts_mask])
        else:
            new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_thermal_features_dc = self._thermal_features_dc[selected_pts_mask] if self.has_thermal_branch else None
        new_thermal_features_rest = self._thermal_features_rest[selected_pts_mask] if self.has_thermal_branch else None
        
        new_opacities = self._opacity[selected_pts_mask]
        # original_opacities = self._opacity[selected_pts_mask]

        # # new_opacities = 1 - torch.sqrt(1 - self.opacity_activation(original_opacities))
        # # new_opacities = self.inverse_opacity_activation(new_opacities)

        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # cloned_opacity_vals = self.opacity_activation(original_opacities.squeeze(-1))
        # cloned_scaling_vals = self.scaling_activation(new_scaling)
        # recalc_opacity_mask = (cloned_opacity_vals >= 0.5) & (torch.any(cloned_scaling_vals >= 0.01, dim=1))
        # recalc_opacity_mask = recalc_opacity_mask.view(-1, 1)
        # new_opacities = original_opacities.clone()

        # new_opacities[recalc_opacity_mask] = (
        #     1 - torch.sqrt(1 - self.opacity_activation(original_opacities[recalc_opacity_mask]))
        # )
        # new_opacities[recalc_opacity_mask] = self.inverse_opacity_activation(
        #     new_opacities[recalc_opacity_mask]
        # )
        
        # self._opacity[selected_pts_mask] = new_opacities

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, device, new_thermal_features_dc, new_thermal_features_rest)
        return {
            "candidates": candidate_count,
            "selected": selected_count,
            "clipped": clipped_count,
        }

    def needle_shape_perturbance(self, device, ratio_min=0.8, ratio_max=0.999):
        original_scales = self.get_scaling
        if original_scales.numel() == 0:
            return {"selected": 0, "ratio_mean": 0.0, "ratio_p99": 0.0}

        max_scales, max_idx = torch.max(original_scales, dim=1)
        min_scales, _ = torch.min(original_scales, dim=1)
        denom = (torch.sum(original_scales, dim=1) - max_scales - min_scales).clamp_min(1e-8)
        needle_degree = max_scales / denom
        needle_ratio = max_scales / torch.sum(original_scales, dim=1).clamp_min(1e-8)
        selected_mask = torch.logical_and(needle_ratio > float(ratio_min), needle_ratio < float(ratio_max))
        selected_idx = torch.where(selected_mask)[0]

        ratio_values = needle_ratio.detach()
        stats = {
            "selected": int(selected_idx.numel()),
            "ratio_mean": float(ratio_values.mean().item()),
            "ratio_p99": float(torch.quantile(ratio_values, 0.99).item()) if ratio_values.numel() > 0 else 0.0,
        }
        if selected_idx.numel() == 0:
            return stats

        half_degree = (needle_degree / 2).unsqueeze(1)
        new_scales = original_scales.clone()
        new_scales[selected_idx] *= half_degree[selected_idx]
        new_scales[selected_idx, max_idx[selected_idx]] = original_scales[selected_idx, max_idx[selected_idx]]
        new_scales = self.scaling_inverse_activation(new_scales)

        optimizable_tensors = self.replace_tensor_to_optimizer(new_scales, "scaling")
        self._scaling = optimizable_tensors["scaling"]
        return stats

    def _densify_grad_stats(self, values, threshold):
        finite_values = values.detach().reshape(-1)
        finite_values = finite_values[torch.isfinite(finite_values)]
        if finite_values.numel() == 0:
            return {
                "mean": 0.0,
                "p90": 0.0,
                "p99": 0.0,
                "max": 0.0,
                "over_threshold": 0,
            }
        return {
            "mean": float(finite_values.mean().item()),
            "p90": float(torch.quantile(finite_values, 0.90).item()),
            "p99": float(torch.quantile(finite_values, 0.99).item()),
            "max": float(finite_values.max().item()),
            "over_threshold": int((finite_values >= threshold).sum().item()),
        }

    def _normalized_for_budget(self, values):
        values = values.detach().reshape(-1).float()
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        positive = values[values > 0]
        if positive.numel() == 0:
            return torch.zeros_like(values)
        scale = torch.quantile(positive, 0.90).clamp_min(1e-8)
        return (values / scale).clamp(0.0, 2.0)

    def _budget_recycle_prune(self, max_total_points, max_new_points, max_grad, split_threshold,
                              recycle_points=0, recycle_start=0.97, recycle_opacity=0.25,
                              recycle_grad_factor=2.0):
        stats = {
            "budget_ratio": 0.0,
            "budget_remaining_before": 0,
            "budget_remaining_after": 0,
            "recycle_target": 0,
            "recycle_candidates": 0,
            "recycle_pruned": 0,
        }
        if not max_total_points or max_total_points <= 0 or not recycle_points or recycle_points <= 0:
            return stats

        current_points = int(self.get_xyz.shape[0])
        max_total_points = int(max_total_points)
        remaining = max(max_total_points - current_points, 0)
        budget_ratio = current_points / max(max_total_points, 1)
        stats["budget_ratio"] = float(budget_ratio)
        stats["budget_remaining_before"] = int(remaining)
        stats["budget_remaining_after"] = int(remaining)
        if budget_ratio < float(recycle_start):
            return stats

        reserve = int(max_new_points) if max_new_points and max_new_points > 0 else int(recycle_points)
        target = min(int(recycle_points), max(reserve - remaining, 0), current_points)
        stats["recycle_target"] = int(target)
        if target <= 0:
            return stats

        grad_mean = self.xyz_gradient_accum / self.denom
        grad_mean[grad_mean.isnan()] = 0.0
        grad_abs = self.xyz_gradient_accum_abs / self.denom
        grad_abs[grad_abs.isnan()] = 0.0
        grad_score = torch.maximum(
            grad_mean.detach().reshape(-1).abs(),
            grad_abs.detach().reshape(-1).abs(),
        )
        opacity = self.get_opacity.detach().reshape(-1)
        radii = self.max_radii2D.detach().reshape(-1).float()
        scale = self.get_scaling.detach().max(dim=1).values

        grad_limit = max(float(max_grad), float(split_threshold)) * float(recycle_grad_factor)
        candidate_mask = torch.logical_and(opacity <= float(recycle_opacity), grad_score <= grad_limit)
        candidate_count = int(candidate_mask.sum().item())
        if candidate_count < target:
            # Fall back to all non-high-gradient points if opacity alone is too conservative.
            candidate_mask = grad_score <= grad_limit
            candidate_count = int(candidate_mask.sum().item())
        if candidate_count <= 0:
            return stats

        utility = (
            0.50 * self._normalized_for_budget(opacity)
            + 0.30 * self._normalized_for_budget(grad_score)
            + 0.15 * self._normalized_for_budget(radii)
            + 0.05 * self._normalized_for_budget(scale)
        )
        candidate_idx = torch.where(candidate_mask)[0]
        k = min(int(target), int(candidate_idx.numel()))
        selected_local = torch.topk(utility[candidate_idx], k=k, largest=False).indices
        prune_mask = torch.zeros((current_points,), device=self.get_xyz.device, dtype=torch.bool)
        prune_mask[candidate_idx[selected_local]] = True
        pruned = int(prune_mask.sum().item())
        if pruned > 0:
            self.prune_points(prune_mask)

        stats["recycle_candidates"] = int(candidate_count)
        stats["recycle_pruned"] = int(pruned)
        stats["budget_remaining_after"] = int(max(max_total_points - int(self.get_xyz.shape[0]), 0))
        return stats

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, device,
                          max_grad_abs=None, use_dual_gradient=False,
                          use_direction_aware=False, direction_aware_base=0.8,
                          direction_aware_scale=25.0, direction_aware_power=15.0,
                          direction_aware_split_abs=False,
                          density_guided_clone=False, density_guided_clone_scale=1.0,
                          max_new_points=0, max_total_points=0, split_first=False,
                          budget_recycle_points=0, budget_recycle_start=0.97,
                          budget_recycle_opacity=0.25, budget_recycle_grad_factor=2.0):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        consistency_stats = {
            "consistency_mean": 0.0,
            "consistency_p10": 0.0,
            "consistency_p50": 0.0,
            "direction_weight_mean": 1.0,
            "direction_weight_p90": 1.0,
            "direction_weight_max": 1.0,
        }

        if use_direction_aware:
            split_threshold = max_grad_abs if max_grad_abs is not None else max_grad
            consistency = ((grads + 1e-8) / (grads_abs + 1e-8)).clamp(0.0, 1.0)
            direction_weight = float(direction_aware_base) + float(direction_aware_scale) * torch.pow(
                1.0 - consistency, float(direction_aware_power)
            )
            clone_grads = grads / direction_weight
            if direction_aware_split_abs:
                split_grads = grads_abs * direction_weight
                mode = "direction_aware_abs_split"
            else:
                split_grads = grads * direction_weight
                mode = "direction_aware"
            finite_consistency = consistency.detach().reshape(-1)
            finite_weight = direction_weight.detach().reshape(-1)
            finite_consistency = finite_consistency[torch.isfinite(finite_consistency)]
            finite_weight = finite_weight[torch.isfinite(finite_weight)]
            if finite_consistency.numel() > 0:
                consistency_stats = {
                    "consistency_mean": float(finite_consistency.mean().item()),
                    "consistency_p10": float(torch.quantile(finite_consistency, 0.10).item()),
                    "consistency_p50": float(torch.quantile(finite_consistency, 0.50).item()),
                    "direction_weight_mean": float(finite_weight.mean().item()),
                    "direction_weight_p90": float(torch.quantile(finite_weight, 0.90).item()),
                    "direction_weight_max": float(finite_weight.max().item()),
                }
        elif use_dual_gradient:
            split_threshold = max_grad_abs if max_grad_abs is not None else max_grad
            clone_grads = grads
            split_grads = grads_abs
            mode = "dual"
        else:
            split_threshold = max_grad
            clone_grads = grads_abs
            split_grads = grads_abs
            mode = "abs_single"

        recycle_stats = self._budget_recycle_prune(
            max_total_points,
            max_new_points,
            max_grad,
            split_threshold,
            recycle_points=budget_recycle_points,
            recycle_start=budget_recycle_start,
            recycle_opacity=budget_recycle_opacity,
            recycle_grad_factor=budget_recycle_grad_factor,
        )
        if recycle_stats["recycle_pruned"] > 0:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            grads_abs = self.xyz_gradient_accum_abs / self.denom
            grads_abs[grads_abs.isnan()] = 0.0
            if use_direction_aware:
                consistency = ((grads + 1e-8) / (grads_abs + 1e-8)).clamp(0.0, 1.0)
                direction_weight = float(direction_aware_base) + float(direction_aware_scale) * torch.pow(
                    1.0 - consistency, float(direction_aware_power)
                )
                clone_grads = grads / direction_weight
                if direction_aware_split_abs:
                    split_grads = grads_abs * direction_weight
                else:
                    split_grads = grads * direction_weight
            elif use_dual_gradient:
                clone_grads = grads
                split_grads = grads_abs
            else:
                clone_grads = grads_abs
                split_grads = grads_abs

        grad_stats = self._densify_grad_stats(clone_grads, max_grad)
        grad_abs_stats = self._densify_grad_stats(split_grads, split_threshold)

        before = int(self.get_xyz.shape[0])
        clone_budget = int(max_new_points) if max_new_points and max_new_points > 0 else 0
        split_budget = int(max_new_points) if max_new_points and max_new_points > 0 else 0
        effective_split_first = bool(split_first)
        if effective_split_first and max_total_points and max_total_points > 0:
            effective_split_first = before / max(int(max_total_points), 1) >= float(budget_recycle_start)
        if effective_split_first:
            split_stats = self.densify_and_split(
                split_grads,
                split_threshold,
                extent,
                device,
                max_new_points=split_budget,
                max_total_points=max_total_points,
            )
            after_split_first = int(self.get_xyz.shape[0])
            clone_stats = {"candidates": 0, "selected": 0, "clipped": 0}
            after_clone = after_split_first
            split_added = after_split_first - before
            cloned = 0
            after_split = after_clone
        else:
            clone_stats = self.densify_and_clone(
                clone_grads,
                max_grad,
                extent,
                device,
                density_guided_clone=density_guided_clone,
                density_guided_clone_scale=density_guided_clone_scale,
                max_new_points=clone_budget,
                max_total_points=max_total_points,
            )
            after_clone = int(self.get_xyz.shape[0])
            split_stats = self.densify_and_split(
                split_grads,
                split_threshold,
                extent,
                device,
                max_new_points=split_budget,
                max_total_points=max_total_points,
            )
            after_split = int(self.get_xyz.shape[0])
            cloned = after_clone - before
            split_added = after_split - after_clone

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        pruned = int(prune_mask.sum().item())
        self.prune_points(prune_mask)
        after_prune = int(self.get_xyz.shape[0])

        torch.cuda.empty_cache()
        return {
            "before": before,
            "cloned": cloned,
            "split_added": split_added,
            "pruned": pruned,
            "after": after_prune,
            "mode": mode,
            "grad_mean": grad_stats["mean"],
            "grad_p90": grad_stats["p90"],
            "grad_p99": grad_stats["p99"],
            "grad_max": grad_stats["max"],
            "grad_over_threshold": grad_stats["over_threshold"],
            "grad_abs_mean": grad_abs_stats["mean"],
            "grad_abs_p90": grad_abs_stats["p90"],
            "grad_abs_p99": grad_abs_stats["p99"],
            "grad_abs_max": grad_abs_stats["max"],
            "grad_abs_over_threshold": grad_abs_stats["over_threshold"],
            "grad_threshold": float(max_grad),
            "grad_abs_threshold": float(split_threshold),
            "density_guided_clone": bool(density_guided_clone),
            "max_new_points": int(max_new_points) if max_new_points else 0,
            "max_total_points": int(max_total_points) if max_total_points else 0,
            "clone_candidates": clone_stats["candidates"],
            "clone_selected": clone_stats["selected"],
            "clone_clipped": clone_stats["clipped"],
            "split_candidates": split_stats["candidates"],
            "split_selected": split_stats["selected"],
            "split_clipped": split_stats["clipped"],
            "split_first": bool(effective_split_first),
            **recycle_stats,
            **consistency_stats,
        }

    def add_densification_stats(self, viewspace_point_tensor, update_filter, weight=1.0,
                                abs_viewspace_point_tensor=None, denom_weight=None,
                                abs_weight=None):
        grad = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        if abs_viewspace_point_tensor is None:
            abs_grad = grad
        else:
            abs_grad = torch.norm(abs_viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        if torch.is_tensor(weight):
            stat_weight = weight.to(grad.device, dtype=grad.dtype).reshape(-1, 1)
        else:
            stat_weight = float(weight)
        if denom_weight is None:
            denom_stat_weight = stat_weight
        elif torch.is_tensor(denom_weight):
            denom_stat_weight = denom_weight.to(grad.device, dtype=grad.dtype).reshape(-1, 1)
        else:
            denom_stat_weight = float(denom_weight)
        if abs_weight is None:
            abs_stat_weight = stat_weight
        elif torch.is_tensor(abs_weight):
            abs_stat_weight = abs_weight.to(grad.device, dtype=grad.dtype).reshape(-1, 1)
        else:
            abs_stat_weight = float(abs_weight)
        self.xyz_gradient_accum[update_filter] += stat_weight * grad
        self.xyz_gradient_accum_abs[update_filter] += abs_stat_weight * abs_grad
        self.denom[update_filter] += denom_stat_weight

    def adjust_scaling(self, scale_factor):
        self._scaling /= scale_factor

    def reset_scaling(self, scale_factor):
        self._scaling *= scale_factor  
