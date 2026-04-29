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
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, device, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

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

    def densify_and_clone(self, grads, grad_threshold, scene_extent, device):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

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

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, device):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, device)
        self.densify_and_split(grads, max_grad, extent, device)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, weight=1.0):
        self.xyz_gradient_accum[update_filter] += float(weight) * torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                                             keepdim=True)
        self.denom[update_filter] += float(weight)

    def adjust_scaling(self, scale_factor):
        self._scaling /= scale_factor

    def reset_scaling(self, scale_factor):
        self._scaling *= scale_factor  
