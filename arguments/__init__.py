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

from argparse import ArgumentParser, Namespace
import sys
import os
import torch

class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, device, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = device
        self.eval = False
        self.data_branch = "auto"
        self.load_iteration = 0
        self.load_model_path = ""
        self.load2gpu_on_the_fly = False
        self.load_sparse_depth = False
        self.is_blender = False
        self.is_6dof = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if getattr(g, "load_model_path", ""):
            g.load_model_path = os.path.abspath(g.load_model_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.iterations = 30000
        self.warm_up = 3_000
        # self.warm_up = 0
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.atmos_lr_max_steps = 30_000
        self.planck_lr_max_steps = 30_000
        self.radiometric_lr_max_steps = 30_000
        self.geometry_lr_max_steps = 30_000
        self.geometry_time_multires = -1
        self.geometry_conditioning = "view"
        self.geometry_train_until_iter = 1_000_000
        self.geometry_grad_clip_norm = 0.0
        self.temporal_jitter_mode = "stack"
        self.temporal_jitter_scale = 1.0
        self.temporal_jitter_min_factor = 0.0
        self.geometry_motion_reg_weight = 0.0
        self.geometry_motion_reg_start_iter = 3000
        self.geometry_motion_reg_ramp_iters = 1000
        self.geometry_temporal_smooth_weight = 0.0
        self.geometry_temporal_smooth_start_iter = 3000
        self.geometry_temporal_smooth_ramp_iters = 1000
        self.feature_lr = 0.0025
        self.thermal_feature_lr = 0.0025
        self.rgbt_rgb_loss_weight = 0.5
        self.rgbt_rgb_feature_lr_mult = 1.0
        self.rgbt_rgb_detach_geometry = 0
        self.rgbt_save_best_eval_checkpoint = True
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.ir_gradient_loss_weight = 0.0
        self.ir_gradient_loss_start_iter = 500
        self.ir_gradient_loss_ramp_iters = 2000
        self.ir_robust_loss_weight = 0.0
        self.ir_robust_loss_start_iter = 3000
        self.ir_robust_loss_ramp_iters = 1000
        self.ir_robust_loss_beta = 0.03
        self.ir_robust_loss_min_weight = 0.25
        self.ir_rmse_loss_weight = 0.0
        self.ir_rmse_loss_start_iter = 1000
        self.ir_rmse_loss_ramp_iters = 2000
        self.ir_rmse_loss_eps = 1e-8
        self.ir_multiscale_loss_weight = 0.0
        self.ir_multiscale_loss_start_iter = 1000
        self.ir_multiscale_loss_ramp_iters = 2000
        self.ir_multiscale_loss_scales = "2,4"
        self.ir_multiscale_loss_mode = "rmse"
        self.ir_trimmed_loss_weight = 0.0
        self.ir_trimmed_loss_start_iter = 20_000
        self.ir_trimmed_loss_ramp_iters = 1000
        self.ir_trimmed_loss_quantile = 0.85
        self.ir_trimmed_loss_mode = "rmse"
        self.ir_trimmed_loss_smooth_window = 3
        self.ir_sparse_depth_weight = 0.0
        self.ir_sparse_depth_start_iter = 500
        self.ir_sparse_depth_ramp_iters = 1000
        self.ir_sparse_depth_min_acc = 0.30
        self.ir_sparse_depth_huber_beta = 0.02
        self.ir_sparse_depth_use_acc_normalize = True
        self.ir_sparse_depth_scale_factor = 4
        self.ir_late_consistency_start_iter = 20_000
        self.ir_late_consistency_weight = 0.2
        self.ir_late_ssim_weight = 0.2
        self.ir_eval_temporal_ensemble = False
        self.ir_eval_temporal_ensemble_samples = 3
        self.ir_eval_temporal_ensemble_radius = 0.5
        self.ir_eval_opacity_threshold = 0.0
        self.ir_temporal_affine = False
        self.ir_temporal_affine_mode = "mlp"
        self.ir_temporal_affine_start_iter = 1000
        self.ir_temporal_affine_lr = 0.001
        self.ir_temporal_affine_reg_weight = 0.01
        self.ir_temporal_affine_max_scale = 0.20
        self.ir_temporal_affine_max_bias = 0.08
        self.ir_temporal_affine_hidden_dim = 32
        self.ir_temporal_affine_num_freqs = 4
        self.ir_temporal_affine_grid_size = 8
        self.ir_temporal_affine_grid_tv_weight = 0.05
        self.ir_temporal_affine_table_smooth_weight = 0.1
        self.ir_temporal_affine_smooth_weight = 0.0
        self.ir_temporal_affine_smooth_start_iter = 1000
        self.ir_temporal_affine_smooth_ramp_iters = 1000
        self.ir_temporal_pose = False
        self.ir_temporal_pose_mode = "mlp"
        self.ir_temporal_pose_start_iter = 1000
        self.ir_temporal_pose_lr = 0.0005
        self.ir_temporal_pose_reg_weight = 0.01
        self.ir_temporal_pose_max_trans = 0.02
        self.ir_temporal_pose_max_rot_deg = 1.0
        self.ir_temporal_pose_table_smooth_weight = 0.1
        self.rgbt_rgb_gradient_loss_weight = 0.0
        self.diagnostic_log_every = 1000
        self.densify_log_every = 1000
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 20_000
        self.densify_grad_threshold = 0.0007
        self.use_dual_gradient_densification = False
        self.densify_grad_abs_threshold = 0.0004
        self.use_direction_aware_densification = False
        self.direction_aware_split_abs = False
        self.direction_aware_base = 0.8
        self.direction_aware_scale = 25.0
        self.direction_aware_power = 15.0
        self.densify_max_points = 0
        self.densify_max_growth = 0.0
        self.densify_max_new_points = 0
        self.densify_split_first = False
        self.densify_budget_recycle_points = 0
        self.densify_budget_recycle_start = 0.97
        self.densify_budget_recycle_opacity = 0.25
        self.densify_budget_recycle_grad_factor = 2.0
        self.use_density_guided_clone = False
        self.density_guided_clone_scale = 1.0
        self.needle_perturb_interval = 0
        self.needle_perturb_until_iter = 20000
        self.needle_perturb_ratio_min = 0.8
        self.needle_perturb_ratio_max = 0.999
        self.densify_screen_weight_power = 0.0
        self.densify_screen_weight_max = 64.0
        self.densify_residual_weight_power = 0.0
        self.densify_residual_weight_max = 4.0
        self.densify_residual_weight_min = 0.25
        self.densify_residual_weight_quantile = 0.75
        self.densify_residual_weight_start_iter = 3000
        self.densify_residual_weight_blur = 1
        self.densify_residual_weight_split_only = False
        self.phys_probe_every = 1000
        self.phys_probe_steps = 128
        self.phys_probe_lr = 0.05
        self.phys_probe_temp_min = 260.0
        self.phys_probe_temp_max = 360.0
        self.phys_probe_lambda_min_um = 7.5
        self.phys_probe_lambda_max_um = 13.5
        self.phys_probe_e_min = 0.85
        self.phys_probe_e_max = 0.99
        self.phys_probe_e_init = 0.95
        self.phys_probe_delta_max = 0.1
        self.phys_probe_lambda_e_prior = 0.01
        self.phys_probe_lambda_a_prior = 0.01
        self.phys_probe_lambda_delta = 0.02
        self.phys_probe_huber_beta = 0.02
        self.phys_probe_ambient_quantile = 0.1
        self.phys_decouple_clusters = 16
        self.phys_decouple_cluster_samples = 50000
        self.phys_decouple_cluster_iters = 20
        self.phys_decouple_smooth_samples = 60000
        self.phys_decouple_steps = 128
        self.phys_decouple_lr = 0.05
        self.phys_decouple_delta_max = 0.05
        self.phys_decouple_lambda_temp_smooth = 0.05
        self.phys_decouple_temp_smooth_beta = 0.02
        self.phys_decouple_lambda_e_prior = 0.01
        self.phys_decouple_lambda_a_prior = 0.01
        self.phys_decouple_lambda_delta = 0.02
        self.phys_decouple_rgb_feature_weight = 1.0
        self.phys_decouple_xyz_feature_weight = 0.15
        self.phys_decouple_rest_feature_weight = 0.05
        self.phys_guidance_start_iter = 20_000
        self.phys_guidance_ramp_iters = 5_000
        self.phys_guidance_max_weight = 0.0
        self.phys_guidance_q_weight = 1.0
        self.phys_guidance_rest_weight = 0.0
        self.phys_guidance_delta_weight = 0.0
        self.phys_guidance_huber_beta = 0.02
        self.phys_guidance_rest_quantile = 0.75
        self.phys_render_start_iter = 20_000
        self.phys_render_blend_iters = 5_000
        self.phys_render_max_blend = 0.0
        self.phys_main_render = False
        self.phys_eval_with_render = False
        self.phys_aux_start_iter = 20_000
        self.phys_aux_ramp_iters = 5_000
        self.phys_aux_mse_max_weight = 0.05
        self.phys_aux_conf_mix = 0.5
        self.phys_aux_conf_min = 0.75
        self.phys_aux_conf_max = 1.25
        super().__init__(parser, "Optimization Parameters", sentinel)


def get_combined_args(parser: ArgumentParser, device):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass

    cfgfile_string = cfgfile_string.replace("device(type='cuda', index=0)", f"torch.device('cuda:0')") #-----------
    args_cfgfile = eval(cfgfile_string, {'torch': torch, 'device': device, 'Namespace': Namespace}) #---------------

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
