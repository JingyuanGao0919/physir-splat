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
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.is_6dof = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
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
        self.feature_lr = 0.0025
        self.thermal_feature_lr = 0.0025
        self.rgbt_rgb_loss_weight = 0.5
        self.rgbt_rgb_feature_lr_mult = 1.0
        self.rgbt_rgb_detach_geometry = 0
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 20_000
        self.densify_grad_threshold = 0.0007
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
        super().__init__(parser, "Optimization Parameters")


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
