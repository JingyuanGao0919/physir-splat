import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import GeoRefineNet
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class GeoRefineModel:
    def __init__(self, device, is_blender=False, is_6dof=False):
        self.geometry = GeoRefineNet(device, is_blender=is_blender, is_6dof=is_6dof).to(device)
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time, cam_pos, view_dir):
            return self.geometry(xyz, time, cam_pos, view_dir)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.geometry.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "Geometry"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.geometry_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                         lr_final=training_args.position_lr_final,
                                                         lr_delay_mult=training_args.position_lr_delay_mult,
                                                         max_steps=training_args.geometry_lr_max_steps)

    def save_weights(self, model_path, iteration, subdir="Geometry"):
        out_weights_path = os.path.join(model_path, "{}/iteration_{}".format(subdir, iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.geometry.state_dict(), os.path.join(out_weights_path, 'geometry.pth'))

    def load_weights(self, model_path, iteration=-1, subdir="Geometry"):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, subdir))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "{}/iteration_{}/geometry.pth".format(subdir, loaded_iter))
        self.geometry.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "Geometry":
                lr = self.geometry_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
