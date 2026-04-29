import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.radiometric_utils import RadiometricNet
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class RadiometricModel:
    def __init__(self, device):
        self.radiometric = RadiometricNet(device).to(device)
        self.optimizer = None
        self.spatial_lr_scale = 0.1

    def step(self, x):
        return self.radiometric(x)

    def train_setting(self, training_args):
        l = [
             {'params': list(self.radiometric.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "Radiometric"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.radiometric_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                            lr_final=training_args.position_lr_final,
                                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                                            max_steps=training_args.radiometric_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "Radiometric/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.radiometric.state_dict(), os.path.join(out_weights_path, 'radiometric.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "Radiometric"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "Radiometric/iteration_{}/radiometric.pth".format(loaded_iter))
        self.radiometric.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "Radiometric":
                lr = self.radiometric_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
