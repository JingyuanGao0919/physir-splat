import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3
import cv2


class ConvLayer(nn.Module):
    def __init__(self, device, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding).to(device)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class PlanckNet(nn.Module):
    def __init__(self, device):
        super(PlanckNet, self).__init__()
        self.device = device
        self.layer1 = ConvLayer(device, in_channels=1, out_channels=16)
        self.layer2 = ConvLayer(device, in_channels=16, out_channels=16)

    def forward(self, x):
        x = x.to(self.device)
        if x.dim() != 3 or x.size(0) != 3:
            raise ValueError(f"Expected input shape [3, H, W], got {x.shape}")
        
        gray_channel = x[0:1, :, :].unsqueeze(0).to(self.device)  # [1, height, width] -> [1, 1, height, width]

        x1 = self.layer1(gray_channel)  # [1, 16, height, width]
        output = self.layer2(x1).squeeze(0)  # [1, 16, height, width]

        return output
