"""Veta-inspired position offsets for the effective IR rendering geometry."""
import torch
from torch import nn
import torch.nn.functional as F


def embed(value):
    terms = [value]
    for band in range(10):
        terms.extend((torch.sin(value * 2 ** band), torch.cos(value * 2 ** band)))
    return torch.cat(terms, dim=-1)


class ViewDeformation(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 63 + 21 + 3 + 3
        self.layers = nn.ModuleList([
            nn.Linear(channels if index == 0 else 256 + channels if index == 5 else 256, 256)
            for index in range(8)])
        self.position = nn.Linear(256, 3)
        nn.init.zeros_(self.position.weight)
        nn.init.zeros_(self.position.bias)

    def forward(self, xyz, frame, center, direction):
        inputs = torch.cat((embed(xyz), embed(frame), center, direction), dim=-1)
        hidden = inputs
        for index, layer in enumerate(self.layers):
            hidden = F.relu(layer(hidden))
            if index == 4:
                hidden = torch.cat((inputs, hidden), dim=-1)
        return self.position(hidden)


class ShiftedGaussians:
    def __init__(self, gaussians, displacement):
        self.base = gaussians
        self.get_xyz = gaussians.get_xyz + displacement

    def __getattr__(self, name):
        return getattr(self.base, name)


class SubsetGaussians:
    """Pointwise radiative fields on the native projection support."""
    def __init__(self, gaussians, selected):
        self.base = gaussians
        self.selected = selected

    def __getattr__(self, name):
        value = getattr(self.base, name)
        if name in ('get_xyz', 'get_scaling', 'get_rotation', 'source', 'source_sh', 'reflectivity', 'source_spline', 'source_detail', 'source_fine', 'source_finer'):
            return value[self.selected]
        return value
