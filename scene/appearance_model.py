import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.system_utils import searchForMaxIteration


class TemporalAffineNet(nn.Module):
    def __init__(self, max_scale=0.20, max_bias=0.08, hidden_dim=32, num_freqs=4):
        super().__init__()
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.num_freqs = int(num_freqs)
        in_dim = 1 + 2 * self.num_freqs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _encode_time(self, fid):
        t = fid.reshape(1, 1).to(next(self.parameters()).device)
        features = [t]
        for i in range(self.num_freqs):
            freq = float(2 ** i)
            features.append(torch.sin(freq * torch.pi * t))
            features.append(torch.cos(freq * torch.pi * t))
        return torch.cat(features, dim=-1)

    def forward(self, image, fid, update_stats=True):
        raw = self.net(self._encode_time(fid))
        scale_delta = self.max_scale * torch.tanh(raw[:, 0]).view(1, 1, 1)
        bias = self.max_bias * torch.tanh(raw[:, 1]).view(1, 1, 1)
        corrected = image * (1.0 + scale_delta) + bias
        reg = scale_delta.pow(2).mean() + bias.pow(2).mean()
        return corrected, scale_delta, bias, reg


class TemporalAffineTableNet(nn.Module):
    def __init__(self, train_fids, max_scale=0.20, max_bias=0.08, smooth_weight=0.1):
        super().__init__()
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.smooth_weight = float(smooth_weight)
        if train_fids is None or len(train_fids) == 0:
            fids = torch.linspace(0.0, 1.0, 2)
        else:
            fids = torch.as_tensor(train_fids, dtype=torch.float32).reshape(-1)
            fids = torch.unique(torch.sort(fids).values)
            if fids.numel() == 1:
                fids = torch.cat([fids, (fids + 1e-3).clamp(max=1.0)])
        self.register_buffer("fids", fids)
        self.raw_affine = nn.Parameter(torch.zeros(fids.shape[0], 2))

    def _interpolate_raw(self, fid):
        t = fid.reshape(()).to(self.fids.device, dtype=self.fids.dtype)
        if self.fids.numel() == 1:
            return self.raw_affine[0]
        hi = torch.searchsorted(self.fids, t, right=True).clamp(1, self.fids.numel() - 1)
        lo = hi - 1
        denom = (self.fids[hi] - self.fids[lo]).clamp_min(1e-6)
        w = ((t - self.fids[lo]) / denom).clamp(0.0, 1.0)
        return self.raw_affine[lo] * (1.0 - w) + self.raw_affine[hi] * w

    def forward(self, image, fid, update_stats=True):
        raw = self._interpolate_raw(fid)
        scale_delta = self.max_scale * torch.tanh(raw[0]).view(1, 1, 1)
        bias = self.max_bias * torch.tanh(raw[1]).view(1, 1, 1)
        corrected = image * (1.0 + scale_delta) + bias
        affine_reg = scale_delta.pow(2).mean() + bias.pow(2).mean()
        if self.raw_affine.shape[0] > 1 and self.smooth_weight > 0.0:
            table_scale = self.max_scale * torch.tanh(self.raw_affine[:, 0])
            table_bias = self.max_bias * torch.tanh(self.raw_affine[:, 1])
            smooth_reg = (table_scale[1:] - table_scale[:-1]).pow(2).mean()
            smooth_reg = smooth_reg + (table_bias[1:] - table_bias[:-1]).pow(2).mean()
            affine_reg = affine_reg + self.smooth_weight * smooth_reg
        return corrected, scale_delta, bias, affine_reg


class TemporalAffineTablePolyNet(nn.Module):
    def __init__(self, train_fids, max_scale=0.20, max_bias=0.08, smooth_weight=0.1):
        super().__init__()
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.smooth_weight = float(smooth_weight)
        self.num_basis = 6
        self._basis_cache = {}
        self._last_stats = {}
        if train_fids is None or len(train_fids) == 0:
            fids = torch.linspace(0.0, 1.0, 2)
        else:
            fids = torch.as_tensor(train_fids, dtype=torch.float32).reshape(-1)
            fids = torch.unique(torch.sort(fids).values)
            if fids.numel() == 1:
                fids = torch.cat([fids, (fids + 1e-3).clamp(max=1.0)])
        self.register_buffer("fids", fids)
        self.raw_coeff = nn.Parameter(torch.zeros(fids.shape[0], 2, self.num_basis))

    def _basis(self, height, width, device, dtype):
        key = (int(height), int(width), str(device), dtype)
        cached = self._basis_cache.get(key)
        if cached is not None:
            return cached
        y = torch.linspace(-1.0, 1.0, int(height), device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, int(width), device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        basis = torch.stack(
            [
                torch.ones_like(xx),
                xx,
                yy,
                xx * yy,
                xx.square() - 1.0 / 3.0,
                yy.square() - 1.0 / 3.0,
            ],
            dim=0,
        ).unsqueeze(0)
        self._basis_cache[key] = basis
        return basis

    def _interpolate_raw(self, fid):
        t = fid.reshape(()).to(self.fids.device, dtype=self.fids.dtype)
        if self.fids.numel() == 1:
            return self.raw_coeff[0]
        hi = torch.searchsorted(self.fids, t, right=True).clamp(1, self.fids.numel() - 1)
        lo = hi - 1
        denom = (self.fids[hi] - self.fids[lo]).clamp_min(1e-6)
        w = ((t - self.fids[lo]) / denom).clamp(0.0, 1.0)
        return self.raw_coeff[lo] * (1.0 - w) + self.raw_coeff[hi] * w

    def forward(self, image, fid, update_stats=True):
        _, height, width = image.shape
        basis = self._basis(height, width, image.device, image.dtype)
        raw = self._interpolate_raw(fid).view(1, 2, self.num_basis, 1, 1)
        scale_map = self.max_scale * torch.tanh((raw[:, 0] * basis).sum(dim=1, keepdim=True))
        bias_map = self.max_bias * torch.tanh((raw[:, 1] * basis).sum(dim=1, keepdim=True))
        corrected = image * (1.0 + scale_map.squeeze(0)) + bias_map.squeeze(0)
        affine_reg = scale_map.pow(2).mean() + bias_map.pow(2).mean()
        coeff_reg = 1e-4 * raw.pow(2).mean()
        if self.raw_coeff.shape[0] > 1 and self.smooth_weight > 0.0:
            table_scale = self.max_scale * torch.tanh(self.raw_coeff[:, 0])
            table_bias = self.max_bias * torch.tanh(self.raw_coeff[:, 1])
            smooth_reg = (table_scale[1:] - table_scale[:-1]).pow(2).mean()
            smooth_reg = smooth_reg + (table_bias[1:] - table_bias[:-1]).pow(2).mean()
            affine_reg = affine_reg + self.smooth_weight * smooth_reg
        if update_stats:
            with torch.no_grad():
                table_scale = self.max_scale * torch.tanh(self.raw_coeff[:, 0])
                table_bias = self.max_bias * torch.tanh(self.raw_coeff[:, 1])
                self._last_stats = {
                    "poly_scale_mean": float(scale_map.detach().mean().item()),
                    "poly_scale_std": float(scale_map.detach().std(unbiased=False).item()),
                    "poly_scale_min": float(scale_map.detach().min().item()),
                    "poly_scale_max": float(scale_map.detach().max().item()),
                    "poly_bias_mean": float(bias_map.detach().mean().item()),
                    "poly_bias_std": float(bias_map.detach().std(unbiased=False).item()),
                    "poly_bias_min": float(bias_map.detach().min().item()),
                    "poly_bias_max": float(bias_map.detach().max().item()),
                    "table_poly_scale_coeff_std": float(table_scale.std(unbiased=False).item()),
                    "table_poly_scale_coeff_max": float(table_scale.abs().max().item()),
                    "table_poly_bias_coeff_std": float(table_bias.std(unbiased=False).item()),
                    "table_poly_bias_coeff_max": float(table_bias.abs().max().item()),
                }
        return corrected, scale_map, bias_map, affine_reg + coeff_reg

    def diagnostic_stats(self):
        return dict(self._last_stats)


class TemporalAffinePolyNet(nn.Module):
    def __init__(self, max_scale=0.20, max_bias=0.08, hidden_dim=32, num_freqs=4):
        super().__init__()
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.num_freqs = int(num_freqs)
        self.num_basis = 6
        self._basis_cache = {}
        self._last_stats = {}
        in_dim = 1 + 2 * self.num_freqs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.num_basis),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _encode_time(self, fid):
        t = fid.reshape(1, 1).to(next(self.parameters()).device)
        features = [t]
        for i in range(self.num_freqs):
            freq = float(2 ** i)
            features.append(torch.sin(freq * torch.pi * t))
            features.append(torch.cos(freq * torch.pi * t))
        return torch.cat(features, dim=-1)

    def _basis(self, height, width, device, dtype):
        key = (int(height), int(width), str(device), dtype)
        cached = self._basis_cache.get(key)
        if cached is not None:
            return cached
        y = torch.linspace(-1.0, 1.0, int(height), device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, int(width), device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        basis = torch.stack(
            [
                torch.ones_like(xx),
                xx,
                yy,
                xx * yy,
                xx.square() - 1.0 / 3.0,
                yy.square() - 1.0 / 3.0,
            ],
            dim=0,
        ).unsqueeze(0)
        self._basis_cache[key] = basis
        return basis

    def forward(self, image, fid, update_stats=True):
        _, height, width = image.shape
        basis = self._basis(height, width, image.device, image.dtype)
        raw = self.net(self._encode_time(fid)).view(1, 2, self.num_basis, 1, 1)
        scale_map = self.max_scale * torch.tanh((raw[:, 0] * basis).sum(dim=1, keepdim=True))
        bias_map = self.max_bias * torch.tanh((raw[:, 1] * basis).sum(dim=1, keepdim=True))
        corrected = image * (1.0 + scale_map.squeeze(0)) + bias_map.squeeze(0)
        affine_reg = scale_map.pow(2).mean() + bias_map.pow(2).mean()
        coeff_reg = 1e-3 * raw.pow(2).mean()
        if update_stats:
            self._last_stats = {
                "poly_scale_mean": float(scale_map.detach().mean().item()),
                "poly_scale_std": float(scale_map.detach().std(unbiased=False).item()),
                "poly_scale_min": float(scale_map.detach().min().item()),
                "poly_scale_max": float(scale_map.detach().max().item()),
                "poly_bias_mean": float(bias_map.detach().mean().item()),
                "poly_bias_std": float(bias_map.detach().std(unbiased=False).item()),
                "poly_bias_min": float(bias_map.detach().min().item()),
                "poly_bias_max": float(bias_map.detach().max().item()),
            }
        return corrected, scale_map, bias_map, affine_reg + coeff_reg

    def diagnostic_stats(self):
        return dict(self._last_stats)


class TemporalAffineTableGridNet(nn.Module):
    def __init__(self, train_fids, max_scale=0.20, max_bias=0.08, smooth_weight=0.1,
                 grid_size=8, tv_weight=0.05):
        super().__init__()
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.smooth_weight = float(smooth_weight)
        self.grid_size = int(grid_size)
        self.tv_weight = float(tv_weight)
        self._last_stats = {}
        if train_fids is None or len(train_fids) == 0:
            fids = torch.linspace(0.0, 1.0, 2)
        else:
            fids = torch.as_tensor(train_fids, dtype=torch.float32).reshape(-1)
            fids = torch.unique(torch.sort(fids).values)
            if fids.numel() == 1:
                fids = torch.cat([fids, (fids + 1e-3).clamp(max=1.0)])
        self.register_buffer("fids", fids)
        self.raw_grid = nn.Parameter(torch.zeros(fids.shape[0], 2, self.grid_size, self.grid_size))

    def _interpolate_raw(self, fid):
        t = fid.reshape(()).to(self.fids.device, dtype=self.fids.dtype)
        if self.fids.numel() == 1:
            return self.raw_grid[0]
        hi = torch.searchsorted(self.fids, t, right=True).clamp(1, self.fids.numel() - 1)
        lo = hi - 1
        denom = (self.fids[hi] - self.fids[lo]).clamp_min(1e-6)
        w = ((t - self.fids[lo]) / denom).clamp(0.0, 1.0)
        return self.raw_grid[lo] * (1.0 - w) + self.raw_grid[hi] * w

    def _tv_reg(self, field):
        dx = field[..., :, 1:] - field[..., :, :-1]
        dy = field[..., 1:, :] - field[..., :-1, :]
        return dx.pow(2).mean() + dy.pow(2).mean()

    def forward(self, image, fid, update_stats=True):
        _, height, width = image.shape
        raw = self._interpolate_raw(fid).view(1, 2, self.grid_size, self.grid_size)
        low_scale = self.max_scale * torch.tanh(raw[:, 0:1])
        low_bias = self.max_bias * torch.tanh(raw[:, 1:2])
        scale_map = F.interpolate(low_scale, size=(height, width), mode="bilinear", align_corners=True)
        bias_map = F.interpolate(low_bias, size=(height, width), mode="bilinear", align_corners=True)
        corrected = image * (1.0 + scale_map.squeeze(0)) + bias_map.squeeze(0)
        affine_reg = scale_map.pow(2).mean() + bias_map.pow(2).mean()
        tv_reg = self.tv_weight * (self._tv_reg(low_scale) + self._tv_reg(low_bias))
        coeff_reg = 1e-4 * raw.pow(2).mean()
        if self.raw_grid.shape[0] > 1 and self.smooth_weight > 0.0:
            table_scale = self.max_scale * torch.tanh(self.raw_grid[:, 0:1])
            table_bias = self.max_bias * torch.tanh(self.raw_grid[:, 1:2])
            temporal_smooth = (table_scale[1:] - table_scale[:-1]).pow(2).mean()
            temporal_smooth = temporal_smooth + (table_bias[1:] - table_bias[:-1]).pow(2).mean()
            affine_reg = affine_reg + self.smooth_weight * temporal_smooth
        if update_stats:
            with torch.no_grad():
                table_scale = self.max_scale * torch.tanh(self.raw_grid[:, 0:1])
                table_bias = self.max_bias * torch.tanh(self.raw_grid[:, 1:2])
                table_delta = (
                    (table_scale[1:] - table_scale[:-1]).pow(2).mean()
                    + (table_bias[1:] - table_bias[:-1]).pow(2).mean()
                    if self.raw_grid.shape[0] > 1
                    else torch.zeros((), device=image.device, dtype=image.dtype)
                )
                self._last_stats = {
                    "grid_scale_mean": float(scale_map.detach().mean().item()),
                    "grid_scale_std": float(scale_map.detach().std(unbiased=False).item()),
                    "grid_scale_min": float(scale_map.detach().min().item()),
                    "grid_scale_max": float(scale_map.detach().max().item()),
                    "grid_bias_mean": float(bias_map.detach().mean().item()),
                    "grid_bias_std": float(bias_map.detach().std(unbiased=False).item()),
                    "grid_bias_min": float(bias_map.detach().min().item()),
                    "grid_bias_max": float(bias_map.detach().max().item()),
                    "grid_tv": float(tv_reg.detach().item()),
                    "table_grid_scale_std": float(table_scale.detach().std(unbiased=False).item()),
                    "table_grid_scale_max": float(table_scale.detach().abs().max().item()),
                    "table_grid_bias_std": float(table_bias.detach().std(unbiased=False).item()),
                    "table_grid_bias_max": float(table_bias.detach().abs().max().item()),
                    "table_grid_temporal_delta": float(table_delta.detach().item()),
                }
        return corrected, scale_map, bias_map, affine_reg + tv_reg + coeff_reg

    def diagnostic_stats(self):
        return dict(self._last_stats)


class TemporalAffineGridNet(nn.Module):
    def __init__(self, max_scale=0.20, max_bias=0.08, hidden_dim=32, num_freqs=4,
                 grid_size=8, tv_weight=0.05):
        super().__init__()
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.num_freqs = int(num_freqs)
        self.grid_size = int(grid_size)
        self.tv_weight = float(tv_weight)
        self._last_stats = {}
        in_dim = 1 + 2 * self.num_freqs
        out_dim = 2 * self.grid_size * self.grid_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _encode_time(self, fid):
        t = fid.reshape(1, 1).to(next(self.parameters()).device)
        features = [t]
        for i in range(self.num_freqs):
            freq = float(2 ** i)
            features.append(torch.sin(freq * torch.pi * t))
            features.append(torch.cos(freq * torch.pi * t))
        return torch.cat(features, dim=-1)

    def _tv_reg(self, field):
        dx = field[..., :, 1:] - field[..., :, :-1]
        dy = field[..., 1:, :] - field[..., :-1, :]
        return dx.pow(2).mean() + dy.pow(2).mean()

    def forward(self, image, fid, update_stats=True):
        _, height, width = image.shape
        raw = self.net(self._encode_time(fid)).view(1, 2, self.grid_size, self.grid_size)
        low_scale = self.max_scale * torch.tanh(raw[:, 0:1])
        low_bias = self.max_bias * torch.tanh(raw[:, 1:2])
        scale_map = F.interpolate(low_scale, size=(height, width), mode="bilinear", align_corners=True)
        bias_map = F.interpolate(low_bias, size=(height, width), mode="bilinear", align_corners=True)
        corrected = image * (1.0 + scale_map.squeeze(0)) + bias_map.squeeze(0)
        affine_reg = scale_map.pow(2).mean() + bias_map.pow(2).mean()
        tv_reg = self.tv_weight * (self._tv_reg(low_scale) + self._tv_reg(low_bias))
        coeff_reg = 1e-4 * raw.pow(2).mean()
        if update_stats:
            self._last_stats = {
                "grid_scale_mean": float(scale_map.detach().mean().item()),
                "grid_scale_std": float(scale_map.detach().std(unbiased=False).item()),
                "grid_scale_min": float(scale_map.detach().min().item()),
                "grid_scale_max": float(scale_map.detach().max().item()),
                "grid_bias_mean": float(bias_map.detach().mean().item()),
                "grid_bias_std": float(bias_map.detach().std(unbiased=False).item()),
                "grid_bias_min": float(bias_map.detach().min().item()),
                "grid_bias_max": float(bias_map.detach().max().item()),
                "grid_tv": float(tv_reg.detach().item()),
            }
        return corrected, scale_map, bias_map, affine_reg + tv_reg + coeff_reg

    def diagnostic_stats(self):
        return dict(self._last_stats)


class TemporalAffineModel:
    def __init__(self, device, max_scale=0.20, max_bias=0.08, hidden_dim=32, num_freqs=4,
                 mode="mlp", train_fids=None, table_smooth_weight=0.1,
                 grid_size=8, grid_tv_weight=0.05):
        self.device = device
        self.max_scale = max_scale
        self.max_bias = max_bias
        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        self.train_fids = train_fids
        self.table_smooth_weight = table_smooth_weight
        self.grid_size = grid_size
        self.grid_tv_weight = grid_tv_weight
        self.mode = str(mode).lower()
        self.appearance = self._build(self.mode, train_fids).to(device)
        self.optimizer = None

    def _build(self, mode, train_fids=None):
        if str(mode).lower() == "table":
            self.mode = "table"
            return TemporalAffineTableNet(
                train_fids=train_fids,
                max_scale=self.max_scale,
                max_bias=self.max_bias,
                smooth_weight=self.table_smooth_weight,
            )
        elif str(mode).lower() in ("table_poly", "tablepoly", "poly_table"):
            self.mode = "table_poly"
            return TemporalAffineTablePolyNet(
                train_fids=train_fids,
                max_scale=self.max_scale,
                max_bias=self.max_bias,
                smooth_weight=self.table_smooth_weight,
            )
        elif str(mode).lower() in ("table_grid", "tablegrid", "grid_table"):
            self.mode = "table_grid"
            return TemporalAffineTableGridNet(
                train_fids=train_fids,
                max_scale=self.max_scale,
                max_bias=self.max_bias,
                smooth_weight=self.table_smooth_weight,
                grid_size=self.grid_size,
                tv_weight=self.grid_tv_weight,
            )
        elif str(mode).lower() == "poly":
            self.mode = "poly"
            return TemporalAffinePolyNet(
                max_scale=self.max_scale,
                max_bias=self.max_bias,
                hidden_dim=self.hidden_dim,
                num_freqs=self.num_freqs,
            )
        elif str(mode).lower() == "grid":
            self.mode = "grid"
            return TemporalAffineGridNet(
                max_scale=self.max_scale,
                max_bias=self.max_bias,
                hidden_dim=self.hidden_dim,
                num_freqs=self.num_freqs,
                grid_size=self.grid_size,
                tv_weight=self.grid_tv_weight,
            )
        else:
            self.mode = "mlp"
            return TemporalAffineNet(
                max_scale=self.max_scale,
                max_bias=self.max_bias,
                hidden_dim=self.hidden_dim,
                num_freqs=self.num_freqs,
            )

    def step(self, image, fid, update_stats=True):
        return self.appearance(image, fid, update_stats=update_stats)

    def smoothness_reg(self, image, fid, delta):
        delta = float(delta)
        if delta <= 0.0:
            return image.new_tensor(0.0)

        image_for_fields = image.detach()
        fid_tensor = fid.reshape(()).to(image_for_fields.device, dtype=image_for_fields.dtype)
        _, scale_center, bias_center, _ = self.step(image_for_fields, fid_tensor, update_stats=False)
        regs = []
        for sign in (-1.0, 1.0):
            neighbor_fid = (fid_tensor + sign * delta).clamp(0.0, 1.0)
            if torch.abs(neighbor_fid - fid_tensor) <= 1e-8:
                continue
            _, scale_neighbor, bias_neighbor, _ = self.step(image_for_fields, neighbor_fid, update_stats=False)
            regs.append(
                (scale_center - scale_neighbor).pow(2).mean()
                + (bias_center - bias_neighbor).pow(2).mean()
            )
        if not regs:
            return image.new_tensor(0.0)
        return torch.stack(regs).mean()

    def diagnostic_stats(self):
        if hasattr(self.appearance, "diagnostic_stats"):
            return self.appearance.diagnostic_stats()
        if self.mode != "table" or not hasattr(self.appearance, "raw_affine"):
            return {}
        with torch.no_grad():
            raw = self.appearance.raw_affine.detach()
            scale = self.appearance.max_scale * torch.tanh(raw[:, 0])
            bias = self.appearance.max_bias * torch.tanh(raw[:, 1])
            return {
                "table_scale_std": float(scale.std(unbiased=False).item()),
                "table_scale_min": float(scale.min().item()),
                "table_scale_max": float(scale.max().item()),
                "table_bias_std": float(bias.std(unbiased=False).item()),
                "table_bias_min": float(bias.min().item()),
                "table_bias_max": float(bias.max().item()),
            }

    def train_setting(self, training_args):
        lr = getattr(training_args, "ir_temporal_affine_lr", 0.001)
        self.optimizer = torch.optim.Adam(
            [{"params": list(self.appearance.parameters()), "lr": lr, "name": "TemporalAffine"}],
            lr=0.0,
            eps=1e-15,
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "TemporalAffine/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(
            {"mode": self.mode, "state_dict": self.appearance.state_dict()},
            os.path.join(out_weights_path, "temporal_affine.pth"),
        )

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "TemporalAffine"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(
            model_path, "TemporalAffine/iteration_{}/temporal_affine.pth".format(loaded_iter)
        )
        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            saved_mode = str(state.get("mode", self.mode)).lower()
            state_dict = state["state_dict"]
            if saved_mode != self.mode:
                train_fids = self.train_fids
                if saved_mode in ("table", "table_poly", "table_grid") and "fids" in state_dict:
                    train_fids = state_dict["fids"].detach().cpu().tolist()
                self.appearance = self._build(saved_mode, train_fids).to(self.device)
            self.appearance.load_state_dict(state_dict)
            self.mode = saved_mode
        else:
            self.appearance.load_state_dict(state)
        return True
