import os

import torch
import torch.nn as nn

from utils.system_utils import searchForMaxIteration


def _encode_time(fid, num_freqs):
    t = fid.reshape(1, 1)
    features = [t]
    for i in range(int(num_freqs)):
        freq = float(2 ** i)
        features.append(torch.sin(freq * torch.pi * t))
        features.append(torch.cos(freq * torch.pi * t))
    return torch.cat(features, dim=-1)


def _axis_angle_to_matrix(rotvec):
    rotvec = rotvec.reshape(3)
    theta = torch.linalg.norm(rotvec).clamp_min(1e-9)
    axis = rotvec / theta
    x, y, z = axis.unbind()
    zero = torch.zeros((), device=rotvec.device, dtype=rotvec.dtype)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    return eye + torch.sin(theta) * k + (1.0 - torch.cos(theta)) * (k @ k)


class TemporalPoseNet(nn.Module):
    def __init__(self, max_trans=0.02, max_rot_deg=1.0, hidden_dim=32, num_freqs=4):
        super().__init__()
        self.max_trans = float(max_trans)
        self.max_rot = float(max_rot_deg) * torch.pi / 180.0
        self.num_freqs = int(num_freqs)
        in_dim = 1 + 2 * self.num_freqs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fid, extent):
        raw = self.net(_encode_time(fid.to(next(self.parameters()).device), self.num_freqs)).reshape(6)
        trans = float(extent) * self.max_trans * torch.tanh(raw[:3])
        rotvec = self.max_rot * torch.tanh(raw[3:])
        reg = (trans / max(float(extent), 1e-6)).pow(2).mean() + rotvec.pow(2).mean()
        return trans, rotvec, reg


class TemporalPoseTableNet(nn.Module):
    def __init__(self, train_fids, max_trans=0.02, max_rot_deg=1.0, smooth_weight=0.1):
        super().__init__()
        self.max_trans = float(max_trans)
        self.max_rot = float(max_rot_deg) * torch.pi / 180.0
        self.smooth_weight = float(smooth_weight)
        if train_fids is None or len(train_fids) == 0:
            fids = torch.linspace(0.0, 1.0, 2)
        else:
            fids = torch.as_tensor(train_fids, dtype=torch.float32).reshape(-1)
            fids = torch.unique(torch.sort(fids).values)
            if fids.numel() == 1:
                fids = torch.cat([fids, (fids + 1e-3).clamp(max=1.0)])
        self.register_buffer("fids", fids)
        self.raw_pose = nn.Parameter(torch.zeros(fids.shape[0], 6))

    def _interpolate_raw(self, fid):
        t = fid.reshape(()).to(self.fids.device, dtype=self.fids.dtype)
        if self.fids.numel() == 1:
            return self.raw_pose[0]
        hi = torch.searchsorted(self.fids, t, right=True).clamp(1, self.fids.numel() - 1)
        lo = hi - 1
        denom = (self.fids[hi] - self.fids[lo]).clamp_min(1e-6)
        w = ((t - self.fids[lo]) / denom).clamp(0.0, 1.0)
        return self.raw_pose[lo] * (1.0 - w) + self.raw_pose[hi] * w

    def _decoded_table(self, extent):
        trans = float(extent) * self.max_trans * torch.tanh(self.raw_pose[:, :3])
        rotvec = self.max_rot * torch.tanh(self.raw_pose[:, 3:])
        return trans, rotvec

    def forward(self, fid, extent):
        raw = self._interpolate_raw(fid)
        trans = float(extent) * self.max_trans * torch.tanh(raw[:3])
        rotvec = self.max_rot * torch.tanh(raw[3:])
        reg = (trans / max(float(extent), 1e-6)).pow(2).mean() + rotvec.pow(2).mean()
        if self.raw_pose.shape[0] > 1 and self.smooth_weight > 0.0:
            table_trans, table_rot = self._decoded_table(extent)
            smooth_reg = ((table_trans[1:] - table_trans[:-1]) / max(float(extent), 1e-6)).pow(2).mean()
            smooth_reg = smooth_reg + (table_rot[1:] - table_rot[:-1]).pow(2).mean()
            reg = reg + self.smooth_weight * smooth_reg
        return trans, rotvec, reg


class TemporalPoseModel:
    def __init__(self, device, max_trans=0.02, max_rot_deg=1.0, hidden_dim=32, num_freqs=4,
                 mode="mlp", train_fids=None, table_smooth_weight=0.1):
        self.device = device
        self.max_trans = max_trans
        self.max_rot_deg = max_rot_deg
        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        self.train_fids = train_fids
        self.table_smooth_weight = table_smooth_weight
        self.mode = str(mode).lower()
        self.pose = self._build(self.mode, train_fids).to(device)
        self.optimizer = None
        self._last_trans = None
        self._last_rotvec = None

    def _build(self, mode, train_fids=None):
        if str(mode).lower() == "table":
            self.mode = "table"
            return TemporalPoseTableNet(
                train_fids=train_fids,
                max_trans=self.max_trans,
                max_rot_deg=self.max_rot_deg,
                smooth_weight=self.table_smooth_weight,
            )
        self.mode = "mlp"
        return TemporalPoseNet(
            max_trans=self.max_trans,
            max_rot_deg=self.max_rot_deg,
            hidden_dim=self.hidden_dim,
            num_freqs=self.num_freqs,
        )

    def step(self, xyz, fid, extent):
        trans, rotvec, reg = self.pose(fid, extent)
        xyz_detached = xyz.detach()
        center = xyz_detached.mean(dim=0, keepdim=True)
        rot = _axis_angle_to_matrix(rotvec)
        rotated = (xyz_detached - center) @ rot.transpose(0, 1) + center
        delta_xyz = rotated - xyz_detached + trans.view(1, 3)
        self._last_trans = trans.detach()
        self._last_rotvec = rotvec.detach()
        return delta_xyz, trans.detach(), rotvec.detach(), reg

    def diagnostic_stats(self, extent=1.0):
        stats = {}
        if self._last_trans is not None:
            stats["trans_norm"] = float(self._last_trans.norm().item())
        if self._last_rotvec is not None:
            stats["rot_deg"] = float(self._last_rotvec.norm().item() * 180.0 / torch.pi)
        if self.mode == "table" and hasattr(self.pose, "_decoded_table"):
            with torch.no_grad():
                trans, rot = self.pose._decoded_table(extent)
                trans_norm = torch.linalg.norm(trans, dim=-1)
                rot_deg = torch.linalg.norm(rot, dim=-1) * 180.0 / torch.pi
                stats.update(
                    {
                        "table_trans_std": float(trans_norm.std(unbiased=False).item()),
                        "table_trans_max": float(trans_norm.max().item()),
                        "table_rot_std": float(rot_deg.std(unbiased=False).item()),
                        "table_rot_max": float(rot_deg.max().item()),
                    }
                )
        return stats

    def train_setting(self, training_args):
        lr = getattr(training_args, "ir_temporal_pose_lr", 0.0005)
        self.optimizer = torch.optim.Adam(
            [{"params": list(self.pose.parameters()), "lr": lr, "name": "TemporalPose"}],
            lr=0.0,
            eps=1e-15,
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "TemporalPose/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(
            {"mode": self.mode, "state_dict": self.pose.state_dict()},
            os.path.join(out_weights_path, "temporal_pose.pth"),
        )

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "TemporalPose"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(
            model_path, "TemporalPose/iteration_{}/temporal_pose.pth".format(loaded_iter)
        )
        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            saved_mode = str(state.get("mode", self.mode)).lower()
            state_dict = state["state_dict"]
            if saved_mode != self.mode:
                train_fids = self.train_fids
                if saved_mode == "table" and "fids" in state_dict:
                    train_fids = state_dict["fids"].detach().cpu().tolist()
                self.pose = self._build(saved_mode, train_fids).to(self.device)
            self.pose.load_state_dict(state_dict)
            self.mode = saved_mode
        else:
            self.pose.load_state_dict(state)
        return True
