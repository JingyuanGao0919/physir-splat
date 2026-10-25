"""Interpolate frozen position fields using only known TRAIN camera metadata."""
import numpy as np
import torch


def pose_records(cameras):
    return [dict(uid=int(view.uid), fid=float(view.fid), R=view.R.tolist(), T=view.T.tolist(),
                 center=view.camera_center.detach().cpu().tolist(),
                 direction=(-view.world_view_transform[:3, 2]).detach().cpu().tolist())
            for view in cameras]


class DeformationPose:
    def __init__(self, records, extent):
        self.records = records
        self.extent = extent
        self.by_uid = {row['uid'] for row in records}
        self.rotations = np.asarray([row['R'] for row in records], dtype=np.float64)
        translations = np.asarray([row['T'] for row in records], dtype=np.float64)
        self.centers = -np.einsum('nij,nj->ni', self.rotations, translations)

    def value(self, view, xyz, network):
        if view.uid in self.by_uid:
            frame = view.fid.reshape(1, 1).expand(len(xyz), 1)
            center = view.camera_center[None].expand(len(xyz), 3)
            direction = -view.world_view_transform[:3, 2][None].expand(len(xyz), 3)
            return network(xyz, frame, center, direction)
        center = -view.R @ view.T
        distance = np.sqrt(((self.centers - center) ** 2).sum(-1) / self.extent ** 2 +
                           .5 * ((self.rotations - view.R) ** 2).sum((1, 2)))
        neighbors = np.argsort(distance)[:2]
        weights = distance[neighbors][::-1] / distance[neighbors].sum()
        offsets = []
        for index in neighbors:
            anchor = self.records[index]
            frame = xyz.new_full((len(xyz), 1), anchor['fid'])
            anchor_center = xyz.new_tensor(anchor['center'])[None].expand(len(xyz), 3)
            direction = xyz.new_tensor(anchor['direction'])[None].expand(len(xyz), 3)
            offsets.append(network(xyz, frame, anchor_center, direction))
        return offsets[0] * float(weights[0]) + offsets[1] * float(weights[1])
