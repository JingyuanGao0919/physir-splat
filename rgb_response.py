"""Camera-metadata interpolation of jointly learned RGB display responses."""
import numpy as np
import torch


def pose_records(cameras, anchors):
    index = {value: i for i, value in enumerate(anchors)}
    return [dict(uid=int(view.uid), anchor_index=index[float(view.fid)],
                 R=view.R.tolist(), T=view.T.tolist()) for view in cameras]


class PoseResponse:
    def __init__(self, records, extent):
        self.extent = extent
        self.rotations = np.asarray([row['R'] for row in records], dtype=np.float64)
        translations = np.asarray([row['T'] for row in records], dtype=np.float64)
        self.centers = -np.einsum('nij,nj->ni', self.rotations, translations)
        self.indices = np.asarray([row['anchor_index'] for row in records])
        self.by_uid = {row['uid']: row['anchor_index'] for row in records}

    def value(self, view, centered_log_gains):
        if view.uid in self.by_uid:
            return centered_log_gains[self.by_uid[view.uid]]
        center = -view.R @ view.T
        distance = np.sqrt(((self.centers - center) ** 2).sum(-1) / self.extent ** 2 +
                           .5 * ((self.rotations - view.R) ** 2).sum((1, 2)))
        neighbors = np.argsort(distance)[:2]
        distances = distance[neighbors]
        weights = distances[::-1] / distances.sum()
        indices = torch.as_tensor(self.indices[neighbors], device=centered_log_gains.device)
        # Pose coefficients retain the precision of the given COLMAP metadata.
        coefficients = centered_log_gains.new_tensor(weights, dtype=torch.float64)
        value = (centered_log_gains[indices].double() * coefficients).sum()
        return value.to(centered_log_gains.dtype)
