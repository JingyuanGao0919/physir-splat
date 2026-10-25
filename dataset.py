"""Given-COLMAP inputs with PhysIR's TI-NSD and published RGBT splits."""
import json
from pathlib import Path

import numpy as np
import torch

from measurement import read_measurement, training_profile
from scene.cameras import Camera
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_text, qvec2rotmat)
from scene.dataset_readers import fetchPly
from utils.graphics_utils import focal2fov


class DualCamera(Camera):
    def load2device(self, data_device='cuda'):
        super().load2device(data_device)
        if self.rgb_image is not None:
            self.rgb_image = self.rgb_image.to(data_device)


class Dataset:
    def __init__(self, source, branch='auto', measurement_profile=None, profile=None,
                 validation_stride=0):
        self.source = Path(source).resolve()
        self.branch = ('rgbt' if (self.source / 'thermal/train').is_dir() else 'ir') if branch == 'auto' else branch
        sparse = self.source / 'sparse/0'
        if not sparse.is_dir():
            sparse = self.source / 'colmap/sparse/0'
        if (sparse / 'images.bin').exists():
            extrinsics = read_extrinsics_binary(str(sparse / 'images.bin'))
            intrinsics = read_intrinsics_binary(str(sparse / 'cameras.bin'))
        else:
            extrinsics = read_extrinsics_text(str(sparse / 'images.txt'))
            intrinsics = read_intrinsics_text(str(sparse / 'cameras.txt'))
        entries = sorted(extrinsics.values(), key=lambda x: Path(x.name).stem)
        records = []
        for index, extrinsic in enumerate(entries):
            name = Path(extrinsic.name).name
            if self.branch == 'rgbt':
                split = 'train' if (self.source / 'thermal/train' / name).exists() else 'test'
                thermal = self.source / 'thermal' / split / name
                rgb = self.source / 'rgb' / split / name
                if not thermal.exists() or not rgb.exists():
                    continue
            else:
                split = 'test' if index % 8 == 0 else 'train'
                thermal, rgb = self.source / 'images' / name, None
            records.append(dict(extrinsic=extrinsic, thermal=thermal, rgb=rgb, split=split))
        # Development views are taken from the original training set. The
        # published test set remains separate and is not scored in dev runs.
        original_train = [r for r in records if r['split'] == 'train']
        if validation_stride:
            for index, record in enumerate(original_train):
                if index % validation_stride == 0:
                    record['split'] = 'validation'
        train_paths = [r['thermal'] for r in records if r['split'] == 'train']
        self.profile = profile or training_profile(train_paths, measurement_profile)
        self.cameras = dict(train=[], validation=[], test=[])
        self.records = []
        centers = []
        for index, record in enumerate(records):
            extrinsic = record['extrinsic']
            intrinsic = intrinsics[extrinsic.camera_id]
            fx = intrinsic.params[0]
            if intrinsic.model == 'PINHOLE':
                fy = intrinsic.params[1]
            elif intrinsic.model == 'SIMPLE_PINHOLE':
                fy = fx
            else:
                raise ValueError('Undistorted PINHOLE or SIMPLE_PINHOLE cameras are required.')
            image = read_measurement(record['thermal'], self.profile)
            if image.shape[-2:] != (intrinsic.height, intrinsic.width):
                raise ValueError('Image size differs from the supplied COLMAP camera: ' + str(record['thermal']))
            R = qvec2rotmat(extrinsic.qvec).T
            T = np.asarray(extrinsic.tvec)
            camera = DualCamera(
                colmap_id=extrinsic.camera_id, R=R, T=T,
                FoVx=focal2fov(fx, intrinsic.width), FoVy=focal2fov(fy, intrinsic.height),
                image=image, gt_alpha_mask=None, image_name=record['thermal'].stem,
                uid=index, data_device='cpu', fid=index / max(len(records) - 1, 1))
            # Native Camera clamps display RGB. Linear DN and calibrated band
            # radiance may legitimately exceed one and must retain their values.
            camera.original_image = image
            camera.rgb_image = (read_measurement(record['rgb'], dict(domain='display_gray'))
                                if record['rgb'] else None)
            self.cameras[record['split']].append(camera)
            if record['split'] == 'train':
                centers.append(-R @ T)
            self.records.append(dict(split=record['split'], name=camera.image_name,
                                     thermal=str(record['thermal']), rgb=str(record['rgb']) if record['rgb'] else None,
                                     R=R.tolist(), T=T.tolist(), FoVx=camera.FoVx, FoVy=camera.FoVy,
                                     width=intrinsic.width, height=intrinsic.height, frame_coordinate=float(camera.fid)))
        centers = np.stack(centers)
        self.extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
        self.ply_path = sparse / 'points3D.ply'
        self.point_cloud = fetchPly(str(self.ply_path))

    def save_inputs(self, model_path):
        report = dict(source=str(self.source), branch=self.branch, point_cloud=str(self.ply_path),
                      extent=self.extent, measurement=self.profile, views=self.records)
        (Path(model_path) / 'inputs.json').write_text(json.dumps(report, indent=2))
