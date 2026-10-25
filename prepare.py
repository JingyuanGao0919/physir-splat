"""Export the provided COLMAP point cloud to PLY if it is not already present."""
import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from scene.colmap_loader import read_points3D_binary, read_points3D_text


def prepare(source):
    sparse = source / 'sparse/0'
    if not sparse.is_dir():
        sparse = source / 'colmap/sparse/0'
    output = sparse / 'points3D.ply'
    if output.exists():
        return
    if (sparse / 'points3D.bin').exists():
        xyz, rgb, _ = read_points3D_binary(str(sparse / 'points3D.bin'))
    else:
        xyz, rgb, _ = read_points3D_text(str(sparse / 'points3D.txt'))
    vertices = np.empty(len(xyz), dtype=[(name, 'f4') for name in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
                        + [(name, 'u1') for name in ['red', 'green', 'blue']])
    attributes = np.concatenate((xyz, np.zeros_like(xyz), rgb), axis=1)
    vertices[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(output))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    prepare(parser.parse_args().source)
