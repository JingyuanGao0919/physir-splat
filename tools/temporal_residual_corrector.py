#!/usr/bin/env python3
"""Train-set cross-validated temporal residual correction for IR renders.

The corrector learns no test-image information. It builds residual maps from
training renders (GT - render), selects low-frequency temporal interpolation
hyperparameters by leave-one-out validation on training views, and applies the
selected setting to held-out test renders.
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.dataset_readers import readColmapSceneInfo, readRGBTSceneInfo


def _image_paths(folder):
    return sorted(Path(folder).glob("*.png"))


def _read_rgb(path):
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.float32) / 255.0


def _write_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image, 0.0, 1.0)
    imageio.imwrite(path, (image * 255.0 + 0.5).astype(np.uint8))


def _psnr(pred, gt):
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


def _resize_residual(residual, grid_size, out_shape):
    if grid_size <= 0:
        return residual
    h, w = residual.shape[:2]
    low = cv2.resize(residual, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(low, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR)


def _scene_fids(source_path, data_branch):
    data_branch = data_branch.lower()
    if data_branch == "rgbt":
        info = readRGBTSceneInfo(source_path, None, True)
    else:
        info = readColmapSceneInfo(source_path, "images", True)
    train = sorted(info.train_cameras, key=lambda c: c.image_name)
    test = sorted(info.test_cameras, key=lambda c: c.image_name)
    return (
        np.asarray([float(c.fid) for c in train], dtype=np.float32),
        np.asarray([float(c.fid) for c in test], dtype=np.float32),
        [str(c.image_name) for c in train],
        [str(c.image_name) for c in test],
    )


def _load_pair_stack(render_dir, gt_dir):
    render_paths = _image_paths(render_dir)
    gt_paths = _image_paths(gt_dir)
    if len(render_paths) != len(gt_paths):
        raise RuntimeError(f"render/gt count mismatch: {render_dir}={len(render_paths)} {gt_dir}={len(gt_paths)}")
    renders = [_read_rgb(path) for path in render_paths]
    gts = [_read_rgb(path) for path in gt_paths]
    return render_paths, gt_paths, renders, gts


def _neighbor_weights(query_fid, train_fids, k, sigma, exclude_idx=None):
    distances = np.abs(train_fids - float(query_fid))
    if exclude_idx is not None:
        distances[int(exclude_idx)] = np.inf
    valid = np.isfinite(distances)
    if not np.any(valid):
        raise RuntimeError("no valid residual neighbors available")
    order = np.argsort(distances)
    order = order[np.isfinite(distances[order])][: max(int(k), 1)]
    if len(order) == 0:
        order = np.asarray([int(np.nanargmin(distances))], dtype=np.int64)
    sigma = max(float(sigma), 1e-8)
    weights = np.exp(-0.5 * (distances[order] / sigma) ** 2)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights, dtype=np.float32)
    weights = weights.astype(np.float32)
    weights = weights / weights.sum()
    return order, weights


def _predict_residual(query_fid, train_fids, residuals, grid_size, k, sigma, out_shape, exclude_idx=None):
    indices, weights = _neighbor_weights(query_fid, train_fids, k, sigma, exclude_idx=exclude_idx)
    pred = np.zeros(out_shape, dtype=np.float32)
    for idx, weight in zip(indices, weights):
        pred += float(weight) * _resize_residual(residuals[int(idx)], grid_size, out_shape[:2])
    return pred, indices.tolist(), weights.tolist()


def _candidate_name(iteration, grid_size, k, sigma_mult, strength, clamp):
    return (
        f"ours_{iteration}_tres"
        f"_g{int(grid_size)}_k{int(k)}_s{str(float(sigma_mult)).replace('.', 'p')}"
        f"_a{str(float(strength)).replace('.', 'p')}_c{str(float(clamp)).replace('.', 'p')}"
    )


def run(args):
    method = getattr(args, "source_method", "") or f"ours_{args.iteration}"
    train_root = Path(args.model_path) / "train" / method
    test_root = Path(args.model_path) / "test" / method
    train_render_dir = train_root / "renders"
    train_gt_dir = train_root / "gt"
    test_render_dir = test_root / "renders"
    test_gt_dir = test_root / "gt"

    train_fids, test_fids, train_names, test_names = _scene_fids(args.source_path, args.data_branch)
    train_render_paths, train_gt_paths, train_renders, train_gts = _load_pair_stack(train_render_dir, train_gt_dir)
    test_render_paths, test_gt_paths, test_renders, test_gts = _load_pair_stack(test_render_dir, test_gt_dir)

    if len(train_fids) != len(train_renders):
        raise RuntimeError(f"train fid/render count mismatch: fids={len(train_fids)} renders={len(train_renders)}")
    if len(test_fids) != len(test_renders):
        raise RuntimeError(f"test fid/render count mismatch: fids={len(test_fids)} renders={len(test_renders)}")

    residuals = [gt - render for render, gt in zip(train_renders, train_gts)]
    baseline_train = np.asarray([_psnr(render, gt) for render, gt in zip(train_renders, train_gts)], dtype=np.float32)
    baseline_test = np.asarray([_psnr(render, gt) for render, gt in zip(test_renders, test_gts)], dtype=np.float32)
    median_dt = float(np.median(np.diff(np.sort(train_fids)))) if len(train_fids) > 1 else 1.0
    cv_indices = np.arange(0, len(train_fids), max(int(args.cv_stride), 1), dtype=np.int64)

    print(
        "[TRES_HYPOTHESIS] temporal train residuals may capture sensor/exposure/low-frequency "
        "rendering errors that are coherent across adjacent fids. Hyperparameters are selected "
        "only by train leave-one-out CV, then frozen for held-out test rendering."
    )
    print(
        "[TRES_DATA] train={} test={} median_dt={:.8f} train_psnr={:.4f} test_psnr_raw={:.4f} "
        "cv_stride={} cv_views={}".format(
            len(train_fids),
            len(test_fids),
            median_dt,
            float(baseline_train.mean()),
            float(baseline_test.mean()),
            int(args.cv_stride),
            len(cv_indices),
        )
    )

    candidates = []
    for grid_size in args.grid_sizes:
        for k in args.ks:
            for sigma_mult in args.sigma_multipliers:
                sigma = float(sigma_mult) * median_dt
                for strength in args.strengths:
                    for clamp in args.clamps:
                        psnrs = []
                        gains = []
                        for idx in cv_indices:
                            pred_resid, _, _ = _predict_residual(
                                train_fids[int(idx)],
                                train_fids,
                                residuals,
                                int(grid_size),
                                int(k),
                                sigma,
                                train_renders[int(idx)].shape,
                                exclude_idx=int(idx),
                            )
                            pred_resid = np.clip(pred_resid, -float(clamp), float(clamp))
                            corrected = np.clip(train_renders[int(idx)] + float(strength) * pred_resid, 0.0, 1.0)
                            cur = _psnr(corrected, train_gts[int(idx)])
                            psnrs.append(cur)
                            gains.append(cur - float(baseline_train[int(idx)]))
                        candidates.append(
                            {
                                "grid_size": int(grid_size),
                                "k": int(k),
                                "sigma_multiplier": float(sigma_mult),
                                "sigma": sigma,
                                "strength": float(strength),
                                "clamp": float(clamp),
                                "cv_psnr": float(np.mean(psnrs)),
                                "cv_gain": float(np.mean(gains)),
                            }
                        )

    candidates = sorted(candidates, key=lambda item: (item["cv_psnr"], item["cv_gain"]), reverse=True)
    top = candidates[: min(8, len(candidates))]
    for rank, item in enumerate(top, start=1):
        print(
            "[TRES_CV][rank {}] grid={} k={} sigma_mult={:.3f} strength={:.3f} clamp={:.3f} "
            "cv_psnr={:.4f} cv_gain={:+.4f}".format(
                rank,
                item["grid_size"],
                item["k"],
                item["sigma_multiplier"],
                item["strength"],
                item["clamp"],
                item["cv_psnr"],
                item["cv_gain"],
            )
        )

    selected = top[0]
    out_method = args.output_method or _candidate_name(
        args.iteration,
        selected["grid_size"],
        selected["k"],
        selected["sigma_multiplier"],
        selected["strength"],
        selected["clamp"],
    )
    out_root = Path(args.model_path) / "test" / out_method
    out_render_dir = out_root / "renders"
    out_gt_dir = out_root / "gt"
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_render_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    corrected_psnrs = []
    per_view = []
    for idx, (fid, render, gt, gt_path) in enumerate(zip(test_fids, test_renders, test_gts, test_gt_paths)):
        pred_resid, nn_idx, nn_weights = _predict_residual(
            fid,
            train_fids,
            residuals,
            selected["grid_size"],
            selected["k"],
            selected["sigma"],
            render.shape,
            exclude_idx=None,
        )
        pred_resid = np.clip(pred_resid, -selected["clamp"], selected["clamp"])
        corrected = np.clip(render + selected["strength"] * pred_resid, 0.0, 1.0)
        out_path = out_render_dir / f"{idx:05d}.png"
        _write_rgb(out_path, corrected)
        shutil.copy2(gt_path, out_gt_dir / f"{idx:05d}.png")
        cur_psnr = _psnr(corrected, gt)
        corrected_psnrs.append(cur_psnr)
        per_view.append(
            {
                "idx": int(idx),
                "image_name": test_names[idx] if idx < len(test_names) else str(idx),
                "fid": float(fid),
                "baseline_psnr": float(baseline_test[idx]),
                "corrected_psnr": float(cur_psnr),
                "gain": float(cur_psnr - baseline_test[idx]),
                "neighbors": [
                    {
                        "train_idx": int(nidx),
                        "image_name": train_names[int(nidx)] if int(nidx) < len(train_names) else str(int(nidx)),
                        "weight": float(weight),
                    }
                    for nidx, weight in zip(nn_idx, nn_weights)
                ],
            }
        )

    corrected_psnrs = np.asarray(corrected_psnrs, dtype=np.float32)
    stats = {
        "method": out_method,
        "source_method": method,
        "selected": selected,
        "raw_test_psnr": float(baseline_test.mean()),
        "corrected_test_psnr": float(corrected_psnrs.mean()),
        "test_gain": float(corrected_psnrs.mean() - baseline_test.mean()),
        "raw_test_psnr_min": float(baseline_test.min()),
        "corrected_test_psnr_min": float(corrected_psnrs.min()),
        "per_view": per_view,
        "top_candidates": top,
    }
    with open(out_root / "temporal_residual_corrector.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(
        "[TRES_APPLY] output={} selected_grid={} k={} sigma_mult={:.3f} strength={:.3f} clamp={:.3f} "
        "test_psnr={:.4f}->{:.4f} gain={:+.4f} worst={:.4f}->{:.4f}".format(
            out_root,
            selected["grid_size"],
            selected["k"],
            selected["sigma_multiplier"],
            selected["strength"],
            selected["clamp"],
            float(baseline_test.mean()),
            float(corrected_psnrs.mean()),
            float(corrected_psnrs.mean() - baseline_test.mean()),
            float(baseline_test.min()),
            float(corrected_psnrs.min()),
        )
    )
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal train-residual correction for rendered test images.")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--source_method", default="")
    parser.add_argument("--data_branch", choices=["ir", "rgbt"], default="ir")
    parser.add_argument("--grid_sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--ks", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--sigma_multipliers", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--clamps", nargs="+", type=float, default=[0.03, 0.06, 0.10])
    parser.add_argument("--cv_stride", type=int, default=8)
    parser.add_argument("--output_method", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
