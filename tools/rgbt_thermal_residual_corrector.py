#!/usr/bin/env python3
"""RGBT-only train-CV thermal residual correction.

This corrector keeps the IR-only TRES path untouched.  For RGBT scenes it uses
only train-view supervision to learn a low-frequency thermal residual model from
rendered thermal/RGB features, optionally blended with a temporal train-residual
prior.  Test thermal GT is used only for the final diagnostic log.
"""

import argparse
import json
import math
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

from scene.dataset_readers import readRGBTSceneInfo


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
    imageio.imwrite(path, (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def _psnr(pred, gt):
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


def _scene_fids(source_path):
    info = readRGBTSceneInfo(source_path, None, True)
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
    return render_paths, gt_paths, [_read_rgb(p) for p in render_paths], [_read_rgb(p) for p in gt_paths]


def _load_render_stack(render_dir):
    paths = _image_paths(render_dir)
    return paths, [_read_rgb(p) for p in paths]


def _resize(image, grid_size, interpolation=cv2.INTER_AREA):
    return cv2.resize(image, (int(grid_size), int(grid_size)), interpolation=interpolation)


def _upsample(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _luma(image):
    return (
        0.299 * image[..., 0:1]
        + 0.587 * image[..., 1:2]
        + 0.114 * image[..., 2:3]
    ).astype(np.float32)


def _feature_image(thermal_render, rgb_render):
    h, w = thermal_render.shape[:2]
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    xx = xx[..., None]
    yy = yy[..., None]
    thermal_luma = _luma(thermal_render)
    rgb_luma = _luma(rgb_render)
    diff_luma = rgb_luma - thermal_luma
    return np.concatenate(
        [
            thermal_render.astype(np.float32),
            rgb_render.astype(np.float32),
            thermal_luma,
            rgb_luma,
            diff_luma,
            thermal_luma * thermal_luma,
            rgb_luma * rgb_luma,
            xx,
            yy,
            xx * yy,
            xx * xx,
            yy * yy,
        ],
        axis=-1,
    )


def _stack_features(thermal_renders, rgb_renders, indices, grid_size):
    features = []
    for idx in indices:
        thermal_low = _resize(thermal_renders[int(idx)], grid_size)
        rgb_low = _resize(rgb_renders[int(idx)], grid_size)
        features.append(_feature_image(thermal_low, rgb_low).reshape(-1, _feature_image(thermal_low, rgb_low).shape[-1]))
    return np.concatenate(features, axis=0).astype(np.float32)


def _stack_targets(residuals, indices, grid_size):
    targets = []
    for idx in indices:
        targets.append(_resize(residuals[int(idx)], grid_size).reshape(-1, 3))
    return np.concatenate(targets, axis=0).astype(np.float32)


def _fit_ridge(features, targets, alpha):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    x = (features - mean) / std
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    reg = np.eye(x.shape[1], dtype=np.float32) * float(alpha)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + reg, x.T @ targets).astype(np.float32)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "weights": weights}


def _predict_ridge(model, thermal_render, rgb_render, grid_size):
    thermal_low = _resize(thermal_render, grid_size)
    rgb_low = _resize(rgb_render, grid_size)
    features = _feature_image(thermal_low, rgb_low).reshape(-1, model["mean"].shape[1])
    x = (features - model["mean"]) / model["std"]
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    pred = x @ model["weights"]
    return pred.reshape(int(grid_size), int(grid_size), 3).astype(np.float32)


def _neighbor_weights(query_fid, train_fids, k, sigma, exclude_idx=None):
    distances = np.abs(train_fids - float(query_fid))
    if exclude_idx is not None:
        distances[int(exclude_idx)] = np.inf
    order = np.argsort(distances)
    order = order[np.isfinite(distances[order])][: max(int(k), 1)]
    if len(order) == 0:
        raise RuntimeError("no valid temporal neighbors")
    sigma = max(float(sigma), 1e-8)
    weights = np.exp(-0.5 * (distances[order] / sigma) ** 2).astype(np.float32)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights, dtype=np.float32)
    weights /= weights.sum()
    return order, weights


def _temporal_low_residual(query_fid, train_fids, residual_lows, k, sigma, exclude_idx=None):
    order, weights = _neighbor_weights(query_fid, train_fids, k, sigma, exclude_idx=exclude_idx)
    pred = np.zeros_like(residual_lows[0], dtype=np.float32)
    for idx, weight in zip(order, weights):
        pred += float(weight) * residual_lows[int(idx)]
    return pred, order.tolist(), weights.tolist()


def _candidate_name(iteration, item):
    return (
        f"ours_{iteration}_rgbttherm"
        f"_g{int(item['grid_size'])}_k{int(item['k'])}"
        f"_s{str(float(item['sigma_multiplier'])).replace('.', 'p')}"
        f"_l{str(float(item['linear_strength'])).replace('.', 'p')}"
        f"_t{str(float(item['temporal_strength'])).replace('.', 'p')}"
        f"_c{str(float(item['clamp'])).replace('.', 'p')}"
        f"_a{str(float(item['ridge_alpha'])).replace('.', 'p')}"
    )


def _correct_full(render, gt, linear_pred, temporal_pred, linear_strength, temporal_strength, clamp):
    residual = float(linear_strength) * linear_pred + float(temporal_strength) * temporal_pred
    residual = np.clip(residual, -float(clamp), float(clamp))
    residual = _upsample(residual, render.shape[:2])
    corrected = np.clip(render + residual, 0.0, 1.0)
    return _psnr(corrected, gt)


def run(args):
    method = getattr(args, "source_method", "") or f"ours_{args.iteration}"
    train_root = Path(args.model_path) / "train" / method
    test_root = Path(args.model_path) / "test" / method
    train_render_dir = train_root / "renders"
    train_gt_dir = train_root / "gt"
    train_rgb_render_dir = train_root / "renders_rgb"
    test_render_dir = test_root / "renders"
    test_gt_dir = test_root / "gt"
    test_rgb_render_dir = test_root / "renders_rgb"

    train_fids, test_fids, train_names, test_names = _scene_fids(args.source_path)
    _, _, train_renders, train_gts = _load_pair_stack(train_render_dir, train_gt_dir)
    _, _, test_renders, test_gts = _load_pair_stack(test_render_dir, test_gt_dir)
    _, train_rgb_renders = _load_render_stack(train_rgb_render_dir)
    _, test_rgb_renders = _load_render_stack(test_rgb_render_dir)
    if not (len(train_fids) == len(train_renders) == len(train_rgb_renders)):
        raise RuntimeError(
            f"train count mismatch: fids={len(train_fids)} thermal={len(train_renders)} rgb={len(train_rgb_renders)}"
        )
    if not (len(test_fids) == len(test_renders) == len(test_rgb_renders)):
        raise RuntimeError(
            f"test count mismatch: fids={len(test_fids)} thermal={len(test_renders)} rgb={len(test_rgb_renders)}"
        )

    residuals = [gt - render for render, gt in zip(train_renders, train_gts)]
    baseline_train = np.asarray([_psnr(r, g) for r, g in zip(train_renders, train_gts)], dtype=np.float32)
    baseline_test = np.asarray([_psnr(r, g) for r, g in zip(test_renders, test_gts)], dtype=np.float32)
    median_dt = float(np.median(np.diff(np.sort(train_fids)))) if len(train_fids) > 1 else 1.0
    cv_indices = np.arange(0, len(train_fids), max(int(args.cv_stride), 1), dtype=np.int64)
    train_all_indices = np.arange(len(train_fids), dtype=np.int64)

    print(
        "[RGBT_THERM_HYPOTHESIS] RGBT thermal errors can include low-frequency radiometric bias "
        "plus cross-modal geometry/material cues visible in the rendered RGB branch. This corrector "
        "uses rendered RGB only, never test RGB/thermal GT for model selection."
    )
    print(
        "[RGBT_THERM_DATA] train={} test={} median_dt={:.8f} train_psnr={:.4f} test_psnr_raw={:.4f} "
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

    scored = []
    final_models = {}
    final_test_linear = {}
    final_test_temporal = {}
    low_cache = {}

    for grid_size in args.grid_sizes:
        grid_size = int(grid_size)
        train_render_lows = [_resize(img, grid_size) for img in train_renders]
        train_gt_lows = [_resize(img, grid_size) for img in train_gts]
        train_residual_lows = [gt - render for render, gt in zip(train_render_lows, train_gt_lows)]
        low_cache[grid_size] = (train_render_lows, train_gt_lows, train_residual_lows)

        cv_linear = {}
        for alpha in args.ridge_alphas:
            alpha = float(alpha)
            preds = []
            for idx in cv_indices:
                fit_indices = train_all_indices[train_all_indices != int(idx)]
                features = _stack_features(train_renders, train_rgb_renders, fit_indices, grid_size)
                targets = _stack_targets(residuals, fit_indices, grid_size)
                model = _fit_ridge(features, targets, alpha)
                preds.append(_predict_ridge(model, train_renders[int(idx)], train_rgb_renders[int(idx)], grid_size))
            cv_linear[alpha] = preds

            final_features = _stack_features(train_renders, train_rgb_renders, train_all_indices, grid_size)
            final_targets = _stack_targets(residuals, train_all_indices, grid_size)
            final_models[(grid_size, alpha)] = _fit_ridge(final_features, final_targets, alpha)
            final_test_linear[(grid_size, alpha)] = [
                _predict_ridge(final_models[(grid_size, alpha)], thermal, rgb, grid_size)
                for thermal, rgb in zip(test_renders, test_rgb_renders)
            ]

        cv_temporal = {}
        for k in args.ks:
            for sigma_mult in args.sigma_multipliers:
                sigma = float(sigma_mult) * median_dt
                key = (int(k), float(sigma_mult))
                preds = []
                for idx in cv_indices:
                    pred, _, _ = _temporal_low_residual(
                        train_fids[int(idx)],
                        train_fids,
                        train_residual_lows,
                        int(k),
                        sigma,
                        exclude_idx=int(idx),
                    )
                    preds.append(pred)
                cv_temporal[key] = preds
                final_test_temporal[(grid_size, key)] = [
                    _temporal_low_residual(fid, train_fids, train_residual_lows, int(k), sigma)[0]
                    for fid in test_fids
                ]

        for alpha, linear_preds in cv_linear.items():
            for k in args.ks:
                for sigma_mult in args.sigma_multipliers:
                    temporal_preds = cv_temporal[(int(k), float(sigma_mult))]
                    for linear_strength in args.linear_strengths:
                        for temporal_strength in args.temporal_strengths:
                            if float(linear_strength) <= 0.0 and float(temporal_strength) <= 0.0:
                                continue
                            for clamp in args.clamps:
                                psnrs = []
                                gains = []
                                for pos, idx in enumerate(cv_indices):
                                    cur = _correct_full(
                                        train_renders[int(idx)],
                                        train_gts[int(idx)],
                                        linear_preds[pos],
                                        temporal_preds[pos],
                                        float(linear_strength),
                                        float(temporal_strength),
                                        float(clamp),
                                    )
                                    psnrs.append(cur)
                                    gains.append(cur - float(baseline_train[int(idx)]))
                                scored.append(
                                    {
                                        "grid_size": grid_size,
                                        "ridge_alpha": float(alpha),
                                        "k": int(k),
                                        "sigma_multiplier": float(sigma_mult),
                                        "linear_strength": float(linear_strength),
                                        "temporal_strength": float(temporal_strength),
                                        "clamp": float(clamp),
                                        "cv_psnr": float(np.mean(psnrs)),
                                        "cv_gain": float(np.mean(gains)),
                                    }
                                )

    scored.sort(key=lambda item: (item["cv_psnr"], item["cv_gain"]), reverse=True)
    top = scored[: min(10, len(scored))]
    for rank, item in enumerate(top, start=1):
        print(
            "[RGBT_THERM_CV][rank {}] grid={} alpha={} k={} sigma_mult={:.3f} "
            "linear={:.3f} temporal={:.3f} clamp={:.3f} cv_psnr={:.4f} cv_gain={:+.4f}".format(
                rank,
                item["grid_size"],
                item["ridge_alpha"],
                item["k"],
                item["sigma_multiplier"],
                item["linear_strength"],
                item["temporal_strength"],
                item["clamp"],
                item["cv_psnr"],
                item["cv_gain"],
            )
        )

    selected = top[0]
    output_method = args.output_method or _candidate_name(args.iteration, selected)
    out_root = Path(args.model_path) / "test" / output_method
    out_render_dir = out_root / "renders"
    out_gt_dir = out_root / "gt"
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_render_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    grid_size = int(selected["grid_size"])
    alpha = float(selected["ridge_alpha"])
    temporal_key = (int(selected["k"]), float(selected["sigma_multiplier"]))
    test_linear = final_test_linear[(grid_size, alpha)]
    test_temporal = final_test_temporal[(grid_size, temporal_key)]

    corrected_psnrs = []
    per_view = []
    for idx, (render, gt, gt_path) in enumerate(zip(test_renders, test_gts, _image_paths(test_gt_dir))):
        low_residual = (
            float(selected["linear_strength"]) * test_linear[idx]
            + float(selected["temporal_strength"]) * test_temporal[idx]
        )
        low_residual = np.clip(low_residual, -float(selected["clamp"]), float(selected["clamp"]))
        residual = _upsample(low_residual, render.shape[:2])
        corrected = np.clip(render + residual, 0.0, 1.0)
        _write_rgb(out_render_dir / f"{idx:05d}.png", corrected)
        shutil.copy2(gt_path, out_gt_dir / f"{idx:05d}.png")
        cur_psnr = _psnr(corrected, gt)
        corrected_psnrs.append(cur_psnr)
        per_view.append(
            {
                "idx": int(idx),
                "image_name": test_names[idx] if idx < len(test_names) else str(idx),
                "fid": float(test_fids[idx]),
                "baseline_psnr": float(baseline_test[idx]),
                "corrected_psnr": float(cur_psnr),
                "gain": float(cur_psnr - baseline_test[idx]),
            }
        )

    corrected_psnrs = np.asarray(corrected_psnrs, dtype=np.float32)
    stats = {
        "method": output_method,
        "source_method": method,
        "selected": selected,
        "raw_test_psnr": float(baseline_test.mean()),
        "corrected_test_psnr": float(corrected_psnrs.mean()),
        "test_gain": float(corrected_psnrs.mean() - baseline_test.mean()),
        "raw_test_psnr_min": float(baseline_test.min()),
        "corrected_test_psnr_min": float(corrected_psnrs.min()),
        "top_candidates": top,
        "per_view": per_view,
    }
    with open(out_root / "rgbt_thermal_residual_corrector.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(
        "[RGBT_THERM_APPLY] output={} test_psnr={:.4f}->{:.4f} gain={:+.4f} "
        "worst={:.4f}->{:.4f}".format(
            out_root,
            float(baseline_test.mean()),
            float(corrected_psnrs.mean()),
            float(corrected_psnrs.mean() - baseline_test.mean()),
            float(baseline_test.min()),
            float(corrected_psnrs.min()),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="RGBT train-CV thermal residual correction.")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--source_method", default="")
    parser.add_argument("--grid_sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--ridge_alphas", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    parser.add_argument("--ks", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--sigma_multipliers", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--linear_strengths", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--temporal_strengths", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--clamps", nargs="+", type=float, default=[0.06, 0.10, 0.15])
    parser.add_argument("--cv_stride", type=int, default=8)
    parser.add_argument("--output_method", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
