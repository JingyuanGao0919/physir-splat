#!/usr/bin/env python3
"""Train-CV temporal affine plus low-frequency residual correction.

This is a stricter follow-up to temporal_residual_corrector.py. It still uses no
test images for model selection: per-view global affine/color corrections and
low-frequency residuals are estimated on train renders, hyperparameters are
selected by leave-one-out train CV, and the selected setting is then frozen for
held-out test renders.
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


def _scene_fids(source_path, data_branch):
    if data_branch.lower() == "rgbt":
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


def _fit_global_affine(render, gt):
    gains = []
    biases = []
    for channel in range(3):
        x = render[..., channel].reshape(-1)
        y = gt[..., channel].reshape(-1)
        mask = (x > 0.02) & (x < 0.98) & (y > 0.02) & (y < 0.98)
        if int(mask.sum()) < 64:
            mask = np.ones_like(x, dtype=bool)
        x = x[mask]
        y = y[mask]
        mx = float(x.mean())
        my = float(y.mean())
        vx = float(np.mean((x - mx) ** 2))
        cov = float(np.mean((x - mx) * (y - my)))
        gain = cov / max(vx, 1e-6)
        bias = my - gain * mx
        gains.append(float(np.clip(gain, 0.60, 1.60)))
        biases.append(float(np.clip(bias, -0.25, 0.25)))
    return np.asarray(gains, dtype=np.float32), np.asarray(biases, dtype=np.float32)


def _resize_residual(residual, grid_size, out_shape):
    if grid_size <= 0:
        return residual
    h, w = residual.shape[:2]
    low = cv2.resize(residual, (int(grid_size), int(grid_size)), interpolation=cv2.INTER_AREA)
    return cv2.resize(low, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR)


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


def _predict_temporal(query_fid, train_fids, values, k, sigma, exclude_idx=None):
    indices, weights = _neighbor_weights(query_fid, train_fids, k, sigma, exclude_idx=exclude_idx)
    pred = np.zeros_like(values[0], dtype=np.float32)
    for idx, weight in zip(indices, weights):
        pred += float(weight) * values[int(idx)]
    return pred, indices.tolist(), weights.tolist()


def _predict_residual(query_fid, train_fids, residuals, grid_size, k, sigma, out_shape, exclude_idx=None):
    indices, weights = _neighbor_weights(query_fid, train_fids, k, sigma, exclude_idx=exclude_idx)
    pred = np.zeros(out_shape, dtype=np.float32)
    for idx, weight in zip(indices, weights):
        pred += float(weight) * _resize_residual(residuals[int(idx)], int(grid_size), out_shape[:2])
    return pred, indices.tolist(), weights.tolist()


def _apply_candidate(render, train_fids, query_fid, residual_bank, gains, biases, candidate, exclude_idx=None):
    image = render
    nn_idx = []
    nn_weights = []
    sigma = float(candidate["sigma"])
    k = int(candidate["k"])
    affine_alpha = float(candidate.get("affine_strength", 0.0))
    if candidate["family"] in ("affine", "affine_residual"):
        gain, nn_idx, nn_weights = _predict_temporal(query_fid, train_fids, gains, k, sigma, exclude_idx=exclude_idx)
        bias, _, _ = _predict_temporal(query_fid, train_fids, biases, k, sigma, exclude_idx=exclude_idx)
        gain = 1.0 + affine_alpha * (gain - 1.0)
        bias = affine_alpha * bias
        image = np.clip(image * gain.reshape(1, 1, 3) + bias.reshape(1, 1, 3), 0.0, 1.0)

    residual_strength = float(candidate.get("residual_strength", 0.0))
    if candidate["family"] in ("residual", "affine_residual") and residual_strength > 0.0:
        residual, nn_idx, nn_weights = _predict_residual(
            query_fid,
            train_fids,
            residual_bank,
            int(candidate["grid_size"]),
            k,
            sigma,
            render.shape,
            exclude_idx=exclude_idx,
        )
        residual = np.clip(residual, -float(candidate["clamp"]), float(candidate["clamp"]))
        image = np.clip(image + residual_strength * residual, 0.0, 1.0)
    return image, nn_idx, nn_weights


def _candidate_name(iteration, item):
    fam = item["family"].replace("_", "")
    return (
        f"ours_{iteration}_tcorr_{fam}"
        f"_g{int(item.get('grid_size', 0))}_k{int(item['k'])}"
        f"_s{str(float(item['sigma_multiplier'])).replace('.', 'p')}"
        f"_a{str(float(item.get('affine_strength', 0.0))).replace('.', 'p')}"
        f"_r{str(float(item.get('residual_strength', 0.0))).replace('.', 'p')}"
        f"_c{str(float(item.get('clamp', 0.0))).replace('.', 'p')}"
    )


def _build_candidates(args, median_dt):
    candidates = [{"family": "identity", "k": 1, "sigma_multiplier": 1.0, "sigma": median_dt}]
    for k in args.ks:
        for sigma_mult in args.sigma_multipliers:
            sigma = float(sigma_mult) * median_dt
            for affine_strength in args.affine_strengths:
                if float(affine_strength) > 0.0:
                    candidates.append(
                        {
                            "family": "affine",
                            "k": int(k),
                            "sigma_multiplier": float(sigma_mult),
                            "sigma": sigma,
                            "affine_strength": float(affine_strength),
                        }
                    )
            for grid_size in args.grid_sizes:
                for residual_strength in args.residual_strengths:
                    if float(residual_strength) <= 0.0:
                        continue
                    for clamp in args.clamps:
                        candidates.append(
                            {
                                "family": "residual",
                                "grid_size": int(grid_size),
                                "k": int(k),
                                "sigma_multiplier": float(sigma_mult),
                                "sigma": sigma,
                                "residual_strength": float(residual_strength),
                                "clamp": float(clamp),
                            }
                        )
                for affine_strength in args.affine_strengths:
                    if float(affine_strength) <= 0.0:
                        continue
                    for residual_strength in args.residual_strengths:
                        if float(residual_strength) <= 0.0:
                            continue
                        for clamp in args.clamps:
                            candidates.append(
                                {
                                    "family": "affine_residual",
                                    "grid_size": int(grid_size),
                                    "k": int(k),
                                    "sigma_multiplier": float(sigma_mult),
                                    "sigma": sigma,
                                    "affine_strength": float(affine_strength),
                                    "residual_strength": float(residual_strength),
                                    "clamp": float(clamp),
                                }
                            )
    return candidates


def run(args):
    method = f"ours_{args.iteration}"
    train_root = Path(args.model_path) / "train" / method
    test_root = Path(args.model_path) / "test" / method
    train_render_dir = train_root / "renders"
    train_gt_dir = train_root / "gt"
    test_render_dir = test_root / "renders"
    test_gt_dir = test_root / "gt"

    train_fids, test_fids, train_names, test_names = _scene_fids(args.source_path, args.data_branch)
    _, _, train_renders, train_gts = _load_pair_stack(train_render_dir, train_gt_dir)
    _, test_gt_paths, test_renders, test_gts = _load_pair_stack(test_render_dir, test_gt_dir)
    if len(train_fids) != len(train_renders):
        raise RuntimeError(f"train fid/render count mismatch: {len(train_fids)} vs {len(train_renders)}")
    if len(test_fids) != len(test_renders):
        raise RuntimeError(f"test fid/render count mismatch: {len(test_fids)} vs {len(test_renders)}")

    gains = []
    biases = []
    affine_residuals = []
    raw_residuals = []
    for render, gt in zip(train_renders, train_gts):
        gain, bias = _fit_global_affine(render, gt)
        affine_render = np.clip(render * gain.reshape(1, 1, 3) + bias.reshape(1, 1, 3), 0.0, 1.0)
        gains.append(gain)
        biases.append(bias)
        raw_residuals.append(gt - render)
        affine_residuals.append(gt - affine_render)
    gains = np.asarray(gains, dtype=np.float32)
    biases = np.asarray(biases, dtype=np.float32)

    baseline_train = np.asarray([_psnr(r, g) for r, g in zip(train_renders, train_gts)], dtype=np.float32)
    baseline_test = np.asarray([_psnr(r, g) for r, g in zip(test_renders, test_gts)], dtype=np.float32)
    median_dt = float(np.median(np.diff(np.sort(train_fids)))) if len(train_fids) > 1 else 1.0
    cv_indices = np.arange(0, len(train_fids), max(int(args.cv_stride), 1), dtype=np.int64)
    candidates = _build_candidates(args, median_dt)

    print(
        "[TCORR_HYPOTHESIS] temporal global affine terms may capture exposure/contrast drift, "
        "while low-frequency residuals capture coherent spatial bias. Selection uses only train "
        "leave-one-out CV; the selected candidate is frozen before test rendering."
    )
    print(
        "[TCORR_DATA] train={} test={} median_dt={:.8f} train_psnr={:.4f} test_psnr_raw={:.4f} "
        "cv_stride={} cv_views={} candidates={}".format(
            len(train_fids),
            len(test_fids),
            median_dt,
            float(baseline_train.mean()),
            float(baseline_test.mean()),
            int(args.cv_stride),
            len(cv_indices),
            len(candidates),
        )
    )

    scored = []
    for candidate in candidates:
        residual_bank = affine_residuals if candidate["family"] == "affine_residual" else raw_residuals
        psnrs = []
        gains_cv = []
        for idx in cv_indices:
            corrected, _, _ = _apply_candidate(
                train_renders[int(idx)],
                train_fids,
                train_fids[int(idx)],
                residual_bank,
                gains,
                biases,
                candidate,
                exclude_idx=int(idx),
            )
            cur = _psnr(corrected, train_gts[int(idx)])
            psnrs.append(cur)
            gains_cv.append(cur - float(baseline_train[int(idx)]))
        item = dict(candidate)
        item["cv_psnr"] = float(np.mean(psnrs))
        item["cv_gain"] = float(np.mean(gains_cv))
        scored.append(item)

    scored.sort(key=lambda item: (item["cv_psnr"], item["cv_gain"]), reverse=True)
    top = scored[: min(10, len(scored))]
    for rank, item in enumerate(top, start=1):
        print(
            "[TCORR_CV][rank {}] family={} grid={} k={} sigma_mult={:.3f} "
            "affine={:.3f} resid={:.3f} clamp={:.3f} cv_psnr={:.4f} cv_gain={:+.4f}".format(
                rank,
                item["family"],
                int(item.get("grid_size", 0)),
                int(item["k"]),
                float(item["sigma_multiplier"]),
                float(item.get("affine_strength", 0.0)),
                float(item.get("residual_strength", 0.0)),
                float(item.get("clamp", 0.0)),
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

    residual_bank = affine_residuals if selected["family"] == "affine_residual" else raw_residuals
    corrected_psnrs = []
    per_view = []
    for idx, (fid, render, gt, gt_path) in enumerate(zip(test_fids, test_renders, test_gts, test_gt_paths)):
        corrected, nn_idx, nn_weights = _apply_candidate(
            render,
            train_fids,
            fid,
            residual_bank,
            gains,
            biases,
            selected,
            exclude_idx=None,
        )
        _write_rgb(out_render_dir / f"{idx:05d}.png", corrected)
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
    with open(out_root / "temporal_affine_residual_corrector.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(
        "[TCORR_APPLY] output={} family={} test_psnr={:.4f}->{:.4f} gain={:+.4f} "
        "worst={:.4f}->{:.4f}".format(
            out_root,
            selected["family"],
            float(baseline_test.mean()),
            float(corrected_psnrs.mean()),
            float(corrected_psnrs.mean() - baseline_test.mean()),
            float(baseline_test.min()),
            float(corrected_psnrs.min()),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train-CV temporal affine/residual correction.")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--data_branch", choices=["ir", "rgbt"], default="ir")
    parser.add_argument("--grid_sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--ks", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--sigma_multipliers", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--affine_strengths", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--residual_strengths", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--clamps", nargs="+", type=float, default=[0.03, 0.06, 0.10])
    parser.add_argument("--cv_stride", type=int, default=8)
    parser.add_argument("--output_method", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
