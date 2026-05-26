#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
import shutil
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].to(device))
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].to(device))
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, keep_all_methods=False):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            scene_full = {}
            scene_per_view = {}

            for method in sorted(os.listdir(test_dir)):
                if not method.startswith("ours"):
                    continue
                print("Method:", method)

                scene_full[method] = {}
                scene_per_view[method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                scene_full[method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                           "PSNR": torch.tensor(psnrs).mean().item(),
                                           "LPIPS": torch.tensor(lpipss).mean().item()})
                scene_per_view[method].update(
                    {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                     "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                     "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            if scene_full and not keep_all_methods:
                best_method = max(scene_full, key=lambda name: scene_full[name]["PSNR"])
                removed = []
                for method in scene_full:
                    if method != best_method:
                        method_path = test_dir / method
                        if method_path.is_dir():
                            shutil.rmtree(method_path)
                            removed.append(method)
                print(
                    "[METRICS_BEST] keeping only method {} with PSNR {:.7f}; removed={}".format(
                        best_method,
                        scene_full[best_method]["PSNR"],
                        len(removed),
                    )
                )
                full_dict[scene_dir] = {best_method: scene_full[best_method]}
                per_view_dict[scene_dir] = {best_method: scene_per_view[best_method]}
            else:
                full_dict[scene_dir] = scene_full
                per_view_dict[scene_dir] = scene_per_view

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--keep_all_methods', action="store_true")
    args = parser.parse_args()
    evaluate(args.model_paths, keep_all_methods=args.keep_all_methods)
