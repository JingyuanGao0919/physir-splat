"""Render a fixed jointly-trained physical scene with no image fitting."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torchvision

from dataset import Dataset
from measurement import Measurement
from physical_gaussians import PhysicalGaussians
from renderer import RadiativeField, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--split', choices=['test', 'validation', 'train'], default='test')
    args = parser.parse_args()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    saved = torch.load(args.model / ('checkpoint_%d.pt' % args.iteration), map_location='cuda')
    cfg = saved['config']
    dataset = Dataset(cfg['source'], cfg['branch'], profile=saved['profile'],
                      validation_stride=cfg['validation_stride'])
    gaussians = PhysicalGaussians.from_state(saved['gaussians'])
    field = RadiativeField(saved['profile'], saved['extent'], cfg['atmosphere'], cfg['reflection'],
                           cfg.get('temporal_source', False), cfg.get('temporal_frequencies', 10)).cuda()
    field.load_state_dict(saved['radiative_field'])
    measurement = Measurement(saved['profile'], cfg['response_knots'], cfg['psf_sigma'],
                              cfg['digital_sigma'], cfg.get('clip_display', False)).cuda()
    measurement.load_state_dict(saved['measurement'])
    field.eval()
    measurement.eval()
    destination = args.model / args.split / ('ours_%d' % args.iteration)
    folders = ['renders', 'gt'] + (['renders_rgb', 'gt_rgb'] if dataset.branch == 'rgbt' else [])
    for folder in folders:
        (destination / folder).mkdir(parents=True, exist_ok=True)
    is_display = saved['profile']['domain'] in ('display_gray', 'display_palette')
    elapsed = []
    measured = []
    with torch.no_grad():
        for index, view in enumerate(dataset.cameras[args.split]):
            view.load2device()
            torch.cuda.synchronize()
            rendering_start = time.perf_counter()
            result = render(view, gaussians, field, measurement, dataset.branch == 'rgbt', args.iteration)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - rendering_start)
            name = '%05d.png' % index
            torchvision.utils.save_image(result['render'], destination / 'renders' / name)
            torchvision.utils.save_image(view.original_image, destination / 'gt' / name)
            if not is_display:
                np.save(destination / 'renders' / ('%05d.npy' % index), result['render'][0].cpu().numpy())
                np.save(destination / 'gt' / ('%05d.npy' % index), view.original_image[0].cpu().numpy())
                difference = result['render'] - view.original_image
                measured.append(dict(image=view.image_name, MAE=float(difference.abs().mean()),
                                     RMSE=float(difference.square().mean().sqrt())))
            if dataset.branch == 'rgbt':
                torchvision.utils.save_image(result['rgb'], destination / 'renders_rgb' / name)
                torchvision.utils.save_image(view.rgb_image, destination / 'gt_rgb' / name)
            view.load2device('cpu')
            del result
    report = dict(completed=True, iteration=args.iteration, split=args.split, views=len(elapsed),
                  elapsed_seconds=time.perf_counter() - start, render_seconds=elapsed,
                  first_view_seconds=elapsed[0], steady_fps=len(elapsed[5:]) / sum(elapsed[5:]) if len(elapsed) > 5 else None,
                  peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                  peak_reserved_gpu_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                  metric_domain='display PNG' if is_display else 'floating normalized band radiance, calibrated detector signal or DN; PNG previews are not radiometric metrics',
                  scope='CUDA-synchronized full thermal/RGB renderer, including radiative field and measurement. Excludes input transfers, scene loading and file export.')
    if measured:
        report['radiometric_errors'] = measured
    (args.model / ('render_profile_%s.json' % args.split)).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
