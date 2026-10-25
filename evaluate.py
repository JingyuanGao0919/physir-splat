"""Score a fixed raw-render checkpoint with one shared thermal/RGB evaluator."""
import argparse
import json
from pathlib import Path

import lpips
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor

from utils.image_utils import psnr
from utils.loss_utils import ssim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--split', choices=['test', 'validation', 'train'], default='test')
    args = parser.parse_args()
    method = args.model / args.split / ('ours_' + str(args.iteration))
    pairs = [('thermal', 'renders', 'gt')]
    if (method / 'renders_rgb').is_dir():
        pairs.append(('rgb', 'renders_rgb', 'gt_rgb'))
    metric = lpips.LPIPS(net='vgg').cuda().eval()
    report = dict(iteration=args.iteration, split=args.split, model=str(args.model), modalities={},
                  convention='8-bit PNG; native LPIPS uses [0,1] without normalization; standard LPIPS uses normalize=True. No color alignment or checkpoint selection.')
    with torch.no_grad():
        for modality, rendered_dir, gt_dir in pairs:
            rows = []
            for path in sorted((method / rendered_dir).glob('*.png')):
                with Image.open(path) as image:
                    rendered = to_tensor(image)[:3].unsqueeze(0).cuda()
                with Image.open(method / gt_dir / path.name) as image:
                    gt = to_tensor(image)[:3].unsqueeze(0).cuda()
                rows.append(dict(
                    image=path.name, PSNR=float(psnr(rendered, gt).mean()),
                    SSIM=float(ssim(rendered, gt)),
                    LPIPS_native=float(metric(rendered, gt)),
                    LPIPS_standard=float(metric(rendered, gt, normalize=True)),
                ))
            means = {name: sum(row[name] for row in rows) / len(rows)
                     for name in ['PSNR', 'SSIM', 'LPIPS_native', 'LPIPS_standard']}
            report['modalities'][modality] = dict(count=len(rows), mean=means, per_view=rows)
            print(modality, json.dumps(means), flush=True)
    output_name = 'benchmark_metrics.json' if args.split == 'test' else 'benchmark_metrics_%s.json' % args.split
    (args.model / output_name).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
