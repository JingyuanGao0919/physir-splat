import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_image(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).item())


def make_coord(height, width, device):
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=0)


class ResidualRefiner(nn.Module):
    def __init__(self, hidden=32, max_residual=0.08):
        super().__init__()
        self.max_residual = float(max_residual)
        self.net = nn.Sequential(
            nn.Conv2d(5, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image, coord):
        if coord.dim() == 3:
            coord = coord.unsqueeze(0)
        coord = coord.expand(image.shape[0], -1, image.shape[-2], image.shape[-1])
        residual = self.max_residual * torch.tanh(self.net(torch.cat([image, coord], dim=1)))
        return torch.clamp(image + residual, 0.0, 1.0), residual


def list_pairs(root):
    render_dir = Path(root) / "renders"
    gt_dir = Path(root) / "gt"
    render_paths = sorted(render_dir.glob("*.png"))
    pairs = []
    for render_path in render_paths:
        gt_path = gt_dir / render_path.name
        if gt_path.exists():
            pairs.append((render_path, gt_path))
    if not pairs:
        raise RuntimeError(f"No render/gt png pairs found under {root}")
    return pairs


def load_pairs(root):
    pairs = list_pairs(root)
    renders = [read_image(r) for r, _ in pairs]
    gts = [read_image(g) for _, g in pairs]
    names = [r.name for r, _ in pairs]
    return names, renders, gts


def stack_if_same_size(tensors):
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) != 1:
        return None
    return torch.stack(tensors, dim=0)


def eval_split(model, root, out_root=None, device="cuda", batch_size=1):
    names, renders, gts = load_pairs(root)
    before = []
    after = []
    corrected_images = []
    model.eval()
    with torch.no_grad():
        for name, render_cpu, gt_cpu in zip(names, renders, gts):
            render = render_cpu.unsqueeze(0).to(device)
            gt = gt_cpu.to(device)
            coord = make_coord(render.shape[-2], render.shape[-1], device)
            corrected, residual = model(render, coord)
            corrected = corrected.squeeze(0)
            before.append(psnr(render.squeeze(0), gt))
            after.append(psnr(corrected, gt))
            corrected_images.append((name, corrected.cpu()))
            if out_root is not None:
                save_image(corrected, Path(out_root) / "renders" / name)
                save_image(gt_cpu, Path(out_root) / "gt" / name)
    return {
        "count": len(before),
        "before": sum(before) / len(before),
        "after": sum(after) / len(after),
        "delta": sum(after) / len(after) - sum(before) / len(before),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_residual", type=float, default=0.08)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    method = args.method or f"ours_{args.iteration}"
    train_root = Path(args.model_path) / "train" / method
    test_root = Path(args.model_path) / "test" / method
    out_method = f"{method}_irrefine_s{args.steps}_r{str(args.max_residual).replace('.', 'p')}"
    out_root = Path(args.model_path) / "test" / out_method
    ckpt_path = Path(args.model_path) / "ir_residual_refiner" / f"{out_method}.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    names, renders_cpu, gts_cpu = load_pairs(train_root)
    renders = stack_if_same_size(renders_cpu)
    gts = stack_if_same_size(gts_cpu)
    if renders is None or gts is None:
        raise RuntimeError("Residual refiner currently expects same-size train renders.")
    device = torch.device(args.device)
    renders = renders.to(device)
    gts = gts.to(device)
    _, _, height, width = renders.shape
    crop = min(int(args.crop), height, width)
    model = ResidualRefiner(hidden=args.hidden, max_residual=args.max_residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        "[IR_REFINER] train_pairs={} size={}x{} steps={} crop={} batch={} "
        "max_residual={} lr={}".format(
            len(names), width, height, args.steps, crop, args.batch, args.max_residual, args.lr
        )
    )
    print(
        "[IR_REFINER_HYPOTHESIS] A small bounded residual CNN can learn train-only systematic "
        "IR render errors such as blur, ringing, and low-frequency sensor fields without touching test GT."
    )

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, renders.shape[0], (args.batch,), device=device)
        ys = torch.randint(0, height - crop + 1, (args.batch,), device=device)
        xs = torch.randint(0, width - crop + 1, (args.batch,), device=device)
        batch_render = torch.empty((args.batch, 3, crop, crop), device=device)
        batch_gt = torch.empty((args.batch, 3, crop, crop), device=device)
        for b in range(args.batch):
            y = int(ys[b].item())
            x = int(xs[b].item())
            batch_render[b] = renders[idx[b], :, y:y + crop, x:x + crop]
            batch_gt[b] = gts[idx[b], :, y:y + crop, x:x + crop]
        coord = make_coord(crop, crop, device)
        corrected, residual = model(batch_render, coord)
        mse = F.mse_loss(corrected, batch_gt)
        residual_tv = (
            (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs().mean()
            + (residual[:, :, 1:, :] - residual[:, :, :-1, :]).abs().mean()
        )
        loss = mse + 0.002 * residual_tv + 0.0005 * residual.pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if args.log_every > 0 and (step % args.log_every == 0 or step == 1):
            with torch.no_grad():
                base_psnr = psnr(batch_render, batch_gt)
                refined_psnr = psnr(corrected, batch_gt)
                print(
                    "[IR_REFINER][step {}] loss={:.8f} patch_psnr={:.4f}->{:.4f} "
                    "res_mean={:.6f} res_p95={:.6f}".format(
                        step,
                        float(loss.detach().item()),
                        base_psnr,
                        refined_psnr,
                        float(residual.detach().abs().mean().item()),
                        float(torch.quantile(residual.detach().abs().reshape(-1), 0.95).item()),
                    )
                )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": args.hidden,
            "max_residual": args.max_residual,
            "method": method,
            "steps": args.steps,
        },
        ckpt_path,
    )
    train_stats = eval_split(model, train_root, out_root=None, device=device)
    test_stats = eval_split(model, test_root, out_root=out_root, device=device)
    print(
        "[IR_REFINER_RESULT] train_psnr={:.6f}->{:.6f} ({:+.6f}) "
        "test_psnr={:.6f}->{:.6f} ({:+.6f}) out={}".format(
            train_stats["before"],
            train_stats["after"],
            train_stats["delta"],
            test_stats["before"],
            test_stats["after"],
            test_stats["delta"],
            out_root,
        )
    )


if __name__ == "__main__":
    main()
