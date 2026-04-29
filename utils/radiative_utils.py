import json
import math
import os
from typing import Dict

import torch
import torch.nn.functional as F

from utils.sh_utils import SH2RGB


def _safe_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor(float("nan"), device=x.device)
    return torch.quantile(x.float(), q)


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().reshape(-1).float()
    if x.numel() == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "p5": float("nan"),
            "p95": float("nan"),
            "std": float("nan"),
        }
    return {
        "min": x.min().item(),
        "max": x.max().item(),
        "mean": x.mean().item(),
        "p5": _safe_quantile(x, 0.05).item(),
        "p95": _safe_quantile(x, 0.95).item(),
        "std": x.std(unbiased=False).item(),
    }


def corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().reshape(-1).float()
    y = y.detach().reshape(-1).float()
    if x.numel() == 0 or y.numel() == 0 or x.numel() != y.numel():
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().mean() + 1e-12) * (y.square().mean() + 1e-12))
    if denom.item() == 0.0:
        return float("nan")
    return (x * y).mean().div(denom).item()


def frac_near_bounds(x: torch.Tensor, low: float, high: float, tol: float) -> Dict[str, float]:
    x = x.detach().reshape(-1)
    if x.numel() == 0:
        return {"near_low": float("nan"), "near_high": float("nan")}
    return {
        "near_low": (x <= low + tol).float().mean().item(),
        "near_high": (x >= high - tol).float().mean().item(),
    }


class PlanckLUT:
    """LWIR Planck lookup used only to read a normalized emission proxy as temperature."""

    def __init__(
        self,
        temp_min: float,
        temp_max: float,
        lambda_min_um: float,
        lambda_max_um: float,
        device,
        num_temp: int = 2048,
        num_lambda: int = 256,
    ):
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)

        temps = torch.linspace(self.temp_min, self.temp_max, int(num_temp), dtype=torch.float64)
        lambdas = torch.linspace(float(lambda_min_um), float(lambda_max_um), int(num_lambda), dtype=torch.float64) * 1e-6
        h = 6.62607015e-34
        c = 299792458.0
        k = 1.380649e-23

        lam = lambdas[:, None]
        T = temps[None, :]
        exponent = torch.clamp((h * c) / (lam * k * T), max=80.0)
        spectral = (2.0 * h * c ** 2) / (lam ** 5) / (torch.exp(exponent) - 1.0 + 1e-12)
        radiance = torch.trapz(spectral, lambdas, dim=0).float()
        radiance_norm = (radiance - radiance.min()) / (radiance.max() - radiance.min() + 1e-12)

        self.temps = temps.float().to(device)
        self.radiance_norm = radiance_norm.to(device)

    def norm_radiance_to_temp(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(u.float(), 0.0, 1.0)
        lut_u = self.radiance_norm
        lut_t = self.temps
        idx = torch.searchsorted(lut_u, u.reshape(-1))
        idx = torch.clamp(idx, 1, lut_u.numel() - 1)
        u0 = lut_u[idx - 1]
        u1 = lut_u[idx]
        t0 = lut_t[idx - 1]
        t1 = lut_t[idx]
        w = (u.reshape(-1) - u0) / (u1 - u0 + 1e-12)
        return (t0 + w * (t1 - t0)).reshape_as(u)


def extract_thermal_gaussian_target(gaussians) -> Dict[str, torch.Tensor]:
    dc = gaussians.get_thermal_features_dc[:, 0, :]
    rgb_dc = torch.clamp(SH2RGB(dc), 0.0, 1.0)
    q_target = thermal_rgb_to_gray(rgb_dc)

    thermal_features = gaussians.get_thermal_features
    rest = thermal_features[:, 1:, :].reshape(thermal_features.shape[0], -1)
    rest_energy = torch.norm(rest, dim=1, keepdim=True)
    dc_energy = torch.norm(dc, dim=1, keepdim=True) + 1e-8
    rest_ratio = rest_energy / dc_energy

    return {
        "rgb_dc": rgb_dc,
        "q_target": q_target,
        "rest_ratio": rest_ratio,
    }


def current_thermal_gaussian_luminance(gaussians) -> Dict[str, torch.Tensor]:
    dc = gaussians.get_thermal_features_dc[:, 0, :]
    rgb_dc = torch.clamp(SH2RGB(dc), 0.0, 1.0)
    q = thermal_rgb_to_gray(rgb_dc)
    thermal_features = gaussians.get_thermal_features
    rest = thermal_features[:, 1:, :].reshape(thermal_features.shape[0], -1)
    rest_energy = torch.norm(rest, dim=1, keepdim=True)
    dc_energy = torch.norm(dc, dim=1, keepdim=True) + 1e-8
    rest_ratio = rest_energy / dc_energy
    return {"rgb_dc": rgb_dc, "q": q, "rest_ratio": rest_ratio}


def thermal_rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.shape[-1] == 1:
        return rgb
    weights = rgb.new_tensor([0.299, 0.587, 0.114])
    return (rgb[..., :3] * weights).sum(dim=-1, keepdim=True)


def ramp_value(iteration: int, start_iter: int, ramp_iters: int, max_value: float) -> float:
    if iteration < int(start_iter):
        return 0.0
    span = max(int(ramp_iters), 1)
    progress = min(1.0, max(iteration - int(start_iter) + 1, 0) / float(span))
    return float(max_value) * progress


def physir_render_blend(iteration: int, opt) -> float:
    return ramp_value(
        iteration,
        opt.phys_render_start_iter,
        opt.phys_render_blend_iters,
        opt.phys_render_max_blend,
    )


def radiative_guidance_gate(iteration: int, opt) -> float:
    return ramp_value(
        iteration,
        opt.phys_guidance_start_iter,
        opt.phys_guidance_ramp_iters,
        opt.phys_guidance_max_weight,
    )


def tir_auxiliary_gate(iteration: int, opt) -> float:
    return ramp_value(
        iteration,
        opt.phys_aux_start_iter,
        opt.phys_aux_ramp_iters,
        opt.phys_aux_mse_max_weight,
    )


def radiative_confidence_values(gaussians, opt) -> torch.Tensor:
    device = gaussians.get_xyz.device
    if not gaussians.has_radiative_transfer_attributes():
        return torch.ones((gaussians.get_xyz.shape[0], 1), device=device)

    q_ref = gaussians.get_passband_radiance.detach().to(device).clamp(0.0, 1.0)
    delta_ref = gaussians.get_radiometric_residual.detach().to(device).abs()
    temp_ref = gaussians.get_thermal_temperature.detach().to(device)

    if q_ref.shape[0] != gaussians.get_xyz.shape[0]:
        return torch.ones((gaussians.get_xyz.shape[0], 1), device=device)

    # Mid-range, low-residual, non-boundary physical estimates are the most
    # reliable. Keep the range intentionally narrow so this can only reweight
    # the PSNR-oriented loss mildly, not rewrite the render target.
    q_mid = (1.0 - 2.0 * (q_ref - 0.5).abs()).clamp(0.0, 1.0)
    delta_conf = (1.0 - delta_ref / max(float(opt.phys_probe_delta_max), 1e-6)).clamp(0.0, 1.0)
    temp_unit = (
        (temp_ref - float(opt.phys_probe_temp_min))
        / max(float(opt.phys_probe_temp_max - opt.phys_probe_temp_min), 1e-6)
    ).clamp(0.0, 1.0)
    temp_mid = (1.0 - 2.0 * (temp_unit - 0.5).abs()).clamp(0.0, 1.0)

    confidence = 0.70 + 0.18 * q_mid + 0.08 * delta_conf + 0.04 * temp_mid
    return confidence.clamp(0.0, 1.0)


def radiative_confidence_mse_loss(image: torch.Tensor, gt_image: torch.Tensor, confidence_image: torch.Tensor, opt, iteration: int):
    zero = image.new_tensor(0.0)
    stats = {
        "gate": 0.0,
        "mse": 0.0,
        "weighted_mse": 0.0,
        "total": 0.0,
        "conf_mean": 0.0,
        "conf_min": 0.0,
        "conf_max": 0.0,
    }

    gate = tir_auxiliary_gate(iteration, opt)
    if gate <= 0.0:
        return zero, stats

    err_sq = (image - gt_image).square()
    confidence = confidence_image.detach().mean(dim=0, keepdim=True).clamp_min(0.0)
    conf_mean = confidence.mean().clamp_min(1e-6)
    weights = (confidence / conf_mean).clamp(float(opt.phys_aux_conf_min), float(opt.phys_aux_conf_max))
    weights = weights / weights.mean().clamp_min(1e-6)

    mse = err_sq.mean()
    weighted_mse = (err_sq * weights).mean()
    mix = max(0.0, min(float(opt.phys_aux_conf_mix), 1.0))
    aux = (1.0 - mix) * mse + mix * weighted_mse
    total = gate * aux

    stats = {
        "gate": float(gate),
        "mse": float(mse.detach().item()),
        "weighted_mse": float(weighted_mse.detach().item()),
        "total": float(total.detach().item()),
        "conf_mean": float(confidence.detach().mean().item()),
        "conf_min": float(confidence.detach().min().item()),
        "conf_max": float(confidence.detach().max().item()),
    }
    return total, stats


def rt_guidance_loss(gaussians, opt, iteration: int):
    device = gaussians.get_xyz.device
    zero = gaussians.get_xyz.new_tensor(0.0)
    stats = {
        "gate": 0.0,
        "q_loss": 0.0,
        "rest_loss": 0.0,
        "delta_loss": 0.0,
        "total": 0.0,
    }

    gate = radiative_guidance_gate(iteration, opt)
    if gate <= 0.0 or not gaussians.has_radiative_transfer_attributes():
        return zero, stats

    current = current_thermal_gaussian_luminance(gaussians)
    q = current["q"]
    rest_ratio = current["rest_ratio"]
    q_ref = gaussians.get_passband_radiance.detach().to(device).clamp(0.0, 1.0)
    delta_ref = gaussians.get_radiometric_residual.detach().to(device)

    if q_ref.shape != q.shape or delta_ref.shape != q.shape:
        return zero, stats

    # Saturated endpoints in thermal imagery are less reliable as supervision,
    # so they get a lower but nonzero weight.
    confidence = ((q_ref > 0.02) & (q_ref < 0.98)).float()
    confidence = 0.25 + 0.75 * confidence
    q_err = F.smooth_l1_loss(q, q_ref, beta=opt.phys_guidance_huber_beta, reduction="none")
    q_loss = (confidence * q_err).sum() / (confidence.sum() + 1e-6)

    delta_abs = delta_ref.abs()
    if delta_abs.numel() > 0 and torch.any(delta_abs > 0):
        delta_scale = torch.quantile(delta_abs.reshape(-1), 0.90).clamp_min(1e-6)
        delta_weight = (delta_abs / delta_scale).clamp(0.0, 1.0)
    else:
        delta_weight = torch.zeros_like(delta_abs)

    if rest_ratio.numel() > 0:
        rest_target = torch.quantile(rest_ratio.detach().reshape(-1), opt.phys_guidance_rest_quantile)
        rest_excess = F.relu(rest_ratio - rest_target)
        rest_loss = (delta_weight * rest_excess).sum() / (delta_weight.sum() + 1e-6)
    else:
        rest_loss = zero

    delta_loss = (delta_weight * q_err).sum() / (delta_weight.sum() + 1e-6)
    guided = (
        opt.phys_guidance_q_weight * q_loss
        + opt.phys_guidance_rest_weight * rest_loss
        + opt.phys_guidance_delta_weight * delta_loss
    )
    total = gate * guided
    stats = {
        "gate": float(gate),
        "q_loss": float(q_loss.detach().item()),
        "rest_loss": float(rest_loss.detach().item()),
        "delta_loss": float(delta_loss.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, stats


def fit_rt_attributes(q_target: torch.Tensor, opt, device, lut: PlanckLUT) -> Dict[str, torch.Tensor]:
    q = torch.clamp(q_target.detach().reshape(-1), 1e-4, 1.0 - 1e-4)
    n = q.numel()

    ambient_init = torch.quantile(q, opt.phys_probe_ambient_quantile).item()
    ambient_init = max(min(ambient_init, 0.95), 1e-4)
    e_frac = (opt.phys_probe_e_init - opt.phys_probe_e_min) / max(opt.phys_probe_e_max - opt.phys_probe_e_min, 1e-6)
    e_frac = max(min(float(e_frac), 1.0 - 1e-4), 1e-4)

    u_raw = torch.logit(q).detach().clone().requires_grad_(True)
    e_raw = torch.full((n,), math.log(e_frac / (1.0 - e_frac)), device=device, dtype=torch.float32, requires_grad=True)
    a_raw = torch.full((n,), math.log(ambient_init / (1.0 - ambient_init)), device=device, dtype=torch.float32, requires_grad=True)
    delta_raw = torch.zeros((n,), device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([u_raw, e_raw, a_raw, delta_raw], lr=opt.phys_probe_lr)

    e_init = torch.full((n,), opt.phys_probe_e_init, device=device)
    a_init = torch.full((n,), ambient_init, device=device)

    for _ in range(int(opt.phys_probe_steps)):
        u = torch.sigmoid(u_raw)
        e = opt.phys_probe_e_min + (opt.phys_probe_e_max - opt.phys_probe_e_min) * torch.sigmoid(e_raw)
        a = torch.sigmoid(a_raw)
        delta = opt.phys_probe_delta_max * torch.tanh(delta_raw)
        q_phys = e * u + (1.0 - e) * a + delta

        fit = F.smooth_l1_loss(q_phys, q, beta=opt.phys_probe_huber_beta)
        e_prior = F.smooth_l1_loss(e, e_init, beta=opt.phys_probe_huber_beta)
        a_prior = F.smooth_l1_loss(a, a_init, beta=opt.phys_probe_huber_beta)
        delta_reg = delta.abs().mean()
        loss = (
            fit
            + opt.phys_probe_lambda_e_prior * e_prior
            + opt.phys_probe_lambda_a_prior * a_prior
            + opt.phys_probe_lambda_delta * delta_reg
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u = torch.sigmoid(u_raw)
        e = opt.phys_probe_e_min + (opt.phys_probe_e_max - opt.phys_probe_e_min) * torch.sigmoid(e_raw)
        a = torch.sigmoid(a_raw)
        delta = opt.phys_probe_delta_max * torch.tanh(delta_raw)
        q_phys = e * u + (1.0 - e) * a + delta
        temp = lut.norm_radiance_to_temp(u)

    return {
        "emission": u[:, None],
        "emissivity": e[:, None],
        "ambient": a[:, None],
        "delta": delta[:, None],
        "q_phys": q_phys[:, None],
        "temperature": temp[:, None],
        "fit_abs_err": (q_phys - q).abs()[:, None],
        "fit_sq_err": (q_phys - q).square()[:, None],
        "ambient_init": torch.full((n, 1), ambient_init, device=device),
    }


def _add_prefixed(summary: Dict[str, float], prefix: str, stats: Dict[str, float]) -> None:
    for key, value in stats.items():
        summary[f"{prefix}_{key}"] = value


def gather_rt_summary(iteration, prev_point_count, gaussians, extracted, fit, opt) -> Dict[str, float]:
    q_target = extracted["q_target"]
    q_phys = fit["q_phys"]
    temp = fit["temperature"]
    emission = fit["emission"]
    emissivity = fit["emissivity"]
    ambient = fit["ambient"]
    delta = fit["delta"]
    rest_ratio = extracted["rest_ratio"]
    opacity = gaussians.get_opacity.detach()
    scaling = gaussians.get_scaling.detach()
    scale_max = scaling.max(dim=1, keepdim=True).values
    scale_min = scaling.min(dim=1, keepdim=True).values.clamp_min(1e-8)
    anisotropy = scale_max / scale_min
    radius = torch.norm(gaussians.get_xyz.detach(), dim=1, keepdim=True)

    summary = {
        "iteration": float(iteration),
        "N": float(q_target.shape[0]),
        "dN": float(q_target.shape[0] - prev_point_count),
        "fit_mae": fit["fit_abs_err"].mean().item(),
        "fit_rmse": fit["fit_sq_err"].mean().sqrt().item(),
        "fit_corr": corrcoef(q_target, q_phys),
        "opacity_q_corr": corrcoef(opacity, q_target),
        "rest_q_corr": corrcoef(rest_ratio, q_target),
        "radius_q_corr": corrcoef(radius, q_target),
        "delta_rest_corr": corrcoef(delta, rest_ratio),
        "delta_anis_corr": corrcoef(delta, anisotropy),
        "emission_rest_corr": corrcoef(emission, rest_ratio),
        "emission_anis_corr": corrcoef(emission, anisotropy),
        "ambient_opacity_corr": corrcoef(ambient, opacity),
    }

    eps = 1e-6
    q_safe = q_phys.clamp_min(eps)
    emit_ratio = (emissivity * emission) / q_safe
    ambient_ratio = ((1.0 - emissivity) * ambient) / q_safe
    residual_ratio = delta.abs() / q_safe

    _add_prefixed(summary, "q_target", tensor_stats(q_target))
    _add_prefixed(summary, "q_phys", tensor_stats(q_phys))
    _add_prefixed(summary, "temperature", tensor_stats(temp))
    _add_prefixed(summary, "emission", tensor_stats(emission))
    _add_prefixed(summary, "emissivity", tensor_stats(emissivity))
    _add_prefixed(summary, "ambient", tensor_stats(ambient))
    _add_prefixed(summary, "delta", tensor_stats(delta))
    _add_prefixed(summary, "rest_ratio", tensor_stats(rest_ratio))
    _add_prefixed(summary, "opacity", tensor_stats(opacity))
    _add_prefixed(summary, "scale_max", tensor_stats(scale_max))
    _add_prefixed(summary, "anisotropy", tensor_stats(anisotropy))
    _add_prefixed(summary, "emit_ratio", tensor_stats(emit_ratio))
    _add_prefixed(summary, "ambient_ratio", tensor_stats(ambient_ratio))
    _add_prefixed(summary, "residual_ratio", tensor_stats(residual_ratio))

    temp_bounds = frac_near_bounds(temp, opt.phys_probe_temp_min, opt.phys_probe_temp_max, tol=0.5)
    e_bounds = frac_near_bounds(emissivity, opt.phys_probe_e_min, opt.phys_probe_e_max, tol=1e-3)
    summary["temperature_near_min_frac"] = temp_bounds["near_low"]
    summary["temperature_near_max_frac"] = temp_bounds["near_high"]
    summary["emissivity_near_min_frac"] = e_bounds["near_low"]
    summary["emissivity_near_max_frac"] = e_bounds["near_high"]
    summary["delta_abs_gt_0.02_frac"] = (delta.abs() > 0.02).float().mean().item()
    summary["ambient_init"] = fit["ambient_init"][0].item() if fit["ambient_init"].numel() else float("nan")
    return summary


def summary_to_text(iteration: int, summary: Dict[str, float]) -> str:
    def fmt(name: str, digits: int = 4) -> str:
        value = summary.get(name, float("nan"))
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"

    return (
        f"[ITER {iteration}] PhysIRProbe | "
        f"T mean {fmt('temperature_mean')} p5 {fmt('temperature_p5')} p95 {fmt('temperature_p95')} | "
        f"emissivity mean {fmt('emissivity_mean')} p5 {fmt('emissivity_p5')} p95 {fmt('emissivity_p95')}"
    )


def diag_to_text(iteration: int, summary: Dict[str, float]) -> str:
    def fmt(name: str, digits: int = 4) -> str:
        value = summary.get(name, float("nan"))
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"

    return (
        f"[ITER {iteration}] PhysIRProbeDiag | "
        f"T@min {fmt('temperature_near_min_frac')} T@max {fmt('temperature_near_max_frac')} | "
        f"e@min {fmt('emissivity_near_min_frac')} e@max {fmt('emissivity_near_max_frac')} | "
        f"delta>|0.02| {fmt('delta_abs_gt_0.02_frac')} | "
        f"emit_ratio mean {fmt('emit_ratio_mean')} residual_ratio mean {fmt('residual_ratio_mean')} | "
        f"corr(opacity,q) {fmt('opacity_q_corr')} corr(rest,q) {fmt('rest_q_corr')} corr(radius,q) {fmt('radius_q_corr')}"
    )


def append_physir_summary(log_path: str, summary: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def fit_and_log_physir_attributes(iteration: int, gaussians, scene, opt, model_path: str, prev_point_count: int):
    del scene
    device = gaussians.get_xyz.device
    if gaussians.get_xyz.numel() == 0:
        return None

    lut = PlanckLUT(
        temp_min=opt.phys_probe_temp_min,
        temp_max=opt.phys_probe_temp_max,
        lambda_min_um=opt.phys_probe_lambda_min_um,
        lambda_max_um=opt.phys_probe_lambda_max_um,
        device=device,
    )

    extracted = extract_thermal_gaussian_target(gaussians)
    with torch.enable_grad():
        fit = fit_rt_attributes(extracted["q_target"], opt, device, lut)

    gaussians.update_radiative_transfer_attributes(
        iteration=iteration,
        q_target=extracted["q_target"],
        q_phys=fit["q_phys"],
        temperature=fit["temperature"],
        emissivity=fit["emissivity"],
        emission=fit["emission"],
        ambient=fit["ambient"],
        delta=fit["delta"],
        rest_ratio=extracted["rest_ratio"],
        rgb_dc=extracted["rgb_dc"],
    )

    summary = gather_rt_summary(iteration, prev_point_count, gaussians, extracted, fit, opt)
    text = summary_to_text(iteration, summary)
    print(text)
    append_physir_summary(os.path.join(model_path, "phys_probe", "probe_log.jsonl"), summary)
    gaussians.save_radiative_transfer_attributes(os.path.join(model_path, "phys_probe", f"iteration_{iteration}", "phys_probe.pt"))

    result_file_path = os.path.join(model_path, "result.txt")
    with open(result_file_path, "a", encoding="utf-8") as result_file:
        result_file.write(text + "\n")

    return summary

