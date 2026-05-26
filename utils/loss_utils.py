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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.fftpack import fft2, fftshift

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def image_gradient_xy(img):
    grad_x = img[..., :, 1:] - img[..., :, :-1]
    grad_y = img[..., 1:, :] - img[..., :-1, :]
    return grad_x, grad_y


def gradient_l1_loss(network_output, gt):
    pred_x, pred_y = image_gradient_xy(network_output)
    gt_x, gt_y = image_gradient_xy(gt)
    return torch.abs(pred_x - gt_x).mean() + torch.abs(pred_y - gt_y).mean()


def image_gradient_mean(img):
    grad_x, grad_y = image_gradient_xy(img)
    return 0.5 * (torch.abs(grad_x).mean() + torch.abs(grad_y).mean())


def residual_robust_l1_loss(network_output, gt, beta=0.03, min_weight=0.25):
    abs_diff = torch.abs(network_output - gt)
    residual = abs_diff.detach().mean(dim=0, keepdim=True)
    weight = beta / (residual + beta)
    weight = weight.clamp(min=float(min_weight), max=1.0)
    normalized_weight = weight / weight.mean().clamp_min(1e-6)
    loss = (abs_diff * normalized_weight).mean()
    low_weight_fraction = (weight <= float(min_weight) + 1e-6).float().mean()
    return loss, weight.mean(), low_weight_fraction


def trimmed_residual_loss(network_output, gt, quantile=0.85, mode="rmse", eps=1e-8, smooth_window=3):
    abs_diff = torch.abs(network_output - gt)
    residual = abs_diff.detach().mean(dim=0, keepdim=True)
    q = min(max(float(quantile), 0.05), 0.99)
    threshold = torch.quantile(residual.reshape(-1).float(), q).to(residual.dtype)
    inlier = (residual <= threshold).float().unsqueeze(0)
    window_size = int(smooth_window)
    if window_size > 1:
        if window_size % 2 == 0:
            window_size += 1
        neighbor_mean = F.avg_pool2d(
            inlier,
            kernel_size=window_size,
            stride=1,
            padding=window_size // 2,
        )
        inlier = torch.clamp(inlier + (neighbor_mean > 0.5).float(), 0.0, 1.0)
    inlier = inlier.squeeze(0)
    denom = (inlier.sum() * abs_diff.shape[0]).clamp_min(1.0)
    if str(mode).lower() in ("l1", "mae"):
        loss = (abs_diff * inlier).sum() / denom
    else:
        loss = torch.sqrt(((network_output - gt).pow(2) * inlier).sum() / denom + eps)
    keep_fraction = inlier.mean()
    return loss, keep_fraction, threshold


def find_corners(img):
    inputs = img.unsqueeze(0)*255
    device = img.device
    sobel_x = torch.tensor([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).to(device).repeat(1,3,1,1)
    sobel_y = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).to(device).repeat(1,3,1,1)
    I_x = F.conv2d(inputs, sobel_x, stride=1, padding=1,)
    I_y = F.conv2d(inputs, sobel_y, stride=1, padding=1,)
    k = 0.04
    I_x_squared = I_x * I_x
    I_y_squared = I_y * I_y

    I_x_y = I_x * I_y
    
    det_M = I_x_squared * I_y_squared - I_x_y* I_x_y
    trace_M = I_x_squared + I_y_squared
    R = det_M - k * (trace_M*trace_M)
    return R
    
def compute_planck_consistency_loss(pred, gt, planck_model):

    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    fft_pred = fftshift(fft2(pred))
    fft_gt = fftshift(fft2(gt))

    magnitude_pred = np.log(np.abs(fft_pred) + 1)
    magnitude_gt = np.log(np.abs(fft_gt) + 1)

    magnitude_pred = np.transpose(magnitude_pred, (1, 2, 0))  # Convert to (H, W, 3)
    magnitude_gt = np.transpose(magnitude_gt, (1, 2, 0))  # Convert to (H, W, 3)

    magnitude_pred = np.transpose(magnitude_pred, (2, 0, 1))  # Convert to (H, W, 3)
    magnitude_gt = np.transpose(magnitude_gt, (2, 0, 1))  # Convert to (H, W, 3)

    magnitude_pred = torch.tensor(magnitude_pred, dtype=torch.float32, device=planck_model.device)
    magnitude_gt = torch.tensor(magnitude_gt, dtype=torch.float32, device=planck_model.device)

    pred_features = planck_model.step(magnitude_pred)
    gt_features = planck_model.step(magnitude_gt)

    pred_flat = pred_features.view(pred_features.size(0), -1)
    gt_flat = gt_features.view(gt_features.size(0), -1)

    cosine_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1)

    normalized_cosine_sim = (cosine_sim + 1) / 2
    passband_similarity = normalized_cosine_sim.mean()

    return passband_similarity

def compute_radiometric_consistency_loss(pred, gt, radiometric_model):
    pred_features = radiometric_model.step(pred)
    gt_features = radiometric_model.step(gt)

    pred_flat = pred_features.view(pred_features.size(0), -1)
    gt_flat = gt_features.view(gt_features.size(0), -1)

    cosine_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1)

    normalized_cosine_sim = (cosine_sim + 1) / 2
    appearance_similarity = normalized_cosine_sim.mean()

    return appearance_similarity

def compute_atmos_consistency_loss(pred, gt, atmos_model, device):
    pred_np = pred.squeeze().detach().cpu().numpy()
    gt_np = gt.squeeze().detach().cpu().numpy()

    pred_channel = pred_np[0]  
    gt_channel = gt_np[0]  

    pred_edges = cv2.Canny((pred_channel * 255).astype('uint8'), 100, 200)
    gt_edges = cv2.Canny((gt_channel * 255).astype('uint8'), 100, 200)

    pred_edges_3channel = np.stack([pred_edges] * 3, axis=0)
    gt_edges_3channel = np.stack([gt_edges] * 3, axis=0)

    pred_edges_tensor = torch.tensor(pred_edges_3channel, dtype=torch.float32, device=device) / 255.0
    gt_edges_tensor = torch.tensor(gt_edges_3channel, dtype=torch.float32, device=device) / 255.0

    pred_features = atmos_model.step(pred_edges_tensor)
    gt_features = atmos_model.step(gt_edges_tensor)

    pred_flat = pred_features.view(pred_features.size(0), -1)
    gt_flat = gt_features.view(gt_features.size(0), -1)

    cosine_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1)

    normalized_cosine_sim = (cosine_sim + 1) / 2

    edge_similarity = normalized_cosine_sim.mean()

    return edge_similarity


def compute_rt_consistency_loss(
    img,
    gt,
    atmos_model,
    planck_model,
    radiometric_model,
    device,
):
    l_sim = compute_planck_consistency_loss(img, gt, planck_model)
    s_sim = compute_radiometric_consistency_loss(img, gt, radiometric_model)
    e_sim = compute_atmos_consistency_loss(img, gt, atmos_model, device)

    total_loss = (1 - l_sim) * (1 - s_sim) * (1 - e_sim)
    return total_loss


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
