"""Sensor/display adaptation, fitted only from the permitted training images.

JPEG display values constrain relative band radiance, not absolute temperature.
Calibrated counts and brightness-temperature files require an explicit profile.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


def band_radiance(temperature, wavelength):
    wavelength_m = wavelength * 1e-6
    spectrum = (2 * 6.62607015e-34 * 299792458.0 ** 2 / wavelength_m ** 5) / torch.expm1(
        6.62607015e-34 * 299792458.0 / (wavelength_m * 1.380649e-23 * temperature[..., None]))
    return torch.trapz(spectrum, wavelength_m, dim=-1)


def flir_blackbody_signal(temperature, calibration):
    """Calibrated blackbody signal before the camera's count offset.

    This is a detector-signal unit, not an assumed rectangular-band radiance.
    The empirical calibration is used within the camera's calibrated range.
    """
    c = calibration
    reciprocal_exp = torch.exp(-c['PlanckB'] / temperature)
    return (c['PlanckR1'] / c['PlanckR2']) * reciprocal_exp / (1 - c['PlanckF'] * reciprocal_exp)


def read_measurement(path, profile):
    with Image.open(path) as image:
        if image.mode == 'P':
            image = image.convert('RGB')
        array = np.asarray(image).copy()
    domain = profile['domain']
    if domain in ('display_gray', 'display_palette'):
        if array.dtype != np.uint8:
            raise ValueError('Display profiles require uint8 images; specify a DN or temperature profile.')
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        return torch.from_numpy(array[..., :3]).permute(2, 0, 1).float() / 255
    values = torch.from_numpy(array.astype(np.float32))
    if values.ndim != 2:
        raise ValueError('Radiometric profiles require a single-channel measurement array.')
    if domain == 'linear_dn':
        values = (values - profile['dn_offset']) / profile['dn_scale']
    elif domain == 'brightness_temperature':
        temperature = values * profile['temperature_scale'] + profile['temperature_offset_kelvin']
        wavelength = torch.linspace(*profile['band_um'], 32)
        values = band_radiance(temperature, wavelength) / band_radiance(torch.tensor(300.), wavelength)
    elif domain == 'flir_counts':
        c = profile['calibration']
        reference = flir_blackbody_signal(torch.tensor(300.), c)
        values = (values + c['PlanckO']) / reference
    else:
        raise ValueError('Unknown measurement domain: ' + domain)
    return values.unsqueeze(0).expand(3, -1, -1)


def training_profile(paths, profile_path=None):
    if profile_path:
        return json.loads(Path(profile_path).read_text())
    samples = []
    for index in np.linspace(0, len(paths) - 1, min(16, len(paths)), dtype=int):
        with Image.open(paths[index]) as image:
            if image.mode == 'P':
                image = image.convert('RGB')
            array = np.asarray(image)
        if array.dtype != np.uint8:
            raise ValueError('Non-display data needs --measurement-profile with its units and scale.')
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        samples.append(array[::8, ::8, :3].reshape(-1, 3).astype(np.float64) / 255)
    pixels = np.concatenate(samples)
    chroma = np.quantile(pixels.max(1) - pixels.min(1), 0.95)
    # Color range is used to distinguish grayscale encodings from pseudocolor,
    # never to infer temperature calibration or a camera's spectral response.
    domain = 'display_gray' if chroma < 0.05 else 'display_palette'
    profile = dict(domain=domain, source='training images only',
                   calibrated=False, band_um=[7.5, 13.5], band_assumption='rectangular, uncalibrated')
    if domain == 'display_gray':
        offsets = np.median(pixels - pixels.mean(1, keepdims=True), axis=0)
        profile['channel_offsets'] = offsets.tolist()
    else:
        # A brightness-ordered initial palette is an encoding model, not an
        # inverse temperature map. Joint optimization refines this shared LUT.
        luma = pixels @ np.array([0.299, 0.587, 0.114])
        bins = np.clip(np.rint(luma * 255), 0, 255).astype(int)
        counts = np.bincount(bins, minlength=256)
        populated = np.flatnonzero(counts)
        palette = np.stack([
            np.interp(np.arange(256), populated,
                      np.bincount(bins, weights=pixels[:, c], minlength=256)[populated] / counts[populated])
            for c in range(3)], axis=-1)
        profile['palette'] = palette.tolist()
        profile['palette_mean_squared_encoding_error'] = float(np.mean((palette[bins] - pixels) ** 2))
    return profile


def blur(image, sigma):
    if sigma == 0:
        return image
    radius = max(1, int(np.ceil(3 * sigma)))
    axis = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-axis.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = len(image)
    x = F.pad(image[None], (radius, radius, radius, radius), mode='replicate')
    x = F.conv2d(x, kernel[None, None, None, :].expand(channels, 1, 1, -1), groups=channels)
    return F.conv2d(x, kernel[None, None, :, None].expand(channels, 1, -1, 1), groups=channels)[0]


class Measurement(nn.Module):
    def __init__(self, profile, response_knots=8, psf_sigma=0., digital_sigma=0.,
                 clip_display=False):
        super().__init__()
        self.profile = profile
        self.psf_sigma = psf_sigma
        self.digital_sigma = digital_sigma
        self.clip_display = clip_display
        self.raw_response = nn.Parameter(torch.zeros(response_knots))
        palette = torch.tensor(profile.get('palette', []), dtype=torch.float32).reshape(-1, 3)
        self.palette = nn.Parameter(palette)
        self.register_buffer('initial_palette', palette.clone())
        self.register_buffer('offsets', torch.tensor(profile.get('channel_offsets', [0., 0., 0.]))[:, None, None])

    def response(self, image):
        increments = F.softplus(self.raw_response)
        knots = torch.cat((increments.new_zeros(1), increments.cumsum(0) / increments.sum()))
        coord = image * len(increments)
        index = coord.floor().long().clamp(0, len(increments) - 1)
        fraction = coord - index
        return knots[index] * (1 - fraction) + knots[index + 1] * fraction

    def forward(self, radiance):
        signal = blur(radiance, self.psf_sigma)
        domain = self.profile['domain']
        if domain not in ('display_gray', 'display_palette'):
            return signal.expand(3, -1, -1)
        value = self.response(signal[0])
        if self.clip_display:
            # Display saturation belongs before palette lookup. Radiometric
            # signals above unity remain valid and bypass this display branch.
            value = value.clamp(0, 1)
        if domain == 'display_palette':
            coord = value * (len(self.palette) - 1)
            index = coord.floor().long().clamp(0, len(self.palette) - 2)
            fraction = (coord - index)[..., None]
            encoded = (self.palette[index] * (1 - fraction) + self.palette[index + 1] * fraction).permute(2, 0, 1)
        else:
            encoded = value[None].expand(3, -1, -1) + self.offsets
        return blur(encoded, self.digital_sigma)

    def regularization(self):
        increments = F.softplus(self.raw_response)
        loss = 1e-4 * (increments / increments.mean() - 1).square().mean()
        if self.palette.numel():
            loss = loss + 1e-3 * (self.palette - self.initial_palette).square().mean()
            loss = loss + 1e-3 * (self.palette[2:] - 2 * self.palette[1:-1] + self.palette[:-2]).square().mean()
        return loss
