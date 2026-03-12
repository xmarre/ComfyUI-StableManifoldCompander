from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _ensure_bhwc(image: Tensor) -> Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dims [B,H,W,C], got shape {tuple(image.shape)}")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected channel-last tensor, got shape {tuple(image.shape)}")
    return image


def _ensure_mask_bhw(mask: Tensor) -> Tensor:
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Expected mask tensor [B,H,W] or [H,W], got shape {tuple(mask.shape)}")
    return mask


def _bhwc_to_bchw(image: Tensor) -> Tensor:
    return image.permute(0, 3, 1, 2).contiguous()


def _bchw_to_bhwc(image: Tensor) -> Tensor:
    return image.permute(0, 2, 3, 1).contiguous()


def _match_batch(tensor: Tensor, batch: int) -> Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        reps = [batch] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*reps)
    raise ValueError(f"Cannot match batch {tensor.shape[0]} to target {batch}")


def _make_gaussian_kernel1d(radius: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    radius = max(int(radius), 0)
    if radius == 0:
        return torch.ones((1,), device=device, dtype=dtype)
    sigma = float(max(sigma, 1e-4))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum().clamp_min(1e-8)
    return kernel


def gaussian_blur_bchw(image: Tensor, radius: int, sigma: Optional[float] = None) -> Tensor:
    radius = max(int(radius), 0)
    if radius == 0:
        return image
    sigma = float(radius) / 2.0 if sigma is None else float(sigma)
    kernel_1d = _make_gaussian_kernel1d(radius, sigma, image.device, image.dtype)
    kernel_h = kernel_1d.view(1, 1, -1, 1)
    kernel_w = kernel_1d.view(1, 1, 1, -1)

    channels = image.shape[1]
    kernel_h = kernel_h.repeat(channels, 1, 1, 1)
    kernel_w = kernel_w.repeat(channels, 1, 1, 1)

    padded = F.pad(image, (0, 0, radius, radius), mode="reflect")
    blurred = F.conv2d(padded, kernel_h, groups=channels)
    padded = F.pad(blurred, (radius, radius, 0, 0), mode="reflect")
    blurred = F.conv2d(padded, kernel_w, groups=channels)
    return blurred


def resize_bhwc(image: Tensor, height: int, width: int, mode: str = "bilinear") -> Tensor:
    image = _ensure_bhwc(image)
    bchw = _bhwc_to_bchw(image)
    align_corners = False if mode in ("bilinear", "bicubic") else None
    resized = F.interpolate(bchw, size=(height, width), mode=mode, align_corners=align_corners)
    return _bchw_to_bhwc(resized)


def resize_mask(mask: Tensor, height: int, width: int) -> Tensor:
    mask = _ensure_mask_bhw(mask).unsqueeze(1)
    resized = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    return resized[:, 0].clamp(0.0, 1.0)


def rgb_to_oklab(rgb: Tensor) -> Tensor:
    # rgb: [B,C,H,W], assumed 0..1 sRGB
    rgb = rgb.clamp(0.0, 1.0)
    threshold = 0.04045
    linear = torch.where(rgb <= threshold, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = torch.sign(l) * torch.abs(l).clamp_min(1e-8).pow(1.0 / 3.0)
    m_ = torch.sign(m) * torch.abs(m).clamp_min(1e-8).pow(1.0 / 3.0)
    s_ = torch.sign(s) * torch.abs(s).clamp_min(1e-8).pow(1.0 / 3.0)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b2 = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return torch.cat([L, a, b2], dim=1)


def oklab_to_rgb(oklab: Tensor) -> Tensor:
    L, a, b = oklab[:, 0:1], oklab[:, 1:2], oklab[:, 2:3]

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b2 = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    linear = torch.cat([r, g, b2], dim=1)

    threshold = 0.0031308
    srgb = torch.where(linear <= threshold, 12.92 * linear, 1.055 * linear.clamp_min(0.0).pow(1.0 / 2.4) - 0.055)
    return srgb.clamp(0.0, 1.0)


def build_low_high_layers(image_bhwc: Tensor, blur_radius: int) -> Tuple[Tensor, Tensor]:
    image = _ensure_bhwc(image_bhwc)
    bchw = _bhwc_to_bchw(image)
    low = gaussian_blur_bchw(bchw, radius=blur_radius)
    high = bchw - low
    return _bchw_to_bhwc(low), _bchw_to_bhwc(high)


def mask_center_and_extent(mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    mask = _ensure_mask_bhw(mask)
    batch, height, width = mask.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=mask.device, dtype=mask.dtype),
        torch.linspace(-1.0, 1.0, width, device=mask.device, dtype=mask.dtype),
        indexing="ij",
    )
    yy = yy.unsqueeze(0).expand(batch, -1, -1)
    xx = xx.unsqueeze(0).expand(batch, -1, -1)

    mass = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    cx = (mask * xx).sum(dim=(1, 2), keepdim=True) / mass
    cy = (mask * yy).sum(dim=(1, 2), keepdim=True) / mass

    dx = xx - cx
    dy = yy - cy
    rx = ((mask * (dx * dx)).sum(dim=(1, 2), keepdim=True) / mass).sqrt().clamp_min(1e-3)
    ry = ((mask * (dy * dy)).sum(dim=(1, 2), keepdim=True) / mass).sqrt().clamp_min(1e-3)
    return cx, cy, torch.cat([rx, ry], dim=-1)


def signedish_mask_falloff(mask: Tensor, blur_radius: int) -> Tensor:
    mask = _ensure_mask_bhw(mask).unsqueeze(1)
    if blur_radius <= 0:
        return mask[:, 0].clamp(0.0, 1.0)
    smooth = gaussian_blur_bchw(mask, radius=blur_radius)[:, 0]
    smooth = smooth / smooth.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return smooth.clamp(0.0, 1.0)


def _build_base_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid


def warp_image_bhwc(image: Tensor, flow_xy_norm: Tensor, mode: str = "bicubic") -> Tensor:
    image = _ensure_bhwc(image)
    batch, height, width, _ = image.shape
    if flow_xy_norm.shape != (batch, height, width, 2):
        raise ValueError(
            f"Expected flow shape {(batch, height, width, 2)}, got {tuple(flow_xy_norm.shape)}"
        )
    base_grid = _build_base_grid(batch, height, width, image.device, image.dtype)
    sample_grid = (base_grid + flow_xy_norm).clamp(-1.25, 1.25)
    bchw = _bhwc_to_bchw(image)
    warped = F.grid_sample(
        bchw,
        sample_grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )
    return _bchw_to_bhwc(warped)


@dataclass
class CompandConfig:
    anchor_mp: float = 1.0
    base_blur_radius: int = 9
    mask_falloff_radius: int = 24
    warp_strength: float = 0.65
    radial_strength: float = 1.0
    anisotropy_strength: float = 0.15
    lowfreq_anchor_mix: float = 0.72
    chroma_restore: float = 0.35
    contrast_restore: float = 0.20
    detail_preservation: float = 1.0
    max_inward_shift_px: float = 6.0
    edge_softness: int = 6

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "anchor_mp": self.anchor_mp,
            "base_blur_radius": self.base_blur_radius,
            "mask_falloff_radius": self.mask_falloff_radius,
            "warp_strength": self.warp_strength,
            "radial_strength": self.radial_strength,
            "anisotropy_strength": self.anisotropy_strength,
            "lowfreq_anchor_mix": self.lowfreq_anchor_mix,
            "chroma_restore": self.chroma_restore,
            "contrast_restore": self.contrast_restore,
            "detail_preservation": self.detail_preservation,
            "max_inward_shift_px": self.max_inward_shift_px,
            "edge_softness": self.edge_softness,
        }


@dataclass
class CompandDebug:
    anchor_image: Tensor
    corrected_base: Tensor
    warped_high_image: Tensor
    flow_visual: Tensor
    debug_mask: Tensor
    metrics: Dict[str, float]


def build_anchor_size(height: int, width: int, target_mp: float, round_to: int = 16) -> Tuple[int, int]:
    target_pixels = float(max(target_mp, 1e-4)) * 1_000_000.0
    current_pixels = float(height * width)
    if current_pixels <= 0:
        raise ValueError("Invalid image size")
    scale = math.sqrt(target_pixels / current_pixels)
    anchor_h = max(round_to, int(round(height * scale / round_to) * round_to))
    anchor_w = max(round_to, int(round(width * scale / round_to) * round_to))
    return anchor_h, anchor_w


def estimate_radial_flow(mask: Tensor, expansion_ratio: Tensor, config: CompandConfig) -> Tensor:
    mask = _ensure_mask_bhw(mask)
    batch, height, width = mask.shape
    cx, cy, extent = mask_center_and_extent(mask)
    rx = extent[..., 0:1].view(batch, 1, 1)
    ry = extent[..., 1:2].view(batch, 1, 1)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=mask.device, dtype=mask.dtype),
        torch.linspace(-1.0, 1.0, width, device=mask.device, dtype=mask.dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(batch, -1, -1)
    yy = yy.unsqueeze(0).expand(batch, -1, -1)

    dx = xx - cx.view(batch, 1, 1)
    dy = yy - cy.view(batch, 1, 1)
    ex = dx / rx
    ey = dy / ry
    radius = torch.sqrt(ex * ex + ey * ey + 1e-8)
    direction_x = dx / (torch.sqrt(dx * dx + dy * dy + 1e-8))
    direction_y = dy / (torch.sqrt(dx * dx + dy * dy + 1e-8))

    smooth_mask = signedish_mask_falloff(mask, blur_radius=config.mask_falloff_radius)
    radial_envelope = torch.exp(-0.5 * (radius / 1.15) ** 2)
    outward = (expansion_ratio.view(batch, 1, 1) - 1.0).clamp(min=0.0)
    max_shift_norm_x = 2.0 * config.max_inward_shift_px / max(width - 1, 1)
    max_shift_norm_y = 2.0 * config.max_inward_shift_px / max(height - 1, 1)

    raw_strength = smooth_mask * radial_envelope * outward * config.warp_strength * config.radial_strength
    shift_x = (-direction_x * raw_strength).clamp(-max_shift_norm_x, max_shift_norm_x)
    shift_y = (-direction_y * raw_strength).clamp(-max_shift_norm_y, max_shift_norm_y)

    # A light anisotropic term reduces “vertical ballooning” / “horizontal ballooning” mismatch.
    aniso_x = (-dx * smooth_mask * outward * config.anisotropy_strength).clamp(-max_shift_norm_x, max_shift_norm_x)
    aniso_y = (-dy * smooth_mask * outward * config.anisotropy_strength).clamp(-max_shift_norm_y, max_shift_norm_y)

    shift_x = shift_x + aniso_x
    shift_y = shift_y + aniso_y
    return torch.stack([shift_x, shift_y], dim=-1)


def estimate_expansion_ratio(
    anchor_base: Tensor,
    high_base: Tensor,
    mask: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    # Estimate how much “broader / flatter” the high-res output became than the anchor.
    # Uses weighted second moments over luminance gradients inside the mask.
    anchor = _bhwc_to_bchw(anchor_base)
    high = _bhwc_to_bchw(high_base)
    mask_b = _ensure_mask_bhw(mask)

    def grad_energy(img: Tensor) -> Tensor:
        gray = 0.2126 * img[:, 0] + 0.7152 * img[:, 1] + 0.0722 * img[:, 2]
        gx = gray[:, :, 1:] - gray[:, :, :-1]
        gy = gray[:, 1:, :] - gray[:, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return (gx * gx + gy * gy).clamp_min(1e-8)

    ea = grad_energy(anchor)
    eh = grad_energy(high)

    batch, height, width = mask_b.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=mask_b.device, dtype=mask_b.dtype),
        torch.linspace(-1.0, 1.0, width, device=mask_b.device, dtype=mask_b.dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(batch, -1, -1)
    yy = yy.unsqueeze(0).expand(batch, -1, -1)

    def weighted_radius(energy: Tensor) -> Tensor:
        w = (energy * mask_b).clamp_min(1e-8)
        mass = w.sum(dim=(1, 2)).clamp_min(1e-6)
        cx = (w * xx).sum(dim=(1, 2)) / mass
        cy = (w * yy).sum(dim=(1, 2)) / mass
        r2 = (xx - cx[:, None, None]) ** 2 + (yy - cy[:, None, None]) ** 2
        return torch.sqrt((w * r2).sum(dim=(1, 2)) / mass).clamp_min(1e-6)

    ra = weighted_radius(ea)
    rh = weighted_radius(eh)
    ratio = (rh / ra).clamp(0.85, 1.25)

    metrics = {
        "anchor_radius_mean": float(ra.mean().item()),
        "high_radius_mean": float(rh.mean().item()),
        "estimated_expansion_mean": float(ratio.mean().item()),
    }
    return ratio, metrics


def restore_low_frequency_color(
    corrected_high_base: Tensor,
    anchor_base: Tensor,
    flow_xy: Tensor,
    mask: Tensor,
    config: CompandConfig,
) -> Tensor:
    high = _bhwc_to_bchw(corrected_high_base)
    anchor = _bhwc_to_bchw(anchor_base)
    mask_b = _ensure_mask_bhw(mask).unsqueeze(1)
    smooth_mask = signedish_mask_falloff(mask, blur_radius=max(config.edge_softness, 1)).unsqueeze(1)

    high_ok = rgb_to_oklab(high)
    anchor_ok = rgb_to_oklab(anchor)

    Lh, ah, bh = high_ok[:, 0:1], high_ok[:, 1:2], high_ok[:, 2:3]
    La, aa, ba = anchor_ok[:, 0:1], anchor_ok[:, 1:2], anchor_ok[:, 2:3]

    Ch = torch.sqrt(ah * ah + bh * bh + 1e-8)
    Ca = torch.sqrt(aa * aa + ba * ba + 1e-8)

    divergence_proxy = torch.sqrt(flow_xy[..., 0] ** 2 + flow_xy[..., 1] ** 2).unsqueeze(1)
    divergence_proxy = divergence_proxy / divergence_proxy.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)

    contrast_gain = 1.0 + config.contrast_restore * divergence_proxy
    chroma_gain = 1.0 + config.chroma_restore * divergence_proxy

    mean_anchor = (La * mask_b).sum(dim=(2, 3), keepdim=True) / mask_b.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    Lcorr = mean_anchor + (Lh - mean_anchor) * contrast_gain
    Ccorr = Ch * chroma_gain

    hue_x = ah / Ch.clamp_min(1e-5)
    hue_y = bh / Ch.clamp_min(1e-5)
    acorr = hue_x * Ccorr
    bcorr = hue_y * Ccorr

    corrected = torch.cat([Lcorr, acorr, bcorr], dim=1)
    mixed = high_ok * (1.0 - smooth_mask) + (corrected * (1.0 - config.lowfreq_anchor_mix) + anchor_ok * config.lowfreq_anchor_mix) * smooth_mask
    rgb = oklab_to_rgb(mixed)
    return _bchw_to_bhwc(rgb)


def make_flow_visual(flow_xy: Tensor) -> Tensor:
    mag = torch.sqrt(flow_xy[..., 0] ** 2 + flow_xy[..., 1] ** 2)
    ang = torch.atan2(flow_xy[..., 1], flow_xy[..., 0])
    hue = (ang / (2.0 * math.pi) + 0.5) % 1.0
    sat = torch.ones_like(hue)
    val = (mag / mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)).clamp(0.0, 1.0)

    h6 = hue * 6.0
    i = torch.floor(h6).to(torch.int64)
    f = h6 - i
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)

    i_mod = i % 6
    r = torch.where(i_mod == 0, val, torch.where(i_mod == 1, q, torch.where(i_mod == 2, p, torch.where(i_mod == 3, p, torch.where(i_mod == 4, t, val)))))
    g = torch.where(i_mod == 0, t, torch.where(i_mod == 1, val, torch.where(i_mod == 2, val, torch.where(i_mod == 3, q, torch.where(i_mod == 4, p, p)))))
    b = torch.where(i_mod == 0, p, torch.where(i_mod == 1, p, torch.where(i_mod == 2, t, torch.where(i_mod == 3, val, torch.where(i_mod == 4, val, q)))))
    return torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)


def stable_manifold_compand(
    high_image: Tensor,
    mask: Tensor,
    anchor_image: Optional[Tensor],
    config: CompandConfig,
) -> Tuple[Tensor, CompandDebug]:
    high_image = _ensure_bhwc(high_image)[..., :3]
    batch, height, width, _ = high_image.shape
    mask = _match_batch(_ensure_mask_bhw(mask), batch).to(device=high_image.device, dtype=high_image.dtype)

    if anchor_image is None:
        anchor_h, anchor_w = build_anchor_size(height, width, config.anchor_mp)
        anchor_small = resize_bhwc(high_image, anchor_h, anchor_w, mode="bicubic")
        anchor_image = resize_bhwc(anchor_small, height, width, mode="bicubic")
    else:
        anchor_image = resize_bhwc(_match_batch(_ensure_bhwc(anchor_image)[..., :3], batch), height, width, mode="bicubic")

    anchor_base, _ = build_low_high_layers(anchor_image, blur_radius=config.base_blur_radius)
    high_base, high_high = build_low_high_layers(high_image, blur_radius=config.base_blur_radius)

    expansion_ratio, metrics = estimate_expansion_ratio(anchor_base, high_base, mask)
    flow_xy = estimate_radial_flow(mask, expansion_ratio, config)
    warped_high = warp_image_bhwc(high_image, flow_xy, mode="bicubic")
    warped_high_base, warped_high_high = build_low_high_layers(warped_high, blur_radius=config.base_blur_radius)

    restored_base = restore_low_frequency_color(warped_high_base, anchor_base, flow_xy, mask, config)
    detail = warped_high_high * config.detail_preservation
    corrected = (restored_base + detail).clamp(0.0, 1.0)

    smooth_mask = signedish_mask_falloff(mask, blur_radius=max(config.edge_softness, 1)).unsqueeze(-1)
    output = (high_image * (1.0 - smooth_mask) + corrected * smooth_mask).clamp(0.0, 1.0)

    flow_visual = make_flow_visual(flow_xy)
    debug = CompandDebug(
        anchor_image=anchor_image,
        corrected_base=restored_base,
        warped_high_image=warped_high,
        flow_visual=flow_visual,
        debug_mask=smooth_mask,
        metrics=metrics,
    )
    return output, debug
