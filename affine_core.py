
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _ensure_nhwc_image(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor image, got {type(image)!r}")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor with 3 or 4 dims, got shape {tuple(image.shape)}")
    channels = image.shape[-1]
    if channels not in (1, 3):
        raise ValueError(f"Expected IMAGE tensor in NHWC layout with 1 or 3 channels, got shape {tuple(image.shape)}")
    return image


def _image_to_nchw(image: torch.Tensor) -> torch.Tensor:
    return _ensure_nhwc_image(image).permute(0, 3, 1, 2).contiguous()


def _image_from_nchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape {tuple(image.shape)}")
    return image.permute(0, 2, 3, 1).contiguous()


def _expand_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    if x.shape[0] == batch:
        return x
    if x.shape[0] == 1:
        return x.expand(batch, *x.shape[1:]).clone()
    raise ValueError(f"Cannot broadcast batch dimension from {x.shape[0]} to {batch}")


def ensure_mask(mask: Optional[torch.Tensor], batch: int, height: int, width: int, device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch, 1, height, width), device=device, dtype=dtype)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        if mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)
        elif mask.shape[1] != 1:
            raise ValueError(f"Expected single-channel mask, got shape {tuple(mask.shape)}")
    else:
        raise ValueError(f"Unsupported mask rank {mask.ndim} for shape {tuple(mask.shape)}")

    mask = mask.to(device=device, dtype=dtype)
    mask = _expand_batch(mask, batch)
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


def resize_image_like(image: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    image = _ensure_nhwc_image(image)
    ref = _ensure_nhwc_image(ref)
    batch, height, width = ref.shape[0], ref.shape[1], ref.shape[2]
    x = _image_to_nchw(image).to(device=ref.device, dtype=ref.dtype)
    x = _expand_batch(x, batch)
    if x.shape[-2:] != (height, width):
        x = F.interpolate(
            x,
            size=(height, width),
            mode=mode,
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
        )
    return _image_from_nchw(x)


def _gaussian_kernel_1d(radius: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(radius))
    sigma = max(float(sigma), 1e-4)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


def gaussian_blur_nchw(x: torch.Tensor, sigma: float, radius: Optional[int] = None) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    if radius is None:
        radius = max(1, int(math.ceil(sigma * 3.0)))
    kernel = _gaussian_kernel_1d(radius, sigma, x.device, x.dtype)
    channels = x.shape[1]
    kernel_x = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel_x, groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel_y, groups=channels)
    return x


def gaussian_blur_image(image: torch.Tensor, sigma: float, radius: Optional[int] = None) -> torch.Tensor:
    return _image_from_nchw(gaussian_blur_nchw(_image_to_nchw(image), sigma, radius))


def frequency_split(image: torch.Tensor, sigma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    low = gaussian_blur_image(image, sigma=sigma)
    return low, _ensure_nhwc_image(image) - low


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def rgb_to_oklab(image: torch.Tensor, assume_srgb: bool = True) -> torch.Tensor:
    image = _ensure_nhwc_image(image)
    if image.shape[-1] != 3:
        raise ValueError(f"rgb_to_oklab expects 3 channels, got shape {tuple(image.shape)}")
    rgb = srgb_to_linear(image) if assume_srgb else image.clamp(0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = torch.sign(l) * torch.abs(l).clamp_min(1e-12).pow(1.0 / 3.0)
    m_ = torch.sign(m) * torch.abs(m).clamp_min(1e-12).pow(1.0 / 3.0)
    s_ = torch.sign(s) * torch.abs(s).clamp_min(1e-12).pow(1.0 / 3.0)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return torch.stack([L, A, B], dim=-1)


def oklab_to_rgb(image: torch.Tensor, assume_srgb: bool = True) -> torch.Tensor:
    image = _ensure_nhwc_image(image)
    if image.shape[-1] != 3:
        raise ValueError(f"oklab_to_rgb expects 3 channels, got shape {tuple(image.shape)}")
    L, A, B = image[..., 0], image[..., 1], image[..., 2]
    l_ = L + 0.3963377774 * A + 0.2158037573 * B
    m_ = L - 0.1055613458 * A - 0.0638541728 * B
    s_ = L - 0.0894841775 * A - 1.2914855480 * B
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_
    r = (+4.0767416621 * l) + (-3.3077115913 * m) + (0.2309699292 * s)
    g = (-1.2684380046 * l) + (+2.6097574011 * m) + (-0.3413193965 * s)
    b = (-0.0041960863 * l) + (-0.7034186147 * m) + (+1.7076147010 * s)
    rgb_linear = torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)
    return linear_to_srgb(rgb_linear) if assume_srgb else rgb_linear


def rgb_to_luma(image: torch.Tensor) -> torch.Tensor:
    image = _ensure_nhwc_image(image)
    if image.shape[-1] != 3:
        raise ValueError(f"rgb_to_luma expects 3 channels, got shape {tuple(image.shape)}")
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(-1)


def make_interior_weight(mask_nchw: torch.Tensor, feather_px: float, power: float) -> torch.Tensor:
    if feather_px <= 0.0:
        return mask_nchw.clamp(0.0, 1.0)
    sigma = max(feather_px * 0.35, 0.5)
    blurred = gaussian_blur_nchw(mask_nchw, sigma=sigma)
    inner = (mask_nchw - (1.0 - blurred)).clamp(0.0, 1.0)
    return inner.pow(max(power, 1e-4)).clamp(0.0, 1.0)


def make_boundary_ring(mask_nchw: torch.Tensor, inner_weight: torch.Tensor) -> torch.Tensor:
    ring = (mask_nchw - inner_weight).clamp(0.0, 1.0)
    denom = ring.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return ring / denom


def coordinate_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def sobel_gradient_magnitude(luma: torch.Tensor) -> torch.Tensor:
    device, dtype = luma.device, luma.dtype
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    padded = F.pad(luma, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(padded, kx)
    gy = F.conv2d(padded, ky)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def masked_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x * w).sum(dim=(-2, -1), keepdim=True) / w.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def masked_std(x: torch.Tensor, w: torch.Tensor, mean: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mean is None:
        mean = masked_mean(x, w)
    var = ((x - mean).pow(2) * w).sum(dim=(-2, -1), keepdim=True) / w.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return torch.sqrt(var.clamp_min(1e-8))


def build_structure_feature(image_low: torch.Tensor, mask_nchw: torch.Tensor,
                            grad_weight: float = 0.65, dev_weight: float = 0.35) -> torch.Tensor:
    luma_nchw = _image_to_nchw(rgb_to_luma(image_low))
    grad = sobel_gradient_magnitude(luma_nchw)
    mean = masked_mean(luma_nchw, mask_nchw)
    dev = (luma_nchw - mean).abs()
    feat = mask_nchw * (grad_weight * grad + dev_weight * dev)
    return gaussian_blur_nchw(feat, sigma=1.0) + mask_nchw * 1e-6


def _weighted_centroid_and_cov(feature: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = feature.clamp_min(1e-9)
    wsum = w.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    mean = (coords * w).sum(dim=(-2, -1), keepdim=True) / wsum
    centered = coords - mean
    xx = (centered[:, 0:1] * centered[:, 0:1] * w).sum(dim=(-2, -1), keepdim=True) / wsum
    xy = (centered[:, 0:1] * centered[:, 1:2] * w).sum(dim=(-2, -1), keepdim=True) / wsum
    yy = (centered[:, 1:2] * centered[:, 1:2] * w).sum(dim=(-2, -1), keepdim=True) / wsum
    cov = torch.cat([xx, xy, xy, yy], dim=1).view(-1, 2, 2)
    return mean.view(-1, 2), cov


def _sym_eigh(mat: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    sym = 0.5 * (mat + mat.transpose(-1, -2))
    vals, vecs = torch.linalg.eigh(sym)
    return vals.clamp_min(eps), vecs


def _mat_sqrt(vals: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
    return vecs @ torch.diag_embed(torch.sqrt(vals)) @ vecs.transpose(-1, -2)


def _mat_invsqrt(vals: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
    return vecs @ torch.diag_embed(torch.rsqrt(vals)) @ vecs.transpose(-1, -2)


def estimate_affine_compand(anchor_low: torch.Tensor, target_low: torch.Tensor, mask_nchw: torch.Tensor,
                            geometry_strength: float = 1.0, max_shrink: float = 0.05,
                            max_expand: float = 0.0, anisotropy: float = 0.2) -> Dict[str, torch.Tensor]:
    batch, height, width = target_low.shape[0], target_low.shape[1], target_low.shape[2]
    coords = coordinate_grid(batch, height, width, target_low.device, target_low.dtype)
    feat_a = build_structure_feature(anchor_low, mask_nchw)
    feat_t = build_structure_feature(target_low, mask_nchw)
    ca, cov_a = _weighted_centroid_and_cov(feat_a, coords)
    ct, cov_t = _weighted_centroid_and_cov(feat_t, coords)
    eye = torch.eye(2, device=target_low.device, dtype=target_low.dtype).unsqueeze(0).expand(batch, -1, -1)
    cov_a = cov_a + 1e-6 * eye
    cov_t = cov_t + 1e-6 * eye
    vals_a, vecs_a = _sym_eigh(cov_a)
    vals_t, vecs_t = _sym_eigh(cov_t)
    raw_A = _mat_sqrt(vals_a, vecs_a) @ _mat_invsqrt(vals_t, vecs_t)
    raw_vals, raw_vecs = _sym_eigh(raw_A)
    iso = torch.sqrt(raw_vals[:, 0] * raw_vals[:, 1]).unsqueeze(-1).expand(-1, 2)
    blended = iso * (1.0 - anisotropy) + raw_vals * anisotropy
    if max_expand <= 0.0:
        blended = torch.minimum(blended, torch.ones_like(blended))
    else:
        blended = blended.clamp(max=1.0 + max_expand)
    blended = blended.clamp(min=1.0 - max_shrink)
    scales = 1.0 + geometry_strength * (blended - 1.0)
    scales = scales.clamp(min=1.0 - max_shrink, max=1.0 + max_expand)
    A = raw_vecs @ torch.diag_embed(scales) @ raw_vecs.transpose(-1, -2)
    det = torch.linalg.det(A).clamp_min(1e-8)
    return {
        "centroid_anchor": ca,
        "centroid_target": ct,
        "A": A,
        "scale_vals": scales,
        "det": det,
        "compaction": 1.0 / det,
    }


def apply_companding_warp(image: torch.Tensor, affine: Dict[str, torch.Tensor], inner_weight: torch.Tensor,
                          translate_strength: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    image = _ensure_nhwc_image(image)
    batch, height, width = image.shape[0], image.shape[1], image.shape[2]
    coords = coordinate_grid(batch, height, width, image.device, image.dtype)
    ca = affine["centroid_anchor"].view(batch, 2, 1, 1)
    ct = affine["centroid_target"].view(batch, 2, 1, 1)
    centered = coords - ct
    transformed = torch.einsum("nij,njhw->nihw", affine["A"], centered)
    translated = ca + transformed
    translated = coords + translate_strength * (translated - coords)
    final_coords = coords + inner_weight * (translated - coords)
    grid = final_coords.permute(0, 2, 3, 1)
    warped = F.grid_sample(_image_to_nchw(image), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return _image_from_nchw(warped), final_coords


def restore_low_frequency_color(anchor_low: torch.Tensor, warped_target_low: torch.Tensor, mask_nchw: torch.Tensor,
                                compaction: torch.Tensor, mean_strength: float = 1.0, contrast_strength: float = 0.85,
                                chroma_strength: float = 0.95, anchor_pull: float = 0.18,
                                compaction_to_chroma: float = 0.22, assume_srgb: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    anchor_ok = _image_to_nchw(rgb_to_oklab(anchor_low, assume_srgb=assume_srgb))
    target_ok = _image_to_nchw(rgb_to_oklab(warped_target_low, assume_srgb=assume_srgb))
    mean_a = masked_mean(anchor_ok, mask_nchw)
    mean_t = masked_mean(target_ok, mask_nchw)
    std_a = masked_std(anchor_ok, mask_nchw, mean_a)
    std_t = masked_std(target_ok, mask_nchw, mean_t)
    ratio = (std_a / std_t.clamp_min(1e-4)).clamp(0.70, 1.45)
    compaction = compaction.view(-1, 1, 1, 1)
    chroma_boost = (1.0 + compaction_to_chroma * (compaction - 1.0)).clamp(0.95, 1.35)
    L_t = target_ok[:, 0:1]
    AB_t = target_ok[:, 1:3]
    L_restored = mean_t[:, 0:1] + mean_strength * (mean_a[:, 0:1] - mean_t[:, 0:1]) + (L_t - mean_t[:, 0:1]) * (1.0 + contrast_strength * (ratio[:, 0:1] - 1.0))
    AB_restored = mean_t[:, 1:3] + mean_strength * (mean_a[:, 1:3] - mean_t[:, 1:3]) + (AB_t - mean_t[:, 1:3]) * (1.0 + chroma_strength * ((ratio[:, 1:3] * chroma_boost) - 1.0))
    restored = torch.cat([L_restored, AB_restored], dim=1)
    if anchor_pull > 0.0:
        restored = restored * (1.0 - anchor_pull) + anchor_ok * anchor_pull
    restored_rgb = oklab_to_rgb(_image_from_nchw(restored), assume_srgb=assume_srgb)
    return restored_rgb, {"std_ratio": ratio, "chroma_boost": chroma_boost}


DEFAULT_PROFILE: Dict[str, Any] = {
    "stable_sigma": 8.0,
    "detail_sigma": 8.0,
    "boundary_feather_px": 18.0,
    "interior_power": 1.35,
    "geometry_strength": 1.0,
    "translate_strength": 1.0,
    "max_shrink": 0.05,
    "max_expand": 0.0,
    "anisotropy": 0.20,
    "mean_strength": 1.0,
    "contrast_strength": 0.85,
    "chroma_strength": 0.95,
    "anchor_pull": 0.18,
    "compaction_to_chroma": 0.22,
    "detail_retain": 1.0,
    "mask_global_strength": 1.0,
    "boundary_tether": 0.35,
    "assume_srgb": True,
    "blend_back_to_target": True,
}


@dataclass
class CompandResult:
    corrected: torch.Tensor
    low_corrected: torch.Tensor
    warped_target: torch.Tensor
    diagnostics: torch.Tensor
    field: torch.Tensor
    stats: Dict[str, Any]


def merge_profile(overrides: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    out = dict(DEFAULT_PROFILE)
    if overrides:
        out.update(overrides)
    out.update({k: v for k, v in kwargs.items() if v is not None})
    return out


def _scalar_or_default(x: Any, default: float) -> float:
    if x is None:
        return float(default)
    if isinstance(x, torch.Tensor):
        return float(x.detach().flatten()[0].item())
    return float(x)


def _make_diagnostics(anchor: torch.Tensor, target: torch.Tensor, corrected: torch.Tensor, mask_nchw: torch.Tensor,
                      field: torch.Tensor) -> torch.Tensor:
    batch, height, width = target.shape[0], target.shape[1], target.shape[2]
    mask_img = _image_from_nchw(mask_nchw.expand(batch, 3, height, width))
    delta = (field - coordinate_grid(batch, height, width, target.device, target.dtype)).pow(2).sum(dim=1, keepdim=True).sqrt()
    delta = delta / delta.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    heat = _image_from_nchw(torch.cat([delta, torch.zeros_like(delta), 1.0 - delta], dim=1))
    return torch.cat([anchor, target, corrected, (0.7 * heat + 0.3 * mask_img).clamp(0.0, 1.0)], dim=2)


def _jsonable_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            out[key] = float(value.item()) if value.numel() == 1 else value.tolist()
        elif isinstance(value, (float, int, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def stats_to_json(stats: Dict[str, Any]) -> str:
    return json.dumps(_jsonable_stats(stats), indent=2, sort_keys=True)


def _stable_manifold_compand_impl(anchor_image: torch.Tensor, target_image: torch.Tensor, mask: Optional[torch.Tensor] = None,
                                  original_image: Optional[torch.Tensor] = None, profile: Optional[Dict[str, Any]] = None) -> CompandResult:
    cfg = merge_profile(profile)
    target = _ensure_nhwc_image(target_image).to(dtype=torch.float32)
    anchor = resize_image_like(anchor_image, target)
    original = resize_image_like(original_image, target) if original_image is not None else None
    batch, height, width = target.shape[0], target.shape[1], target.shape[2]
    mask_nchw = ensure_mask(mask, batch, height, width, target.device, target.dtype)
    mask_nchw = (mask_nchw * float(cfg["mask_global_strength"])).clamp(0.0, 1.0)
    inner_weight = make_interior_weight(mask_nchw, float(cfg["boundary_feather_px"]), float(cfg["interior_power"]))
    boundary_ring = make_boundary_ring(mask_nchw, inner_weight)
    stable_sigma = _scalar_or_default(cfg.get("stable_sigma"), DEFAULT_PROFILE["stable_sigma"])
    detail_sigma = _scalar_or_default(cfg.get("detail_sigma"), DEFAULT_PROFILE["detail_sigma"])
    anchor_low = gaussian_blur_image(anchor, sigma=stable_sigma)
    target_low = gaussian_blur_image(target, sigma=stable_sigma)
    affine = estimate_affine_compand(anchor_low, target_low, mask_nchw, float(cfg["geometry_strength"]), float(cfg["max_shrink"]), float(cfg["max_expand"]), float(cfg["anisotropy"]))
    warped_target, field = apply_companding_warp(target, affine, inner_weight, float(cfg["translate_strength"]))
    warped_low = gaussian_blur_image(warped_target, sigma=stable_sigma)
    low_corrected, color_stats = restore_low_frequency_color(anchor_low, warped_low, mask_nchw, affine["compaction"], float(cfg["mean_strength"]), float(cfg["contrast_strength"]), float(cfg["chroma_strength"]), float(cfg["anchor_pull"]), float(cfg["compaction_to_chroma"]), bool(cfg["assume_srgb"]))
    if original is not None and float(cfg["boundary_tether"]) > 0.0:
        original_low = gaussian_blur_image(original, sigma=stable_sigma)
        tether = (boundary_ring * float(cfg["boundary_tether"])).permute(0, 2, 3, 1)
        low_corrected = original_low * tether + low_corrected * (1.0 - tether)
    warped_low_for_detail = gaussian_blur_image(warped_target, sigma=detail_sigma)
    corrected = (low_corrected + (warped_target - warped_low_for_detail) * float(cfg["detail_retain"])).clamp(0.0, 1.0)
    if bool(cfg.get("blend_back_to_target", True)):
        corrected = target * (1.0 - mask_nchw.permute(0, 2, 3, 1)) + corrected * mask_nchw.permute(0, 2, 3, 1)
    diagnostics = _make_diagnostics(anchor, target, corrected, mask_nchw, field)
    stats = {
        "scale_x": affine["scale_vals"][:, 0],
        "scale_y": affine["scale_vals"][:, 1],
        "scale_iso": torch.sqrt(affine["scale_vals"][:, 0] * affine["scale_vals"][:, 1]),
        "compaction": affine["compaction"],
        "det": affine["det"],
        "centroid_anchor": affine["centroid_anchor"],
        "centroid_target": affine["centroid_target"],
        "translation_pixels_x": (affine["centroid_anchor"][:, 0] - affine["centroid_target"][:, 0]) * (width / 2.0),
        "translation_pixels_y": (affine["centroid_anchor"][:, 1] - affine["centroid_target"][:, 1]) * (height / 2.0),
        "luminance_std_ratio": color_stats["std_ratio"][:, 0, 0, 0],
        "chroma_std_ratio_a": color_stats["std_ratio"][:, 1, 0, 0],
        "chroma_std_ratio_b": color_stats["std_ratio"][:, 2, 0, 0],
        "chroma_boost": color_stats["chroma_boost"][:, 0, 0, 0],
    }
    return CompandResult(corrected, low_corrected, warped_target, diagnostics, field, stats)


def stable_manifold_compand(anchor_image: torch.Tensor, target_image: torch.Tensor, mask: Optional[torch.Tensor] = None,
                            original_image: Optional[torch.Tensor] = None, profile: Optional[Dict[str, Any]] = None) -> CompandResult:
    return _stable_manifold_compand_impl(anchor_image, target_image, mask=mask, original_image=original_image, profile=profile)


def make_profile_dict(**kwargs: Any) -> Dict[str, Any]:
    return merge_profile(kwargs)


def megapixels_from_hw(height: int, width: int) -> float:
    return (height * width) / 1_000_000.0


def resize_to_megapixels(image: torch.Tensor, target_megapixels: float, multiple_of: int = 16,
                         mode: str = "bilinear") -> Tuple[torch.Tensor, int, int, float]:
    image = _ensure_nhwc_image(image)
    height, width = image.shape[1], image.shape[2]
    scale = math.sqrt(max(target_megapixels, 1e-6) / max(megapixels_from_hw(height, width), 1e-8))
    new_h = max(multiple_of, int(round(height * scale / multiple_of) * multiple_of))
    new_w = max(multiple_of, int(round(width * scale / multiple_of) * multiple_of))
    ref = torch.zeros((image.shape[0], new_h, new_w, image.shape[-1]), device=image.device, dtype=image.dtype)
    resized = resize_image_like(image, ref, mode=mode)
    return resized, new_w, new_h, megapixels_from_hw(new_h, new_w)
