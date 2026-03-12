from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .core import (
    CompandConfig,
    build_anchor_size,
    build_low_high_layers,
    resize_bhwc,
    resize_mask,
    stable_manifold_compand as anchor_guided_compand,
)
from .affine_core import (
    DEFAULT_PROFILE,
    _ensure_nhwc_image,
    ensure_mask,
    estimate_affine_compand,
    frequency_split,
    gaussian_blur_image,
    make_interior_weight,
    make_profile_dict,
    resize_image_like,
    resize_to_megapixels,
    stable_manifold_compand as affine_compand,
    stats_to_json,
)
from .impact_hook import SMCSelfAnchorDetailerHookProviderNode


PRIMARY_CATEGORY = "StableManifoldCompand"
AFFINE_CATEGORY = "StableManifoldCompand/Affine"
UTILITY_CATEGORY = "StableManifoldCompand/Utility"


# -----------------------------------------------------------------------------
# Shared crop helpers used by the newer anchor-guided path.
# -----------------------------------------------------------------------------

def _compute_mask_bbox(mask: torch.Tensor, threshold: float = 1e-4):
    mask = mask if mask.ndim == 3 else mask.unsqueeze(0)
    boxes = []
    for b in range(mask.shape[0]):
        ys, xs = torch.where(mask[b] > threshold)
        if ys.numel() == 0:
            boxes.append((0, 0, mask.shape[2], mask.shape[1]))
            continue
        y1 = int(ys.min().item())
        y2 = int(ys.max().item()) + 1
        x1 = int(xs.min().item())
        x2 = int(xs.max().item()) + 1
        boxes.append((x1, y1, x2, y2))
    return boxes


class SMCExtractMaskedCropNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {"default": 64, "min": 0, "max": 2048, "step": 1}),
                "round_to": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "SMC_REGION", "STRING")
    RETURN_NAMES = ("crop_image", "crop_mask", "region", "summary")
    FUNCTION = "extract"
    CATEGORY = PRIMARY_CATEGORY

    def extract(self, image, mask, padding, round_to):
        if image.shape[0] != 1:
            raise ValueError("SMC Extract Masked Crop currently supports batch size 1 only.")
        mask = resize_mask(mask, image.shape[1], image.shape[2])
        x1, y1, x2, y2 = _compute_mask_bbox(mask)[0]
        h, w = image.shape[1], image.shape[2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        cw = x2 - x1
        ch = y2 - y1
        if round_to > 1:
            target_w = max(round_to, int(round(cw / round_to) * round_to or round_to))
            target_h = max(round_to, int(round(ch / round_to) * round_to or round_to))
            x2 = min(w, x1 + target_w)
            y2 = min(h, y1 + target_h)
            cw = x2 - x1
            ch = y2 - y1
        crop = image[:, y1:y2, x1:x2, :3]
        crop_mask = mask[:, y1:y2, x1:x2]
        region = [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "src_h": h, "src_w": w, "batch_index": 0}]
        summary = f"({x1},{y1})-({x2},{y2}) {cw}x{ch}"
        return crop, crop_mask, region, summary


class SMCCompositeCropNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "crop_image": ("IMAGE",),
                "crop_mask": ("MASK",),
                "region": ("SMC_REGION",),
                "mask_blur_radius": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = PRIMARY_CATEGORY

    def composite(self, base_image, crop_image, crop_mask, region, mask_blur_radius):
        if base_image.shape[0] != 1 or crop_image.shape[0] != 1:
            raise ValueError("SMC Composite Crop currently supports batch size 1 only.")
        out = base_image[..., :3].clone()
        crop_mask = crop_mask if crop_mask.ndim == 3 else crop_mask.unsqueeze(0)
        reg = region[0]
        x1, y1, x2, y2 = int(reg["x1"]), int(reg["y1"]), int(reg["x2"]), int(reg["y2"])
        cm = crop_mask[:1]
        if mask_blur_radius > 0:
            k = mask_blur_radius * 2 + 1
            cm = torch.nn.functional.avg_pool2d(cm.unsqueeze(1), kernel_size=k, stride=1, padding=mask_blur_radius)[:, 0]
        alpha = cm.clamp(0.0, 1.0).unsqueeze(-1)
        out[:, y1:y2, x1:x2, :] = (
            out[:, y1:y2, x1:x2, :] * (1.0 - alpha)
            + crop_image[:, : y2 - y1, : x2 - x1, :3] * alpha
        )
        return (out.clamp(0.0, 1.0),)


# -----------------------------------------------------------------------------
# Newer anchor-guided path (primary workflow).
# -----------------------------------------------------------------------------

class SMCConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anchor_mp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "base_blur_radius": ("INT", {"default": 9, "min": 1, "max": 63, "step": 1}),
                "mask_falloff_radius": ("INT", {"default": 24, "min": 0, "max": 128, "step": 1}),
                "warp_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 3.0, "step": 0.01}),
                "radial_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "anisotropy_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lowfreq_anchor_mix": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chroma_restore": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "contrast_restore": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 2.0, "step": 0.01}),
                "detail_preservation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_inward_shift_px": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 64.0, "step": 0.1}),
                "edge_softness": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("SMC_CONFIG",)
    FUNCTION = "build"
    CATEGORY = PRIMARY_CATEGORY

    def build(self, **kwargs):
        config = CompandConfig(**kwargs)
        return (config.to_dict(),)


class SMCAnchorResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_mp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "round_to": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("anchor_height", "anchor_width", "summary")
    FUNCTION = "compute"
    CATEGORY = UTILITY_CATEGORY

    def compute(self, image, target_mp, round_to):
        _, h, w, _ = image.shape
        ah, aw = build_anchor_size(h, w, target_mp, round_to=round_to)
        summary = f"{aw}x{ah} (~{aw * ah / 1_000_000:.3f} MP)"
        return ah, aw, summary


class SMCMakeAnchorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_mp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "round_to": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "upsample_mode": (["bicubic", "bilinear", "nearest-exact"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("anchor_small", "anchor_resized", "summary")
    FUNCTION = "make"
    CATEGORY = PRIMARY_CATEGORY

    def make(self, image, target_mp, round_to, upsample_mode):
        _, h, w, _ = image.shape
        ah, aw = build_anchor_size(h, w, target_mp, round_to=round_to)
        anchor_small = resize_bhwc(image, ah, aw, mode="bicubic")
        anchor_resized = resize_bhwc(anchor_small, h, w, mode=upsample_mode)
        summary = f"anchor {aw}x{ah} (~{aw * ah / 1_000_000:.3f} MP)"
        return anchor_small, anchor_resized, summary


class SMCFrequencySplitNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {"default": 9, "min": 1, "max": 63, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("low_frequency", "high_frequency")
    FUNCTION = "split"
    CATEGORY = UTILITY_CATEGORY

    def split(self, image, blur_radius):
        low, high = build_low_high_layers(image[..., :3], blur_radius=blur_radius)
        high_vis = (high * 0.5 + 0.5).clamp(0.0, 1.0)
        return low, high_vis


class SMCCompandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_image": ("IMAGE",),
                "mask": ("MASK",),
                "config": ("SMC_CONFIG",),
            },
            "optional": {
                "anchor_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = (
        "output",
        "anchor_image",
        "warped_high_image",
        "flow_visual",
        "debug_mask",
        "metrics",
    )
    FUNCTION = "run"
    CATEGORY = PRIMARY_CATEGORY

    def run(self, high_image, mask, config, anchor_image=None):
        cfg = CompandConfig(**config)
        output, debug = anchor_guided_compand(high_image[..., :3], mask, anchor_image, cfg)
        metrics = ", ".join(f"{k}={v:.6f}" for k, v in debug.metrics.items())
        return output, debug.anchor_image, debug.warped_high_image, debug.flow_visual, debug.debug_mask[..., 0], metrics


class SMCCompandBlendNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "mask": ("MASK",),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_blur_radius": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = UTILITY_CATEGORY

    def blend(self, original_image, processed_image, mask, blend_strength, mask_blur_radius):
        mask = resize_mask(mask, original_image.shape[1], original_image.shape[2]).unsqueeze(-1)
        if mask_blur_radius > 0:
            k = mask_blur_radius * 2 + 1
            mask_bchw = mask.permute(0, 3, 1, 2)
            mask_bchw = torch.nn.functional.avg_pool2d(mask_bchw, kernel_size=k, stride=1, padding=mask_blur_radius)
            mask = mask_bchw.permute(0, 2, 3, 1)
        alpha = mask.clamp(0.0, 1.0) * blend_strength
        out = original_image[..., :3] * (1.0 - alpha) + processed_image[..., :3] * alpha
        return (out.clamp(0.0, 1.0),)


class SMCDescribeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": ("SMC_CONFIG",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe"
    CATEGORY = UTILITY_CATEGORY

    def describe(self, config):
        cfg = CompandConfig(**config)
        lines = [
            "Stable Manifold Compand configuration",
            f"anchor_mp={cfg.anchor_mp}",
            f"base_blur_radius={cfg.base_blur_radius}",
            f"mask_falloff_radius={cfg.mask_falloff_radius}",
            f"warp_strength={cfg.warp_strength}",
            f"radial_strength={cfg.radial_strength}",
            f"anisotropy_strength={cfg.anisotropy_strength}",
            f"lowfreq_anchor_mix={cfg.lowfreq_anchor_mix}",
            f"chroma_restore={cfg.chroma_restore}",
            f"contrast_restore={cfg.contrast_restore}",
            f"detail_preservation={cfg.detail_preservation}",
            f"max_inward_shift_px={cfg.max_inward_shift_px}",
            f"edge_softness={cfg.edge_softness}",
        ]
        return ("\n".join(lines),)


# -----------------------------------------------------------------------------
# Earlier affine/profile path retained as a secondary toolset.
# -----------------------------------------------------------------------------

class SMCAffineProfileNode:
    CATEGORY = AFFINE_CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("SMC_PROFILE", "STRING")
    RETURN_NAMES = ("profile", "profile_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stable_sigma": ("FLOAT", {"default": DEFAULT_PROFILE["stable_sigma"], "min": 0.0, "max": 64.0, "step": 0.1}),
                "detail_sigma": ("FLOAT", {"default": DEFAULT_PROFILE["detail_sigma"], "min": 0.0, "max": 64.0, "step": 0.1}),
                "boundary_feather_px": ("FLOAT", {"default": DEFAULT_PROFILE["boundary_feather_px"], "min": 0.0, "max": 256.0, "step": 0.5}),
                "interior_power": ("FLOAT", {"default": DEFAULT_PROFILE["interior_power"], "min": 0.1, "max": 8.0, "step": 0.05}),
                "geometry_strength": ("FLOAT", {"default": DEFAULT_PROFILE["geometry_strength"], "min": 0.0, "max": 2.0, "step": 0.01}),
                "translate_strength": ("FLOAT", {"default": DEFAULT_PROFILE["translate_strength"], "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_shrink": ("FLOAT", {"default": DEFAULT_PROFILE["max_shrink"], "min": 0.0, "max": 0.25, "step": 0.001}),
                "max_expand": ("FLOAT", {"default": DEFAULT_PROFILE["max_expand"], "min": 0.0, "max": 0.25, "step": 0.001}),
                "anisotropy": ("FLOAT", {"default": DEFAULT_PROFILE["anisotropy"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "mean_strength": ("FLOAT", {"default": DEFAULT_PROFILE["mean_strength"], "min": 0.0, "max": 2.0, "step": 0.01}),
                "contrast_strength": ("FLOAT", {"default": DEFAULT_PROFILE["contrast_strength"], "min": 0.0, "max": 2.0, "step": 0.01}),
                "chroma_strength": ("FLOAT", {"default": DEFAULT_PROFILE["chroma_strength"], "min": 0.0, "max": 2.0, "step": 0.01}),
                "anchor_pull": ("FLOAT", {"default": DEFAULT_PROFILE["anchor_pull"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "compaction_to_chroma": ("FLOAT", {"default": DEFAULT_PROFILE["compaction_to_chroma"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_retain": ("FLOAT", {"default": DEFAULT_PROFILE["detail_retain"], "min": 0.0, "max": 2.0, "step": 0.01}),
                "mask_global_strength": ("FLOAT", {"default": DEFAULT_PROFILE["mask_global_strength"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "boundary_tether": ("FLOAT", {"default": DEFAULT_PROFILE["boundary_tether"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "assume_srgb": ("BOOLEAN", {"default": True}),
                "blend_back_to_target": ("BOOLEAN", {"default": True}),
            }
        }

    def build(self, **kwargs: Any):
        profile = make_profile_dict(**kwargs)
        return profile, stats_to_json(profile)


class SMCAffineAnchorResizeNode:
    CATEGORY = AFFINE_CATEGORY
    FUNCTION = "resize"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("anchor_image", "width", "height", "megapixels")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 32.0, "step": 0.01}),
                "multiple_of": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "upscale_mode": (["bilinear", "bicubic"], {"default": "bilinear"}),
            }
        }

    def resize(self, image: torch.Tensor, target_megapixels: float, multiple_of: int, upscale_mode: str):
        return resize_to_megapixels(image, target_megapixels, multiple_of, upscale_mode)


class SMCAffineEstimateNode:
    CATEGORY = AFFINE_CATEGORY
    FUNCTION = "estimate"
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("scale_x", "scale_y", "scale_iso", "compaction", "translation_x_px", "translation_y_px", "stats_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anchor_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "stable_sigma": ("FLOAT", {"default": DEFAULT_PROFILE["stable_sigma"], "min": 0.0, "max": 64.0, "step": 0.1}),
                "boundary_feather_px": ("FLOAT", {"default": DEFAULT_PROFILE["boundary_feather_px"], "min": 0.0, "max": 256.0, "step": 0.5}),
                "geometry_strength": ("FLOAT", {"default": DEFAULT_PROFILE["geometry_strength"], "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_shrink": ("FLOAT", {"default": DEFAULT_PROFILE["max_shrink"], "min": 0.0, "max": 0.25, "step": 0.001}),
                "max_expand": ("FLOAT", {"default": DEFAULT_PROFILE["max_expand"], "min": 0.0, "max": 0.25, "step": 0.001}),
                "anisotropy": ("FLOAT", {"default": DEFAULT_PROFILE["anisotropy"], "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"mask": ("MASK",)},
        }

    def estimate(
        self,
        anchor_image: torch.Tensor,
        target_image: torch.Tensor,
        stable_sigma: float,
        boundary_feather_px: float,
        geometry_strength: float,
        max_shrink: float,
        max_expand: float,
        anisotropy: float,
        mask: Optional[torch.Tensor] = None,
    ):
        target = _ensure_nhwc_image(target_image).to(dtype=torch.float32)
        anchor = resize_image_like(anchor_image, target)
        batch, height, width = target.shape[0], target.shape[1], target.shape[2]
        mask_nchw = ensure_mask(mask, batch, height, width, target.device, target.dtype)
        inner = make_interior_weight(mask_nchw, boundary_feather_px, DEFAULT_PROFILE["interior_power"])
        stats = estimate_affine_compand(
            gaussian_blur_image(anchor, stable_sigma),
            gaussian_blur_image(target, stable_sigma),
            inner,
            geometry_strength,
            max_shrink,
            max_expand,
            anisotropy,
        )
        sx = float(stats["scale_vals"][0, 0].item())
        sy = float(stats["scale_vals"][0, 1].item())
        si = float(torch.sqrt(stats["scale_vals"][0, 0] * stats["scale_vals"][0, 1]).item())
        comp = float(stats["compaction"][0].item())
        tx = float(((stats["centroid_anchor"][0, 0] - stats["centroid_target"][0, 0]) * (width / 2.0)).item())
        ty = float(((stats["centroid_anchor"][0, 1] - stats["centroid_target"][0, 1]) * (height / 2.0)).item())
        return sx, sy, si, comp, tx, ty, stats_to_json({
            "scale_x": stats["scale_vals"][:, 0],
            "scale_y": stats["scale_vals"][:, 1],
            "scale_iso": torch.sqrt(stats["scale_vals"][:, 0] * stats["scale_vals"][:, 1]),
            "compaction": stats["compaction"],
            "translation_pixels_x": (stats["centroid_anchor"][:, 0] - stats["centroid_target"][:, 0]) * (width / 2.0),
            "translation_pixels_y": (stats["centroid_anchor"][:, 1] - stats["centroid_target"][:, 1]) * (height / 2.0),
        })


class SMCAffineCompandNode:
    CATEGORY = AFFINE_CATEGORY
    FUNCTION = "compand"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("corrected", "warped_target", "diagnostics", "stats_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"anchor_image": ("IMAGE",), "target_image": ("IMAGE",)},
            "optional": {"mask": ("MASK",), "original_image": ("IMAGE",), "profile": ("SMC_PROFILE",)},
        }

    def compand(
        self,
        anchor_image: torch.Tensor,
        target_image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        original_image: Optional[torch.Tensor] = None,
        profile: Optional[Dict[str, Any]] = None,
    ):
        result = affine_compand(
            anchor_image,
            target_image,
            mask=mask,
            original_image=original_image,
            profile=profile,
        )
        return result.corrected, result.warped_target, result.diagnostics, stats_to_json(result.stats)


class SMCAffineLowHighSplitNode:
    CATEGORY = AFFINE_CATEGORY
    FUNCTION = "split"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("low", "high")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigma": ("FLOAT", {"default": DEFAULT_PROFILE["stable_sigma"], "min": 0.0, "max": 64.0, "step": 0.1}),
            }
        }

    def split(self, image: torch.Tensor, sigma: float):
        return frequency_split(image, sigma)


class SMCAffineRecombineNode:
    CATEGORY = AFFINE_CATEGORY
    FUNCTION = "recombine"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low": ("IMAGE",),
                "high": ("IMAGE",),
                "high_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
            },
            "optional": {"mask": ("MASK",), "base_image": ("IMAGE",)},
        }

    def recombine(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        high_strength: float,
        mask: Optional[torch.Tensor] = None,
        base_image: Optional[torch.Tensor] = None,
    ):
        low = _ensure_nhwc_image(low).to(dtype=torch.float32)
        high = resize_image_like(high, low).to(dtype=torch.float32)
        image = (low + high * high_strength).clamp(0.0, 1.0)
        if mask is not None:
            mask_nchw = ensure_mask(mask, image.shape[0], image.shape[1], image.shape[2], image.device, image.dtype)
            base = low if base_image is None else resize_image_like(base_image, image).to(dtype=torch.float32)
            image = base * (1.0 - mask_nchw.permute(0, 2, 3, 1)) + image * mask_nchw.permute(0, 2, 3, 1)
        return (image,)


NODE_CLASS_MAPPINGS = {
    # Primary anchor-guided path.
    "SMCExtractMaskedCrop": SMCExtractMaskedCropNode,
    "SMCCompositeCrop": SMCCompositeCropNode,
    "SMCConfig": SMCConfigNode,
    "SMCAnchorResolution": SMCAnchorResolutionNode,
    "SMCMakeAnchor": SMCMakeAnchorNode,
    "SMCFrequencySplit": SMCFrequencySplitNode,
    "SMCCompand": SMCCompandNode,
    "SMCCompandBlend": SMCCompandBlendNode,
    "SMCDescribe": SMCDescribeNode,

    # Affine/profile toolset.
    "SMCAffineProfile": SMCAffineProfileNode,
    "SMCAffineAnchorResize": SMCAffineAnchorResizeNode,
    "SMCAffineEstimate": SMCAffineEstimateNode,
    "SMCAffineCompand": SMCAffineCompandNode,
    "SMCAffineLowHighSplit": SMCAffineLowHighSplitNode,
    "SMCAffineRecombine": SMCAffineRecombineNode,

    # Impact Pack integration.
    "SMCSelfAnchorDetailerHookProvider": SMCSelfAnchorDetailerHookProviderNode,

    # Backward-compat aliases from the earlier pack.
    "StableManifoldProfile": SMCAffineProfileNode,
    "StableManifoldAnchorResize": SMCAffineAnchorResizeNode,
    "StableManifoldEstimate": SMCAffineEstimateNode,
    "StableManifoldCompand": SMCAffineCompandNode,
    "StableManifoldLowHighSplit": SMCAffineLowHighSplitNode,
    "StableManifoldRecombine": SMCAffineRecombineNode,
    "StableManifoldSelfAnchorDetailerHookProvider": SMCSelfAnchorDetailerHookProviderNode,

    # Compatibility alias for the later pack's placeholder name; now routes to the functional self-anchor hook.
    "SMCDetailerHookProvider": SMCSelfAnchorDetailerHookProviderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Primary anchor-guided path.
    "SMCExtractMaskedCrop": "SMC Extract Masked Crop",
    "SMCCompositeCrop": "SMC Composite Crop",
    "SMCConfig": "SMC Config",
    "SMCAnchorResolution": "SMC Anchor Resolution",
    "SMCMakeAnchor": "SMC Make Anchor",
    "SMCFrequencySplit": "SMC Frequency Split",
    "SMCCompand": "SMC Anchor-Guided Compand",
    "SMCCompandBlend": "SMC Compand Blend",
    "SMCDescribe": "SMC Describe Config",

    # Affine/profile path.
    "SMCAffineProfile": "SMC Affine Profile",
    "SMCAffineAnchorResize": "SMC Affine Anchor Resize",
    "SMCAffineEstimate": "SMC Affine Estimate Drift",
    "SMCAffineCompand": "SMC Affine Compand",
    "SMCAffineLowHighSplit": "SMC Affine Low / High Split",
    "SMCAffineRecombine": "SMC Affine Recombine",

    # Impact.
    "SMCSelfAnchorDetailerHookProvider": "SMC Self-Anchor Detailer Hook Provider",

    # Backward-compat labels.
    "StableManifoldProfile": "Stable Manifold Profile",
    "StableManifoldAnchorResize": "Stable Manifold Anchor Resize",
    "StableManifoldEstimate": "Stable Manifold Estimate Drift",
    "StableManifoldCompand": "Stable Manifold Compand",
    "StableManifoldLowHighSplit": "Stable Manifold Low / High Split",
    "StableManifoldRecombine": "Stable Manifold Recombine",
    "StableManifoldSelfAnchorDetailerHookProvider": "Stable Manifold Self-Anchor Detailer Hook",
    "SMCDetailerHookProvider": "SMC Self-Anchor Detailer Hook Provider",
}
