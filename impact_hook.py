from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .affine_core import merge_profile, stable_manifold_compand

try:
    from impact.hooks import DetailerHook  # type: ignore
    IMPACT_AVAILABLE = True
except Exception:
    IMPACT_AVAILABLE = False

    class DetailerHook:  # type: ignore
        def post_upscale(self, pixels, mask=None):
            return pixels

        def post_decode(self, pixels):
            return pixels

        def post_paste(self, image):
            return image

        def cycle_latent(self, latent):
            return latent

        def post_detection(self, segs):
            return segs

        def get_custom_noise(self, seed, noise, is_touched):
            return noise, is_touched

        def get_custom_sampler(self):
            return None

        def get_skip_sampling(self):
            return False

        def should_retry_patch(self, patch):
            return False


class StableManifoldSelfAnchorDetailerHook(DetailerHook):
    """
    Practical fallback hook for Impact Pack.

    It stores the post-upscale crop as a self-anchor, then after decode runs the
    older affine/profile compand path against that stored anchor. This is less
    ideal than a true separate <=1 MP model-generated anchor pass, but it is the
    only version from the two candidate packs that actually performs a meaningful
    correction inside the available hook surface.
    """

    def __init__(self, profile: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.profile = merge_profile(profile)
        self._anchor_upscaled: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None

    def post_upscale(self, pixels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self._anchor_upscaled = pixels.detach().clone()
        self._mask = None if mask is None else mask.detach().clone()
        return pixels

    def post_decode(self, pixels: torch.Tensor):
        if self._anchor_upscaled is None:
            return pixels
        result = stable_manifold_compand(
            anchor_image=self._anchor_upscaled,
            target_image=pixels,
            mask=self._mask,
            original_image=self._anchor_upscaled,
            profile=self.profile,
        )
        return result.corrected


class SMCSelfAnchorDetailerHookProviderNode:
    CATEGORY = "StableManifoldCompand/Impact"
    FUNCTION = "build"
    RETURN_TYPES = ("DETAILER_HOOK",)
    RETURN_NAMES = ("detailer_hook",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"profile": ("SMC_PROFILE",)}}

    def build(self, profile: Optional[Dict[str, Any]] = None):
        if not IMPACT_AVAILABLE:
            raise RuntimeError(
                "SMC Self-Anchor Detailer Hook Provider requires ComfyUI-Impact-Pack. "
                "Install Impact Pack or use the standalone Stable Manifold Compand nodes instead."
            )
        return (StableManifoldSelfAnchorDetailerHook(profile=profile),)
