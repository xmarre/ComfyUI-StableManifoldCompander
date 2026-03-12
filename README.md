# ComfyUI Stable Manifold Compander

Stable Manifold Compander is a ComfyUI custom node pack for correcting the high-resolution drift that can show up in FLUX.2-style workflows, especially with detail crops, masked edits, and larger full-frame renders.

Instead of only resizing the image, the pack tries to pull the high-resolution result back toward a lower-resolution reference state that is usually compositionally more stable. In practice this helps counter issues such as:

- outward zoom or expansion drift
- low-frequency washout or relighting
- reduced color richness or chroma flattening
- crop-to-image mismatch after compositing a detailed region back into the source image

The pack includes a primary anchor-guided workflow for general use, an affine diagnostic/fallback workflow, and an optional Impact Pack hook for in-detailer correction.

## What it does

The core idea is simple:

1. derive or provide a lower-resolution anchor that represents the more stable version of the image or crop
2. compare the high-resolution result against that anchor
3. apply a controlled correction that pulls geometry and low-frequency structure inward without throwing away high-frequency detail

This makes the pack useful for both whole-image correction and crop-based detailing workflows.

## Included nodes

### Main workflow
- **SMC Config**
- **SMC Make Anchor**
- **SMC Anchor Resolution**
- **SMC Anchor-Guided Compand**
- **SMC Extract Masked Crop**
- **SMC Composite Crop**
- **SMC Frequency Split**
- **SMC Compand Blend**
- **SMC Describe Config**

### Affine / diagnostic workflow
- **SMC Affine Profile**
- **SMC Affine Anchor Resize**
- **SMC Affine Estimate Drift**
- **SMC Affine Compand**
- **SMC Affine Low / High Split**
- **SMC Affine Recombine**

### Optional Impact Pack integration
- **SMC Self-Anchor Detailer Hook Provider**

## Recommended workflows

### 1. Masked crop detailing
This is the most useful workflow when a face, subject, or region drifts after high-resolution detailing.

1. Create a mask for the region you want to process.
2. Use **SMC Extract Masked Crop** to pull out the image region and crop mask.
3. Run your normal upscale, detail, or edit pass on that crop.
4. Use **SMC Anchor-Guided Compand** on the edited crop.
   - Supply an explicit anchor if you already have a low-resolution reference.
   - Otherwise let the node derive one from the configured anchor megapixel target.
5. Paste the corrected crop back with **SMC Composite Crop**.

This workflow is usually the best fit for reducing crop expansion, relighting drift, and paste mismatch.

### 2. Whole-image correction
Use this when the full image drifts as resolution rises.

1. Build a mask for the affected subject or area, or use a broad mask for the whole image.
2. Create settings with **SMC Config**.
3. Run **SMC Anchor-Guided Compand** on the high-resolution image.
4. Use **SMC Compand Blend** if you want a softer or partial blend back to the original.

### 3. Diagnostic or fallback correction
Use the affine nodes when you want a more explicit and interpretable correction path.

The affine workflow is useful for:
- estimating scale and translation drift directly
- testing correction strength in a more controlled way
- older experiments where you want a clearer diagnostic readout
- the optional Impact Pack self-anchor hook

## Node overview

### SMC Config
Builds the main correction configuration used by the anchor-guided path.

Important controls:
- **anchor_mp**: target size of the internal low-resolution anchor
- **warp_strength**: overall strength of the inward correction
- **radial_strength**: how strongly the correction behaves like inward compaction
- **lowfreq_anchor_mix**: how much low-frequency structure is pulled toward the anchor
- **chroma_restore** / **contrast_restore**: restore color and tonal presence after correction
- **detail_preservation**: keeps high-frequency detail from collapsing
- **max_inward_shift_px**: safety cap on geometric correction magnitude

### SMC Make Anchor
Creates a low-resolution anchor and also returns a resized version at the original resolution for inspection.

### SMC Anchor Resolution
Calculates the anchor resolution for a given megapixel target while preserving aspect ratio.

### SMC Anchor-Guided Compand
Main correction node. It takes a high-resolution image, a mask, and a config, then outputs:
- corrected image
- anchor image used internally
- warped intermediate
- flow visualization
- debug mask
- numeric metrics summary

### SMC Extract Masked Crop / SMC Composite Crop
Utility nodes for crop-based workflows. These make it easier to extract a masked region, process it externally, and composite it back cleanly.

### SMC Frequency Split / SMC Compand Blend
Inspection and finishing utilities. Use them to preview low/high components or blend the processed result back more gently.

### Affine nodes
These expose a more explicit correction path based on drift estimation and affine-style compaction. They are especially useful when you want to inspect the estimated scale and translation values directly.

### SMC Self-Anchor Detailer Hook Provider
Optional Impact Pack hook.

It stores the post-upscale crop as a self-anchor, then runs the affine compand path after decode. This is a practical fallback for FaceDetailer-style workflows when you want some correction inside the hook surface itself.

## Installation

Clone or copy this repository into:

`ComfyUI/custom_nodes/ComfyUI-Stable-Manifold-Compander`

Then restart ComfyUI.

## Requirements

No extra Python dependencies are required beyond a normal ComfyUI PyTorch installation.

For **SMC Self-Anchor Detailer Hook Provider**, you also need **ComfyUI-Impact-Pack** installed.

## Notes and limitations

- **SMC Extract Masked Crop** and **SMC Composite Crop** currently support batch size 1 only.
- The main anchor-guided method is training-free and heuristic. It is not a learned deformation model, optical flow solver, or landmark-based warp.
- The Impact hook is a practical fallback, not a replacement for a true two-pass low-resolution anchor branch.
- Results depend heavily on mask quality and on choosing an anchor megapixel range that matches the model's stable regime.

## When to use which path

Use the **anchor-guided path** when:
- you want the main intended workflow
- you are correcting FLUX.2 klein high-resolution drift directly
- you are working with masked crops or broad masked regions

Use the **affine path** when:
- you want direct drift estimates
- you need a more diagnostic or interpretable correction route
- you want the optional Impact Pack self-anchor hook

## Files

- `core.py` — anchor-guided compand implementation
- `affine_core.py` — affine/profile correction implementation
- `impact_hook.py` — Impact Pack self-anchor hook
- `nodes.py` — ComfyUI node registration and UI surface
- `__init__.py`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`
