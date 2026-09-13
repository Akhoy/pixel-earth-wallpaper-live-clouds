"""Weighted blend of N simultaneous geostationary sources onto one global
equirect frame.

Unlike pipeline/merge.py's darkest-of-N (built for suppressing transient
haze/smoke/glint across multiple daily passes of the *same* orbital
sensor), these sources are one look each, taken at roughly the same
moment, from different vantage points. So instead of "darkest wins", every
source contributes proportionally to its per-pixel confidence weight
(reproject.py: highest at its own sub-satellite point, fading to 0 near
the edge of its usable view) -- overlap zones blend smoothly, tone-matched
first, rather than picking a winner.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from merge import match_histogram  # noqa: E402

# Weighted-average normalization divides by weight_sum, so if a stretch of
# the globe has only one source with any coverage at all, that source
# gets scaled up to full opacity there even if its actual confidence was
# nearly 0 (e.g. a satellite's extreme, heavily-distorted limb). Below
# this total-confidence floor, treat the pixel as a gap instead --
# feathered, not a hard cutoff, so it fades into cache-fill/inpainting
# rather than snapping off.
MIN_TRUSTED_WEIGHT = 0.12


def weighted_blend(entries: list[tuple[np.ndarray, np.ndarray]], reference_index: int = 0):
    """entries: list of (rgb uint8 HxWx3, weight float32 HxW), all same shape.

    Returns (merged_rgb uint8, still_gap float32) where still_gap is 1.0
    wherever every source had zero weight (caller should fall back to
    cache/inpainting there, same as the VIIRS pipeline).
    """
    ref_rgb, ref_weight = entries[reference_index]
    ref_rgb_f = ref_rgb.astype(np.float32)

    normalized = []
    for i, (rgb, weight) in enumerate(entries):
        rgb_f = rgb.astype(np.float32)
        if i == reference_index:
            normalized.append((rgb_f, weight))
            continue
        # Tone-match each source to the reference using their overlap
        # region, so a blended seam isn't also a brightness/color jump.
        matched = match_histogram(rgb_f, ref_rgb_f, weight, ref_weight)
        normalized.append((matched, weight))

    weight_stack = np.stack([w for _, w in normalized], axis=0)       # (N,H,W)
    rgb_stack = np.stack([r for r, _ in normalized], axis=0)          # (N,H,W,3)

    weight_sum = weight_stack.sum(axis=0)                             # (H,W)
    still_gap = np.clip((MIN_TRUSTED_WEIGHT - weight_sum) / MIN_TRUSTED_WEIGHT, 0.0, 1.0)
    safe_sum = np.clip(weight_sum, 1e-6, None)

    merged = (rgb_stack * weight_stack[..., None]).sum(axis=0) / safe_sum[..., None]
    # Where weight_sum is exactly 0 the numerator is 0 too (unreached
    # pixels are initialized black), so this is already black there --
    # left un-zeroed elsewhere so still_gap's partial (0,1) values can
    # feather smoothly into cache-fill/inpainting instead of a hard cut.
    return np.clip(merged, 0, 255).astype(np.uint8), still_gap
