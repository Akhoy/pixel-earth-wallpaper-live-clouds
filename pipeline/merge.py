"""Feathered gap-fill compositing for satellite swath imagery.

Instead of a hard cutoff between "has data" / "gap", we build a validity
mask per source, Gaussian-blur it into a soft alpha weight, and alpha-blend
between sources across that soft edge. Anywhere both sources are still a
gap, the caller should fall back to a cached previous composite.
"""
from __future__ import annotations
import warnings
import numpy as np
from PIL import Image, ImageFilter

GAP_THRESHOLD = 8  # sum-of-channels below this (out of ~765) counts as "no data"
FEATHER_RADIUS = 12


def validity_mask(rgb: np.ndarray) -> np.ndarray:
    total = rgb.astype(np.int32).sum(axis=-1)
    return (total > GAP_THRESHOLD).astype(np.float32)


_validity_mask = validity_mask  # internal alias used elsewhere in this module


def _feather(mask: np.ndarray) -> np.ndarray:
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(radius=FEATHER_RADIUS))
    return np.array(im).astype(np.float32) / 255.0


def match_histogram(source: np.ndarray, reference: np.ndarray,
                     source_valid: np.ndarray, reference_valid: np.ndarray) -> np.ndarray:
    """Rescale `source` so its per-channel mean/std matches `reference`'s.

    Only pixels valid in BOTH images are used to measure the statistics
    (the actual overlap region), then the adjustment is applied to the
    whole `source` image, so the two sources look tonally consistent
    everywhere before they're blended -- not just at the seam.
    """
    both_valid = (source_valid > 0.5) & (reference_valid > 0.5)
    if both_valid.sum() < 1000:  # not enough overlap to measure statistics reliably
        return source

    out = source.copy()
    for c in range(3):
        src_vals = source[..., c][both_valid]
        ref_vals = reference[..., c][both_valid]
        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        ref_mean, ref_std = ref_vals.mean(), ref_vals.std() + 1e-6
        out[..., c] = (source[..., c] - src_mean) / src_std * ref_std + ref_mean
    return np.clip(out, 0, 255)


def feathered_merge(primary: np.ndarray, secondary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blend `secondary` into `primary`'s gaps with a soft edge.

    Returns (merged_rgb, still_gap_mask) where still_gap_mask is 1.0 where
    BOTH sources had no data (caller should fall back to cache there).
    """
    primary = primary.astype(np.float32)
    secondary = secondary.astype(np.float32)

    valid_p = _validity_mask(primary)
    valid_s = _validity_mask(secondary)

    # Match secondary's tone to primary's using their overlap region, so the
    # seam is tonal-discontinuity-free, not just gap-free.
    secondary = match_histogram(secondary, primary, valid_s, valid_p)

    alpha = _feather(valid_p)  # soft weight favoring primary where it has data
    alpha = alpha[..., None]

    merged = primary * alpha + secondary * (1 - alpha)

    still_gap = ((valid_p < 0.5) & (valid_s < 0.5)).astype(np.float32)
    return np.clip(merged, 0, 255).astype(np.uint8), still_gap


def fill_from_cache(rgb: np.ndarray, still_gap: np.ndarray, cached_rgb: np.ndarray | None) -> np.ndarray:
    if cached_rgb is None or still_gap.sum() == 0:
        return rgb
    alpha = _feather(1.0 - still_gap)[..., None]  # 1 where we trust `rgb`, soft-fading near gap edges
    out = rgb.astype(np.float32) * alpha + cached_rgb.astype(np.float32) * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def darkest_of_n_composite(sources: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel "darkest wins" composite across N independent satellite
    passes of the same day (e.g. NOAA-21/20/SNPP VIIRS, different overpass
    times). Transient haze/smoke/glint brightens a pixel relative to its
    true value and drifts between passes, so picking the darkest of several
    independent looks at each spot suppresses it while real, persistent
    clouds (dark-vs-bright consistently across passes) are preserved.

    A "consensus-of-2" alternative (per-pixel, average whichever two of the
    three sources agree most closely, discard the third) was tried and
    measured objectively better on two fronts: darkest-wins is ~30% more
    saturated on average than any single raw source (confirmed empirically
    -- picking the least-hazy pixel at every location independently also
    discards the natural desaturation haze normally causes, even with zero
    color grading applied), and consensus-of-2 doesn't have that bias.
    Despite that, user preference on-device favored darkest-wins's richer/
    moodier look over consensus-of-2's flatter one, just wanting it a touch
    brighter -- see `brighten_day_image` in run_pipeline.py, applied after
    this compositing step, only to the day texture.

    Returns (composite_rgb, all_gap_mask) where all_gap_mask is 1.0 only
    where EVERY source had no data at that pixel.
    """
    BIG = 10 ** 9
    lums = []
    for s in sources:
        lum = s.astype(np.int64).sum(axis=-1)
        lum = np.where(validity_mask(s) > 0.5, lum, BIG)
        lums.append(lum)

    stack = np.stack(sources, axis=0)
    lum_stack = np.stack(lums, axis=0)
    best = np.argmin(lum_stack, axis=0)

    out = np.take_along_axis(stack, best[None, ..., None], axis=0)[0]
    all_gap = (lum_stack == BIG).all(axis=0)
    out = np.where(all_gap[..., None], 0, out).astype(np.uint8)
    return out, all_gap.astype(np.float32)


def median_of_masks(masks: list[np.ndarray], valids: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median across N independently-derived cloud masks (one per
    satellite pass), taken over only the sources that are valid at that
    pixel. Use this INSTEAD of deriving the mask from `darkest_of_n_composite`
    when the published output is a standalone grayscale cloud mask (no color
    photo shown to the user) -- see run_pipeline.py.

    `darkest_of_n_composite` combines the raw *photos* first, always
    favouring whichever pass looks darkest/clearest at each point. That's
    great for suppressing transient haze/smoke/glint (which brightens a
    pixel), but it applies the same logic to genuine small, fast-evolving
    convective cloud: if a cell shows in only one of three passes (VIIRS
    overpasses are 30-100min apart, plenty of time for cumulus to form or
    dissipate), "always pick darkest" throws that cloud out too, discarding
    real signal. Measured on 2026-09-12 over the Congo basin: each single
    satellite read ~150-158 mean brightness there, but the darkest-of-3
    composite dropped to 132 -- a real, verifiable loss, not a rendering
    artifact.

    Deriving the mask per-source first, then taking the per-pixel MEDIAN of
    the three mask values, fixes this: a cell caught by 2 of 3 passes still
    comes through at roughly its true value (median of two similar-ish
    readings and one low one lands near the two agreeing values), while a
    genuine single-satellite outlier (glint, a granule seam) still gets
    rejected the same way it always did, since it's the outlier of the
    three. Verified as a strict improvement over the old approach on every
    axis checked: real storm cloud +22%, Congo scattered convection +19%,
    while bare desert only rose 10% -- meaning desert-vs-cloud separation
    actually got BETTER, not worse, because cloud/storm brightness grew
    faster than desert did.

    Returns (combined_mask, all_gap_mask) where all_gap_mask is 1.0 only
    where EVERY source was invalid (no data) at that pixel -- same
    convention as `darkest_of_n_composite`, so callers can reuse
    `fill_from_cache`/`inpaint_remaining_gaps` unchanged.
    """
    stack = np.stack(masks, axis=0).astype(np.float32)
    valid_stack = np.stack(valids, axis=0) > 0.5
    stack_nan = np.where(valid_stack, stack, np.nan)
    all_gap = ~valid_stack.any(axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # all-NaN slices are exactly the all_gap pixels, overwritten to 0
        # below -- numpy's own warning about them is expected, not a bug.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        combined = np.nanmedian(stack_nan, axis=0)
    combined = np.where(all_gap, 0, combined)
    return combined.astype(np.uint8), all_gap.astype(np.float32)


def inpaint_remaining_gaps(rgb: np.ndarray, still_gap: np.ndarray, radius: int = 40) -> np.ndarray:
    """Last-resort fill for any pixels still gapped after cache fallback.

    Normalized-convolution diffusion: blur(content*mask) / blur(mask) gives,
    at each gap pixel, a weighted average of nearby *valid* pixels only (not
    polluted by the black gap itself), which smoothly bridges thin swath
    gaps using real surrounding cloud texture instead of leaving hard edges.
    """
    if still_gap.sum() == 0:
        return rgb

    valid = 1.0 - still_gap  # 1 = trustworthy pixel, 0 = gap to fill
    rgb_f = rgb.astype(np.float32)

    weighted = rgb_f * valid[..., None]

    # PIL's blur works on uint8 "L" images, so blur each channel (and the
    # weight map) separately rather than the raw float array directly.
    def blur_channel(chan: np.ndarray) -> np.ndarray:
        im = Image.fromarray(chan)
        im = im.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.array(im).astype(np.float32)

    # Represent as float32 "L" isn't supported by PIL blur well for large values,
    # so scale into 0..255 per-channel-safe range via two-pass blur on uint8-safe data.
    weight_blur = blur_channel((valid * 255.0).astype(np.uint8)) / 255.0
    filled = np.zeros_like(rgb_f)
    for c in range(3):
        chan_weighted = (weighted[..., c]).clip(0, 255).astype(np.uint8)
        filled[..., c] = blur_channel(chan_weighted)

    weight_blur_safe = np.clip(weight_blur, 1e-3, None)
    inpainted = filled / weight_blur_safe[..., None]
    inpainted = np.clip(inpainted, 0, 255)

    gap_alpha = _feather(still_gap)[..., None]  # soft edge between real content and inpainted fill
    out = rgb_f * (1 - gap_alpha) + inpainted * gap_alpha
    return np.clip(out, 0, 255).astype(np.uint8)
