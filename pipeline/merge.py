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


def find_polar_hole(still_gap: np.ndarray, min_lat: float = 60.0, radius: int = 20) -> np.ndarray:
    """Mask (1.0 = hole) of the large no-data caps over the polar-night / no-coverage poles.

    VIIRS images only daylit ground, so around the winter pole (and Antarctica's
    coverage edge) there is a big contiguous gap that nothing can fill honestly: cache/inpaint either leave
    it black or smear a fake ring over it. Callers should leave this region
    out of gap-filling and publish "no cloud" there, so the app's own
    dayMap (which has real polar data) shows through.

    Only blobs, not thin swath slivers (those still get the normal fill):
    a gap pixel counts as hole only if most of its neighbourhood is gap too.
    """
    h = still_gap.shape[0]
    lat = 90 - (np.arange(h) + 0.5) / h * 180
    in_cap = (np.abs(lat) >= min_lat)[:, None]
    solid = _feather_radius(still_gap, radius) > 0.5
    return (in_cap & solid & (still_gap > 0.5)).astype(np.float32)


def fade_to_hole(face: np.ndarray, hole_face: np.ndarray, radius: int = 40) -> np.ndarray:
    """Fade a cube face's cloud values to 0 at a hole's edge.

    `hole_face` is the hole mask reprojected onto the same face (0..1). The
    blurred mask is exactly 0.5 on the edge, so 1 - 2*blur is 0 there and
    rises to 1 over ~radius px outside it, giving clouds a soft end instead
    of a hard circle. Done per face (not on the equirect) because an equirect
    blur is much narrower east-west than north-south near a pole.
    """
    if hole_face.max() <= 0:
        return face
    w = 1.0 - np.clip(2.0 * _feather_radius(hole_face, radius), 0.0, 1.0)
    return np.clip(face.astype(np.float32) * w[..., None], 0, 255).astype(np.uint8)


def _feather_radius(mask: np.ndarray, radius: int) -> np.ndarray:
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(im).astype(np.float32) / 255.0


DATELINE_BAND_FRAC = 24 / 4096   # half-width of the cross-fade, as a share of the map width (~2 degrees)


def soften_dateline(img: np.ndarray, gap: np.ndarray | None = None, band_frac: float = DATELINE_BAND_FRAC) -> np.ndarray:
    """Cross-fade the left/right edges of an equirect image (the date line).

    A daily mosaic is built from one day of orbits, so at +/-180 deg the two
    edges of the map were photographed ~24h apart and the clouds don't match:
    a hard straight line that shows up on the cube faces nx/nz (the meridian
    through them is the date line). Each pixel within `band` columns of the
    seam is mixed with its mirror across the seam, with weight 0.5 at the seam
    falling to 0 at the band edge, so both edges meet at the same value.
    `gap` (1 = no data) pixels are left alone, and so are their mirrors.
    """
    h, w = img.shape[:2]
    band = max(1, int(round(w * band_frac)))
    d = np.arange(band)
    a = 0.5 * (1 - (d + 0.5) / band)
    if img.ndim == 3:
        a = a[None, :, None]
    else:
        a = a[None, :]
    left, right = img[:, d].astype(np.float32), img[:, w - 1 - d].astype(np.float32)
    if gap is not None:
        ok = ((gap[:, d] < 0.5) & (gap[:, w - 1 - d] < 0.5)).astype(np.float32)
        if img.ndim == 3:
            ok = ok[..., None]
        a = a * ok
    out = img.astype(np.float32).copy()
    out[:, d] = (1 - a) * left + a * right
    out[:, w - 1 - d] = (1 - a) * right + a * left
    return np.clip(out, 0, 255).astype(img.dtype) if img.dtype == np.uint8 else out
