"""Derive a grayscale cloud-density mask from true-color imagery.

Heuristic: clouds are bright AND low-saturation (white/gray). Land and
ocean are comparatively saturated (green/brown/blue). So:

    cloud = brightness * (1 - saturation)

with a slight gamma push so partial/thin cloud doesn't wash out to gray.
"""
import numpy as np
from PIL import Image, ImageFilter


def _vertical_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Per-column moving average along axis 0 (rows), via cumulative sum."""
    h = arr.shape[0]
    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="reflect")
    csum = np.cumsum(padded, axis=0, dtype=np.float64)
    csum = np.vstack([np.zeros_like(csum[:1]), csum])
    top = csum[window:window + h]
    bottom = csum[0:h]
    return (top - bottom) / window


def destripe(gray: np.ndarray, window: int = 250, strength: float = 1.0) -> np.ndarray:
    """Remove granule-to-granule banding (visible even within a single
    satellite's own daily mosaic -- NASA stitches many short orbital swath
    segments together, each with a slightly different tone, into bands that
    run roughly top-to-bottom / along-track).

    An isotropic blur can't isolate this: within any local neighborhood
    there's still plenty of *real* cloud detail, so subtracting it just acts
    like a high-pass sharpen rather than flattening the banding.

    Instead, average heavily ALONG each stripe's long axis (here: straight
    down each column) with a wide window -- long enough that real cloud
    features average out -- while doing zero averaging ACROSS stripes
    (columns stay independent). That isolates the actual per-stripe
    brightness offset, which we then subtract and re-center.
    """
    gray_f = gray.astype(np.float64)
    along_track_trend = _vertical_moving_average(gray_f, window)
    global_mean = float(gray_f.mean())

    corrected = gray_f - strength * (along_track_trend - global_mean)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def fix_antarctic_cap(gray: np.ndarray, fade_start_lat: float = -60, fade_end_lat: float = -70,
                       target_value: int = 245) -> np.ndarray:
    """South of Antarctica's satellite-coverage edge, gap-inpainting only has
    a handful of thin real slivers to diffuse from, so it produces a flat,
    muddy mid-gray filler (visibly different from real cloud/ice texture,
    and darker than real ice) rather than usable texture. Since that region
    is realistically ice/cloud-covered almost year-round anyway, replace it
    with a clean near-white value -- smoothly feathered in, not a hard cut.
    """
    h, w = gray.shape
    lat = 90 - (np.arange(h) + 0.5) / h * 180
    # 0 above fade_start_lat, ramps to 1 by fade_end_lat
    blend = np.clip((fade_start_lat - lat) / (fade_start_lat - fade_end_lat), 0, 1)

    out = gray.astype(np.float32) * (1 - blend[:, None]) + target_value * blend[:, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def rgb_to_cloud_mask(rgb: np.ndarray, clear_sky=None) -> np.ndarray:
    """rgb: HxWx3 uint8. Returns HxW uint8 grayscale cloud density.

    With a `clear_sky` reference (a cloud-free image of the same scene, see
    clearsky.py) cloud is measured as how far a pixel has been pushed from
    the known bare surface toward white. That is what separates cloud from
    merely bright ground: the Sahara is blinding, but it is exactly as
    blinding in the reference, so it contributes nothing.

    Without a reference it falls back to the old standalone heuristic
    (bright and unsaturated == cloud), which works at sea but paints a white
    veil over deserts, snow and ice.
    """
    arr = rgb.astype(np.float32)
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    saturation = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0.0)

    if clear_sky is None:
        cloud = (maxc / 255.0) * (1.0 - saturation)
        cloud = np.clip(cloud, 0.0, 1.0)
        # Exponent 1.6 (heavily compressive) was tuned for a since-reverted
        # architecture where this mask was painted a second time on top of a
        # live true-color day texture (DayMapProvider), to avoid double-
        # brightening cloud already visible in that photo. The day surface is
        # back to the static bundled basemap now, so that reason no longer
        # applies -- and 1.6 was, as a side effect, squashing real moderate/
        # thin cloud into thin, patchy streaks (verified: a solid frontal
        # cloud band off South America read as a thin eroded line instead of
        # the thick mass NASA's own photo shows).
        #
        # 1.2 recovers most of that lost thickness/texture while keeping
        # enough separation that bare desert (Sahara/Arabia) still reads
        # clearly darker than genuine storm cloud (measured ratio 0.85 vs.
        # 0.94 at the old pre-DayMapProvider value of 0.8, where desert and
        # storm cloud become nearly indistinguishable -- the exact problem
        # that got this exponent added in the first place). Values near 1.0
        # (dense storm-top cloud) stay close to unchanged either way, so the
        # night-side storm gate in earth.frag (fires above cloudColor
        # 0.72-0.82) still clears normally.
        cloud = cloud ** 1.2
        return (cloud * 255.0).astype(np.uint8)

    scene = arr.mean(axis=-1)
    reference = clear_sky.astype(np.float32).mean(axis=-1)

    # The two images are not exposure-matched (the reference is a stylised
    # basemap), so rescale the reference onto the scene before differencing,
    # using the median ratio over land/lit pixels rather than a fitted curve.
    lit = reference > 10
    gain = float(np.median(scene[lit] / np.clip(reference[lit], 1.0, None))) if lit.any() else 1.0
    baseline = reference * gain

    # fraction of the remaining headroom to white that this pixel has taken up
    excess = (scene - baseline) / np.clip(255.0 - baseline, 20.0, None)

    # cloud is white; keep penalising saturated pixels so coloured ground that
    # happens to be brighter than the reference (vegetation flush, sun glint
    # off water) does not read as cloud.
    cloud = np.clip(excess, 0.0, 1.0) * np.clip(1.0 - saturation * 1.6, 0.0, 1.0)
    cloud = np.clip(cloud, 0.0, 1.0) ** 0.75
    return (cloud * 255.0).astype(np.uint8)
