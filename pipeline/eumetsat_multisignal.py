"""Fetch a global cloud-density mask combining two independent EUMETSAT
signals (temperature + a second, independently-processed product), with a
static terrain-elevation correction -- reverse-engineered in spirit from
matteason/live-cloud-maps' public source (index.js), since our own single-
channel IR approach (eumetsat_ir.py) couldn't tell faint/thin cloud apart
from naturally-cool clear ground without either missing real cloud or
painting deserts white (see eumetsat_ir.py's version history).

Signals combined:

1. mumi:worldcloudmap_ir108 -- temperature (IR). Cold reads bright.
2. mumi:wideareacoverage_rgb_dust -- a second, independently-derived
   EUMETSAT product, screen-blended with IR (matteason's formula) so either
   signal registering strongly is enough to call a pixel cloud, catching
   what IR alone misses -- the same redundancy principle as compositing 3
   VIIRS satellites instead of trusting one.

A third signal -- a real daylight color photo, used to boost pixels that
look white/grey like real cloud -- was tried and REMOVED. It's a legitimate
idea (matteason's code does this too) but on this data it mistook sun
glint (sunlight reflecting off open ocean, also bright and colourless) for
cloud, painting huge stretches of clear ocean white. Fixing that needs
either a land/ocean mask (skip the glint-prone surface entirely) or real
glint-geometry exclusion -- neither built yet, so the signal is off rather
than shipping something that actively makes things worse. See
HANDOFF.md / conversation history for the full investigation.

Terrain-elevation correction: temperature alone can't tell "this is cold
because it's a cloud" from "this is cold because it's 4500m up." High
terrain (Himalaya/Tibetan Plateau, Andes, Rockies) reads as permanently
"cloudy" otherwise -- measured 2026-10-31, the Tibetan Plateau ran ~37
IR-units colder than lowland at the same latitude, purely from altitude,
nothing to do with weather. Corrected using NASA GIBS's SRTM_Color_Index
layer (free, no auth, same reliable snapshot API as the old VIIRS pipeline
used) as an elevation proxy: its red channel tracks elevation monotonically
(measured same day: R=0 Amazon lowland, R~130 Sahara, R~165-190 Himalaya/
Andes peaks) well enough to use directly without decoding the exact color
ramp. ELEVATION_K is calibrated to roughly cancel the measured Tibetan-
Plateau-vs-lowland gap. This is a STATIC correction (terrain doesn't move)
cached to disk and fetched only once, unlike the IR/dust layers which are
re-fetched every run.

Both EUMETSAT layers are fetched in two WEST/EAST hemisphere halves rather
than one global request, mirroring matteason's own fetch pattern -- our own
single-shot global requests to this GeoServer have been measurably flaky/
slow (eumetsat_ir.py's retry-and-shrink logic exists because of it), so
this matches the pattern his already-reliable, months-running service uses.
"""
import re
from pathlib import Path

import numpy as np
import requests
from io import BytesIO
from PIL import Image

WMS_URL = "https://view.eumetsat.int/geoserver/wms"
IR_LAYER = "mumi:worldcloudmap_ir108"
DUST_LAYER = "mumi:wideareacoverage_rgb_dust"

GIBS_SNAPSHOT_URL = "https://wvs.earthdata.nasa.gov/api/v1/snapshot"
ELEVATION_LAYER = "SRTM_Color_Index"
ELEVATION_CACHE = Path(__file__).parent.parent / "cache" / "elevation_r.npy"
# Applied once, to the final combined IR+dust output, not to either input
# individually -- measured 2026-10-31 that BOTH signals carry their own
# independent elevation bias (dust channel alone: Tibetan Plateau ~174 vs
# Indo-Gangetic lowland ~22, even larger than IR's own ~37-unit gap), so
# correcting only one input left the other still pushing high terrain
# toward "cloud." Calibrated against the combined output directly: with
# elevation-proxy gap ~134 (Tibet ~173 vs lowland ~39) producing a combined-
# output gap of ~127, 127/134 ~= 0.95.
ELEVATION_K = 0.95

# Same geostationary-ring viewing-angle limit as eumetsat_ir.py -- see that
# module for the measurement this is based on.
POLE_LIMIT_DEG = 81.0


def _latest_layer_time(layer: str, timeout: int = 30) -> str:
    resp = requests.get(WMS_URL, params={
        "service": "WMS", "version": "1.3.0", "request": "GetCapabilities",
    }, timeout=timeout)
    resp.raise_for_status()
    xml = resp.text
    idx = xml.find(f"<Name>{layer}</Name>")
    if idx == -1:
        raise ValueError(f"layer {layer!r} not found in EUMETSAT capabilities")
    block = xml[idx:idx + 3000]
    m = re.search(r'<Dimension name="time"[^>]*default="([^"]+)"', block)
    if not m:
        raise ValueError(f"no time dimension default found for layer {layer!r}")
    return m.group(1)


def _fetch_hemisphere(layer: str, west: float, east: float, size: int, time_str: str) -> np.ndarray:
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetMap",
        "layers": layer, "styles": "", "crs": "EPSG:4326",
        "bbox": f"-90,{west},90,{east}",
        "width": str(size), "height": str(size),
        "format": "image/png", "time": time_str,
    }
    resp = requests.get(WMS_URL, params=params, timeout=60)
    resp.raise_for_status()
    return np.array(Image.open(BytesIO(resp.content)).convert("RGB"))


def _fetch_global(layer: str, width: int, time_str: str) -> np.ndarray:
    """Fetch a layer as one HxWx3 equirect array, as two west/east hemisphere
    halves (matteason's pattern -- see module docstring)."""
    half_size = width // 2
    west = _fetch_hemisphere(layer, -180, 0, half_size, time_str)
    east = _fetch_hemisphere(layer, 0, 180, half_size, time_str)
    return np.concatenate([west, east], axis=1)


def _screen(a, b):
    return (1 - (1 - a / 255.0) * (1 - b / 255.0)) * 255.0


def _gamma(g, x):
    return 255.0 * np.power(np.clip(x, 0, 255) / 255.0, 1.0 / g)


def _interpolate(from_a, to_a, from_b, to_b, x):
    t = np.clip((x - from_a) / (to_a - from_a), 0, 1)
    return from_b + (to_b - from_b) * t


def _elevation_proxy(width: int, height: int) -> np.ndarray:
    """Terrain-elevation proxy (0-255, higher = higher terrain), fetched
    once from NASA GIBS and cached indefinitely -- terrain doesn't change
    run to run, unlike the EUMETSAT signals above."""
    if ELEVATION_CACHE.exists():
        cached = np.load(ELEVATION_CACHE)
        if cached.shape == (height, width):
            return cached

    params = {
        "REQUEST": "GetSnapshot", "LAYERS": ELEVATION_LAYER, "CRS": "EPSG:4326",
        "TIME": "2026-10-31", "WRAP": "DAY", "BBOX": "-90,-180,90,180",
        "FORMAT": "image/png", "WIDTH": str(width), "HEIGHT": str(height),
    }
    resp = requests.get(GIBS_SNAPSHOT_URL, params=params, timeout=90)
    resp.raise_for_status()
    rgb = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
    elevation_r = rgb[..., 0].astype(np.float32)

    ELEVATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(ELEVATION_CACHE, elevation_r)
    return elevation_r


def fetch_cloud_mask(width: int = 4096):
    """Returns (mask, still_gap) in the same convention as eumetsat_ir.py."""
    height = width // 2
    time_str = _latest_layer_time(IR_LAYER)

    ir_rgb = _fetch_global(IR_LAYER, width, time_str)
    dust_rgb = _fetch_global(DUST_LAYER, width, time_str)

    ir_raw = ir_rgb[..., 0].astype(np.float32)  # IR layer is already grayscale (R=G=B)

    # -- dust channel (matteason's formula) --
    dust_r = dust_rgb[..., 0].astype(np.float32)
    dust_b = 255.0 - dust_rgb[..., 2].astype(np.float32)
    dust_r_masked = (dust_r / 255.0) * (dust_b / 255.0) * 255.0
    dust = _screen(dust_r_masked, 0.5 * dust_b)

    # -- IR channel --
    ir = _interpolate(72, 178, 0, 255, ir_raw)
    ir_gamma = _gamma(1.46, ir)

    # -- combine IR + dust --
    output = _gamma(2.0, _screen(dust, ir_gamma * 0.77))

    # -- cancel the elevation-driven cold bias, once, on the combined result --
    # (not on either input individually -- both IR and dust independently
    # carry this bias, see ELEVATION_K's comment above)
    elevation_r = _elevation_proxy(width, height)
    output = output - ELEVATION_K * elevation_r

    mask = np.clip(output, 0, 255).astype(np.uint8)

    lat = 90 - (np.arange(height) + 0.5) / height * 180
    pole_rows = np.abs(lat) > POLE_LIMIT_DEG
    still_gap = np.zeros((height, width), dtype=np.float32)
    still_gap[pole_rows, :] = 1.0

    return mask, still_gap
