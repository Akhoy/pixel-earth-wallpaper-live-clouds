"""Fetch a global cloud-density mask directly from EUMETSAT's own raw
Multimission Geostationary Ring IR10.8um mosaic (view.eumetsat.int WMS,
layer mumi:worldcloudmap_ir108) -- genuinely global (already stitched
across all EUMETSAT-participating geostationary satellites), refreshed
every 3h, no auth required.

Built after matteason/live-cloud-maps (clouds.jpg/-alpha.png, built from
this same kind of EUMETSAT data) turned out to badly over-classify clear
land as cloud -- measured 2026-10-31: 54-90% of pixels over deserts/Europe
still read as "visibly whitened" cloud even after a tuned tone-curve
correction (contrast + gamma) on top of their product. The raw IR mosaic
underneath has none of that problem: measured the same day, Sahara/Europe/
Arabia/Australia-interior all read correctly dark (a plain linear stretch,
no gamma needed, gets all four under 3% "visibly whitened"), while the
top 1% brightest pixels (genuine storm cores) stay near-white (~250/255).
So this derives our own mask directly from the raw product, the same way
gibs.py's VIIRS true-color used to get turned into a mask via
cloud_mask.py, instead of trusting matteason's own classification.
"""
import re
from pathlib import Path

import numpy as np
import requests
from io import BytesIO
from PIL import Image

WMS_URL = "https://view.eumetsat.int/geoserver/wms"
LAYER = "mumi:worldcloudmap_ir108"

# Per-pixel clear-sky baseline: the running MINIMUM raw value ever observed
# at each pixel. Clouds only ever make a pixel colder-looking (brighter, in
# this layer's rendering) than the bare ground underneath, never warmer --
# so the lowest value seen at a location over time is its best estimate of
# "what this exact spot looks like with no cloud on it," adapting per
# location instead of judging the whole planet against one fixed threshold.
# This is what actually catches faint/thin cloud that a fixed STRETCH_LOW
# misses: a wisp over a naturally-cool spot raises it only slightly above
# that spot's OWN baseline, which still registers, where a global threshold
# would need to clear the same bar as a wisp over a naturally-warm desert.
#
# Needs real history to be any good -- a single day can't distinguish
# "clear" from "persistently cloudy all day" at a given spot, so this
# necessarily starts rough and only gets accurate as more genuinely-clear
# moments get recorded over following days. Updated every run; never reset.
CLEARSKY_CACHE = Path(__file__).parent.parent / "cache" / "eumetsat_clearsky_baseline.npy"
CLEARSKY_RANGE = 80.0  # raw-value span above baseline that maps to full white

# Re-tuned twice on 2026-10-31 -- each version traded off against the last:
#
# v1: LOW=120/HIGH=190, calibrated purely against clear-region brightness
#     (Sahara/Europe/Arabia/Australia-interior). Cut false "cloud over
#     clear desert" to near zero, but also clipped most real thin/moderate
#     cloud to black (a lot of genuine cloud sits in the 60-120 raw range
#     this stretch discarded), leaving a globe nearly as bare as the dead
#     stock texture.
# v2: LOW=60/HIGH=160, re-tuned to match NASA VIIRS's own published mask's
#     statistical profile (mean/std/Europe-region close to NASA's:
#     119.6/81.1/120.1 here vs. 113.4/90.2/116.0) instead of minimizing
#     false positives alone. Fixed the bare-globe problem, but introduced a
#     new one: large, fairly-uniform cloud systems (e.g. a big stratus deck
#     over Europe/Russia) read as flat mid-gray "haze" rather than crisp
#     white, because their raw brightness sits well below HIGH -- looked
#     structurally different from genuinely thick storm cloud even though
#     both are real cloud.
# v3 (current): LOW=60/HIGH=140. Saturates to white at a lower raw value so
#     moderate/weak cloud reads with the same visual strength as thick
#     cloud (Europe's "solid white" fraction roughly doubles, 8.7%->20.4%)
#     instead of fading to haze. This necessarily makes some clear desert
#     read whiter too (Sahara's "solid white" fraction also roughly
#     doubles, 7.8%->15.8%) -- weak-cloud and warm-clear-desert brightness
#     genuinely overlap in this single IR channel, so there is no curve
#     that fixes one without moving the other; this is the accepted
#     middle-ground point between the v1 and v2 extremes.
STRETCH_LOW = 60.0
STRETCH_HIGH = 140.0

# The ring mosaic has no coverage past this latitude in either hemisphere
# -- a geostationary viewing-angle limit, not a seasonal/daylight gap like
# VIIRS's. Measured 2026-10-31 at exactly 81.4 deg (rows of flat, zero-
# variance no-data pixels beyond it) -- a small, fixed cap rather than
# VIIRS's larger, season-dependent polar-night hole. A little margin is
# kept inside the measured edge.
POLE_LIMIT_DEG = 81.0


def _latest_layer_time(timeout: int = 30) -> str:
    """The WMS time dimension uses nearestValue, but an arbitrary "now"
    timestamp can land between valid scans and 502 -- read the layer's
    current `default` time out of GetCapabilities first, then request
    exactly that."""
    resp = requests.get(WMS_URL, params={
        "service": "WMS", "version": "1.3.0", "request": "GetCapabilities",
    }, timeout=timeout)
    resp.raise_for_status()
    xml = resp.text
    idx = xml.find(f"<Name>{LAYER}</Name>")
    if idx == -1:
        raise ValueError(f"layer {LAYER!r} not found in EUMETSAT capabilities")
    block = xml[idx:idx + 3000]
    m = re.search(r'<Dimension name="time"[^>]*default="([^"]+)"', block)
    if not m:
        raise ValueError(f"no time dimension default found for layer {LAYER!r}")
    return m.group(1)


def fetch_cloud_mask(width: int = 4096):
    """Returns (mask, still_gap): mask is an HxW uint8 grayscale cloud-
    density array (H = width // 2); still_gap is an HxW float32 mask
    (1.0 = no real data -- the polar caps past POLE_LIMIT_DEG) in the same
    convention merge.py/polar.py already expect from the old VIIRS pipeline.

    GetMap for this layer at the full global extent is a heavy render on
    EUMETSAT's own GeoServer -- measured 2026-10-31, width=4096 sometimes
    hangs or 500s under load while width=2048 has been reliable, so this
    retries a few times before falling back to a smaller, safer width
    rather than letting one slow render break an unattended scheduled run.
    """
    height = width // 2
    time_str = _latest_layer_time()

    last_exc = None
    for attempt_width in (width, width, 2048 if width > 2048 else width // 2):
        attempt_height = attempt_width // 2
        params = {
            "service": "WMS", "version": "1.1.1", "request": "GetMap",
            "layers": LAYER, "srs": "EPSG:4326",
            "bbox": "-180,-90,180,90",
            "width": str(attempt_width), "height": str(attempt_height),
            "format": "image/png", "time": time_str,
        }
        try:
            resp = requests.get(WMS_URL, params=params, timeout=60)
            resp.raise_for_status()
            gray = np.array(Image.open(BytesIO(resp.content)).convert("L")).astype(np.float32)
            width, height = attempt_width, attempt_height
            break
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
    else:
        raise last_exc

    baseline = None
    if CLEARSKY_CACHE.exists():
        cached = np.load(CLEARSKY_CACHE)
        if cached.shape == gray.shape:
            baseline = cached
    if baseline is None:
        # No usable history yet at this resolution -- bootstrap from this
        # run's own reading (every pixel starts at "no cloud detected" and
        # sharpens up as genuinely-clearer moments get recorded over the
        # following runs).
        baseline = gray.copy()

    stretched = (gray - baseline) / CLEARSKY_RANGE
    mask = np.clip(stretched, 0, 1) * 255.0

    baseline = np.minimum(baseline, gray)
    CLEARSKY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CLEARSKY_CACHE, baseline)

    lat = 90 - (np.arange(height) + 0.5) / height * 180
    pole_rows = np.abs(lat) > POLE_LIMIT_DEG
    still_gap = np.zeros((height, width), dtype=np.float32)
    still_gap[pole_rows, :] = 1.0

    return np.clip(mask, 0, 255).astype(np.uint8), still_gap
