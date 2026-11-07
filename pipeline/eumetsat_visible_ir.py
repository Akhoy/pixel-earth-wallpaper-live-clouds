"""Cloud-density mask from EUMETSAT's visible + IR layers directly (no dust
channel). Matches matteason/live-cloud-maps' own compositing approach: IR is
the always-present base signal (it alone decides night-side cloud cover),
and visible is only ever screened in on top as a brightness *boost* -- never
swapped in as a replacement signal. Visible naturally contributes nothing at
night (there's no sunlight for it to reflect, so it reads near-black and
`screen()` with a near-black value is a no-op), so there's no sun-angle
calculation, day/night switchover point, or brightness calibration needed to
keep the day and night sides consistent -- there's simply nothing to splice.
An earlier version of this module computed a per-pixel solar day-weight and
blended `day_weight*vis + (1-day_weight)*ir`, which meant vis and ir had to
be forced onto the same brightness scale or the switchover itself would read
as a seam. Dropping the switch in favor of always-base-plus-boost removed
the need for that calibration entirely, and removed the seam.

This intentionally drops the dust channel eumetsat_multisignal.py mixes in:
dust's one real job is catching thin, high clouds over hot deserts at
night, when nothing else can see them. Losing that is a known, accepted
gap.

Known remaining gap: bright snow/ice can still read as cloud under the
visible formula (same ambiguity IR has with cold terrain, just inverted --
brightness instead of temperature). Land/vegetation also used to read as
cloud under the visible formula (checked against live data: Indonesia's
islands scored close to real thin cloud nearby) -- that part is now fixed
within EUMETSAT's own cloud-mask product's regional coverage (see
CLM_SECTORS), by suppressing visible there entirely wherever that product
confirms land. Outside that coverage (roughly the Pacific and the
Americas), and for snow/ice everywhere, the formula's own saturation
penalty is the only defense, and it's imperfect by nature -- a real,
global land/ice reference mask would fix both, which this module still
does not have.

**Not wired into run_pipeline.py.** This is a standalone alternative to
eumetsat_ir.py/eumetsat_multisignal.py, developed and visually verified
against live data in a separate project, then ported here. Unlike those two
modules, this one fills its own polar caps completely (see
_fix_polar_caps) rather than reporting a still_gap for polar.py to handle
-- fetch_cloud_mask()'s still_gap return is always all-zero, kept only so
the function signature matches its siblings' `(mask, still_gap)` contract
for easy swapping in later. See HANDOFF.md for the full session history
behind every design choice below.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"
CACHE_FILE = Path(__file__).parent.parent / "cache" / "eumetsat_visible_ir_last_good.npy"
# One-time real NASA VIIRS photos of each pole (see gibs.py), taken at its
# own hemisphere's summer solstice when it has 24h daylight and so full
# real coverage right up to the pole -- used to fill NODATA_THRESHOLD's cap
# in _fix_polar_caps. Not live (a live daily GIBS fetch was tried and
# dropped -- not worth a second network call and a day-old mosaic just for
# a few percent of the image at the very edge of the frame when a real
# photo from a different day already looks right), but a real photo of ice
# and cloud beats a mirrored guess. Covers lat 65..90 / -90..-65, already
# column-reordered to this module's canvas convention (see _lon_to_col) and
# with the source mosaic's own date-line wrap-edge artifact repaired (both
# fixes applied once when the assets were generated, not at load time).
POLAR_REFERENCE_LAT_SPAN = 25.0
POLAR_REFERENCES = {"north": "arctic_reference.jpg", "south": "antarctic_reference.jpg"}

WMS_URL = "https://view.eumetsat.int/geoserver/ows"
LAYER_VISIBLE = "mumi:wideareacoverage_rgb_natural"
LAYER_IR = "mumi:worldcloudmap_ir108"

# bbox = minLat,minLon,maxLat,maxLon (EUMETSAT's WMS 1.3.0 + EPSG:4326 uses
# lat,lon axis order -- NOT lon,lat. Confirmed empirically: lon,lat order
# silently returns a plausible-looking but wrongly-scaled/skewed crop
# instead of an error, so this is easy to get backwards without noticing.)
BBOX_EAST = (-90, 0, 90, 180)     # lon 0 -> 180
BBOX_WEST = (-90, -180, 90, 0)    # lon -180 -> 0

# EUMETSAT's full-disk mosaic has a hard view-angle cutoff a little past
# +-81 deg latitude -- beyond it every pixel is a flat (255,255,255) no-data
# fill, not real data. Checked directly against live tiles: real data right
# up to that edge is valid and often noisy-looking (view angle gets more
# oblique, not "bad"), so this is detected from the data itself per-fetch
# (NODATA_THRESHOLD) rather than assumed at a fixed latitude -- an earlier
# version faded a fixed, much wider lat 60->90 band to a flat gray constant,
# which threw away ~20 degrees of real data and replaced it with an
# artificial flat band that read as a seam across the whole southern ocean.
NODATA_THRESHOLD = 254.0

# EUMETSAT's own pre-computed cloud mask product (clear-land / clear-ocean /
# cloudy classification, not a brightness heuristic) -- used here only for
# its land/ocean classification, to suppress the visible formula's land
# false-positives (see _compute_mask). Two regional sectors; combined they
# cover lon -77..118.5, lat -77..77. Outside that the visible formula's own
# saturation penalty is the only defense.
CLM_WMS_URL = "https://view.eumetsat.int/geoserver/wms"
CLM_SECTORS = [
    ("msg_iodc:clm", -77.0, 77.0, 0.0, 118.5),   # lat south/north, lon west/east
    ("msg_fes:clm", -77.0, 77.0, -77.0, 0.0),
]
# product convention: green = clear land, blue = clear ocean, white = cloudy,
# black = no data (night side / outside the sector).
CLM_LAND_GREEN_MARGIN = 30


def _fetch_layer(layer: str, bbox: tuple[float, float, float, float], time_iso: str, size: int) -> np.ndarray:
    min_lat, min_lon, max_lat, max_lon = bbox
    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.3.0",
        "layers": layer,
        "styles": "",
        "format": "image/png",
        "crs": "EPSG:4326",
        "bbox": f"{min_lat},{min_lon},{max_lat},{max_lon}",
        "width": str(size),
        "height": str(size),
        "time": time_iso,
    }
    resp = requests.get(WMS_URL, params=params, timeout=90)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)


def _stitch(east: np.ndarray, west: np.ndarray) -> np.ndarray:
    """east covers lon 0->180, west covers lon -180->0. Placed east-then-west
    so the antimeridian join (east's right edge meeting west's left edge)
    lands at the center of the canvas, and the prime-meridian join (east's
    left edge / west's right edge) lands at the canvas's own wrap edges,
    where a modulo-indexed column lookup already treats them as adjacent.

    No gradient patch: both tiles are crops of one single underlying
    EUMETSAT mosaic, not independently-captured photos, so adjacent crops
    already line up pixel-for-pixel at the join -- checked directly against
    live data (mean/std per column within +-3px of the seam showed no jump
    at all). An earlier version of this function manufactured a gradient
    patch here on the assumption the join would mismatch; it didn't, and
    the patch's own single-column stretch was what produced a visible
    stripe artifact. Plain concatenation is both simpler and correct.
    """
    return np.concatenate([east, west], axis=1)


def _lon_to_col(lon: float, width: int) -> int:
    """Inverse of the column->longitude mapping the rest of this module
    uses (column 0 = lon 0, increasing east through lon 180 at the
    midpoint, continuing from lon -180 back to lon 0 at the far edge --
    matching _stitch's east-then-west canvas layout)."""
    return int(round((lon % 360.0) / 360.0 * width))


def _fetch_land_mask(width: int, height: int, time_iso: str) -> tuple[np.ndarray, np.ndarray]:
    """Land/ocean classification from EUMETSAT's own Cloud Mask (CLM)
    product, placed onto a canvas aligned with the rest of this module's
    vis/ir canvas. Returns (is_land, covered) -- covered is False outside
    CLM_SECTORS' combined coverage (see its docstring), where the caller
    falls back to the saturation-penalty heuristic alone.
    """
    is_land = np.zeros((height, width), dtype=bool)
    covered = np.zeros((height, width), dtype=bool)

    for layer, south, north, west, east in CLM_SECTORS:
        col0, col1 = _lon_to_col(west, width), _lon_to_col(east, width)
        row0 = int(round((90.0 - north) / 180.0 * height))
        row1 = int(round((90.0 - south) / 180.0 * height))
        px_w, px_h = col1 - col0, row1 - row0
        if px_w <= 0 or px_h <= 0:
            continue

        resp = requests.get(CLM_WMS_URL, params={
            "service": "WMS", "version": "1.3.0", "request": "GetMap",
            "layers": layer, "styles": "", "crs": "EPSG:4326",
            "bbox": f"{south},{west},{north},{east}",
            "width": str(px_w), "height": str(px_h),
            "format": "image/png", "time": time_iso,
        }, timeout=90)
        resp.raise_for_status()
        rgb = np.array(Image.open(BytesIO(resp.content)).convert("RGB")).astype(np.int16)
        sector_land = (rgb[..., 1] > rgb[..., 0] + CLM_LAND_GREEN_MARGIN) & \
            (rgb[..., 1] > rgb[..., 2] + CLM_LAND_GREEN_MARGIN)
        # Anything the sector actually rendered (not transparent/no-data
        # black) counts as covered, cloudy or clear, land or ocean alike --
        # land suppression only needs to know land vs. not-land.
        sector_covered = rgb.any(axis=-1)

        is_land[row0:row1, col0:col1] = sector_land
        covered[row0:row1, col0:col1] = sector_covered

    return is_land, covered


def _truecolor_score(rgb: np.ndarray) -> np.ndarray:
    """Same shape of formula as vis_score in _compute_mask, for the same
    reason (bright + not-colorful reads as cloud/ice) -- keeps the static
    polar reference photos (see POLAR_REFERENCES) on a comparable tone to
    the rest of the image before _fix_polar_caps' own tone-matching trims
    whatever gap is left.
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    maxc = rgb_f.max(axis=-1)
    minc = rgb_f.min(axis=-1)
    sat = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0.0)
    return np.clip(maxc * (1 - sat) ** 2, 0, 1) ** 0.8


def _static_polar_score(pole: str, width: int, height: int) -> np.ndarray | None:
    """The fallback score for one pole's cap from POLAR_REFERENCES, resized
    to this fetch's resolution and placed at the top (north) or bottom
    (south) of a full HxW canvas (zero elsewhere -- the caller only ever
    reads it within that pole's own cap rows). Returns None if the asset
    can't be loaded, so the caller can fall back further to the mirror.
    """
    try:
        img = Image.open(ASSETS_DIR / POLAR_REFERENCES[pole]).convert("RGB")
        band_h = int(round(POLAR_REFERENCE_LAT_SPAN / 180.0 * height))
        if img.size != (width, band_h):
            img = img.resize((width, band_h), Image.LANCZOS)
        band_score = _truecolor_score(np.array(img))
    except Exception:
        logger.warning("polar reference image %s failed to load", pole, exc_info=True)
        return None

    full = np.zeros((height, width), dtype=np.float32)
    if pole == "north":
        full[:band_h, :] = band_score
    else:
        full[height - band_h:, :] = band_score
    return full


def _fix_polar_caps(gray: np.ndarray, ir_gray: np.ndarray) -> np.ndarray:
    """Replace only the true no-data disk edge (see NODATA_THRESHOLD) with a
    mirror reflection of that column's own real data just inside it.

    Two other approaches were tried and rejected first:
    - A flat fade to a constant tone (the original version of this
      function): looked like an artificial, texture-less band against the
      organic cloud texture everywhere else, however well its brightness
      was tuned -- flatness itself was the tell, not the tone.
    - Filling with a column-averaged tone (real tone, but still flat):
      better, but still a smooth patch next to turbulent cloud texture, and
      any texture-less patch reads as a seam regardless of how well its
      brightness matches its surroundings.

    Mirroring keeps real texture in the cap. Using each column's own exact
    boundary works, but the boundary itself has a fine sawtooth in it --
    checked directly against the raw IR tile, the jaggedness is really there
    in the satellite's own edge, not something this module introduces -- and
    mirroring doubles it into a visible comb. A blurred version was tried to
    hide that, but blurring the whole cap replaced one artificial-looking
    thing (a comb) with another (an obviously hazy, texture-less patch next
    to sharp cloud) -- not an improvement, just a different tell.

    Better: smooth the *boundary line* (a median filter over the boundary
    row-index, not over pixels) so most columns mirror around a gently
    curved line instead of the sawtooth, which is what actually produces the
    comb. That smoothing can undershoot the true boundary at a few columns
    (the real edge pokes out further than its smoothed neighbors there),
    which would leave a sliver of genuine no-data pixels exposed -- so any
    such sliver gets patched with its own column's nearest valid pixel
    afterward. That patch only ever touches the few leftover pixels between
    the smoothed and true boundary (checked: a handful of rows at a handful
    of columns), never the whole cap, so it doesn't reintroduce the comb
    itself.
    """
    h, w = gray.shape
    not_nodata = ir_gray < NODATA_THRESHOLD
    any_valid = not_nodata.any(axis=0)
    north_true = np.where(any_valid, np.argmax(not_nodata, axis=0), h)
    south_true = np.where(any_valid, np.argmax(not_nodata[::-1, :], axis=0), h)
    north = median_filter(north_true, size=25, mode="wrap")
    south = median_filter(south_true, size=25, mode="wrap")

    out = gray.copy()
    for c in range(w):
        n = int(north[c])
        if 0 < n < h:
            length = min(n, h - n)
            out[n - length:n, c] = out[n:n + length, c][::-1]
        s = int(south[c])
        if 0 < s < h:
            boundary = h - s
            length = min(s, boundary)
            out[boundary:boundary + length, c] = out[boundary - length:boundary, c][::-1]

    for c in range(w):
        true_n, smooth_n = int(north_true[c]), int(north[c])
        if smooth_n < true_n:
            out[smooth_n:true_n, c] = out[true_n, c]
        true_s, smooth_s = int(south_true[c]), int(south[c])
        if smooth_s < true_s:
            out[h - true_s:h - smooth_s, c] = out[h - true_s - 1, c]

    # The mirror above is a guess; a real photo (see POLAR_REFERENCES) is
    # strictly better, so blend the true no-data cap toward one wherever the
    # asset loads. Southern summer means Antarctica's real exposure happens
    # to land close to the rest of the image's tone on its own (checked
    # against live data: within ~1/255 without any adjustment), but the
    # Arctic reference was taken at its OWN summer -- a different season
    # from whatever this fetch's actual date is -- and its exposure doesn't
    # just happen to match: checked against live data, it came in about
    # 11/255 dimmer than the real pixels right next to it. Every splice here
    # is tone-matched first (shift that pole's reference by the difference
    # between its own mean right at the boundary and the real data's mean
    # on the other side of it), then blended in gradually over a wide band
    # straddling the boundary rather than swapped in with a hard cut --
    # checked against live data, even a perfectly tone-matched hard cut
    # still read as a seam, because the static photo's texture (smooth
    # ice/cloud) doesn't carry the turbulence of the organic cloud texture
    # right next to it; fading between the two over ~60 rows instead of
    # switching in one lets the turbulent texture itself taper off instead
    # of ending abruptly.
    out = out.astype(np.float32)
    rows = np.arange(h)[:, None].astype(np.float32)
    transition = 30.0
    for pole, is_north in (("north", True), ("south", False)):
        static_score = _static_polar_score(pole, w, h)
        if static_score is None:
            continue
        # Use the same *smoothed* boundary the mirror above used, not the
        # raw sawtooth edge -- using the unsmoothed one here reintroduced
        # exactly the jagged, stairstepped cut line (with slivers of
        # unreplaced, still-bad edge pixels peeking through at its notches)
        # that median-filtering the boundary was already written to avoid.
        boundary = north.astype(np.float32)[None, :] if is_north else (h - south.astype(np.float32))[None, :]
        depth = (boundary - rows) if is_north else (rows - boundary)
        t = np.clip((depth + transition) / (2 * transition), 0.0, 1.0)
        weight = t * t * (3 - 2 * t)  # smoothstep: 1 deep in the cap, 0 past the boundary

        cap = np.zeros((h, w), dtype=bool)
        edge = np.zeros((h, w), dtype=bool)
        for c in range(w):
            b = int(north[c]) if is_north else h - int(south[c])
            if is_north and b > 0:
                cap[:b, c] = True
                edge[max(0, b - 10):b, c] = True
            elif not is_north and b < h:
                cap[b:, c] = True
                edge[b:min(h, b + 10), c] = True
        if not cap.any() or not edge.any():
            continue

        offset = out[edge].mean() - static_score[cap].mean() * 255
        static_gray = np.clip(static_score * 255 + offset, 0, 255)
        out = weight * static_gray + (1 - weight) * out

    return np.clip(out, 0, 255).astype(np.uint8)


def _screen(base: np.ndarray, boost: np.ndarray) -> np.ndarray:
    """matteason's own blend: `boost` only ever brightens `base`, never
    replaces it -- screen(x, 0) == x, so a near-black boost (visible at
    night, with no sunlight to reflect) is a no-op by construction. No
    day/night switch, so nothing can seam at a switchover point.
    """
    return 1.0 - (1.0 - base) * (1.0 - boost)


def _compute_mask(vis: np.ndarray, ir: np.ndarray, is_land: np.ndarray, land_covered: np.ndarray) -> np.ndarray:
    rgb = vis.astype(np.float32) / 255.0
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    sat = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0.0)
    # Squared, not linear: land/vegetation (muted green/tan, sat around
    # 0.1-0.2) only got a mild discount from a linear (1-sat) and still
    # scored close to real thin cloud -- checked against live data, a
    # sampled Indonesia land pixel scored 0.46 against 0.07 for actual thin
    # cloud nearby. Squaring the penalty pushes moderate-saturation land
    # down harder while barely touching already-low-saturation thick cloud.
    vis_score = np.clip(maxc * (1 - sat) ** 2, 0, 1) ** 0.8
    # Where EUMETSAT's own cloud-mask product (not a brightness heuristic)
    # confirms a pixel is land, drop the visible formula's contribution
    # there entirely -- the saturation penalty above is a same-formula
    # mitigation and still lets plenty of land through on its own.
    vis_score = np.where(is_land & land_covered, 0.0, vis_score)
    # Gain of 1 (vis_score used as-is): a 0.6x damping was tried here after
    # a 1.4x gain blew out a moderately-cloudy region, but that blowout was
    # actually caused by *also* having ir_lifted's floor raised to 0.23 at
    # the same time (see below) -- the two stacked through `screen()`. With
    # ir_lifted back to its gentler floor, 0.6x turned out to have
    # overcorrected: checked against live data, the combined output then
    # differed from ir_lifted alone by only ~7/255 on average (screen()'s
    # own math already damps a boost's effect wherever the IR base is
    # already bright, since its slope is `1 - ir_lifted`), making the
    # visible channel's contribution nearly invisible in the final image --
    # exactly the "looks identical to IR" complaint it produced. Gain 1
    # keeps real visible detail without re-introducing the old floor-stack
    # problem.
    vis_boost = np.clip(vis_score, 0, 1)

    ir_gray = ir.astype(np.float32).mean(axis=-1)
    ir_stretched = np.clip((ir_gray - 72) / (178 - 72), 0, 1) * 255
    ir_score = np.clip((ir_stretched / 255.0) ** (1 / 1.46), 0, 1)
    # IR's clear-sky floor reads noticeably darker than visible's (checked
    # against live data), which otherwise leaves the night side looking
    # like a different, darker image rather than the same grayscale tone.
    # Raise the floor and compress the range so it sits on the same scale.
    # Kept modest: a stronger lift (0.23/0.77) narrowed the day/night gap a
    # little further, but stacked with vis_boost through `screen()` and
    # pushed moderately-cloudy regions into the same washed-out brightness
    # problem vis_boost's gain caused on its own -- not worth it for a few
    # points of gap-closing.
    ir_lifted = np.clip(0.15 + ir_score * 0.85, 0, 1)

    final = _screen(ir_lifted, vis_boost)
    gray = (np.clip(final, 0, 1) * 255).astype(np.uint8)
    return _fix_polar_caps(gray, ir_gray)


def fetch_cloud_mask(width: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mask, still_gap) in the same convention as eumetsat_ir.py
    and eumetsat_multisignal.py -- but still_gap is always all-zero here,
    since _fix_polar_caps already fills both poles itself (real photo where
    one's available, a textured mirror otherwise). Falls back to the last
    successfully computed mask on any fetch/decode failure, rather than
    raising and leaving a scheduled run with nothing to publish.
    """
    utc = datetime.now(timezone.utc)
    half_size = width // 2
    time_iso = utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    try:
        vis_east = _fetch_layer(LAYER_VISIBLE, BBOX_EAST, time_iso, half_size)
        vis_west = _fetch_layer(LAYER_VISIBLE, BBOX_WEST, time_iso, half_size)
        ir_east = _fetch_layer(LAYER_IR, BBOX_EAST, time_iso, half_size)
        ir_west = _fetch_layer(LAYER_IR, BBOX_WEST, time_iso, half_size)

        vis = _stitch(vis_east, vis_west)
        ir = _stitch(ir_east, ir_west)
        is_land, land_covered = _fetch_land_mask(width, vis.shape[0], time_iso)

        mask = _compute_mask(vis, ir, is_land, land_covered)
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE_FILE, mask)
    except Exception:
        logger.exception("EUMETSAT visible+IR fetch failed")
        if CACHE_FILE.exists():
            logger.warning("falling back to last cached cloud mask")
            mask = np.load(CACHE_FILE)
        else:
            raise

    still_gap = np.zeros(mask.shape, dtype=np.float32)
    return mask, still_gap
