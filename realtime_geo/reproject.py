"""Place each source onto a common global equirect canvas as (rgb, valid,
weight) triples, where `weight` is a per-pixel confidence in [0, 1] that
fades from 1 at the source's sub-satellite point toward 0 near the edge of
its usable view -- so where two satellites' disks overlap, the blend
naturally favors whichever one has the more head-on (less oblique, less
parallax-distorted) look at that pixel, instead of a hard seam.

Equirect convention used everywhere here (matches pipeline/cubemap.py):
column 0 = lon -180, row 0 = lat +90 (north pole), endpoint=False sampling.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from geos_reproject import lonlat_to_scan_angle, reproject_fulldisk_to_equirect, SCAN_LIMIT  # noqa: E402

GAP_THRESHOLD = 8  # matches pipeline/merge.py's convention for "no data" pixels

# Fraction of SCAN_LIMIT (GOES) / bbox half-width (EUMETSAT) beyond which
# confidence starts fading to 0. The true geometric limb (r=1) is where a
# pixel stops being visible at all, but imagery gets extremely oblique and
# distorted well before that -- fading out earlier keeps that garbage from
# ever winning a normalized blend just because it's the only nonzero
# source left in a stretch no one else reaches (see merge_realtime.py's
# MIN_TRUSTED_WEIGHT for the other half of that fix).
USABLE_R = 0.75


def reproject_goes_to_equirect(img: np.ndarray, sat_lon_deg: float,
                                out_width: int, out_height: int):
    """Returns (rgb, valid, weight), each (out_height, out_width, ...)."""
    rgb, valid = reproject_fulldisk_to_equirect(img, sat_lon_deg, out_width, out_height)

    lons = np.linspace(np.radians(-180), np.radians(180), out_width, endpoint=False)
    lats = np.linspace(np.radians(90), np.radians(-90), out_height, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    x, y, _ = lonlat_to_scan_angle(lon_grid, lat_grid, sat_lon_deg)

    r = np.sqrt(x ** 2 + y ** 2) / SCAN_LIMIT  # 0 at nadir, ~1 approaching the visible limb
    weight = np.clip((1.0 - r) / (1.0 - USABLE_R), 0.0, 1.0) ** 1.5
    weight = (weight * valid).astype(np.float32)
    return rgb, valid, weight


def place_eumetsat_on_equirect(tile_rgb: np.ndarray, bbox: tuple,
                                sat_lon_deg: float, out_width: int, out_height: int):
    """tile_rgb already covers `bbox` in plate-carree; resize it onto the
    matching slice of the global canvas and build validity + confidence
    weight masks (confidence fades toward the bbox edges)."""
    minx, miny, maxx, maxy = bbox
    canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    valid = np.zeros((out_height, out_width), dtype=np.float32)

    col0 = int(round((minx + 180.0) / 360.0 * out_width))
    col1 = int(round((maxx + 180.0) / 360.0 * out_width))
    row0 = int(round((90.0 - maxy) / 180.0 * out_height))
    row1 = int(round((90.0 - miny) / 180.0 * out_height))
    col0, col1 = np.clip([col0, col1], 0, out_width)
    row0, row1 = np.clip([row0, row1], 0, out_height)

    resized = np.array(
        Image.fromarray(tile_rgb).resize((col1 - col0, row1 - row0), Image.BILINEAR)
    )
    canvas[row0:row1, col0:col1] = resized

    tile_valid = (resized.astype(np.int32).sum(axis=-1) > GAP_THRESHOLD).astype(np.float32)
    valid[row0:row1, col0:col1] = tile_valid

    lons = np.linspace(-180.0, 180.0, out_width, endpoint=False)
    lats = np.linspace(90.0, -90.0, out_height, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    half_w = (maxx - minx) / 2.0
    half_h = (maxy - miny) / 2.0
    dx = np.abs(lon_grid - sat_lon_deg) / half_w
    dy = np.abs(lat_grid - (miny + maxy) / 2.0) / half_h
    r = np.maximum(dx, dy)
    weight = (np.clip((1.0 - r) / (1.0 - USABLE_R), 0.0, 1.0) ** 1.5 * valid).astype(np.float32)

    return canvas, valid, weight
