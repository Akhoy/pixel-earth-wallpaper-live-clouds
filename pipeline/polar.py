"""Fill the two polar no-data holes in the cloud cubemap.

VIIRS true-color needs daylight, so each pole is blank for part of the year. The app draws that side of the
Earth dark anyway (it is the night side), so the fill only has to look like clouds and match what surrounds it.
Earlier versions fetched thermal infrared and tuned fills and quality limits on a few weeks of data; each new
season or data outage broke a rule. This version has nothing to tune day to day and nothing to fetch:

  1. Measure the hole in that day's data (solid no-data blob + the dark half-lit fringe around it).
  2. Fill it: the surrounding clouds' brightness spread inward (distance-weighted, nearest edge first), plus the
     fine texture of a FIXED real-cloud photo (pipeline/assets/polar_texture.png) scaled to the contrast of the
     surroundings.
  3. If that fill looks broken (mostly black / a hard edge) or raises, reuse the previously PUBLISHED polar tiles
     for the hole, and if those aren't available fade the hole to "no cloud".

Only the two polar faces are ever touched, and any error here leaves the other four faces exactly as they were:
apply_polar_fixes never raises.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import (binary_closing, binary_dilation, distance_transform_edt, gaussian_filter, label, zoom)

from cubemap import equirect_to_face
from dxt1 import decode_dds_tile
from merge import fade_to_hole

FACE_SIZE = 1024
REF_WIDTH = 4096              # map width the pixel-sized constants below were tuned at (they scale with the width)
TEXTURE_PATH = Path(__file__).parent / "assets" / "polar_texture.png"

# --- measuring the hole
SOUTH_FRINGE = 215            # half-lit pixels darker than this, touching the hole, join it
SOUTH_GROW_STEPS, SOUTH_MARGIN = 3, 8
BAND_RECOVER = 0.8            # the dim band is over where a 6 px ring's mean reaches this share of normal
BAND_MAX_FRAC = 0.4           # never grow by more than this share of the hole radius

# --- the fill
SPREAD_DOWN = 4               # _spread_inward works at 1/4 resolution
TEXTURE_SCALE = 30            # px: detail finer than this comes from the fixed texture
TEXTURE_WEIGHT = 1.0

# --- "does the fill look broken?" -- deliberately loose, it only has to catch a dead fill
DARK_LEVEL = 30               # cloud-mask value below which a pixel counts as black
MAX_DARK_FRAC = 0.5           # more than this share of the hole near-black = dead
MAX_EDGE_STEP = 90.0          # mean brightness just inside vs just outside the hole's edge


def _blur(a, s):
    return gaussian_filter(a.astype(np.float32), s, mode="nearest")


def _face(img2d: np.ndarray, name: str) -> np.ndarray:
    rgb = np.stack([np.clip(img2d, 0, 255).astype(np.uint8)] * 3, axis=-1)
    return equirect_to_face(rgb, FACE_SIZE, name)[..., 0].astype(np.float32)


def _radius_grid() -> np.ndarray:
    yy, xx = np.mgrid[0:FACE_SIZE, 0:FACE_SIZE]
    return np.hypot(xx - (FACE_SIZE - 1) / 2, yy - (FACE_SIZE - 1) / 2)


def _hole_radius(hole_face: np.ndarray) -> float:
    return float(np.sqrt((hole_face > 0.5).sum() / np.pi))


# ------------------------------------------------------------------ measuring the hole

def south_hole(cloud_gray: np.ndarray) -> np.ndarray:
    """Measured per day: the solid no-data blob around the south pole, grown a few steps into the dark
    half-lit fringe, plus a margin. (The north pole uses the same routine on the flipped map: north_hole.)"""
    h, w = cloud_gray.shape
    sc = w / REF_WIDTH                                           # the pixel sizes below are tuned at width 4096
    step, margin, closing = max(1, round(8 * sc)), max(1, round(SOUTH_MARGIN * sc)), max(1, round(6 * sc))
    lat = 90 - (np.arange(h) + 0.5) / h * 180
    south = (lat < -60)[:, None]
    gap = ((cloud_gray == 0) & south).astype(np.float32)
    labels, _ = label(_blur(gap, 12 * sc) > 0.5)
    pole = labels[h - 1][labels[h - 1] > 0]
    if not len(pole):
        return np.zeros(cloud_gray.shape, bool)
    cur = labels == np.bincount(pole).argmax()
    # pad past the pole row so closing/dilation don't erode at the array border
    pad = 2 * (step + margin + closing)
    cur = np.pad(cur, ((0, pad), (0, 0)), mode="edge")
    south = np.pad(south, ((0, pad), (0, 0)), mode="edge")
    gray = np.pad(cloud_gray, ((0, pad), (0, 0)), mode="edge")
    for _ in range(SOUTH_GROW_STEPS):
        ring = binary_dilation(cur, iterations=step) & ~cur
        cur = cur | (ring & south & (gray < SOUTH_FRINGE))
    return binary_dilation(binary_closing(cur, iterations=closing), iterations=margin)[:-pad]


def north_hole(cloud_gray: np.ndarray) -> np.ndarray:
    return south_hole(cloud_gray[::-1])[::-1]


def grow_face_hole(hole_face: np.ndarray, photo_face: np.ndarray | None = None) -> np.ndarray:
    """Grow a big winter hole outward over the dim terminator band.

    Just beyond the no-photo disc the sun is almost on the horizon, so the photo there is very dark
    (cloud-mask values ~20-100 where normal cloud is ~150) for a band whose width varies a lot: ~30 px at the
    Arctic in Dec 2025, ~75 px at Antarctica in Jun 2026, none at the Arctic on Oct 23 2026. Left in, it shows as
    a grey ring around the fill. With the photo, the band is measured: grow to the first 6 px ring outside the
    hole whose mean has recovered to BAND_RECOVER of the normal level 120-200 px out. Without it, 11.5% of the
    radius. Small holes (radius < ~150) are unchanged; the growth ramps in up to full size at radius 230."""
    r = _hole_radius(hole_face)
    ramp = float(np.clip((r - 150) / 80, 0, 1))
    if ramp <= 0:
        return hole_face
    d = distance_transform_edt(hole_face < 0.5)
    margin = 0.115 * r
    far = (d > 120) & (d <= 200)
    if photo_face is not None and far.sum() > 500:
        ref = float(photo_face[far].mean())
        margin = BAND_MAX_FRAC * r
        for lo in range(0, int(BAND_MAX_FRAC * r), 6):
            ring = (d > lo) & (d <= lo + 6)
            if ring.any() and float(photo_face[ring].mean()) >= BAND_RECOVER * ref:
                margin = lo + 4.0
                break
    margin *= ramp
    if margin < 1:
        return hole_face
    return np.maximum(hole_face, (d <= margin).astype(np.float32))


# ------------------------------------------------------------------ the fill

def _spread_inward(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill the invalid region from the valid pixels around it, nearest data first.

    Each pixel takes a Gaussian-weighted average of the valid pixels around it, with a blur radius that grows
    smoothly with its distance from the nearest valid pixel (radius = 0.5 x distance). So it takes its tone from
    the closest edge (a hole in the middle of ice stays ice) instead of drifting to the global mean, and since
    the radius varies continuously there are no concentric steps like a few fixed blur scales leave.
    Works on a 4x smaller copy (the estimate is smooth anyway), then scales back up."""
    f = SPREAD_DOWN
    n = img.shape[0] // f
    vw = valid.reshape(n, f, n, f).mean(axis=(1, 3))
    num = (img * valid).reshape(n, f, n, f).mean(axis=(1, 3))
    dist = distance_transform_edt(vw < 0.5)                      # in downsampled pixels
    ladder = [1.0 * 1.4 ** k for k in range(0, 17)]               # sigma 1 .. ~290 downsampled px (4 .. ~1160 px)
    est = np.stack([_blur(num, s) / np.maximum(_blur(vw, s), 1e-4) for s in ladder])
    target = np.clip(0.5 * dist, ladder[0], ladder[-1])
    pos = np.log(target / ladder[0]) / np.log(1.4)
    lo = np.clip(np.floor(pos).astype(int), 0, len(ladder) - 2)
    t = pos - lo
    yy, xx = np.mgrid[0:n, 0:n]
    out = est[lo, yy, xx] * (1 - t) + est[lo + 1, yy, xx] * t
    out = np.where(vw >= 0.5, num / np.maximum(vw, 1e-4), out)
    return zoom(out, f, order=1, mode="nearest")


def _texture(face_name: str):
    """The fixed real-cloud texture (mirrored for the south so the two poles don't look identical), or None."""
    if not TEXTURE_PATH.exists():
        return None
    tex = np.array(Image.open(TEXTURE_PATH).convert("L")).astype(np.float32)
    return tex[:, ::-1] if face_name == "nz" else tex


def fill_hole(photo_face: np.ndarray, hole_face: np.ndarray, face_name: str) -> np.ndarray:
    """Surrounding clouds' brightness spread inward, plus the fixed texture's detail scaled to the contrast of
    the surroundings (so it never paints dark blobs on flat white ice)."""
    A, Hm = photo_face, hole_face
    base = _spread_inward(A, (Hm < 0.5).astype(np.float32))
    tex = _texture(face_name)
    ring = (Hm < 0.5) & (_blur(Hm, 40) > 0.02)                   # live pixels just outside the hole
    if tex is not None and ring.any():
        gm, gs = tex[ring].mean(), tex[ring].std() + 1e-3
        scaled = (tex - gm) / gs * min(A[ring].std(), gs) + A[ring].mean()
        detail = scaled - _blur(scaled, TEXTURE_SCALE)           # keep texture only, not its low frequencies
        base = base + TEXTURE_WEIGHT * detail * np.clip(_blur(Hm, 20) * 2, 0, 1)
    w = np.clip(2 * _blur(Hm, max(12.0, 0.05 * _hole_radius(Hm))), 0, 1)
    return np.clip(A * (1 - w) + base * w, 0, 255).astype(np.uint8)


def fill_looks_ok(face: np.ndarray, hole_face: np.ndarray):
    """(ok, reason). Loose on purpose: it only has to catch a dead fill, so it can't wrongly reject a good one
    (an earlier tight 'darkest pixels' limit did exactly that and sent Google's blocky tile out for three days)."""
    r = _hole_radius(hole_face)
    if r < 8:
        return True, "tiny hole"
    R = _radius_grid()
    f = face.astype(np.float32)
    if not np.isfinite(f).all():
        return False, "non-finite pixels"
    dark_frac = float((f[R < 0.9 * r] < DARK_LEVEL).mean())
    if dark_frac > MAX_DARK_FRAC:
        return False, f"{100 * dark_frac:.0f}% of the hole is near-black"
    step = float(f[(R > r - 12) & (R < r - 2)].mean() - f[(R > r + 2) & (R < r + 12)].mean())
    if abs(step) > MAX_EDGE_STEP:
        return False, f"edge step {step:+.0f}"
    return True, "ok"


# ------------------------------------------------------------------ last published fallback

def fetch_previous_faces(base_url: str, log=print, get=requests.get, timeout: float = 15.0) -> dict:
    """The polar faces (pz, nz) of the version currently published at base_url (its root.json points at the
    tiles; this runs before this run replaces them), decoded from the DDS tiles. {} if unreachable -- local runs
    point at localhost, so this simply returns {} there."""
    try:
        root = get(f"{base_url.rstrip('/')}/root.json", timeout=timeout)
        root.raise_for_status()
        tiles = root.json()["baseUrl"]
        out = {}
        for name, idx in (("pz", 4), ("nz", 5)):
            face = np.zeros((FACE_SIZE, FACE_SIZE), np.uint8)
            for c in range(4):
                for r in range(4):
                    resp = get(f"{tiles}{idx}_2_{c}_{r}.dds", timeout=timeout)
                    resp.raise_for_status()
                    face[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = decode_dds_tile(resp.content)[..., 0]
            out[name] = face
        return out
    except Exception as exc:                                      # network, JSON, decode: all mean "not available"
        log(f"previous published polar tiles unavailable: {exc}")
        return {}


def _reuse_previous(photo_face: np.ndarray, hole_face: np.ndarray, previous: np.ndarray) -> np.ndarray:
    w = np.clip(2 * _blur(hole_face, max(12.0, 0.05 * _hole_radius(hole_face))), 0, 1)
    return np.clip(photo_face * (1 - w) + previous.astype(np.float32) * w, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ driver

def _gray3(a2d: np.ndarray) -> np.ndarray:
    return np.stack([a2d] * 3, axis=-1)


def _fix_pole(kind: str, face: str, hole_eq: np.ndarray, plain: np.ndarray, get_previous, log, fade_radius: int):
    """Returns (uint8 face or None, report). None means leave the plain face (every step failed)."""
    rep = {"hole": True}
    A = plain[..., 0].astype(np.float32)
    Hm = grow_face_hole(_face(hole_eq.astype(np.float32) * 255, face) / 255.0, A)
    rep["radius"] = round(_hole_radius(Hm), 1)
    # 1. the fixed-texture fill
    try:
        cand = fill_hole(A, Hm, face)
        ok, reason = fill_looks_ok(cand, Hm)
        log(f"{kind}: texture fill: {reason}")
        if ok:
            rep["method"] = "fixed texture fill"
            return cand, rep
        rep["rejected"] = reason
    except Exception as exc:
        log(f"{kind}: texture fill raised {exc!r}")
        rep["error"] = repr(exc)
    # 2. the previously published polar tiles
    try:
        prev = (get_previous() if get_previous else {}).get(face)
        if prev is not None:
            cand = _reuse_previous(A, Hm, prev)
            ok, reason = fill_looks_ok(cand, Hm)
            log(f"{kind}: last published tiles: {reason}")
            if ok:
                rep["method"] = "last published"
                return cand, rep
    except Exception as exc:
        log(f"{kind}: last published fallback raised {exc!r}")
    # 3. fade the hole to "no cloud"
    try:
        rep["method"] = "fade to empty"
        return fade_to_hole(plain, Hm, fade_radius)[..., 0], rep
    except Exception as exc:
        log(f"{kind}: fade raised {exc!r}")
    rep["method"] = "plain (error)"
    return None, rep


def apply_polar_fixes(faces: dict, cloud_gray: np.ndarray, log=print, get_previous=None, fade_radius: int = 40):
    """Returns (faces, report). `faces` maps name -> HxWx3 uint8 cube faces built from cloud_gray; only pz and nz
    can change. Never raises: whatever goes wrong at a pole, the other four faces are returned exactly as given.
    get_previous: optional callable returning {"pz": 2-D uint8, "nz": ...}, only called if a fill fails."""
    faces = dict(faces)
    report = {}
    for kind, face, measure in (("arctic", "pz", north_hole), ("antarctic", "nz", south_hole)):
        rep = report[kind] = {"hole": False}
        try:
            hole = measure(cloud_gray)
            if not hole.any():
                continue
            out, rep2 = _fix_pole(kind, face, hole, faces[face], get_previous, log, fade_radius)
            report[kind] = rep2
            if out is not None:
                faces[face] = _gray3(out)
            log(f"{kind}: {rep2['method']}")
        except Exception as exc:                                  # last line of defence: leave this pole plain
            log(f"{kind}: polar fix failed, leaving the plain face: {exc!r}")
            report[kind] = {"hole": True, "method": "plain (error)", "error": repr(exc)}
    return faces, report
