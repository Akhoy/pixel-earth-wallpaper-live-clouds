"""Fill the two polar no-data holes in the cloud cubemap.

VIIRS true-color needs daylight, so each pole is blank for part of the year
(the Arctic now, Antarctica around the equinoxes' winter side). Leaving the
hole empty shows the app's own dayMap through it, which at the Arctic is dark
open ocean -- a visible disc. Guessing a fill with cache/inpaint draws a fake
smeared ring. This module fills each pole on its cube face instead:

  Arctic  (pz): thermal infrared (VIIRS M15), which has data in polar night,
                tone-matched to the cloud mask and laid over the photo with a
                wide fade. It is a cosmetic fill: infrared reads cold ice and
                cold sea as cloud, so some of its white is not weather.
            When the infrared has no data (it only covers the dark side of the
            planet, so there is a gap of ~2 weeks around each equinox while the
            photo hole already exists) the Arctic uses the Antarctic method.
  Antarctic (nz): the hole is measured per day (it varies in size from ~55 px
                to ~165 px on a 1024 face, with a ring of dark half-lit
                pixels around it), then filled by spreading the surrounding
                ice inward, plus real ice texture from Google's stock tile.

Every fill is checked afterwards (check_face). If a check fails the next
safer method is tried: our own clouds plus Google's texture, our own clouds
alone, then Google's whole stock tile tone-matched and softly blended. Only if
even that fails (or its asset is missing) does the hole fade to empty.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_closing, binary_dilation, gaussian_filter, gaussian_filter1d, grey_closing,
                           distance_transform_edt, label, map_coordinates, maximum_filter, shift as nd_shift, zoom)

from cubemap import equirect_to_face
from gibs import IR_LAYERS, fetch_global_infrared
from merge import fade_to_hole, find_polar_hole, soften_dateline

FACE_SIZE = 1024
REF_WIDTH = 4096              # map width the pixel-sized constants below were tuned at
ASSETS = Path(__file__).parent / "assets"

# --- infrared completeness: a day still being processed has holes/wrong tone
IR_MIN_MEAN = 100.0          # a complete day averages ~113; a partial one ~75
IR_MAX_CAP_BLACK = 0.05      # share of black pixels in the top 10 degrees (a partial day had 10%; a small
                             # missing wedge at the pole -- 1.9% on 2025-12-21 -- is repaired, not rejected)
IR_MAX_DAYS_BACK = 5
IR_QA_RETRIES = 3            # older infrared days to try if the newest one's fill fails the check

# --- Arctic overlay (settings chosen by eye across 11 days, Oct 2026)
GAIN = 0.95                  # infrared scaled a little under the photo mask's level
FADE_IN_OFFSET, FADE_WIDTH = -14, 130   # full infrared out to (hole radius - 14) px, photo only past +130 px more
HAZE = 0.10
CAP, PROTECT = 0.35, 0.85    # merge strength over real photo / hold-back over dark water
DETAIL, DETAIL_SCALE = 0.45, 6
CORE, CORE_FEATHER = 62, 30  # bad-pixel zone at the pole that gets replaced
MERGE_STRENGTH = 0.8
PATCH_MAX_SCORE = 22         # worse than this -> smooth ring continuation instead of a donor patch

# --- Antarctic fill
SOUTH_FRINGE = 215           # half-lit pixels darker than this, touching the hole, join it
SOUTH_GROW_STEPS, SOUTH_MARGIN = 3, 8
DONOR_WEIGHT, DONOR_SCALE = 1.0, 30
SPREAD_DOWN = 4              # _spread_inward works at 1/4 resolution

# --- quality check (see check_face)
MIN_STEP_RADIUS_PX = 30      # holes smaller than this (Feb 2026: 12 px) only get the dark-pixel test
LARGE_HOLE_PX = 200          # winter-size holes (radius ~300 px) are judged by their edge, not centre-vs-ring
QA_LARGE = {"max_step": 25.0, "min_dark": 25.0}     # provisional, calibrated on winter 2025/26 in the harness
QA = {
    # measured over 23 days (Sept 15 - Oct 19 2026): south steps -2.2..+1.8, darkest 207+;
    # north steps +18..+31, darkest 75+. A broken fill (dark ring/disc) lands far outside both.
    "south": {"max_step": 8.0, "min_dark": 170.0},
    "north": {"max_step": 45.0, "min_dark": 50.0},
}


def _blur(a, s):
    return gaussian_filter(a.astype(np.float32), s, mode="nearest")


def _face(img2d: np.ndarray, name: str) -> np.ndarray:
    rgb = np.stack([np.clip(img2d, 0, 255).astype(np.uint8)] * 3, axis=-1)
    return equirect_to_face(rgb, FACE_SIZE, name)[..., 0].astype(np.float32)


def _radius_grid() -> np.ndarray:
    yy, xx = np.mgrid[0:FACE_SIZE, 0:FACE_SIZE]
    return np.hypot(xx - (FACE_SIZE - 1) / 2, yy - (FACE_SIZE - 1) / 2)


# ---------------------------------------------------------------- infrared

def ir_is_complete(ir: np.ndarray) -> bool:
    cap = ir[: int(ir.shape[0] * 10 / 180)]
    return float(ir.mean()) >= IR_MIN_MEAN and float((cap < 3).mean()) < IR_MAX_CAP_BLACK


def fill_ir_gaps(ir: np.ndarray) -> np.ndarray:
    """Repair small no-data wedges (black) in the polar cap of the infrared by spreading the valid pixels
    around them, so a missing sliver of a satellite pass doesn't become a dark streak in the fill."""
    h = ir.shape[0]
    cap = np.zeros(ir.shape, bool)
    cap[: int(h * 20 / 180)] = True                          # top 20 degrees only
    bad = (ir < 3) & cap
    if not bad.any():
        return ir
    valid = (~bad).astype(np.float32)
    out, done = ir.astype(np.float32) * valid, valid.copy()
    base = out.copy()
    for s in (4, 8, 16, 32, 64):
        den = _blur(valid, s)
        take = (done < 0.5) & (den > 0.02)
        base = np.where(take, _blur(ir * valid, s) / np.maximum(den, 1e-4), base)
        done = np.maximum(done, take.astype(np.float32))
    return np.where(bad, base, ir).astype(np.float32)


def fetch_polar_infrared(target_day, width: int, fetch=fetch_global_infrared, log=print, south: bool = False):
    """Newest complete infrared day at or before target_day, as (array, day),
    or None. Averages the satellites that answer, which also smooths the seams
    between their passes. south=True returns the map flipped top-to-bottom, so the
    south pole is on the top rows where the completeness/repair code looks."""
    for back in range(IR_MAX_DAYS_BACK + 1):
        day = target_day - timedelta(days=back)
        layers = []
        for sat in IR_LAYERS:
            try:
                layers.append(fetch(sat, day.isoformat(), width))
            except Exception as exc:  # network / GIBS errors: try what remains
                log(f"infrared {sat} {day}: {exc}")
        if not layers:
            continue
        ir = np.mean(layers, axis=0)
        if south:
            ir = ir[::-1]
        if ir_is_complete(ir):
            ir = fill_ir_gaps(soften_dateline(ir))              # same date-line seam as the photo mosaic
            if back:
                log(f"infrared for {target_day} incomplete -- using {day}")
            return ir, day
        log(f"infrared {day}: mean {ir.mean():.0f}, incomplete")
    return None


# ------------------------------------------------------------------ Arctic

def grow_face_hole(hole_face: np.ndarray) -> np.ndarray:
    """Grow a big winter hole outward over the dim terminator band.

    Just beyond the no-photo disc the sun is almost on the horizon, so the photo there is very dark
    (cloud-mask values ~30-100 where normal cloud is ~170) for a band ~0.115 of the hole radius wide
    (35 px at the Dec/Jun solstice). Left in, it shows as a grey ring around the fill. Small Sept/Oct
    holes (radius < ~150) are unchanged; the growth ramps in up to full size at radius 230."""
    r = float(np.sqrt((hole_face > 0.5).sum() / np.pi))
    margin = 0.115 * r * float(np.clip((r - 150) / 80, 0, 1))
    if margin < 1:
        return hole_face
    d = distance_transform_edt(hole_face < 0.5)
    return np.maximum(hole_face, (d <= margin).astype(np.float32))


def north_hole(cloud_gray: np.ndarray) -> np.ndarray:
    h, w = cloud_gray.shape
    sc = w / REF_WIDTH
    gap = (cloud_gray == 0).astype(np.float32)
    gap[int(30 / 180 * h):] = 0
    # also cover the invalid fringe at the rim
    return maximum_filter(find_polar_hole(gap, radius=max(2, round(20 * sc))), size=max(3, round(15 * sc)))


def arctic_overlay(photo_face: np.ndarray, hole_face: np.ndarray, ir: np.ndarray, cloud_gray: np.ndarray,
                   force_smooth: bool = False):
    """Returns (uint8 face, info). photo_face/hole_face are float32 2-D pz faces."""
    h = cloud_gray.shape[0]
    R = _radius_grid()
    # infrared -> cloud-mask scale, matched on the daylit band 58-70N
    band = slice(int((90 - 70) / 180 * h), int((90 - 58) / 180 * h))
    if float((cloud_gray[band] == 0).mean()) > 0.3:             # deep winter: the usual band has no photo
        rows = np.nonzero((cloud_gray[: h // 2] == 0).mean(axis=1) > 0.9)[0]
        edge = rows.max() + 1 if len(rows) else int(h * 20 / 180)    # row where the no-photo cap ends
        band = slice(max(0, edge + int(2 / 180 * h)), min(h // 2, edge + int(13 / 180 * h)))   # 2-13 deg outside it
    q = np.linspace(0, 100, 101)
    fill = np.interp(ir, np.percentile(ir[band], q), np.percentile(cloud_gray[band], q)).astype(np.float32)
    B = _face(fill, "pz")
    r = 2                                                       # fills thin dark pass-edge lines
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    B = grey_closing(B, footprint=(xx ** 2 + yy ** 2 <= r * r))

    # unreliable last degrees at the pole: copy real infrared texture from the
    # nearby spot that matches the ring around it (edge, brightness, contrast)
    ann, core = (R >= CORE) & (R < CORE + 12), R < CORE
    ann_mean, ann_std = B[ann].mean(), B[ann].std()
    best = None
    if not force_smooth:
        for k in range(24):
            ang = k * 2 * np.pi / 24
            for mag in (1.9 * CORE, 2.4 * CORE, 3.0 * CORE, 3.8 * CORE):
                shifted = nd_shift(B, (-mag * np.sin(ang), -mag * np.cos(ang)), order=1, mode="nearest")
                score = (np.abs(shifted[ann] - B[ann]).mean()
                         + 1.5 * abs(shifted[core].mean() - ann_mean)
                         + 0.5 * abs(shifted[core].std() - ann_std))
                if best is None or score < best[0]:
                    best = (score, shifted)
    used_patch = best is not None and best[0] <= PATCH_MAX_SCORE
    if used_patch:
        patch = best[1]
    else:                                                       # smooth continuation of the ring into the core
        nt, rm = 720, int(CORE + CORE_FEATHER + 30)
        rr, tt = np.meshgrid(np.arange(rm), np.arange(nt) / nt * 2 * np.pi, indexing="ij")
        c = (FACE_SIZE - 1) / 2
        P = map_coordinates(B, [c + rr * np.sin(tt), c + rr * np.cos(tt)], order=1, mode="nearest")
        ring = gaussian_filter1d(P[int(CORE):int(CORE) + 14].mean(axis=0), 10, mode="wrap")
        Pc = P.copy()
        for i in range(int(CORE)):
            f = (i / CORE) ** 2
            Pc[i] = ring * f + ring.mean() * (1 - f)
        rf = np.clip(R, 0, rm - 1.001)
        tf = (np.arctan2(*(np.mgrid[0:FACE_SIZE, 0:FACE_SIZE] - c)) % (2 * np.pi)) / (2 * np.pi) * nt
        patch = map_coordinates(Pc, [rf, tf], order=1, mode="wrap")
    wc = np.clip((CORE + CORE_FEATHER - R) / CORE_FEATHER, 0, 1)
    B = (B * (1 - wc) + patch * wc) * GAIN

    # lay the infrared over the photo: full inside FADE_IN px, photo only past FADE_OUT
    r_hole = float(np.sqrt((hole_face > 0.5).sum() / np.pi))
    fade_in = max(r_hole + FADE_IN_OFFSET, 60.0)               # scales with the hole: Oct 2026 r~184 gives 170/300
    fade_out = fade_in + FADE_WIDTH
    t = np.clip((fade_out - R) / (fade_out - fade_in), 0, 1)
    w = t * t * (3 - 2 * t) * MERGE_STRENGTH
    w_full = w.copy()
    Ap = photo_face * (1 - hole_face) + B * hole_face           # inside the hole the "photo" is the infrared
    Hs = np.clip(_blur(hole_face, 80) * 2, 0, 1)
    w = w * (CAP + (1 - CAP) * Hs)                               # weaker over real photo
    w = w * (1 - PROTECT * (1 - Ap / 255.0) * (1 - Hs))          # hold back over dark water
    mixed = Ap * (1 - w) + B * w
    mixed = mixed + DETAIL * w_full * (B - _blur(B, DETAIL_SCALE))   # the infrared's fine structure
    mixed = mixed + (255 - mixed) * HAZE * w                     # slight haze, like Google's polar tiles
    return np.clip(mixed, 0, 255).astype(np.uint8), {"patch": bool(used_patch),
                                                        "patch_score": None if best is None else round(float(best[0]), 1)}


# --------------------------------------------------------------- Antarctica

def south_hole(cloud_gray: np.ndarray) -> np.ndarray:
    """Measured per day: the solid no-data blob around the south pole, grown a
    few steps into the dark half-lit fringe, plus a margin."""
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


def north_hole_grown(cloud_gray: np.ndarray) -> np.ndarray:
    """south_hole's measuring (solid blob + dark fringe + margin) for the north pole."""
    return south_hole(cloud_gray[::-1])[::-1]


def _donor(face_name: str):
    path = ASSETS / f"google_{face_name}.png"
    return np.array(Image.open(path).convert("L")).astype(np.float32) if path.exists() else None


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


def fill_hole(photo_face: np.ndarray, hole_face: np.ndarray, donor_face: str | None = None) -> np.ndarray:
    """Spread the surrounding clouds/ice into the hole; optionally add texture
    from Google's stock tile for that pole (donor_face = 'pz' or 'nz')."""
    A, Hm = photo_face, hole_face
    valid = (Hm < 0.5).astype(np.float32)
    base = _spread_inward(A, valid)
    inside = np.clip(_blur(Hm, 6) * 2, 0, 1)
    base = base * (1 - inside) + _blur(base, 10) * inside        # smooth away any pyramid blockiness in the hole
    donor = _donor(donor_face) if donor_face else None
    ring = (Hm < 0.5) & (_blur(Hm, 40) > 0.02)
    if donor is not None and ring.any():
        gm, gs = donor[ring].mean(), donor[ring].std() + 1e-3     # tone-match Google's tile to the live ice
        tex = (donor - gm) / gs * min(A[ring].std(), gs) + A[ring].mean()
        tex = tex - _blur(tex, DONOR_SCALE)                       # keep texture only, not its low frequencies
        base = base + DONOR_WEIGHT * tex * np.clip(_blur(Hm, 20) * 2, 0, 1)
    w = np.clip(2 * _blur(Hm, 12), 0, 1)
    return np.clip(A * (1 - w) + base * w, 0, 255).astype(np.uint8)


def google_tile_fill(photo_face: np.ndarray, hole_face: np.ndarray, donor_face: str):
    """Last-resort fill: Google's whole stock tile for this pole, tone-matched
    (mean and contrast) to the live pixels around the hole and blended in over
    a wide soft edge. Shows April's clouds, not today's, but a soft blend of
    the wrong clouds is far less visible than an empty disc. None if the tile
    asset is missing."""
    donor = _donor(donor_face)
    # narrow ring just outside the hole's edge: a wide one reaches real ocean on big holes and skews the match
    ring = (hole_face < 0.5) & (_blur(hole_face, 14) > 0.08)
    if donor is None or not ring.any():
        return None
    gm, gs = donor[ring].mean(), donor[ring].std() + 1e-3
    tile = (donor - gm) / gs * min(photo_face[ring].std(), gs) + photo_face[ring].mean()   # never stretch contrast
    r = float(np.sqrt((hole_face > 0.5).sum() / np.pi))
    w = np.clip(2 * _blur(hole_face, max(30.0, 0.12 * r)), 0, 1)     # wider soft edge on a big hole
    return np.clip(photo_face * (1 - w) + tile * w, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- quality check

def check_face(face: np.ndarray, hole_face: np.ndarray, kind: str) -> dict:
    """Quality check of a filled pole.

    Small hole (Sept/Oct, radius < LARGE_HOLE_PX): brightness step between the filled centre and a ring just
    outside its edge, plus the darkest pixels -- a ring or dark disc lands far outside the limits.
    Big winter hole: it spans ocean and cloud as well as ice, so centre-vs-ring means little; instead check
    that there is no visible edge (mean just inside vs just outside the boundary) and no black disc."""
    area = float((hole_face > 0.5).sum())
    radius = float(np.sqrt(area / np.pi))
    if radius < 5:
        return {"ok": True, "radius": radius, "step": 0.0, "dark": 255.0, "reason": "no hole"}
    R = _radius_grid()
    f = face.astype(np.float32)
    dark = float(np.percentile(f[R < 0.9 * radius], 2))
    if radius < MIN_STEP_RADIUS_PX:                              # a dot of a hole: a step over a few pixels is noise
        step, lim = 0.0, QA[kind]
    elif radius < LARGE_HOLE_PX:
        # ring just outside the edge (not scaled with the radius: on big holes that would reach real ocean)
        step = float(f[R < 0.6 * radius].mean() - f[(R > radius + 10) & (R < radius + 35)].mean())
        # a bigger hole sits further from its ring, so some centre-vs-ring difference is natural (Sept 15 2026:
        # radius 184, step +7.6 against a flat limit of 8); the allowance grows with the radius past 120 px
        lim = dict(QA[kind], max_step=QA[kind]["max_step"] + 0.12 * max(0.0, radius - 120))
    else:
        step = float(f[(R > radius - 12) & (R < radius - 2)].mean() - f[(R > radius + 2) & (R < radius + 12)].mean())
        lim = QA_LARGE
    reason = "ok"
    if abs(step) > lim["max_step"]:
        reason = f"{'centre/ring' if radius < LARGE_HOLE_PX else 'edge'} step {step:+.1f} exceeds {lim['max_step']}"
    elif dark < lim["min_dark"]:
        reason = f"dark pixels {dark:.0f} below {lim['min_dark']}"
    return {"ok": reason == "ok", "radius": round(radius, 1), "step": round(step, 1), "dark": round(dark, 1),
            "reason": reason}


# ------------------------------------------------------------------ driver

def _gray3(a2d: np.ndarray) -> np.ndarray:
    return np.stack([a2d] * 3, axis=-1)


def _own_data_fills(A: np.ndarray, Hm: np.ndarray, face: str, kind: str, log, rep: dict, label: str):
    """Fills that need no infrared, best first; returns the accepted face or None (rep gets the method).

    Small hole: spread the surrounding clouds/ice inward plus Google's texture, then spread alone, then
    Google's whole tile. Big winter hole (radius >= LARGE_HOLE_PX): the surroundings are ocean and cloud
    while the pole is ice, so spreading them inward gets the tone wrong -- Google's tile goes first."""
    names = {"pz": ("cloud fill + Google texture", "cloud fill"), "nz": ("ice fill + Google texture", "ice fill")}[face]
    attempts = [(names[0], lambda: fill_hole(A, Hm, face)),
                (names[1], lambda: fill_hole(A, Hm, None)),
                ("Google tile", lambda: google_tile_fill(A, Hm, face))]
    if np.sqrt((Hm > 0.5).sum() / np.pi) >= LARGE_HOLE_PX:
        attempts = [attempts[2], attempts[0], attempts[1]]
    for method, make in attempts:
        cand = make()
        if cand is None:
            log(f"{label} {method}: asset missing")
            continue
        qa = check_face(cand, Hm, kind)
        log(f"{label} {method}: {qa['reason']}")
        if qa["ok"]:
            rep.update(method=method, qa=qa)
            return cand
        rep.setdefault("rejected", []).append({"method": method, **qa})
    return None


def _infrared_fill(A, Hm, cloud_gray, kind, target_day, width, fetch, log, rep, label):
    """Infrared overlay for one pole (kind 'north'/'south'), or None. The south pole is mirrored so the
    same routine serves both (nz == pz of the flipped map, mirrored left-right: verified exactly).
    Tries the newest complete infrared day; if its fill fails the quality check (a day still being
    finished can pass the completeness test and still have holes), tries the next older day."""
    south = kind == "south"
    flip = (lambda a: a[:, ::-1]) if south else (lambda a: a)
    cg = cloud_gray[::-1] if south else cloud_gray
    start = target_day
    for _ in range(IR_QA_RETRIES):
        got = fetch_polar_infrared(start, width, fetch, log, south=south)
        if got is None:
            rep.setdefault("no_infrared", "no complete infrared (layer only covers the dark side)")
            return None
        ir, ir_day = got
        for force_smooth in (False, True):
            cand, info = arctic_overlay(flip(A), flip(Hm), ir, cg, force_smooth)
            cand = flip(cand)
            qa = check_face(cand, Hm, kind)
            log(f"{label} {ir_day} {'smooth' if force_smooth else 'patch'} fill: {qa['reason']} {info}")
            if qa["ok"]:
                rep.update(method="infrared overlay", ir_day=str(ir_day), qa=qa, **info)
                return cand
            rep.setdefault("rejected", []).append({"ir_day": str(ir_day), "smooth": force_smooth, **qa})
        start = ir_day - timedelta(days=1)
    return None


def apply_polar_fixes(faces: dict, cloud_gray: np.ndarray, target_day, width: int, log=print,
                      use_ir: bool = True, fade_radius: int = 40, fetch=fetch_global_infrared):
    """Returns (faces, report). `faces` maps name -> HxWx3 uint8 cube faces
    built from cloud_gray; only pz and nz are touched."""
    faces = dict(faces)
    report = {}

    # ---- Arctic
    hole_n = north_hole(cloud_gray)
    rep = report["north"] = {"hole": bool(hole_n.any())}
    if hole_n.any():
        Hm = grow_face_hole(_face(hole_n * 255, "pz") / 255.0)
        A = faces["pz"][..., 0].astype(np.float32)
        out = None
        if not use_ir:
            rep["no_infrared"] = "disabled"
        else:
            out = _infrared_fill(A, Hm, cloud_gray, "north", target_day, width, fetch, log, rep, "arctic")
        if out is None:                                         # no usable infrared: same fills as Antarctica
            Hg = grow_face_hole(_face(north_hole_grown(cloud_gray).astype(np.float32) * 255, "pz") / 255.0)
            out = _own_data_fills(A, Hg, "pz", "north", log, rep, "arctic")
        if out is not None:
            faces["pz"] = _gray3(out)
        else:
            rep["method"] = "fade (fills failed check)"
            faces["pz"] = fade_to_hole(faces["pz"], Hm, fade_radius)
        log(f"arctic: {rep['method']}")

    # ---- Antarctica
    hole_s = south_hole(cloud_gray)
    rep = report["south"] = {"hole": bool(hole_s.any())}
    if hole_s.any():
        Hm = grow_face_hole(_face(hole_s.astype(np.float32) * 255, "nz") / 255.0)
        A = faces["nz"][..., 0].astype(np.float32)
        out = None
        if use_ir:                                              # the south pole is the dark side in Apr-Sep
            out = _infrared_fill(A, Hm, cloud_gray, "south", target_day, width, fetch, log, rep, "antarctic")
        if out is None:
            out = _own_data_fills(A, Hm, "nz", "south", log, rep, "antarctic")
        if out is not None:
            faces["nz"] = _gray3(out)
        else:
            rep["method"] = "fade (fills failed check)"
            faces["nz"] = fade_to_hole(faces["nz"], Hm, fade_radius)
        log(f"antarctic: {rep['method']}")
    return faces, report
