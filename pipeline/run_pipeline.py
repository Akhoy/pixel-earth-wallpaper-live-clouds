"""End-to-end: fetch -> per-satellite cloud mask -> median-of-3 combine -> cubemap -> tile -> DDS -> root.json.

Usage: venv/bin/python3 pipeline/run_pipeline.py [--width 4096] [--out output/latest]
"""
import argparse
import json
import shutil
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from gibs import fetch_global_truecolor
from merge import darkest_of_n_composite, median_of_masks, fill_from_cache, inpaint_remaining_gaps, validity_mask
from cloud_mask import rgb_to_cloud_mask, fix_antarctic_cap
from cubemap import equirect_to_cubemap, FACE_ORDER
from dxt1 import encode_tile_to_dds

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "cache"
DEFAULT_OUT = ROOT / "output" / "latest"

FACE_SIZE = 1024
TILE_SIZE = 256
GRID = FACE_SIZE // TILE_SIZE  # 4


def log(msg):
    print(f"[pipeline] {msg}", flush=True)


# A fully-processed UTC day composites to roughly 87% coverage -- the missing
# share is inherent (slivers between adjacent orbit tracks, and polar night
# where there is no daylight to image). A day that is still being processed
# scores far below that, so this floor separates the two without tripping on
# the seasonal swing in how much polar night there is.
#
# It also covers the awkward case of running just after midnight UTC, where
# "yesterday" has technically ended but its final orbits are still inside the
# ~3.5h publishing delay. Coverage accrues at roughly 3.6%/hour, so such a day
# lands near 87 - 3.5*3.6 = ~75% -- below this floor, so it gets rejected in
# favour of the day before.
MIN_DAY_COVERAGE = 0.78


def _fetch_and_composite(day, width: int):
    """Fetch the day's three VIIRS mosaics and combine them into one cloud
    mask.

    The mask is derived from each satellite pass INDEPENDENTLY first
    (`rgb_to_cloud_mask` per source), then combined with a per-pixel median
    (`median_of_masks`) -- not by compositing the raw photos first. See
    `median_of_masks`'s docstring in merge.py: combining photos first
    (darkest-of-3) discards real, fast-evolving convective cloud whenever
    only one of the three ~30-100min-apart passes caught it, which
    median-of-masks does not.

    Also builds a darkest-of-3 RGB purely for the debug preview PNG (human
    sanity-check of what the raw composite looked like) -- it plays no part
    in deriving the published mask.

    Returns (cloud_gray, still_gap_mask, coverage, debug_rgb) where coverage
    is the fraction of the globe that ended up with real pixels.
    """
    sources = []
    for sat in ["noaa21", "noaa20", "snpp"]:
        log(f"fetching {sat} VIIRS true-color mosaic for {day} at width={width} ...")
        sources.append(fetch_global_truecolor(sat, day.isoformat(), width=width))

    log("deriving cloud mask per-satellite, combining via per-pixel median ...")
    masks = [rgb_to_cloud_mask(s) for s in sources]
    valids = [validity_mask(s) for s in sources]
    cloud_gray, still_gap = median_of_masks(masks, valids)
    coverage = 1.0 - float(still_gap.mean())
    log(f"{day}: {100*coverage:.1f}% covered ({100*(1-coverage):.1f}% gaps)")

    debug_rgb, _ = darkest_of_n_composite(sources)
    return cloud_gray, still_gap, coverage, debug_rgb


def _write_tileset(equirect_rgb: np.ndarray, dest_dir: Path) -> int:
    """Reproject an equirect RGB array (uint8, or float 0..1) to 6 cube faces
    in the cloudMap sampling convention (Z-polar, via cloudTransformMatrix --
    see cubemap.py) and write DDS tiles + debug face PNGs into dest_dir.
    Returns the tile count.
    """
    if equirect_rgb.dtype != np.uint8:
        equirect_rgb = np.clip(equirect_rgb * 255, 0, 255).astype(np.uint8)
    dest_dir.mkdir(parents=True, exist_ok=True)
    faces = equirect_to_cubemap(equirect_rgb, FACE_SIZE)
    face_index = {name: i for i, name in enumerate(FACE_ORDER)}
    tile_count = 0
    for name, face_arr in faces.items():
        idx = face_index[name]
        Image.fromarray(face_arr).save(dest_dir / f"_debug_face_{name}.png")
        for row in range(GRID):
            for col in range(GRID):
                tile = face_arr[row*TILE_SIZE:(row+1)*TILE_SIZE, col*TILE_SIZE:(col+1)*TILE_SIZE]
                dds_bytes = encode_tile_to_dds(tile)
                fname = f"{idx}_2_{col}_{row}.dds"
                (dest_dir / fname).write_bytes(dds_bytes)
                tile_count += 1
    return tile_count


def build(width: int, out_dir: Path, target_day=None, cache_dir: Path = CACHE_DIR, use_cache_fallback: bool = True,
          base_url_prefix: str = "http://localhost:8765", keep_versions: int = 5,
          min_coverage: float = MIN_DAY_COVERAGE):
    """out_dir is a STABLE directory: it holds root.json (always at the same
    URL) plus a versions/ subfolder. Each run publishes tiles into a NEW
    versions/<timestamp>/ folder and points root.json's baseUrl at it -- so
    the app's own "is baseUrl different from last time?" cache check
    correctly detects that new data is available (matching how Google's own
    root.json pointed at a rotating dated folder rather than a fixed path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    versions_dir = out_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = versions_dir / version_id
    version_dir.mkdir(parents=True, exist_ok=True)
    if target_day is None:
        # GIBS keys its daily mosaics by UTC day, and the current UTC day is
        # still filling in as orbits are processed, so step back to the last
        # completed one. This must be UTC rather than local time: from a UTC+
        # timezone (e.g. IST, +5:30) the local date is already the next day
        # for part of the night, so `date.today() - 1` would silently ask for
        # a UTC day that hasn't finished yet.
        #
        # Just step back one day here. Shortly after midnight UTC that day
        # will still be missing its last orbits (near-real-time imagery lands
        # ~3.5h after observation), but there is no need to guess a cutoff
        # hour for that -- the coverage check below measures what actually
        # came back and steps back again if it is short.
        target_day = datetime.now(timezone.utc).date() - timedelta(days=1)

    cloud_gray, still_gap, coverage, debug_rgb = _fetch_and_composite(target_day, width)

    # Since there is no "day complete" flag to ask for, check the result
    # instead: a finished UTC day lands around 87% covered (the rest is
    # unavoidable -- gaps between orbit tracks, plus polar night), whereas a
    # still-filling day comes in far lower. If we somehow picked a day that
    # hasn't finished, publishing it would put a half-empty globe on screen,
    # so drop back a day and use that if it's actually better.
    if coverage < min_coverage:
        fallback_day = target_day - timedelta(days=1)
        log(f"coverage {100*coverage:.1f}% is below the {100*min_coverage:.0f}% floor "
            f"(a complete day is ~87%) -- retrying with {fallback_day} ...")
        alt_gray, alt_gap, alt_coverage, alt_debug_rgb = _fetch_and_composite(fallback_day, width)
        if alt_coverage > coverage:
            log(f"using {fallback_day} instead ({100*alt_coverage:.1f}% covered)")
            target_day, cloud_gray, still_gap, coverage, debug_rgb = (
                fallback_day, alt_gray, alt_gap, alt_coverage, alt_debug_rgb)
        else:
            log(f"{fallback_day} is no better ({100*alt_coverage:.1f}%) -- keeping {target_day}")

    # fill_from_cache/inpaint_remaining_gaps operate on RGB; run the
    # single-channel mask through them as a replicated-3-channel "pseudo-RGB"
    # rather than duplicating that gap-filling logic for a 2D array.
    cache_file = cache_dir / "last_good_cloud_mask.npy"
    cached = np.load(cache_file) if (use_cache_fallback and cache_file.exists()) else None
    if cached is not None and cached.shape != cloud_gray.shape:
        cached = None

    pseudo = np.stack([cloud_gray] * 3, axis=-1)
    cached_pseudo = np.stack([cached] * 3, axis=-1) if cached is not None else None
    pseudo = fill_from_cache(pseudo, still_gap, cached_pseudo)

    # The cache is always a complete, previously-inpainted field (saved right
    # after inpainting below, every run) -- so if a cache existed at all,
    # fill_from_cache already blended every gapped pixel from it and nothing
    # remains to inpaint. Only a missing/incompatible cache leaves real gaps.
    # (Unlike the old RGB pipeline, we can't detect "still gapped" by
    # checking for near-black output here -- a real, legitimate "zero cloud"
    # reading is also black, so re-deriving gap status from pixel content
    # would misfire. Use the already-computed `still_gap` mask directly.)
    remaining_gap = still_gap if cached is None else np.zeros_like(still_gap)
    if remaining_gap.mean() > 0:
        log(f"inpainting {100*remaining_gap.mean():.2f}% still-gapped pixels ...")
        pseudo = inpaint_remaining_gaps(pseudo, remaining_gap)

    cloud_gray = pseudo[..., 0]
    np.save(cache_file, cloud_gray)
    Image.fromarray(debug_rgb).save(version_dir / "_debug_equirect_merged.png")

    # Publish a grayscale cloud-density mask as the live cloudMap texture
    # (CLOUDS_URL) -- this is what the stock, unmodified app shader expects
    # (a white cloud mask, sampled via .r and blended over the static bundled
    # dayMap/nightMap textures). App-side code and earth.frag are untouched;
    # all quality work happens here, server-side.
    cloud_gray = fix_antarctic_cap(cloud_gray)
    cloud_rgb = np.stack([cloud_gray] * 3, axis=-1)
    Image.fromarray(cloud_gray).save(version_dir / "_debug_cloud_mask.png")

    log("building live cloudMap texture from the mask ...")
    tile_count = _write_tileset(cloud_rgb, version_dir)
    log(f"wrote {tile_count} DDS tiles -> {version_dir}")

    # root.json stays at a STABLE url; only its baseUrl content changes each
    # run, which is what the app compares to detect "is there new data".
    base_url = f"{base_url_prefix}/versions/{version_id}/"
    (out_dir / "root.json").write_text(json.dumps({"baseUrl": base_url}))
    log(f"wrote root.json (baseUrl={base_url})")

    # retention: drop old version folders beyond keep_versions, oldest first
    all_versions = sorted(p for p in versions_dir.iterdir() if p.is_dir())
    for stale in all_versions[:-keep_versions] if keep_versions > 0 else []:
        shutil.rmtree(stale, ignore_errors=True)
        log(f"pruned old version: {stale.name}")

    log("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, defaults to yesterday")
    ap.add_argument("--cache-dir", type=str, default=str(CACHE_DIR))
    ap.add_argument("--no-cache-fallback", action="store_true")
    ap.add_argument("--base-url", type=str, default="http://localhost:8765",
                     help="public URL prefix this out_dir will be served from")
    ap.add_argument("--keep-versions", type=int, default=5)
    ap.add_argument("--min-coverage", type=float, default=MIN_DAY_COVERAGE,
                     help="fall back a day if the chosen day is covered less than this "
                          "(0-1; a complete day is ~0.87). 0 disables the check.")
    args = ap.parse_args()
    target = date.fromisoformat(args.date) if args.date else None
    build(args.width, Path(args.out), target_day=target, cache_dir=Path(args.cache_dir),
          use_cache_fallback=not args.no_cache_fallback, base_url_prefix=args.base_url,
          keep_versions=args.keep_versions, min_coverage=args.min_coverage)
