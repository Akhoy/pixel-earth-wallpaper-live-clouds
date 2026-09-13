"""End-to-end real-time global cloud composite from 4 geostationary
satellites (GOES-West, GOES-East, MTG-I1, MSG-IODC) -> weighted blend ->
cloud mask -> cubemap -> tile -> DDS -> root.json.

Same output format as pipeline/run_pipeline.py (the VIIRS/GIBS daily
pipeline) -- a stable root.json pointing at a versions/<timestamp>/ folder
of DDS tiles -- so this can be pointed at the same app config, or run
side by side while it's being evaluated.

Key difference from the VIIRS pipeline: every source here is a live,
independent look (updated every 10-15 min), not a stitched daily mosaic,
so there's no orbital-swath striping to remove and no "yesterday's data"
staleness -- but coverage tops out at the union of these 4 sources'
usable disks (see realtime_geo/sources.py for the exact gap: roughly
118.5E to 180, i.e. E/SE Asia, Japan, Australia, and the western Pacific
aren't covered by any source here yet).

Usage: venv/bin/python3 realtime_geo/run_realtime_pipeline.py [--width 4096] [--out output/latest_realtime]
"""
import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from sources import SOURCES, SOURCE_ORDER
from fetch_goes import fetch_goes_fulldisk
from fetch_eumetsat import fetch_eumetsat_truecolor
from reproject import reproject_goes_to_equirect, place_eumetsat_on_equirect
from merge_realtime import weighted_blend

from merge import fill_from_cache, inpaint_remaining_gaps, validity_mask  # noqa: E402
from cloud_mask import rgb_to_cloud_mask, fix_antarctic_cap  # noqa: E402
from cubemap import equirect_to_cubemap, FACE_ORDER  # noqa: E402
from dxt1 import encode_tile_to_dds  # noqa: E402

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "cache"
DEFAULT_OUT = ROOT / "output" / "latest_realtime"

FACE_SIZE = 1024
TILE_SIZE = 256
GRID = FACE_SIZE // TILE_SIZE  # 4


def log(msg):
    print(f"[realtime_geo] {msg}", flush=True)


def fetch_source_on_canvas(name: str, out_width: int, out_height: int, px_per_deg: float):
    cfg = SOURCES[name]
    if cfg["kind"] == "abi":
        img = fetch_goes_fulldisk(name)
        return reproject_goes_to_equirect(img, cfg["sat_lon"], out_width, out_height)
    elif cfg["kind"] == "eumetsat_wms":
        tile = fetch_eumetsat_truecolor(cfg["layer"], cfg["bbox"], px_per_deg=px_per_deg)
        return place_eumetsat_on_equirect(tile, cfg["bbox"], cfg["sat_lon"], out_width, out_height)
    else:
        raise ValueError(f"unknown source kind: {cfg['kind']}")


def build(width: int, out_dir: Path, cache_dir: Path = CACHE_DIR, use_cache_fallback: bool = True,
          base_url_prefix: str = "http://localhost:8765", keep_versions: int = 5,
          px_per_deg: float = 12.0):
    height = width // 2
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    versions_dir = out_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = versions_dir / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for name in SOURCE_ORDER:
        label = SOURCES[name]["label"]
        log(f"fetching {label} ({name}) ...")
        try:
            rgb, _valid, weight = fetch_source_on_canvas(name, width, height, px_per_deg)
            log(f"  -> ok, {100 * (weight > 0).mean():.1f}% of canvas covered with nonzero confidence")
        except Exception as exc:
            log(f"  -> FAILED ({exc!r}), treating as fully gapped for this run")
            traceback.print_exc()
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            weight = np.zeros((height, width), dtype=np.float32)
        entries.append((rgb, weight))

    log("weighted blend across all sources (tone-matched, confidence-weighted) ...")
    merged, still_gap = weighted_blend(entries, reference_index=0)
    log(f"gap coverage after blend (no source had confident data): {100 * still_gap.mean():.2f}%")

    cache_file = cache_dir / "last_good_equirect_realtime.npy"
    cached = np.load(cache_file) if (use_cache_fallback and cache_file.exists()) else None
    if cached is not None and cached.shape != merged.shape:
        cached = None
    merged = fill_from_cache(merged, still_gap, cached)

    remaining_gap = 1.0 - validity_mask(merged)
    if remaining_gap.mean() > 0:
        log(f"inpainting {100 * remaining_gap.mean():.2f}% still-gapped pixels ...")
        merged = inpaint_remaining_gaps(merged, remaining_gap)

    np.save(cache_file, merged)
    Image.fromarray(merged).save(version_dir / "_debug_equirect_merged.png")

    log("deriving cloud-density mask ...")
    # No destripe() here: a geostationary full-disk frame is a single
    # continuous stare, not stitched orbital swaths, so there's no
    # granule-to-granule banding to remove (see pipeline/geos_reproject.py).
    cloud_gray = rgb_to_cloud_mask(merged)
    cloud_gray = fix_antarctic_cap(cloud_gray)
    cloud_rgb = np.stack([cloud_gray] * 3, axis=-1)
    Image.fromarray(cloud_gray).save(version_dir / "_debug_cloud_mask.png")

    log("reprojecting to 6 cube faces ...")
    faces = equirect_to_cubemap(cloud_rgb, FACE_SIZE)

    face_index = {name: i for i, name in enumerate(FACE_ORDER)}
    tile_count = 0
    for name, face_arr in faces.items():
        idx = face_index[name]
        Image.fromarray(face_arr).save(version_dir / f"_debug_face_{name}.png")
        for row in range(GRID):
            for col in range(GRID):
                tile = face_arr[row * TILE_SIZE:(row + 1) * TILE_SIZE, col * TILE_SIZE:(col + 1) * TILE_SIZE]
                dds_bytes = encode_tile_to_dds(tile)
                fname = f"{idx}_2_{col}_{row}.dds"
                (version_dir / fname).write_bytes(dds_bytes)
                tile_count += 1
    log(f"wrote {tile_count} DDS tiles -> {version_dir}")

    base_url = f"{base_url_prefix}/versions/{version_id}/"
    root_json = {"baseUrl": base_url}
    (out_dir / "root.json").write_text(json.dumps(root_json))
    log(f"wrote root.json -> {out_dir / 'root.json'} (baseUrl={base_url})")

    all_versions = sorted(p for p in versions_dir.iterdir() if p.is_dir())
    for stale in all_versions[:-keep_versions] if keep_versions > 0 else []:
        shutil.rmtree(stale, ignore_errors=True)
        log(f"pruned old version: {stale.name}")

    log("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--cache-dir", type=str, default=str(CACHE_DIR))
    ap.add_argument("--no-cache-fallback", action="store_true")
    ap.add_argument("--base-url", type=str, default="http://localhost:8765",
                     help="public URL prefix this out_dir will be served from")
    ap.add_argument("--keep-versions", type=int, default=5)
    ap.add_argument("--px-per-deg", type=float, default=12.0,
                     help="resolution for EUMETSAT WMS fetches (MTG-I1 / MSG-IODC)")
    args = ap.parse_args()
    build(args.width, Path(args.out), cache_dir=Path(args.cache_dir),
          use_cache_fallback=not args.no_cache_fallback, base_url_prefix=args.base_url,
          keep_versions=args.keep_versions, px_per_deg=args.px_per_deg)
