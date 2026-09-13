"""End-to-end: GOES GLM + MTG LI + Xweather spot-checks -> clustered ->
StormLocations protobuf, matching the app's exact wire format.

Usage: venv/bin/python3 pipeline/run_storms.py --out output/latest/storm_locations
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from storms import (fetch_glm_flashes, fetch_mtg_li_afa_points, fetch_xweather_spot_points,
                     cluster_points, encode_storm_locations)


def log(msg):
    print(f"[storms] {msg}", flush=True)


def build(out_path: Path, xweather_id: str = None, xweather_secret: str = None,
          cluster_radius_km: float = 120.0):
    today = date.today()
    doy = today.timetuple().tm_yday
    year = today.year

    log("fetching GOES-19 (East) GLM ...")
    g19 = fetch_glm_flashes("goes19", year, doy)
    log(f"  {len(g19)} flashes")

    log("fetching GOES-18 (West) GLM ...")
    g18 = fetch_glm_flashes("goes18", year, doy)
    log(f"  {len(g18)} flashes")

    log("fetching MTG-I2 Lightning Imager (Europe/Africa/Atlantic) ...")
    mtg = fetch_mtg_li_afa_points()
    log(f"  {len(mtg)} flash-area centroids")

    sources = [g19, g18, mtg]

    if xweather_id and xweather_secret:
        log("fetching Xweather spot-checks (India/Asia-Pacific cities) ...")
        xw = fetch_xweather_spot_points(xweather_id, xweather_secret)
        log(f"  {len(xw)} points")
        sources.append(xw)
    else:
        log("no Xweather credentials given, skipping India/Asia-Pacific spot-checks")

    non_empty = [s for s in sources if len(s) > 0]
    all_points = np.concatenate(non_empty, axis=0) if non_empty else np.empty((0, 2))
    log(f"total raw points: {len(all_points)}")

    clustered = cluster_points(all_points, radius_km=cluster_radius_km)
    log(f"clustered to {len(clustered)} storm-cell points (radius={cluster_radius_km}km)")

    pb = encode_storm_locations(clustered)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pb)
    log(f"wrote {len(pb)} bytes -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="output/latest/storm_locations")
    ap.add_argument("--xweather-id", type=str, default=None)
    ap.add_argument("--xweather-secret", type=str, default=None)
    ap.add_argument("--cluster-radius-km", type=float, default=120.0)
    args = ap.parse_args()
    build(Path(args.out), args.xweather_id, args.xweather_secret, args.cluster_radius_km)
