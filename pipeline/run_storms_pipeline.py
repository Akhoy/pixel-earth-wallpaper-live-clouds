"""Fetch recent lightning flashes (GOES-East + GOES-West GLM) and write them
out in the app's exact storm_locations protobuf wire format.

Usage: venv/bin/python3 pipeline/run_storms_pipeline.py [--minutes 5] [--out output/latest]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from glm_lightning import fetch_recent_flashes
from storm_protobuf import encode_storm_locations

ROOT = Path(__file__).parent.parent
DEFAULT_OUT = ROOT / "output" / "latest"


def log(msg):
    print(f"[storms] {msg}", flush=True)


def build(minutes: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    all_flashes = []
    for sat in ["goes19", "goes18"]:
        log(f"fetching last {minutes} min of flashes from {sat} ...")
        flashes = fetch_recent_flashes(sat, minutes=minutes)
        log(f"  -> {len(flashes)} flashes")
        all_flashes.extend(flashes)

    log(f"total flashes: {len(all_flashes)} (Americas/Atlantic/Pacific only -- "
        f"no source yet for Asia-Pacific/Indian Ocean)")

    data = encode_storm_locations(all_flashes)
    out_file = out_dir / "storm_locations"
    out_file.write_bytes(data)
    log(f"wrote {len(data)} bytes -> {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()
    build(args.minutes, Path(args.out))
