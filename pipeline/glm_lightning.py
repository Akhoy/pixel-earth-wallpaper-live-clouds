"""Fetch recent lightning flashes from NOAA GOES GLM (Geostationary Lightning
Mapper) public data on AWS Open Data, for both GOES-East and GOES-West.

Free, public, no auth. Files land roughly every 20 seconds; we pull the most
recent few minutes' worth per satellite to build a "currently active
lightning" snapshot, similar to how live lightning-map apps fade out old
strikes after a few minutes.
"""
import io
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import requests

BUCKETS = {
    "goes19": "noaa-goes19",  # GOES-East
    "goes18": "noaa-goes18",  # GOES-West
}
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _list_recent_keys(bucket: str, minutes: int) -> list[str]:
    now = datetime.now(timezone.utc)
    hours_to_check = sorted({(now - timedelta(minutes=m)) for m in range(0, minutes + 1, 5)},
                             key=lambda d: d)
    seen_prefixes = set()
    keys = []
    for dt in hours_to_check:
        prefix = f"GLM-L2-LCFA/{dt.year}/{dt.timetuple().tm_yday}/{dt.hour:02d}/"
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for contents in root.findall("s3:Contents", NS):
            key = contents.find("s3:Key", NS).text
            last_modified = contents.find("s3:LastModified", NS).text
            lm = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
            if (now - lm) <= timedelta(minutes=minutes):
                keys.append(key)
    return keys


def _extract_flashes(nc_bytes: bytes) -> list[tuple[float, float]]:
    with h5py.File(io.BytesIO(nc_bytes), "r") as f:
        lats = f["flash_lat"][:]
        lons = f["flash_lon"][:]
        quality = f["flash_quality_flag"][:]
    # quality_flag == 0 means "good" data per GLM product spec
    good = quality == 0
    return list(zip(lats[good].tolist(), lons[good].tolist()))


def fetch_recent_flashes(satellite: str, minutes: int = 5) -> list[tuple[float, float]]:
    bucket = BUCKETS[satellite]
    keys = _list_recent_keys(bucket, minutes)
    flashes = []
    for key in keys:
        url = f"https://{bucket}.s3.amazonaws.com/{key}"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            continue
        try:
            flashes.extend(_extract_flashes(resp.content))
        except Exception:
            continue
    return flashes
