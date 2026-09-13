"""Fetch the latest full-disk GeoColor composite for a GOES-R series
satellite from NOAA STAR's public CDN.

No auth, no S3 SDK needed, and no waiting on a daily mosaic: STAR publishes
a new full-disk GeoColor composite every ~10 minutes at a stable
`.../5424x5424.jpg` URL that always points at the latest scan.
"""
import numpy as np
import requests
from io import BytesIO
from PIL import Image

CDN_URL = "https://cdn.star.nesdis.noaa.gov/{sat_dir}/ABI/FD/GEOCOLOR/5424x5424.jpg"
SAT_DIRS = {"goes19": "GOES19", "goes18": "GOES18"}


def fetch_goes_fulldisk(satellite: str, timeout: int = 60) -> np.ndarray:
    """Returns the raw square full-disk image (H,W,3) uint8 RGB, still in
    the satellite's native fixed-grid view (not yet reprojected)."""
    url = CDN_URL.format(sat_dir=SAT_DIRS[satellite])
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)
