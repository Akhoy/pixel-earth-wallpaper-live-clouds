"""Fetch true-color imagery from EUMETSAT's public View Service WMS
(view.eumetsat.int) for MTG-I1 and MSG-IODC.

No auth required. Unlike the GOES CDN images, the WMS already returns the
data reprojected to EPSG:4326 (plate carree) -- no GEOS fixed-grid
navigation math needed, just placing the returned crop onto the global
equirect canvas at the right pixel offset.

The service's time dimension uses `nearestValue`, but in practice an
arbitrary "now" timestamp can land between valid scans and 502. The robust
approach (and the one used here) is to read the layer's current `default`
time out of GetCapabilities first, then request exactly that time.
"""
import re
import numpy as np
import requests
from io import BytesIO
from PIL import Image

WMS_URL = "https://view.eumetsat.int/geoserver/wms"


def latest_layer_time(layer: str, timeout: int = 30) -> str:
    """Returns the ISO8601 timestamp of the most recent available scan for
    `layer`, read from the WMS GetCapabilities document."""
    # 1.3.0 capabilities use <Dimension default="...">; 1.1.1 uses a
    # separate <Extent default="..."> element instead -- request 1.3.0 so
    # the regex below has one consistent shape to parse.
    resp = requests.get(WMS_URL, params={
        "service": "WMS", "version": "1.3.0", "request": "GetCapabilities",
    }, timeout=timeout)
    resp.raise_for_status()
    xml = resp.text

    idx = xml.find(f"<Name>{layer}</Name>")
    if idx == -1:
        raise ValueError(f"layer {layer!r} not found in EUMETSAT capabilities")
    block = xml[idx:idx + 3000]
    m = re.search(r'<Dimension name="time"[^>]*default="([^"]+)"', block)
    if not m:
        raise ValueError(f"no time dimension default found for layer {layer!r}")
    return m.group(1)


def fetch_eumetsat_truecolor(layer: str, bbox: tuple, px_per_deg: float = 12.0,
                              timeout: int = 60) -> np.ndarray:
    """bbox = (minx, miny, maxx, maxy) in degrees. Returns (H,W,3) uint8 RGB
    covering exactly that lon/lat box, row 0 = north edge (maxy), col 0 =
    west edge (minx) -- standard WMS GetMap raster convention."""
    minx, miny, maxx, maxy = bbox
    width = max(1, round((maxx - minx) * px_per_deg))
    height = max(1, round((maxy - miny) * px_per_deg))

    time_str = latest_layer_time(layer, timeout=timeout)
    params = {
        "service": "WMS", "version": "1.1.1", "request": "GetMap",
        "layers": layer, "srs": "EPSG:4326",
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "width": str(width), "height": str(height),
        "format": "image/png", "time": time_str,
    }
    resp = requests.get(WMS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)
