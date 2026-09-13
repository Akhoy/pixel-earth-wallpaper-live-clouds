"""Build StormsProvider's storm_locations protobuf from live data:
  - GOES-19/18 GLM (Americas, Atlantic, Pacific) -- direct flash lat/lon
  - EUMETSAT MTG-I2 Lightning Imager AFA raster (Europe/Africa/Atlantic)
  - Xweather free-tier spot checks (India/Asia-Pacific cities, best-effort)

Nearby flashes are clustered into storm-cell centroids (matching the
sparser look of Google's original ~90-point global dataset) before being
encoded into the app's exact protobuf wire format.
"""
from __future__ import annotations
import struct
import requests
import numpy as np
from io import BytesIO
from scipy import ndimage
from PIL import Image

GLM_BUCKETS = {"goes19": "noaa-goes19", "goes18": "noaa-goes18"}


def _list_s3(bucket: str, prefix: str, delimiter: str = "", max_keys: int = 1000):
    params = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
    if delimiter:
        params["delimiter"] = delimiter
    resp = requests.get(f"https://{bucket}.s3.amazonaws.com/", params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def _latest_glm_key(bucket: str, year: int, doy: int) -> str | None:
    import re
    for hour in range(23, -1, -1):
        xml = _list_s3(bucket, f"GLM-L2-LCFA/{year}/{doy:03d}/{hour:02d}/")
        keys = re.findall(r"<Key>([^<]+)</Key>", xml)
        if keys:
            return sorted(keys)[-1]
    return None


def fetch_glm_flashes(satellite: str, year: int, doy: int) -> np.ndarray:
    """Returns Nx2 array of (lat, lon) for the most recent ~20s GLM file."""
    import netCDF4
    bucket = GLM_BUCKETS[satellite]
    key = _latest_glm_key(bucket, year, doy)
    if key is None:
        return np.empty((0, 2))
    resp = requests.get(f"https://{bucket}.s3.amazonaws.com/{key}", timeout=30)
    resp.raise_for_status()
    ds = netCDF4.Dataset("inmemory.nc", memory=resp.content)
    lat = np.array(ds.variables["flash_lat"][:])
    lon = np.array(ds.variables["flash_lon"][:])
    return np.stack([lat, lon], axis=1)


def fetch_mtg_li_afa_points(threshold: int = 40) -> np.ndarray:
    """Fetch MTG Lightning Imager's Accumulated Flash Area raster and
    extract lat/lon centroids of connected bright (flash) regions."""
    bbox = (-77, -77, 77, 77)  # minx,miny,maxx,maxy matching msg_fes:rgb_natural extent
    width = height = 1024
    url = "https://view.eumetsat.int/geoserver/wms"
    params = {
        "service": "WMS", "version": "1.1.1", "request": "GetMap",
        "layers": "mtg_fd:li_afa", "srs": "EPSG:4326",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "width": str(width), "height": str(height), "format": "image/png",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    img = np.array(Image.open(BytesIO(resp.content)).convert("L"))

    mask = img > threshold
    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.empty((0, 2))
    centroids = ndimage.center_of_mass(mask, labeled, range(1, n + 1))

    points = []
    for cy, cx in centroids:
        lon = bbox[0] + (cx / width) * (bbox[2] - bbox[0])
        lat = bbox[3] - (cy / height) * (bbox[3] - bbox[1])
        points.append((lat, lon))
    return np.array(points) if points else np.empty((0, 2))


# (query, longitude) -- longitude is used to approximate local solar time so
# we only spend API budget on cities that are currently in the dark (this
# feature only renders lightning glow on the night side of the globe anyway).
XWEATHER_SPOT_CITIES = [
    # India: major metros + state capitals + high-thunderstorm-frequency regions
    ("mumbai,in", 72.88), ("delhi,in", 77.10), ("bengaluru,in", 77.59), ("hyderabad,in", 78.49), ("chennai,in", 80.27),
    ("kolkata,in", 88.36), ("pune,in", 73.86), ("ahmedabad,in", 72.57), ("jaipur,in", 75.79), ("lucknow,in", 80.95),
    ("kanpur,in", 80.35), ("nagpur,in", 79.09), ("indore,in", 75.86), ("thane,in", 72.97), ("bhopal,in", 77.41),
    ("visakhapatnam,in", 83.22), ("patna,in", 85.14), ("vadodara,in", 73.18), ("ghaziabad,in", 77.45), ("ludhiana,in", 75.86),
    ("agra,in", 78.01), ("nashik,in", 73.79), ("faridabad,in", 77.31), ("meerut,in", 77.71), ("rajkot,in", 70.80),
    ("varanasi,in", 82.98), ("srinagar,in", 74.80), ("amritsar,in", 74.87), ("guwahati,in", 91.75), ("bhubaneswar,in", 85.83),
    ("ranchi,in", 85.33), ("raipur,in", 81.63), ("jodhpur,in", 73.02), ("chandigarh,in", 76.78), ("gwalior,in", 78.18),
    ("vijayawada,in", 80.65), ("coimbatore,in", 76.96), ("madurai,in", 78.12), ("kochi,in", 76.27), ("thiruvananthapuram,in", 76.94),
    ("imphal,in", 93.94), ("shillong,in", 91.89), ("agartala,in", 91.28), ("dehradun,in", 78.03), ("shimla,in", 77.17),
    # Other South Asia
    ("karachi,pk", 67.03), ("lahore,pk", 74.36), ("islamabad,pk", 73.06), ("peshawar,pk", 71.58), ("multan,pk", 71.47),
    ("dhaka,bd", 90.41), ("chittagong,bd", 91.80), ("sylhet,bd", 91.87),
    ("colombo,lk", 79.86), ("kandy,lk", 80.63),
    ("kathmandu,np", 85.32), ("pokhara,np", 83.99),
    ("thimphu,bt", 89.63),
    ("kabul,af", 69.21), ("kandahar,af", 65.71),
    ("male,mv", 73.51),
    # Southeast Asia
    ("jakarta,id", 106.85), ("manila,ph", 120.98), ("bangkok,th", 100.50), ("singapore,sg", 103.82), ("kuala lumpur,my", 101.69),
    ("ho chi minh city,vn", 106.63), ("hanoi,vn", 105.85), ("yangon,mm", 96.20),
    # Japan
    ("tokyo,jp", 139.69), ("osaka,jp", 135.50), ("nagoya,jp", 136.91), ("yokohama,jp", 139.64), ("sapporo,jp", 141.35),
    ("fukuoka,jp", 130.42), ("sendai,jp", 140.87), ("hiroshima,jp", 132.46), ("naha,jp", 127.68), ("kyoto,jp", 135.77),
    # China
    ("beijing,cn", 116.41), ("shanghai,cn", 121.47), ("guangzhou,cn", 113.26), ("shenzhen,cn", 114.06), ("chengdu,cn", 104.07),
    ("chongqing,cn", 106.55), ("wuhan,cn", 114.30), ("xian,cn", 108.95), ("nanjing,cn", 118.80), ("hangzhou,cn", 120.15),
    ("tianjin,cn", 117.20), ("shenyang,cn", 123.43), ("qingdao,cn", 120.38), ("kunming,cn", 102.71), ("harbin,cn", 126.53),
    ("xiamen,cn", 118.09), ("hong kong,hk", 114.17), ("macau,mo", 113.55),
    # Australia / New Zealand
    ("sydney,au", 151.21), ("melbourne,au", 144.96), ("brisbane,au", 153.03), ("perth,au", 115.86), ("adelaide,au", 138.60),
    ("darwin,au", 130.84), ("cairns,au", 145.77), ("gold coast,au", 153.43), ("canberra,au", 149.13),
    ("auckland,nz", 174.76), ("wellington,nz", 174.78), ("christchurch,nz", 172.64),
]


def _is_local_night(longitude: float, utc_now) -> bool:
    """Approximate local solar time from longitude (15 deg/hour from UTC) --
    astronomically matches how the app's own day/night terminator works,
    rather than political timezones. Night = 19:00-05:00 local solar time."""
    utc_hour = utc_now.hour + utc_now.minute / 60.0
    local_hour = (utc_hour + longitude / 15.0) % 24.0
    return local_hour >= 19.0 or local_hour < 5.0


def fetch_xweather_spot_points(client_id: str, client_secret: str, utc_now=None) -> np.ndarray:
    from datetime import datetime, timezone
    utc_now = utc_now or datetime.now(timezone.utc)

    night_cities = [city for city, lon in XWEATHER_SPOT_CITIES if _is_local_night(lon, utc_now)]

    points = []
    for city in night_cities:
        try:
            resp = requests.get(
                "https://data.api.xweather.com/lightning/flash",
                params={"p": city, "limit": 20, "client_id": client_id, "client_secret": client_secret},
                timeout=15,
            )
            data = resp.json()
            if data.get("success"):
                for ob in data.get("response", []):
                    loc = ob.get("loc", {})
                    if "lat" in loc and "long" in loc:
                        points.append((loc["lat"], loc["long"]))
        except Exception:
            continue
    return np.array(points) if points else np.empty((0, 2))


def cluster_points(points: np.ndarray, radius_km: float = 80.0) -> np.ndarray:
    """Grid-based clustering: snap points to a radius_km grid, average
    within each cell. Cheap approximation of DBSCAN, fine for this density."""
    if len(points) == 0:
        return points
    deg_per_km = 1.0 / 111.0  # rough, fine for clustering purposes
    cell = radius_km * deg_per_km
    keys = np.round(points / cell).astype(np.int64)
    clusters = {}
    for pt, key in zip(points, map(tuple, keys)):
        clusters.setdefault(key, []).append(pt)
    return np.array([np.mean(v, axis=0) for v in clusters.values()])


# --- protobuf encoding: StormLocations { repeated LatLng locations = 1; }
#     LatLng { float lat_deg = 1; float lng_deg = 2; } ---

def _encode_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7f
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _encode_latlng(lat: float, lng: float) -> bytes:
    # field 1, wire type 5 (fixed32): tag = (1<<3)|5 = 0x0d
    out = bytes([0x0d]) + struct.pack("<f", lat)
    out += bytes([0x15]) + struct.pack("<f", lng)  # field 2, wire type 5: tag=(2<<3)|5=0x15
    return out


def encode_storm_locations(points: np.ndarray) -> bytes:
    out = bytearray()
    for lat, lng in points:
        entry = _encode_latlng(float(lat), float(lng))
        # field 1, wire type 2 (length-delimited): tag = (1<<3)|2 = 0x0a
        out += bytes([0x0a]) + _encode_varint(len(entry)) + entry
    return bytes(out)
