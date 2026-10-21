"""Fetch global daily true-color mosaics from NASA GIBS's snapshot API."""
import requests
import numpy as np
from PIL import Image
from io import BytesIO

SNAPSHOT_URL = "https://wvs.earthdata.nasa.gov/api/v1/snapshot"

LAYERS = {
    "terra": "MODIS_Terra_CorrectedReflectance_TrueColor",
    "aqua": "MODIS_Aqua_CorrectedReflectance_TrueColor",
    "noaa21": "VIIRS_NOAA21_CorrectedReflectance_TrueColor",
    "noaa20": "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
    "snpp": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
}

# Thermal-infrared (M15) layers. Unlike true color they have data in polar
# night, which is what polar.py uses to fill the Arctic no-photo hole.
# NOAA-21's layer returns blank, so only NOAA-20 and SNPP are used.
IR_LAYERS = {
    "noaa20": "VIIRS_NOAA20_DayNightBand_AtSensor_M15",
    "snpp": "VIIRS_SNPP_DayNightBand_AtSensor_M15",
}


def fetch_global_truecolor(satellite: str, date_str: str, width: int = 4096) -> np.ndarray:
    """date_str: 'YYYY-MM-DD'. Returns HxWx3 uint8 RGB equirect array."""
    height = width // 2
    params = {
        "REQUEST": "GetSnapshot",
        "LAYERS": LAYERS[satellite],
        "CRS": "EPSG:4326",
        "TIME": date_str,
        "WRAP": "DAY",
        "BBOX": "-90,-180,90,180",
        "FORMAT": "image/png",
        "WIDTH": str(width),
        "HEIGHT": str(height),
    }
    resp = requests.get(SNAPSHOT_URL, params=params, timeout=120)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)


def fetch_global_infrared(satellite: str, date_str: str, width: int = 4096) -> np.ndarray:
    """date_str: 'YYYY-MM-DD'. Returns HxW float32 equirect array: the blue
    channel of GIBS's false-colour M15 render (a monotone 'how cold' value,
    not calibrated temperature)."""
    params = {
        "REQUEST": "GetSnapshot", "LAYERS": IR_LAYERS[satellite], "CRS": "EPSG:4326",
        "TIME": date_str, "WRAP": "DAY", "BBOX": "-90,-180,90,180", "FORMAT": "image/png",
        "WIDTH": str(width), "HEIGHT": str(width // 2),
    }
    resp = requests.get(SNAPSHOT_URL, params=params, timeout=180)
    resp.raise_for_status()
    return np.array(Image.open(BytesIO(resp.content)).convert("RGB"))[..., 2].astype(np.float32)
