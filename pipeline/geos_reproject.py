"""Reproject a GOES-R ABI full-disk image (satellite fixed-grid view) into
equirectangular lat/lon, using the standard navigation formulas from the
GOES-R Product User Guide (the same math used by Satpy/goes2go/etc.).

Unlike Terra/Aqua/VIIRS, a geostationary satellite's full-disk image is a
single continuous stare -- no orbital-swath stitching -- so it has no
granule-to-granule striping by construction.
"""
import numpy as np

# WGS84 / GRS80 ellipsoid + standard GOES-R ABI fixed-grid parameters
R_EQ = 6378137.0
R_POL = 6356752.31414
H = 42164160.0  # perspective point height from Earth's center, meters
E2 = 1 - (R_POL ** 2) / (R_EQ ** 2)

# Standard full-disk scan angle extent (radians), same for GOES-East/West
SCAN_LIMIT = 0.151858


def lonlat_to_scan_angle(lon, lat, sat_lon_deg):
    """Forward navigation: geodetic lon/lat (radians) -> ABI scan angle x,y.

    Returns (x, y, visible) where `visible` is False for points on the far
    side of the Earth the satellite can't actually see.
    """
    sat_lon = np.radians(sat_lon_deg)
    phi_c = np.arctan((R_POL ** 2 / R_EQ ** 2) * np.tan(lat))
    rc = R_POL / np.sqrt(1 - (E2 * np.cos(phi_c) ** 2))

    sx = H - rc * np.cos(phi_c) * np.cos(lon - sat_lon)
    sy = -rc * np.cos(phi_c) * np.sin(lon - sat_lon)
    sz = rc * np.sin(phi_c)

    visible = (H * (H - sx)) > (sy ** 2 + (R_EQ ** 2 / R_POL ** 2) * sz ** 2)

    dist = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    x = np.arcsin(np.clip(-sy / dist, -1, 1))
    y = np.arctan(sz / sx)
    return x, y, visible


def reproject_fulldisk_to_equirect(img: np.ndarray, sat_lon_deg: float,
                                    out_width: int, out_height: int,
                                    lon_range=(-180, 180), lat_range=(90, -90)) -> tuple[np.ndarray, np.ndarray]:
    """img: square full-disk image (H,W,3), assumed to exactly span
    [-SCAN_LIMIT, SCAN_LIMIT] in both scan-angle axes.

    Returns (equirect_rgb, valid_mask) sized (out_height, out_width, ...).
    """
    lons = np.linspace(np.radians(lon_range[0]), np.radians(lon_range[1]), out_width, endpoint=False)
    lats = np.linspace(np.radians(lat_range[0]), np.radians(lat_range[1]), out_height, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    x, y, visible = lonlat_to_scan_angle(lon_grid, lat_grid, sat_lon_deg)

    src_h, src_w = img.shape[0], img.shape[1]
    # x maps to column, y maps to row (row 0 = top = +y in ABI convention)
    col = (x + SCAN_LIMIT) / (2 * SCAN_LIMIT) * src_w
    row = (SCAN_LIMIT - y) / (2 * SCAN_LIMIT) * src_h

    in_bounds = (col >= 0) & (col < src_w) & (row >= 0) & (row < src_h)
    valid = visible & in_bounds

    col_c = np.clip(col, 0, src_w - 1).astype(np.int64)
    row_c = np.clip(row, 0, src_h - 1).astype(np.int64)

    out = img[row_c, col_c]
    out = np.where(valid[..., None], out, 0)
    return out.astype(np.uint8), valid.astype(np.float32)
