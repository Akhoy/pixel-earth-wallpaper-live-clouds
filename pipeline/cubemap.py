"""Equirectangular (plate carree) -> 6-face cubemap reprojection.

Uses the standard OpenGL cube-map face/direction convention (the same one
libGDX's Cubemap / GL_TEXTURE_CUBE_MAP_* targets use), so faces come out
seamless and correctly oriented relative to each other. Face order matches
this app's FACE_NAMES: px, nx, py, ny, pz, nz.

Equirect image assumed: column 0 = lon -180, column W = lon +180;
row 0 = lat +90 (north pole), row H = lat -90 (south pole).

IMPORTANT -- the cloud cubemap is NOT sampled with the sphere's raw normal.
earth.frag samples it as `textureCube(cloudMap, vCloudNormal)`, where
earth.vert builds vCloudNormal as `(vec4(a_normal,1.) * cloudTransformMatrix)`.
That is a row-vector product, i.e. transpose(M) * n, and EarthShaderProvider
builds M as identity.rotate(X,-90).rotate(Z,-90). Working it through,
transpose(M) = Rz(+90)*Rx(+90), so the lookup direction is

    d = (n.z, n.x, n.y)

which lands in a Z-polar frame. Two independent checks confirm it:
  * decoding the shipped dayMap-*.ktx cubemap and reprojecting it back to
    equirect only yields a correct (non-mirrored) world map under this rule;
  * the app's own storm placement (LightningStorms) converts lat/lng with
    x=cos(lng)cos(lat), y=sin(lng)cos(lat), z=sin(lat) -- the same frame,
    since stormsMap is sampled with that very same vCloudNormal.

So for a cube direction d the geography is lat = asin(d.z), lon = atan2(d.y, d.x):
+Z is the north pole and longitude runs from +X toward +Y.
"""
import numpy as np

FACE_ORDER = ["px", "nx", "py", "ny", "pz", "nz"]


def _face_direction_grid(face: str, size: int) -> np.ndarray:
    """Returns (size,size,3) unit direction vectors for each texel of a face."""
    i = (np.arange(size) + 0.5) / size * 2.0 - 1.0  # sc, columns
    j = (np.arange(size) + 0.5) / size * 2.0 - 1.0  # tc, rows
    sc, tc = np.meshgrid(i, j)  # sc varies along x (cols), tc along y (rows)

    if face == "px":
        x, y, z = np.ones_like(sc), -tc, -sc
    elif face == "nx":
        x, y, z = -np.ones_like(sc), -tc, sc
    elif face == "py":
        x, y, z = sc, np.ones_like(sc), tc
    elif face == "ny":
        x, y, z = sc, -np.ones_like(sc), -tc
    elif face == "pz":
        x, y, z = sc, -tc, np.ones_like(sc)
    elif face == "nz":
        x, y, z = -sc, -tc, -np.ones_like(sc)
    else:
        raise ValueError(face)

    d = np.stack([x, y, z], axis=-1)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return d


def direction_to_lonlat(d: np.ndarray):
    """Cube-map lookup direction -> geographic lon/lat (radians).

    See the module docstring: the cloud cubemap is sampled through
    cloudTransformMatrix, which puts the lookup in a Z-polar frame.
    """
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lat = np.arcsin(np.clip(z, -1, 1))              # -pi/2..pi/2, +Z = north pole
    lon = np.arctan2(y, x)                           # -pi..pi, from +X toward +Y
    return lon, lat


def direction_to_lonlat_raw(d: np.ndarray):
    """Cube-map lookup direction -> geographic lon/lat (radians), for the
    dayMap/nightMap textures.

    Unlike the cloud cubemap, earth.vert samples these with the raw sphere
    normal (`textureCube(dayMap, N)`/`nightMap`, no cloudTransformMatrix),
    a Y-polar frame: N = (cos(lat)sin(lon), sin(lat), cos(lat)cos(lon)).
    Confirmed by decoding the shipped dayMap/nightMap KTX cubemaps and
    reprojecting under this rule -> correct, non-mirrored world map with
    recognisable coastlines (see pipeline/preview_globe.py, which also
    samples day/night with the raw normal directly).
    """
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lat = np.arcsin(np.clip(y, -1, 1))
    lon = np.arctan2(x, z)
    return lon, lat


def sample_equirect(img: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinear-sample an equirect image (H,W,C) at given lon/lat radians."""
    h, w = img.shape[0], img.shape[1]
    u = (lon + np.pi) / (2 * np.pi) * w - 0.5
    v = (np.pi / 2 - lat) / np.pi * h - 0.5

    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1

    fu = (u - u0)[..., None]
    fv = (v - v0)[..., None]

    u0m = np.mod(u0, w)
    u1m = np.mod(u1, w)
    v0c = np.clip(v0, 0, h - 1)
    v1c = np.clip(v1, 0, h - 1)

    c00 = img[v0c, u0m].astype(np.float32)
    c01 = img[v0c, u1m].astype(np.float32)
    c10 = img[v1c, u0m].astype(np.float32)
    c11 = img[v1c, u1m].astype(np.float32)

    top = c00 * (1 - fu) + c01 * fu
    bot = c10 * (1 - fu) + c11 * fu
    return top * (1 - fv) + bot * fv


def equirect_to_cubemap(img: np.ndarray, face_size: int, frame: str = "cloud") -> dict:
    """Returns {face_name: (face_size,face_size,C) array} matching img dtype range.

    frame="cloud" (default): the cloud cubemap's Z-polar sampling convention.
    frame="raw": dayMap/nightMap's Y-polar convention (raw sphere normal).
    """
    to_lonlat = direction_to_lonlat if frame == "cloud" else direction_to_lonlat_raw
    faces = {}
    for face in FACE_ORDER:
        d = _face_direction_grid(face, face_size)
        lon, lat = to_lonlat(d)
        sampled = sample_equirect(img, lon, lat)
        faces[face] = np.clip(sampled, 0, 255).astype(np.uint8)
    return faces
