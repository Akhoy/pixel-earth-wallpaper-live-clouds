"""Build a cloud-free "clear sky" reference image of the Earth's surface.

Deriving cloud from true-colour alone (bright AND low-saturation == cloud)
mistakes any bright, washed-out surface for cloud: deserts score as high as
real convective cloud, and snow/ice score higher still. On the globe that
paints a white veil over the Sahara, Arabia, Greenland and Antarctica.

The fix is to know what the ground looks like with no cloud on it, and only
call something cloud when it is markedly brighter than that. A ready-made,
well-registered cloud-free Earth already ships inside the wallpaper itself
(assets/earth/dayMap-*.ktx -- the surface texture the app draws underneath
our clouds), so use that as the reference rather than trying to synthesise
one from a long run of daily mosaics.

The reference only has to be built once; it is cached as .npy and reused.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# KTX cubemap faces are stored in GL order: +X, -X, +Y, -Y, +Z, -Z
FACE_ORDER = ["px", "nx", "py", "ny", "pz", "nz"]
KTX_IDENTIFIER = bytes([0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A])


def _read_ktx_cubemap(path: Path, level: int = 1) -> dict:
    """Decode one mip level of an ETC2 cubemap into {face: HxWx3 float32}."""
    import texture2ddecoder  # only needed when (re)building the reference

    data = path.read_bytes()
    if data[:12] != KTX_IDENTIFIER:
        raise ValueError(f"{path} is not a KTX 1.1 file")
    (_endian, _gltype, _gltypesize, _glfmt, _glint, _glbase,
     width, height, _depth, _arr, faces, mips, kvlen) = struct.unpack("<13I", data[12:64])

    offset = 64 + kvlen
    out = {}
    for lvl in range(mips):
        lw, lh = max(1, width >> lvl), max(1, height >> lvl)
        (image_size,) = struct.unpack("<I", data[offset:offset + 4])
        offset += 4
        for face_index in range(faces):
            blob = data[offset:offset + image_size]
            offset += image_size + (3 - ((image_size + 3) % 4))  # cubePadding
            if lvl == level:
                decoded = texture2ddecoder.decode_etc2(blob, lw, lh)
                arr = np.frombuffer(decoded, np.uint8).reshape(lh, lw, 4)
                out[FACE_ORDER[face_index]] = arr[:, :, [2, 1, 0]].astype(np.float32)  # BGRA -> RGB
    return out


def _cubemap_to_equirect(faces: dict, width: int, height: int) -> np.ndarray:
    """Sample a cubemap into equirect.

    Note this uses the *dayMap* frame (the raw sphere normal: lat = asin(y),
    lon = atan2(x, z)), not the rotated frame the cloud cubemap is sampled
    through -- the surface texture is looked up with the untransformed normal
    in earth.frag.
    """
    lons = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lats = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(lons, lats)

    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    size = faces["px"].shape[0]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    eps = 1e-12
    out = np.zeros((height, width, 3), np.float32)

    def place(mask, face, sc, tc):
        col = np.clip(((sc + 1) / 2 * size).astype(int), 0, size - 1)
        row = np.clip(((tc + 1) / 2 * size).astype(int), 0, size - 1)
        out[mask] = faces[face][row[mask], col[mask]]

    place((ax >= ay) & (ax >= az) & (x > 0), "px", -z / (ax + eps), -y / (ax + eps))
    place((ax >= ay) & (ax >= az) & (x < 0), "nx",  z / (ax + eps), -y / (ax + eps))
    place((ay >= ax) & (ay >= az) & (y > 0), "py",  x / (ay + eps),  z / (ay + eps))
    place((ay >= ax) & (ay >= az) & (y < 0), "ny",  x / (ay + eps), -z / (ay + eps))
    place((az >= ax) & (az >= ay) & (z > 0), "pz",  x / (az + eps), -y / (az + eps))
    place((az >= ax) & (az >= ay) & (z < 0), "nz", -x / (az + eps), -y / (az + eps))
    return out


def build_reference(daymap_ktx: Path, width: int, height: int) -> np.ndarray:
    faces = _read_ktx_cubemap(Path(daymap_ktx), level=1)
    return _cubemap_to_equirect(faces, width, height)


def load_or_build(cache_path: Path, daymap_ktx: Path, width: int, height: int):
    """Return the clear-sky reference at (height,width,3), or None if it
    cannot be produced (no cached copy and no source texture available)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        ref = np.load(cache_path)
        if ref.shape == (height, width, 3):
            return ref

    if not Path(daymap_ktx).exists():
        return None

    ref = build_reference(daymap_ktx, width, height)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, ref)
    return ref
