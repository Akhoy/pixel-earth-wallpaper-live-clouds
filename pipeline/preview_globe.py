"""Render what the wallpaper will actually look like, without a phone.

Takes a cloud-mask equirect (the same image the pipeline turns into DDS
tiles), projects it through the cube-map convention the app samples with,
and runs earth.frag's day/night/terminator/atmosphere maths over the top of
the app's own shipped dayMap and nightMap textures. The point is to be able
to judge a change to the cloud mask on this machine instead of rebuilding an
APK and squinting at a phone.

Usage:
  venv/bin/python3 pipeline/preview_globe.py CLOUDMASK.png [--out out.png]
        [--lat 20 --lon 78] [--width 1080 --height 2400]
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
ASSETS = ROOT.parent / "decoded_wallpaper_test" / "assets" / "earth"
FACE_ORDER = ["px", "nx", "py", "ny", "pz", "nz"]
KTX_IDENTIFIER = bytes([0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A])

# colours read out of EarthShaderProvider$EarthShader
TERMINATOR_1 = (np.array([0.39, 0.44, 0.50]), 0.8)
TERMINATOR_2 = (np.array([0.929412, 0.274510, 0.058824]), 0.8)
ATMOSPHERE_1 = (np.array([0.678431, 0.823529, 0.941176]), 0.4)
ATMOSPHERE_2 = (np.array([0.380392, 0.694118, 0.949020]), 0.85)
CLOUD_INTENSITY_DAY = 0.9
CLOUD_INTENSITY_NIGHT = 0.6
CLOUD_SHADOW_INTENSITY = 0.5
NIGHT_CLOUD_COLOR = np.array([0.26, 0.29, 0.34])


def read_ktx_cubemap(path: Path, level: int = 1) -> dict:
    import texture2ddecoder

    data = Path(path).read_bytes()
    assert data[:12] == KTX_IDENTIFIER, f"{path} is not KTX 1.1"
    (_e, _t, _ts, _f, _if, _bif, w, h, _d, _a, faces, mips, kvlen) = struct.unpack("<13I", data[12:64])
    off = 64 + kvlen
    out = {}
    for lvl in range(mips):
        lw, lh = max(1, w >> lvl), max(1, h >> lvl)
        (size,) = struct.unpack("<I", data[off:off + 4])
        off += 4
        for fi in range(faces):
            blob = data[off:off + size]
            off += size + (3 - ((size + 3) % 4))
            if lvl == level:
                arr = np.frombuffer(texture2ddecoder.decode_etc2(blob, lw, lh), np.uint8).reshape(lh, lw, 4)
                out[FACE_ORDER[fi]] = arr[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    return out


def sample_cube(faces: dict, d: np.ndarray) -> np.ndarray:
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    size = faces["px"].shape[0]
    ax, ay, az = np.abs(dx), np.abs(dy), np.abs(dz)
    eps = 1e-12
    out = np.zeros(dx.shape + (3,), np.float32)

    def place(mask, face, sc, tc):
        if not mask.any():
            return
        col = np.clip(((sc + 1) / 2 * size).astype(int), 0, size - 1)
        row = np.clip(((tc + 1) / 2 * size).astype(int), 0, size - 1)
        out[mask] = faces[face][row[mask], col[mask]]

    place((ax >= ay) & (ax >= az) & (dx > 0), "px", -dz / (ax + eps), -dy / (ax + eps))
    place((ax >= ay) & (ax >= az) & (dx < 0), "nx",  dz / (ax + eps), -dy / (ax + eps))
    place((ay >= ax) & (ay >= az) & (dy > 0), "py",  dx / (ay + eps),  dz / (ay + eps))
    place((ay >= ax) & (ay >= az) & (dy < 0), "ny",  dx / (ay + eps), -dz / (ay + eps))
    place((az >= ax) & (az >= ay) & (dz > 0), "pz",  dx / (az + eps), -dy / (az + eps))
    place((az >= ax) & (az >= ay) & (dz < 0), "nz", -dx / (az + eps), -dy / (az + eps))
    return out


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _mix(a, b, t):
    return a * (1 - t) + b * t


def _rot_x(v, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.stack([v[..., 0], c * v[..., 1] - s * v[..., 2], s * v[..., 1] + c * v[..., 2]], -1)


def geo_normal(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)])


def render(clouds, day, night, center_lat, center_lon, utc,
           width=1080, height=2400, globe_frac=0.44):
    doy = utc.timetuple().tm_yday
    declination = 23.44 * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
    sub_lon = -15.0 * (utc.hour + utc.minute / 60.0 - 12.0)
    sun = geo_normal(declination, sub_lon)

    fwd = geo_normal(center_lat, center_lon)
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    right /= np.linalg.norm(right)
    up_cam = np.cross(fwd, right)

    radius = globe_frac * width
    xs = (np.arange(width) - width / 2) / radius
    ys = (height / 2 - np.arange(height)) / radius
    U, V = np.meshgrid(xs, ys)
    r2 = U ** 2 + V ** 2
    inside = r2 <= 1.0
    w = np.sqrt(np.clip(1 - r2, 0, 1))

    N = U[..., None] * right + V[..., None] * up_cam + w[..., None] * fwd
    eye = fwd * 6.0
    E = eye[None, None, :] - N
    E /= np.linalg.norm(E, axis=-1, keepdims=True)

    # earth.vert: clouds are sampled through cloudTransformMatrix
    cloud_dir = np.stack([N[..., 2], N[..., 0], N[..., 1]], -1)
    shadow_dir = _rot_x(cloud_dir, 0.3)

    eye_light = np.abs(np.sum(N * E, -1))[..., None]
    light_dir = np.sum(N * sun[None, None, :], -1)[..., None]
    inv_light = 1.0 - light_dir

    cloud = sample_cube(clouds, cloud_dir)[..., 0:1]
    cloud_shadow = sample_cube(clouds, shadow_dir)[..., 0:1]

    day_diff = sample_cube(day, N)
    day_diff = _mix(day_diff, np.zeros(3), cloud_shadow * CLOUD_INTENSITY_DAY * CLOUD_SHADOW_INTENSITY)
    day_diff = _mix(day_diff, np.ones(3), cloud * CLOUD_INTENSITY_DAY)
    base = np.where(light_dir > -0.25, day_diff, np.zeros_like(day_diff))

    rim = _mix(TERMINATOR_2[0], TERMINATOR_1[0], eye_light)
    rim_a = _mix(TERMINATOR_2[1], TERMINATOR_1[1], eye_light)
    base = np.where(inv_light >= 0.6, _mix(base, base * rim, _smoothstep(0.6, 1.1, inv_light) * rim_a), base)
    base = base - _smoothstep(0.6, 1.0, eye_light) * 0.02

    night_diff = sample_cube(night, N)
    night_diff = night_diff - _smoothstep(0.0, 0.8, 1.0 - eye_light) * 0.05
    night_diff = _mix(night_diff, NIGHT_CLOUD_COLOR, cloud * CLOUD_INTENSITY_NIGHT)
    night_diff = night_diff * _smoothstep(0.0, 1.0, eye_light)
    lnorm = np.linalg.norm(night_diff, axis=-1, keepdims=True) / 1.73205
    night_dark = _mix(night_diff, np.minimum(night_diff, np.array([0.08, 0.09, 0.13])),
                      _smoothstep(0.0, 0.13, lnorm))
    dusk = np.clip(_smoothstep(-0.1, 0.25, -light_dir), 0, 1)
    sunset = np.clip(_smoothstep(-0.25, 0.1, -light_dir), 0, 1)
    blended = _mix(_mix(base, night_dark, sunset), night_diff, dusk)
    base = np.where(light_dir < 0.25, blended, base)

    glow = _smoothstep(0.12, 1.0, 1.0 - eye_light)
    glow = glow * (1.0 - _smoothstep(0.8, 0.9, glow)) * np.clip(light_dir, 0, None)
    edge = _smoothstep(0.1, 0.38, eye_light)
    base = _mix(base, _mix(ATMOSPHERE_2[0], ATMOSPHERE_1[0], edge),
                _mix(ATMOSPHERE_2[1], ATMOSPHERE_1[1], edge) * glow)

    img = np.where(inside[..., None], np.clip(base, 0, 1), 0.0)
    return (img * 255).astype(np.uint8)


def main():
    import sys
    from datetime import datetime, timezone
    sys.path.insert(0, str(Path(__file__).parent))
    from cubemap import equirect_to_cubemap

    ap = argparse.ArgumentParser()
    ap.add_argument("cloudmask", help="equirect cloud-mask PNG")
    ap.add_argument("--out", default="preview.png")
    ap.add_argument("--lat", type=float, default=20.0)
    ap.add_argument("--lon", type=float, default=78.0)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=2400)
    ap.add_argument("--face-size", type=int, default=512)
    args = ap.parse_args()

    mask = np.array(Image.open(args.cloudmask).convert("L"))
    faces = equirect_to_cubemap(np.stack([mask] * 3, -1), args.face_size)
    clouds = {k: v.astype(np.float32) / 255.0 for k, v in faces.items()}
    day = read_ktx_cubemap(ASSETS / "dayMap-Summer.ktx", 1)
    night = read_ktx_cubemap(ASSETS / "nightMap.ktx", 1)

    img = render(clouds, day, night, args.lat, args.lon,
                 datetime.now(timezone.utc), args.width, args.height)
    Image.fromarray(img).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
