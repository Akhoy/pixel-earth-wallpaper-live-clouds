"""Minimal DXT1 (BC1) encoder + DDS container writer, numpy-vectorized.

Quality note: uses the classic "min/max luminance" range-fit heuristic
(not full PCA), which is what fast encoders like stb_dxt's non-refined
path use. Good enough for photographic cloud textures at 256px tiles.
"""
import struct
import numpy as np


def _rgb888_to_565(r, g, b):
    r5 = (r.astype(np.uint32) >> 3) & 0x1F
    g6 = (g.astype(np.uint32) >> 2) & 0x3F
    b5 = (b.astype(np.uint32) >> 3) & 0x1F
    return (r5 << 11) | (g6 << 5) | b5


def _565_to_rgb888(v):
    r5 = (v >> 11) & 0x1F
    g6 = (v >> 5) & 0x3F
    b5 = v & 0x1F
    r = (r5 * 255 + 15) // 31
    g = (g6 * 255 + 31) // 63
    b = (b5 * 255 + 15) // 31
    return r, g, b


def encode_dxt1(rgb: np.ndarray) -> bytes:
    """rgb: HxWx3 uint8 array, H and W must be multiples of 4."""
    h, w, _ = rgb.shape
    assert h % 4 == 0 and w % 4 == 0, "dimensions must be multiples of 4"

    blocks_y = h // 4
    blocks_x = w // 4
    # Reshape into (blocks_y, 4, blocks_x, 4, 3) -> (num_blocks, 16, 3)
    blk = rgb.reshape(blocks_y, 4, blocks_x, 4, 3).transpose(0, 2, 1, 3, 4)
    blk = blk.reshape(blocks_y * blocks_x, 16, 3).astype(np.int32)

    lum = blk[:, :, 0] * 299 + blk[:, :, 1] * 587 + blk[:, :, 2] * 114  # 0..255*1000
    min_idx = np.argmin(lum, axis=1)
    max_idx = np.argmax(lum, axis=1)

    n = blk.shape[0]
    c_min = blk[np.arange(n), min_idx]  # darkest pixel per block -> c1
    c_max = blk[np.arange(n), max_idx]  # brightest pixel per block -> c0

    c0_565 = _rgb888_to_565(c_max[:, 0], c_max[:, 1], c_max[:, 2]).astype(np.uint16)
    c1_565 = _rgb888_to_565(c_min[:, 0], c_min[:, 1], c_min[:, 2]).astype(np.uint16)

    # Degenerate blocks (flat color): nudge c1 down so c0_565 != c1_565,
    # otherwise DXT1 switches into 1-bit-alpha mode semantics.
    equal = c0_565 == c1_565
    c1_565 = np.where(equal & (c1_565 > 0), c1_565 - 1, c1_565)
    c1_565 = np.where(equal & (c1_565 == 0) & (c0_565 == 0), 1, c1_565)

    r0, g0, b0 = _565_to_rgb888(c0_565.astype(np.int32))
    r1, g1, b1 = _565_to_rgb888(c1_565.astype(np.int32))

    # Build the 4 palette colors per block: col0, col1, 2/3 mix, 1/3 mix
    col0 = np.stack([r0, g0, b0], axis=1).astype(np.float32)  # (n,3)
    col1 = np.stack([r1, g1, b1], axis=1).astype(np.float32)
    col2 = (2 * col0 + col1) / 3.0
    col3 = (col0 + 2 * col1) / 3.0
    palette = np.stack([col0, col1, col2, col3], axis=1)  # (n,4,3)

    pixels = blk.astype(np.float32)  # (n,16,3)
    # distance from each pixel to each of the 4 palette colors -> (n,16,4)
    diff = pixels[:, :, None, :] - palette[:, None, :, :]
    dist = np.sum(diff * diff, axis=3)
    indices = np.argmin(dist, axis=2).astype(np.uint32)  # (n,16) values 0..3

    # pack 16 2-bit indices (pixel 0 = lowest bits) into a uint32 per block
    packed = np.zeros(n, dtype=np.uint32)
    for i in range(16):
        packed |= (indices[:, i] & 0x3) << (i * 2)

    out = bytearray(n * 8)
    c0_bytes = c0_565.astype("<u2").tobytes()
    c1_bytes = c1_565.astype("<u2").tobytes()
    idx_bytes = packed.astype("<u4").tobytes()
    mv = memoryview(out)
    for i in range(n):
        o = i * 8
        mv[o:o + 2] = c0_bytes[i * 2:i * 2 + 2]
        mv[o + 2:o + 4] = c1_bytes[i * 2:i * 2 + 2]
        mv[o + 4:o + 8] = idx_bytes[i * 4:i * 4 + 4]

    # Blocks are currently ordered (blocks_y, blocks_x) row-major already
    # matching how DDS expects block scan order (left-to-right, top-to-bottom).
    return bytes(out)


DDS_MAGIC = b"DDS "
DDPF_FOURCC = 0x4
DDSCAPS_TEXTURE = 0x1000
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PIXELFORMAT = 0x1000
DDSD_LINEARSIZE = 0x80000


def write_dds(width: int, height: int, dxt1_data: bytes) -> bytes:
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    pitch_or_linear_size = len(dxt1_data)

    # DDS_HEADER core fields (7 x uint32 after dwSize): flags, height, width,
    # pitchOrLinearSize, depth, mipMapCount, then dwReserved1[11].
    core = struct.pack("<7I", flags, height, width, pitch_or_linear_size, 0, 1, 0)
    reserved1 = b"\x00" * (10 * 4)

    # DDS_PIXELFORMAT (32 bytes): size, flags, fourCC, rgbBitCount, 4x bitmasks
    pixelformat = struct.pack("<2I4s5I", 32, DDPF_FOURCC, b"DXT1", 0, 0, 0, 0, 0)

    # trailing dwCaps, dwCaps2, dwCaps3, dwCaps4, dwReserved2
    caps = struct.pack("<5I", DDSCAPS_TEXTURE, 0, 0, 0, 0)

    header = struct.pack("<I", 124) + core + reserved1 + pixelformat + caps
    assert len(header) == 124, len(header)
    return DDS_MAGIC + header + dxt1_data


def encode_tile_to_dds(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[0], rgb.shape[1]
    data = encode_dxt1(rgb)
    return write_dds(w, h, data)
