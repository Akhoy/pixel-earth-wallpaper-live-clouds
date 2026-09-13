"""Encode (lat, lng) pairs into the exact protobuf wire format the app's
StormProtos.StormLocations expects.

Verified against a real payload pulled from Google's own (now-frozen)
storm_locations endpoint: each entry is
    0x0a 0x0a 0x0d <lat float32 LE> 0x15 <lng float32 LE>
i.e. StormLocations.locations (field 1, length-delimited) containing a
LatLng message with lat_deg (field 1, 32-bit/float) and lng_deg (field 2,
32-bit/float).
"""
import struct


def _encode_latlng(lat: float, lng: float) -> bytes:
    inner = struct.pack("<Bf", 0x0D, lat) + struct.pack("<Bf", 0x15, lng)
    assert len(inner) == 10
    return struct.pack("<BB", 0x0A, len(inner)) + inner


def encode_storm_locations(coords: list[tuple[float, float]]) -> bytes:
    return b"".join(_encode_latlng(lat, lng) for lat, lng in coords)
