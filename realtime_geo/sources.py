"""Registry of geostationary satellite sources for the real-time global
cloud composite.

Coverage math (usable longitude span per source, at the equator):

    GOES-West (goes18, -137.2 deg)  ~ -180 .. -60
    GOES-East (goes19,  -75.2 deg)  ~ -140 .. -10
    MTG-I1    (mtg_i1,     0.0 deg) ~  -70 ..  70   (EUMETSAT-published bbox)
    MSG-IODC  (msg_iodc,  45.5 deg) ~  -35.5 .. 118.5 (EUMETSAT-published bbox)

Union: -180 .. 118.5, with real overlap at every seam (so no hard cut
between sources) -- but NOT a full 360 deg tiling. The remaining gap,
roughly 118.5E to 180 (E/SE Asia, Japan, Korea, maritime SE Asia,
Australia, and the western Pacific), is real and not covered by any
source here.

INSAT-3S was the fourth satellite originally proposed for this gap, but it
has no free/public, unauthenticated real-time image API (MOSDAC requires
a registered login and session cookie) -- so it's not wired up. It's also
largely redundant with MSG-IODC's Indian Ocean view (both are centered
over the Indian Ocean region), so it wouldn't have closed the Asia-Pacific
gap even with credentials. Himawari-9 (~140.7E, JMA) is the satellite that
would actually close that gap; see README.md for what's needed to add it.
"""

SOURCES = {
    "goes18": {
        "kind": "abi",
        "label": "GOES-West",
        "sat_lon": -137.2,
    },
    "goes19": {
        "kind": "abi",
        "label": "GOES-East",
        "sat_lon": -75.2,
    },
    "mtg_i1": {
        "kind": "eumetsat_wms",
        "label": "MTG-I1 (0 deg E)",
        "layer": "mtg_fd:rgb_truecolour",
        "sat_lon": 0.0,
        "bbox": (-70.0, -70.0, 70.0, 70.0),  # minx, miny, maxx, maxy (deg)
    },
    "msg_iodc": {
        "kind": "eumetsat_wms",
        "label": "MSG-IODC (45.5 deg E)",
        "layer": "msg_iodc:rgb_natural",
        "sat_lon": 45.5,
        "bbox": (-35.5, -77.0, 118.5, 77.0),
    },
}

# Order controls the reference source used for tone-matching during blend
# (first entry) and drawing order for debug output -- not blend priority,
# which is driven by the per-pixel confidence weights instead.
SOURCE_ORDER = ["goes18", "goes19", "mtg_i1", "msg_iodc"]
