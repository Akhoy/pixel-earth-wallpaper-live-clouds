# realtime_geo — 4-satellite real-time cloud composite

A prototype alternative to `pipeline/run_pipeline.py` (the VIIRS/GIBS daily
pipeline). Instead of NASA GIBS's once-a-day true-color mosaics, this pulls
a live look from 4 geostationary satellites and blends them into one
global equirect frame, updated as often as you re-run it (each source
refreshes every 10-15 minutes).

```
venv/bin/python3 realtime_geo/run_realtime_pipeline.py --width 4096 --out output/latest_realtime
```

Same output contract as the existing pipeline: a stable `root.json` +
`versions/<timestamp>/` of DDS cube-face tiles, so it's a drop-in swap for
whatever serves `output/latest` today, once you're happy with quality.

## Sources

| Source | Position | Kind | Fetch |
|---|---|---|---|
| GOES-West (`goes18`) | 137.2°W | ABI full-disk GeoColor | NOAA STAR CDN, no auth |
| GOES-East (`goes19`) | 75.2°W | ABI full-disk GeoColor | NOAA STAR CDN, no auth |
| MTG-I1 (`mtg_i1`) | 0° | FCI true color | EUMETSAT View WMS, no auth |
| MSG-IODC (`msg_iodc`) | 45.5°E | SEVIRI natural color | EUMETSAT View WMS, no auth |

All four are free, public, and require no registration or API key.

### Why MSG-IODC instead of INSAT-3S

The original ask was GOES-East + GOES-West + MTG + INSAT-3S. INSAT-3S has
no free, unauthenticated real-time image API — MOSDAC requires a
registered login and session cookie, which isn't something this pipeline
can rely on unattended. It would also be largely redundant here: INSAT-3S
sits at 83°E, squarely inside the same Indian Ocean view MSG-IODC already
covers from 45.5°E. Swapping in MSG-IODC (also free/public via the same
EUMETSAT WMS used for MTG) gets equivalent Indian Ocean coverage today,
with no credentials needed.

### The coverage gap that's still real

These four sources tile roughly **-180° to +118.5°E**, with genuine
overlap at every seam (GOES-West/East overlap around -140 to -60; GOES-East/MTG
around -70 to -10; MTG/MSG-IODC around -35.5 to 70). What's **not**
covered: roughly **118.5°E to 180°** — East/Southeast Asia, Japan, Korea,
Indonesia, Australia, and the western Pacific. You'll see this in output
as a visible seam where real imagery meets inpainted fill (the existing
`pipeline/merge.py` gap-fill was built for VIIRS's thin swath gaps, not a
structural ~60° gap, so it shows up honestly rather than convincingly).

Closing it needs a 5th source over East Asia/the Pacific — Himawari-9
(140.7°E, JMA) is the natural candidate. It has no simple no-auth JPEG/WMS
endpoint like the ones used here (its AWS Open Data bucket ships raw AHI
L1b HSD segments, which need `satpy` or equivalent to decode into an
image); the quicker JMA public viewer endpoints weren't confirmed working
in the time spent on this pass. That's the next piece of work if you want
true 360° tiling.

## Design notes

- **Reprojection**: GOES images are in the satellite's native fixed-grid
  view and go through `reproject.py` (wraps `pipeline/geos_reproject.py`'s
  navigation math). EUMETSAT's WMS already returns EPSG:4326 (plate
  carree) imagery directly, so those sources just get resized and placed
  onto the matching slice of the global canvas — no navigation math
  needed.
- **Confidence weighting**: every source gets a per-pixel weight that's
  highest at its sub-satellite point and fades toward 0 approaching the
  edge of its *usable* view (`USABLE_R` in `reproject.py`, deliberately
  tighter than the true geometric limb — imagery gets extremely oblique
  and distorted well before the hard visibility cutoff).
- **Blend**: `merge_realtime.py` does a confidence-weighted average across
  all sources (not "darkest wins" — these are simultaneous single looks,
  not repeated daily passes, so there's no transient haze/glint to
  suppress by picking a winner). Each source is tone-matched to a
  reference source first (reusing `pipeline/merge.py`'s
  `match_histogram`) so overlap zones don't show a brightness/color
  seam. A `MIN_TRUSTED_WEIGHT` floor treats total-confidence-too-low
  pixels as a gap rather than letting a lone, barely-visible source get
  normalized up to full opacity — without this, a satellite's
  heavily-distorted extreme limb would dominate any stretch nothing else
  reaches.
- **Gap-fill and cloud masking**: reused as-is from `pipeline/merge.py` /
  `pipeline/cloud_mask.py`. No `destripe()` step here — a geostationary
  full-disk frame is one continuous stare, not stitched orbital swaths,
  so there's no granule banding to remove.

## Known limitation: cloud mask quality on this data

`pipeline/cloud_mask.py`'s heuristic (`cloud = brightness * (1 -
saturation)`) was tuned against VIIRS/MODIS daytime true-color mosaics.
Two things degrade it here that are worth knowing about before treating
this as production-ready:

1. **Land/desert false-positives** — bright, low-saturation land (desert,
   salt flats, snow) reads as "cloud" under this heuristic. This is a
   pre-existing limitation shared with the VIIRS pipeline, not something
   new introduced here, but it's more visible in this prototype's debug
   output since it hasn't been through the same visual tuning pass.
2. **Nighttime imagery isn't handled** — the VIIRS/GIBS pipeline only ever
   composites daytime passes. A geostationary full disk always has a
   night side, and GOES's GeoColor product renders that side from IR
   channels (plus city lights), which doesn't fit the
   brightness/saturation assumption at all. Right now the night
   hemisphere still gets a mask value, just not a very meaningful one —
   real clouds are usually visible as brighter swirls against a darker
   background, but it hasn't been validated the way the daytime heuristic
   has. A day/night-aware mask (e.g. blend in the IR channel directly, or
   apply a different heuristic on the night side keyed off a
   solar-zenith-angle mask) is the natural next step.

## Adding a 5th/replacement source later

`sources.py` is a plain registry — add an entry with `kind: "abi"` (needs
a `reproject_goes_to_equirect`-style navigation, i.e. a fixed-grid GEOS
sensor) or `kind: "eumetsat_wms"` (already-equirect WMS imagery), plus a
fetch function, and it'll join the weighted blend automatically. An
INSAT-3S entry could be wired up the same way once MOSDAC credentials are
available (its fetch would need session-cookie auth, unlike the four
here).
