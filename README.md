# pixel-earth-wallpaper-live-clouds

A data pipeline that fetches real satellite cloud imagery daily and publishes
it as a cloud-density mask, in the exact cube-face tile format that Google's
"Marvelous Marble" Earth live wallpaper (`com.breel.wallpapers18`, a Pixel
wallpaper) expects. The app's own two live-data endpoints (cloud tiles and
lightning-storm locations) both stopped updating years ago; this project
replaces the cloud-tile side with real, current data.

**This repo is the pipeline only.** It has no dependency on, and does not
contain, any of Google's app code or assets — the app side (pointing its
`CloudsProvider` at the URL this pipeline publishes to) is a separate,
local patch not published here.

## How it works

Once a day, [GitHub Actions](.github/workflows/pipeline.yml) runs
[`pipeline/run_pipeline.py`](pipeline/run_pipeline.py), which:

1. **Fetches** the latest full-Earth true-color mosaic from three
   independent polar-orbiting satellites — NOAA-20, NOAA-21, and Suomi
   NPP (VIIRS instrument), via NASA's public GIBS API (`pipeline/gibs.py`).
   Three sources are used because any single satellite's daily mosaic has
   real defects: granule-to-granule stitching seams, transient haze/smoke,
   and sun-glint.
2. **Derives a grayscale cloud-density mask from each satellite
   independently** (`pipeline/cloud_mask.py`) — clouds are bright *and*
   low-saturation (white/gray), while land and ocean are comparatively
   colorful, so `brightness × (1 − saturation)` (with a tuned gamma curve)
   gives a per-pixel "how cloud-like is this" score.
3. **Combines the three masks via a per-pixel median**
   (`pipeline/merge.py`). This is deliberately done *after* deriving each
   mask, not before combining the raw photos — combining photos first
   (e.g. always picking the darkest/clearest reading) also discards real,
   fast-evolving convective cloud whenever only one of the three
   ~30–100-minute-apart passes caught it. Taking the median of independently
   derived masks keeps that cloud while still rejecting genuine
   single-satellite outliers (glint, a stitching seam).
4. **Fills any remaining gaps** (missing swaths, polar night) from the
   previous run's cache, then inpaints anything still missing, and cleans
   up the Antarctic coverage edge specifically (`fix_antarctic_cap`).
5. **Reprojects** the equirectangular mask onto 6 cube faces in the same
   sampling convention the app's shader uses (`pipeline/cubemap.py`), and
   **encodes** each face into 256×256 DXT1/DDS tiles with a hand-written
   encoder verified byte-for-byte against the app's original tile format
   (`pipeline/dxt1.py`).
6. **Publishes** the tiles plus a `root.json` pointer file. `root.json`
   lives at a stable URL; its `baseUrl` field points at a fresh
   `versions/<timestamp>/` folder each run, so the app's own "has the
   baseUrl changed?" check correctly notices new data is available.

GitHub Pages then serves the published output directly from the repo's
`gh-pages` branch — live at:

```
https://akhoy.github.io/pixel-earth-wallpaper-live-clouds/root.json
```

## Repo layout

- `pipeline/run_pipeline.py` — the daily cloud-mask job (entry point run by CI)
- `pipeline/gibs.py` — fetches VIIRS true-color mosaics from NASA GIBS
- `pipeline/cloud_mask.py` — per-satellite cloud-density derivation
- `pipeline/merge.py` — median-of-masks combining, cache gap-fill, inpainting
- `pipeline/cubemap.py` — equirectangular → 6-cube-face reprojection
- `pipeline/dxt1.py` — hand-written DXT1/DDS tile encoder
- `pipeline/serve.py` — small local dev server (rewrites `root.json`'s
  `baseUrl` to match whatever host it's reached on — used only for local
  testing; production serving is GitHub Pages)
- `pipeline/preview_globe.py` — renders a candidate cloud mask through the
  same day/night/terminator/atmosphere shader math the app uses, so changes
  can be judged without a phone
- `pipeline/clearsky.py` — an experimental multi-day clear-sky reference
  builder (evaluated, not currently wired into the production mask — see
  inline docstrings for why)
- `pipeline/storms.py`, `run_storms.py`, `run_storms_pipeline.py`,
  `storm_protobuf.py`, `glm_lightning.py`, `geos_reproject.py` — a separate,
  currently **deprioritized** pipeline for replacing the app's frozen
  lightning-storm feed (GOES GLM + EUMETSAT + Xweather spot-checks). Not
  part of the scheduled GitHub Actions job yet.
- `realtime_geo/` — an earlier, abandoned experiment overlaying live INSAT/
  GOES/EUMETSAT geostationary imagery instead of polar-orbiting VIIRS (kept
  for reference; see `realtime_geo/README.md`)

## Running locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build one day's cloud mask into output/latest/
python3 pipeline/run_pipeline.py --width 4096 --out output/latest

# Serve it locally (rewrites baseUrl to whatever host/IP it's reached on)
python3 pipeline/serve.py --port 8765 --dir output/latest
```

## Status / known limitations

- **Storms/lightning**: not yet part of the scheduled cloud-only pipeline
  above. The app's `StormsProvider` still points at a local dev server, not
  a public host — revisiting this is a separate, later task.
- **Cache continuity in CI**: each GitHub Actions run starts from a clean
  checkout, so the "fill gaps from yesterday's cache" step has nothing to
  fall back on between scheduled runs (falls straight through to
  inpainting instead). Works fine in practice — VIIRS coverage is normally
  well above the pipeline's minimum-coverage floor — but isn't as robust as
  a long-running server with a persistent cache would be.
