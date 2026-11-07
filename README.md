# pixel-earth-wallpaper-live-clouds

This project fixes Google's "Marvelous Marble" Earth live wallpaper
(`com.breel.wallpapers18`, a Pixel wallpaper). The app shows a spinning Earth
with real-looking clouds and lightning storms, but Google's own servers for
that data stopped updating years ago — so the "live" wallpaper has been
showing the same frozen cloud pattern since then. This project replaces the
cloud data with something that's actually current, refreshed automatically
every few hours.

**This repo contains only the data pipeline** — the code that fetches
satellite photos and turns them into cloud images the app can use. No Google
code or app assets live here. The app itself is a small patch to Google's
original APK (just pointing it at this project's data instead of Google's
dead server); that patched APK is **not** part of this source, it's attached
as a download on the [Releases page](../../releases) — see
[Download the patched app](#download-the-patched-app).

## How it works, in plain terms

Several times a day, [GitHub Actions](.github/workflows/pipeline.yml)
automatically runs [`pipeline/run_pipeline.py`](pipeline/run_pipeline.py),
which does this:

1. **Downloads two photos of the whole Earth** from EUMETSAT, a European
   weather-satellite agency that gives this data away for free. One photo is
   ordinary infrared — it doesn't see light, it sees *heat* (cold cloud-tops
   glow bright, warm ground stays dark), so it works day or night
   (`pipeline/eumetsat_ir.py`).
2. **Turns that photo into a "how cloudy is it here" map.** Colder spots in
   the infrared photo are scored as more likely to be cloud. Instead of
   judging every spot on Earth against one fixed brightness cutoff, each
   pixel is compared against the *coldest that exact spot has ever read* —
   its own personal "clear sky" baseline — so a wisp of cloud over a place
   that's naturally cold (like a mountain) still stands out, without
   mistaking the mountain itself for cloud the way a single global cutoff
   would.
3. **Fills in small gaps.** The satellite can't quite see all the way to
   either pole, so that sliver gets patched with a real cloud photo's
   texture (`pipeline/polar.py`) rather than being left blank, and the
   seam where the map wraps around (the international date line) is
   smoothed so it doesn't show as a hard line.
4. **Reshapes the flat map onto a cube** — six square faces, the way the
   wallpaper's 3D globe actually expects its texture (`pipeline/cubemap.py`)
   — and **compresses** each face into the exact tiny image format the app
   reads, matched byte-for-byte to what Google's own app originally used
   (`pipeline/dxt1.py`).
5. **Publishes** the result to a public web address. The app checks that
   address periodically and swaps in the new clouds automatically — no
   update to the app itself is ever needed, only new data.

There's also a newer, experimental version of step 1-2
(`pipeline/eumetsat_visible_ir.py`) that additionally looks at an ordinary
daylight photo, not just infrared, so clouds the heat-camera alone might
miss (low, thin, or the same temperature as the ground) still show up where
there's daylight. It was built and tested carefully in a separate project
and is included here for reference, but **isn't turned on yet** — the live
pipeline above still runs on infrared alone.

GitHub Pages then serves the published output directly from the repo's
`gh-pages` branch — live at:

```
https://akhoy.github.io/pixel-earth-wallpaper-live-clouds/root.json
```

## Download the patched app

A ready-to-install, working APK is attached to the latest
[release](../../releases/latest). It is Google's original "Marvelous Marble"
Earth wallpaper (`com.breel.wallpapers18`) with two small patches:

- `CloudsProvider` fetches its cloud tiles from this project's GitHub Pages
  URL (above) instead of Google's dead endpoint.
- `targetSdkVersion` is raised from 31 to 35 so current Android versions
  (tested on Android 17) don't flag it as built for an older version.

Everything else — shaders, textures, day/night maps — is Google's stock app,
unmodified (the storms feed keeps Google's original, now-dead URL). The lightning-storm feed is **not** replaced yet (see
[Status](#status--known-limitations)).

**Install:** download the `.apk` from the release, open it on your Android
device and allow "install unknown apps" when prompted, then choose the
wallpaper from Wallpaper & style → live wallpapers. The app has no launcher
icon (same as Google's original); it only appears in the wallpaper picker. The APK is signed with a
debug key, so it will not update an official Google build in place — uninstall
that copy first if one is installed.

**Note:** this is an unofficial, unaffiliated modification of Google's app,
shared for preservation and personal use. All rights to the original app and
its assets remain with Google.

## Polar holes and the date line

The satellite mosaic can't quite see past about 81° latitude at either pole — a small, fixed sliver, not a
seasonal thing — and the flat map's left/right edges (the international date line) can show a visible seam.

`pipeline/polar.py` fills each pole's hole with the surrounding clouds' brightness plus the fine texture of a
**fixed** real-cloud photo (`pipeline/assets/polar_texture.png`, baked ahead of time by
`scripts/make_polar_texture.py`). Nothing is fetched and nothing needs tuning from day to day. The polar step only
ever changes the two polar faces and can never stop an update: if the fill itself crashes for a pole, that pole
reuses the previously published polar tiles (read from the site before the new version replaces them), and only if
those are unavailable too is it published as plain (the other four faces are always published untouched either
way). `pipeline/merge.py`'s `soften_dateline` cross-fades the date-line seam. Each run writes
`_debug_polar_report.json` describing what happened; `--no-polar` skips the polar step entirely.

Tests: `python -m unittest discover -s tests`. Needs `scipy` (in requirements.txt).

## Repo layout

- `pipeline/run_pipeline.py` — the scheduled cloud-mask job (entry point run by CI)
- `pipeline/eumetsat_ir.py` — the module actually in use: derives a cloud
  map from EUMETSAT's infrared satellite photo
- `pipeline/eumetsat_multisignal.py` — a richer version of the above (adds a
  second EUMETSAT product plus a terrain-elevation correction); built and
  working, not yet switched on in `run_pipeline.py`
- `pipeline/eumetsat_visible_ir.py` — a further alternative that also uses
  ordinary daylight photos, not just infrared (see "How it works" above);
  also not yet switched on
- `pipeline/polar.py` — fills the small pole gaps with a fixed real-cloud
  texture (see above)
- `pipeline/merge.py` — gap-filling from cache, inpainting, and the
  date-line seam fade
- `pipeline/cubemap.py` — equirectangular → 6-cube-face reprojection
- `pipeline/dxt1.py` — hand-written DXT1/DDS tile encoder
- `pipeline/serve.py` — small local dev server (rewrites `root.json`'s
  `baseUrl` to match whatever host it's reached on — used only for local
  testing; production serving is GitHub Pages)
- `pipeline/preview_globe.py` — renders a candidate cloud mask through the
  same day/night/terminator/atmosphere shader math the app uses, so changes
  can be judged without a phone
- `pipeline/storms.py`, `run_storms.py`, `run_storms_pipeline.py`,
  `storm_protobuf.py`, `glm_lightning.py`, `geos_reproject.py` — a separate,
  currently **deprioritized** pipeline for replacing the app's frozen
  lightning-storm feed (GOES GLM + EUMETSAT + Xweather spot-checks). Not
  part of the scheduled GitHub Actions job yet.

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
  inpainting instead). Works fine in practice — EUMETSAT's own coverage gap
  is small and fixed (just the polar slivers `polar.py` already handles) —
  but isn't as robust as a long-running server with a persistent cache
  would be.
- **Two better cloud-mask versions exist but aren't switched on yet**
  (`eumetsat_multisignal.py` and `eumetsat_visible_ir.py`, see
  [Repo layout](#repo-layout)) — `run_pipeline.py` still runs the simpler
  infrared-only version. Swapping it in is a small, deliberate next step,
  not done in this pass.
