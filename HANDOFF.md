# Handoff notes — Google Wallpapers (com.breel.wallpapers18) live data replacement

## SESSION UPDATE (2026-09-13) — day-texture color grading, rewritten from scratch

Long iterative session entirely about how the live day texture (real photo,
`DayMapProvider`/`root_day.json`) looks. No shader/smali touched. Ends with
the color grading in `run_pipeline.py`'s `brighten_day_image()` completely
replaced from the version at the top of this file as of yesterday — that
whole `enhance_day_image()` function (saturation/contrast/S-curve, tuned by
eye, several escalating rounds) is gone. Read this section before touching
day-texture grading again; it's a record of several dead ends so they don't
get re-tried.

### Compositing: darkest-of-3 kept, two alternatives tried and rejected
Explored replacing `darkest_of_n_composite` (in `merge.py`) because it
measurably over-saturates (~30% higher mean saturation than any single raw
satellite image — picking the least-hazy pixel at every location
independently also discards the natural desaturation haze normally causes,
even with zero color grading applied).

- **median-of-3** (per-channel median across the three sources): fixed the
  saturation bias, but computing R/G/B independently means a single output
  pixel can draw different channels from different sources, which
  measurably softened fine texture (~4% lower local high-pass std than any
  single source).
- **consensus-of-2** (per-pixel: average whichever two of the three sources
  agree most closely, discard the third — implemented in `merge.py`,
  since removed): fixed both problems, verified against a real defect (see
  below), and was numerically better than darkest-of-3 on every axis
  measured. **Rejected anyway** — user preference on-device favored
  darkest-of-3's richer/moodier look over consensus-of-2's flatter one.
  darkest-of-3 is back in `merge.py`; the consensus-of-2 code was deleted
  (not left as dead code), but the technique worked and is worth
  revisiting if the saturation bias becomes a problem again.

Verified real defect used to validate both alternatives: a sharp diagonal
tonal seam over Central Asia (~55–115°E, 35–55°N) was confirmed present in
NOAA-20's raw daily mosaic alone (absent in NOAA-21 and SNPP for the same
day) — a genuine single-satellite granule defect, not real terrain. Both
median-of-3 and consensus-of-2 erased it cleanly; darkest-of-3 does too.
Separately, a "1 shade darker" patch in that same region was found to
persist across *all three* compositing methods at nearly the same
magnitude — concluded to be genuine 3-way disagreement between the actual
satellite measurements (most likely sun-angle/shadow differences on
mountainous terrain between the three overpasses, ~30–100 min apart), not
a fixable compositing artifact.

Also confirmed (separately, don't re-litigate): a hazy diagonal band off
the Mauritania/W. Sahara coast is a real, evolving Saharan dust plume
(verified by tracking the same crop across 5 consecutive days — it visibly
drifts/changes shape/intensity day to day, which a fixed sensor artifact
cannot do), and the different-looking India/Pakistan border tonal boundary
is the real, well-documented Indo-Gangetic Plain haze boundary (confirmed
identical in all three raw satellite sources independently). Neither
needs fixing.

### Color grading: the whole saga, in order
1. Mild `ImageEnhance` pass (Color 1.35, Contrast 1.15, Brightness 1.03) —
   fixed an initial "bland" complaint, approved for the ocean specifically.
2. Progressively pushed further per "still bland" feedback on land
   specifically (Arabian Peninsula/Central Asia) — several rounds up to an
   S-curve (strength 0.30) + Color 2.0 + Contrast 1.28, plus a
   cloud-protection blend (blend the graded pixel back toward the original
   in proportion to `brightness*(1-saturation)`, the same heuristic
   `rgb_to_cloud_mask` uses) added partway through because the saturation
   boost was making thin/wispy cloud fade into the terrain underneath —
   keep this cloud-protection technique, it's been re-used successfully in
   every subsequent grading rewrite.
3. Called "too much pop" → halved back down → still "looks ugly" → reverted
   *all the way* to the mild pass from step 1 → **still** rejected ("that's
   fine. lets just...") → then fully removed, day texture = raw
   `darkest_of_n_composite` output, zero grading. Lesson: at this point
   several rounds of "tune by eye against the last on-device impression"
   had been tried and none of them converged — the underlying problem was
   that there was no actual numeric target, just vibes, and vibes drifted
   with every round.
4. **Key discovery**: a screenshot taken right after force-stopping the app
   and reopening the wallpaper picker, *before* tapping "Set wallpaper",
   was assumed to show live data — it doesn't. It's still showing the
   static bundled `dayMap-Summer.ktx` fallback (the live JPEG fetch hasn't
   swapped in yet). Confirmed by rendering that static texture through
   `preview_globe.py`'s shader math and matching its measured
   brightness/saturation almost exactly to that screenshot's. Every "that
   looks rich, get me that" comparison up to this point had actually been
   asking to match Google's own static artwork, not a property of the live
   satellite pipeline at all.
5. Tried matching the static-texture target numerically two ways:
   whole-image `ImageEnhance` grid search (matched the *average* fine,
   107.7/0.263 vs target 105.7/0.256, but ocean specifically was still ~22%
   darker than target because a flat scalar can't fix a region-specific
   gap), then a per-channel gain weighted toward blue to fix the ocean
   specifically. **Rejected** — user called it out as looking artificially
   blue-tinted/wrong; the whole static-texture target was itself the wrong
   thing to chase.
6. **User supplied the actual real target**: the classic Apollo 17 "Blue
   Marble" photo (AS17-148-22727). This is a real photograph, not a
   digital composite, and is fundamentally brighter and *less* saturated
   than anything tried before. Downloaded it, measured ocean/land/cloud
   regions separately (masked to exclude the black-space background):
   ocean brightness 137/saturation 0.155, land brightness 169/saturation
   0.068 (almost gray-pink, much less saturated than any digital target
   tried), cloud brightness 216/saturation 0.025 (near white). Every prior
   attempt had been pushing saturation *up* — the real fix needed *less*
   saturation and far more brightness than a linear scale can deliver
   without clipping highlights.

### Current state (as of this session's end)
`brighten_day_image()` in `run_pipeline.py`: gamma lift (`rgb ** gamma`,
gamma=0.75 — a power curve, chosen over `ImageEnhance.Brightness` because
it lifts shadows/midtones much more than highlights without clipping) +
`ImageEnhance.Color(0.75)` (a **reduction**, not a boost) + the
cloud-protection blend from step 2 above (`protect_power=1.2`). Calibrated
by rendering candidates through the same `preview_globe.py` shader pipeline
used to measure the Apollo photo's stats and comparing numerically, not by
eye alone. gamma=0.6 was the first candidate (measured 138.8/0.124 overall,
close to the reference) but was called "too bright" on-device; backed off
to 0.75 (127.2/0.144). **This was pushed to the phone but not yet confirmed
in a final round-trip screenshot comparison against the Apollo reference —
verify that before considering this closed.**

Reference image saved for future calibration: fetch it fresh if needed —
`https://www.geographyrealm.com/wp-content/uploads/2021/02/Apollo17-Blue-Marble-original-orientation-AS17-148-22727.jpg`.

### Process note for next time
Whenever grading is being tuned again: get a real numeric target *first*
(measure the actual reference image's region-masked brightness/saturation,
the way step 6 above did) before writing any grading code. Every round that
skipped this and tuned by eye against the last on-device screenshot failed
to converge and had to be redone. Region-split stats (ocean vs. land vs.
cloud, not just whole-image mean) matter — a flat scalar transform can
match the whole-image average while still being visibly wrong in one
region, as step 5 found the hard way. Also: the ocean/land pixel classifier
used for region-split validation is itself unreliable across differently
color-graded versions of the same image (changing colors shifts which
pixels fall into which classified bucket) — don't over-trust small
differences in region-split numbers between candidates; confirm visually
too.

## SESSION UPDATE (2026-09-12, later) — two pipeline-only fixes, no shader/smali touched

Both fixes are "API"/data-layer only per the hard rule below — no `earth.frag`,
no smali. Verified numerically and via `preview_globe.py`-style before/after
renders using the cached `cache/last_good_equirect.npy` (no network fetch
needed); not yet re-verified on the actual device.

### Fixed: the known double-brightening issue
`pipeline/cloud_mask.py`'s `rgb_to_cloud_mask()` (the `clear_sky is None`
branch — still the active path) used to raise its output with `** 0.8`
(an expansive gamma, tuned back when this mask alone had to make thin cloud
visible against a stylised basemap). Now that the day surface shows the real
photo directly (`DayMapProvider`/`root_day.json`), that same mask is still
also painted on top of it by the stock shader's day branch
(`mix(dayDiff, vec3(1.0), cloudColor * cloudIntensityDay)`), so raising it
double-brightens cloud that's already visible in the photo.

Changed the exponent to `** 1.6` (compressive, not expansive). Verified on
the cached composite: mean mask value dropped ~28% (136.8→98.5) and the
worst offender (bare Sahara desert reading almost as bright as real cloud)
is now visibly darker/more separated from actual cloud, while the top of
the distribution barely moved (p90 236→219) — dense storm-top cloud stays
close to unchanged, so the night-side storm gate in `earth.frag`
(`smoothstep(0.72, 0.82, cloudColor)`) still fires normally. This is the
same day/night/storm-shared-value trade-off the prior note called out;
resolved by compressing the middle of the curve rather than scaling
everything uniformly, since the mid/low band is what was actually causing
the washed-out double-brightening.

### Fixed: bland/flat VIIRS day composite
Added `enhance_day_image()` in `run_pipeline.py`: boosts color/saturation
(`ImageEnhance.Color`, 1.35x), contrast (`ImageEnhance.Contrast`, 1.15x) and
a touch of brightness (1.03x) via PIL, applied **only** to the RGB pixels
written out as the live day-texture JPEGs — `rgb_to_cloud_mask()` still runs
on the un-enhanced composite upstream, so cloud detection isn't affected.
`ImageEnhance.Color` scales saturation relative to each pixel's own
grayscale value, so already-white/gray cloud isn't pushed toward any tint;
only the more-saturated land/ocean underneath pops more. Before/after
comparison on the cached composite: ocean went from reading near-black to
visibly blue, land tones (desert/forest/ice) noticeably more differentiated,
clouds unchanged/still clean white.

Debug output added: `version_dir/_debug_equirect_day_enhanced.png` (same
convention as the existing `_debug_equirect_merged.png` /
`_debug_cloud_mask.png`).

### Not yet done
- Not re-verified on an actual phone build (would need a fresh
  `run_pipeline.py` run against live GIBS data, rebuild/resign/reinstall).
- The two new constants (gamma `1.6`, Color/Contrast/Brightness
  `1.35/1.15/1.03`) were picked by inspection against one cached scene, not
  swept/tuned across many days — may want revisiting after seeing it live
  on-device across a few different days' imagery.

## SESSION UPDATE (2026-09-12, evening) — read this first, supersedes items below

### HARD RULE from the user this session, carry forward
Do not edit `earth.frag` (or any rendering-logic file) going forward. All
future changes must live in the "API"/data layer: pipeline scripts, and new
provider classes added via the compile-splice technique below, wired into
`EarthEngine.smali` only through small, mechanical edits that mirror
existing, already-proven patterns. This constraint shaped everything below
and should be treated as still in force.

### What happened, in order
1. Tried making the day surface show the real photo directly by editing
   `earth.frag` (bypass the old white-cloud-mask blend, sample `cloudMap`'s
   RGB directly). **Crashed the app**: `IllegalArgumentException: no uniform
   with name 'dayMap' in shader` — removing all references to the `dayMap`
   uniform let the GLSL compiler strip it entirely, and the Java-side
   shader-binding code (`EarthShaderProvider$EarthShader.render`, not
   editable without more smali risk) unconditionally tries to bind a texture
   to a uniform named `dayMap` and throws when it's gone.
2. User set the hard rule above. `earth.frag` was reverted to the exact
   original stock contents (verbatim, from the read earlier in the
   conversation — no diff tool was available since this isn't a git repo).
3. Reverted `pipeline/run_pipeline.py` to the original single-`cloudMap`
   architecture: publishes a **grayscale cloud-density mask** (the
   `rgb_to_cloud_mask` heuristic + `fix_antarctic_cap`, same as before this
   session's detour) to `CLOUDS_URL`/`root.json`, unchanged consumption by
   the stock shader. Rebuilt/resigned/reinstalled, confirmed **no crash**,
   confirmed genuinely live (not stale) by matching a distinctive storm
   spiral north of the Black Sea between the phone screenshot and the
   freshly-fetched satellite composite, and confirming that exact feature is
   **absent** from the static bundled `dayMap-Summer.ktx` at that location.

### Then: added a real live day-image feed, without touching the shader
User asked specifically for the day surface to show the real photo, but
under the hard rule that meant new "API"-layer plumbing, not a shader edit.
Used a technique the user described from another dev's workflow: write the
new class as normal Java in an isolated scratch setup, let a real compiler
generate its bytecode (so register allocation is never hand-done), then
splice the compiler's output into the decompiled app. Concretely:

1. **`daymap_provider_src/`** (project root) — a small Java source tree:
   - `com/breel/wallpapers18/weather/DayMapProvider.java` — the real new
     class. Mirrors `CloudsProvider`'s shape (`Callback` interface,
     `DayMapCubeMap extends FacedCubemapData`, `DayMapUpdatingTask extends
     AsyncTask`), self-throttles to once per 3h same as `CloudsProvider`,
     but **simpler**: fetches `root_day.json` for a `baseUrl`, then
     downloads 6 plain per-face JPEGs (`px.jpg`..`nz.jpg`) directly — no
     DXT1 tiling, no native `CloudsStitching` JNI call (that's a new
     endpoint we fully control, no compatibility need to match Google's old
     tile format). Uses `FacedCubemapData`'s direct 6-`Pixmap` constructor
     (found by reading the decompiled class — simpler than `CloudsProvider`'s
     older manual `PixmapTextureData` approach).
   - Compile-time-only stub classes for the ~6 libGDX types referenced
     (`Gdx`, `Files`, `FileHandle`, `Pixmap`, `CubemapData`,
     `FacedCubemapData`, `GdxRuntimeException`) — empty bodies, signatures
     copied exactly from the app's own decompiled smali for those classes.
     **These stubs are never shipped** — only `DayMapProvider`'s own
     compiled output gets copied into the app; the real gdx classes already
     exist there at runtime. `android.*` and `org.json.*` classes were
     compiled against the real platform jar
     (`~/Library/Android/sdk/platforms/android-37.0/android.jar`, confirmed
     it has `AsyncTask`/`ConnectivityManager`/`JSONObject`) — no stub needed
     for those.
   - Compiled with Android Studio's bundled JBR (`javac --release 8`, no
     system JDK exists on this machine —
     `/Applications/Android Studio.app/Contents/jbr/Contents/Home`).
2. **`daymap_provider_out/`** — the `.class` output.
3. **`daymap_provider_dex/`** — `d8 --release --min-api 24 --lib
   android.jar --classpath daymap_provider_out --output ...`, fed **only**
   the 5 `DayMapProvider*.class` files (not the stub classes) — `--classpath`
   lets d8 resolve references to the stubs without including them in the
   output dex.
4. **`daymap_provider_smali/`** — `apktool d` can't decompile a bare
   `.dex` (wants a zip container), so `classes.dex` got zipped up alone
   (`daymap_provider.apk`, just a zip, no manifest) and `apktool d`'d — this
   works fine, apktool only needs a valid zip with `classes.dex` inside for
   the smali-decoding step. Produced exactly the 5 needed `.smali` files,
   compiler-correct register allocation.
5. Copied those 5 files into
   `decoded_wallpaper_test/smali_classes2/com/breel/wallpapers18/weather/`.
6. **Hand-edited `EarthEngine.smali`** (the only remaining manual smali
   surgery, kept small and mirrored closely off already-working
   `CloudsProvider` wiring):
   - New fields: `dayMapProvider`, `nextDayMap`, `needsDayMapUpdate`.
   - `.implements Lcom/breel/wallpapers18/weather/DayMapProvider$Callback;`
   - Constructor: `needsDayMapUpdate = true` at init (mirrors
     `needsCloudsUpdate`'s init) — required bumping that constructor's
     `.locals 6` → `7` for the one new register.
   - Construct `dayMapProvider` right next to `cloudsProvider`'s
     construction.
   - Call `dayMapProvider.updateDayMap()` right next to
     `cloudsProvider.updateClouds()`.
   - New `onDayMapUpdated()` callback method (mirrors `onCloudsUpdated()`):
     populates `nextDayMap` from `dayMapProvider.getLatest(context)` and
     sets `needsDayMapUpdate = true`.
   - Dispose-on-teardown block for `nextDayMap` (mirrors the existing
     `nextCloudMap` block in the engine's `dispose()`).
   - **The actual swap-in**: in the per-frame update method, right at the
     existing `:cond_4` label (where the cloud-texture swap-in already
     falls through to), added a parallel block — if `nextDayMap` is set and
     `needsDayMapUpdate`, dispose the old `diffuse` Cubemap if present,
     build a new one from `nextDayMap`, reassign `diffuse`, and reapply it
     onto the earth model's `Material` via
     `PlanetTextureAttribute.createDay(...)` — i.e. **overwrite/supersede**
     whatever the existing (untouched) static season-based
     `dayMap-<Season>.ktx` loading code had set, without touching that
     static-load code at all. This is deliberate: static stays as the
     always-present fallback; live data supersedes it once available,
     exactly as asked.
   - Note: like the original app's own cloud-swap logic, this swaps in new
     data **once per successful fetch cycle** (the same
     `needsX/nextX`-consumed-together pattern `CloudsProvider` already has,
     including its same apparent one-shot-per-process-lifetime limitation
     under normal operation) — not a bug introduced here, a deliberate
     mirror of already-shipped behavior.
7. `pipeline/run_pipeline.py` now publishes **two** live things per run:
   the existing grayscale `cloudMap` DDS tileset (`CLOUDS_URL`/`root.json`,
   Z-polar/cloud sampling convention, unchanged), **and** a new `day/`
   subfolder of 6 plain JPEG cube faces (`px.jpg..nz.jpg`) in the
   **dayMap/raw sampling convention** (Y-polar, raw sphere normal — see
   next point), plus `root_day.json`.
8. `pipeline/cubemap.py` gained `direction_to_lonlat_raw()` and a `frame=`
   param on `equirect_to_cubemap()` (`"cloud"` default = Z-polar, `"raw"` =
   Y-polar). **This distinction matters and was easy to get wrong**: the
   cloud cubemap and the day/night cubemaps are sampled by the app with
   genuinely different lookup conventions (established last session for
   clouds; day/night use the raw sphere normal directly, confirmed via
   `preview_globe.py`'s existing render code). Publishing day-image faces
   with the wrong convention would silently produce a geometrically wrong
   (rotated/mirrored) day texture.
9. `pipeline/serve.py` generalized to rewrite `baseUrl` for **both**
   `root.json` and `root_day.json` (previously only handled `root.json`).

### Verified on the actual device, end to end
- Built via `apktool b` → `zipalign` → `apksigner sign` (debug.keystore,
  androiddebugkey/android/android) → `adb install -r`. Same JBR as above for
  all Java-needing steps.
- **No crash.** All 96 cloud DDS tiles + all 6 day-image JPEG faces
  downloaded successfully (confirmed via `serve.py`'s access log and
  `adb logcat` — zero errors, zero 404s on either path).
- Confirmed the day surface is genuinely live (not the static bundled
  texture) the same way as step 3 above: a storm spiral north of the Black
  Sea visible on the phone's actual home-screen screenshot matches the
  freshly-fetched satellite composite exactly, and is **absent** from the
  static `dayMap-Summer.ktx` at that location.
- Also ran `pipeline/run_storms.py` this session (hadn't been run yet this
  session, so `/storm_locations` was 404ing) — produced 237 clustered
  points (92 Africa/EUMETSAT, 53 N.America + 32 CentAm/Caribbean +
  24 S.America from GOES-East/West GLM, 28 Europe/EUMETSAT, 7 Asia from
  Xweather spot-checks — specifically Ahmedabad/Indore/Vadodara/Surat
  (Gujarat/MP cluster), Madurai area, Kathmandu, Bangkok — 1 Pacific).
  Confirmed `StormsProvider` was **already** pointed at our own server
  (`http://192.168.1.8:8765/storm_locations`, from an earlier session, not
  touched this session) — the old Google `gstatic` URL constant is dead/
  vestigial code in `StormsProvider$StormsUpdatingTask.smali`, referenced
  only inside a generic `downloadFile` helper's format-detection branch,
  never actually passed in by the real call site. Confirmed fetch now
  succeeds (200) and no crash after.

### Known open item — not yet fixed
The day view currently **double-renders clouds**: the real photo (now the
live `dayMap`) already shows real clouds, but the still-unmodified shader's
day branch *also* paints a white overlay on top using `cloudMap`'s density
(`mix(dayDiff, vec3(1.0), cloudColor * cloudIntensityDay)`) plus a
shadow-darkening mix — both still run because `earth.frag` is untouched
(per the hard rule). Net effect: cloud regions in the day view likely get a
bit of extra/redundant whitening on top of themselves. Not yet visually
assessed on-device or fixed. Any fix must be "API"-side (e.g. tuning what
density values get published to `cloudMap`) since the shader stays off
limits — this is a real trade-off (any such tuning also proportionally
affects the still-needed night-tint/shadow/storm-gating uses of the same
value) and hasn't been worked out yet.

### Reusable technique for future additions (compile-splice)
This is now a proven, low-risk way to add new behavior without hand-writing
smali register allocation:
1. Write the new class as normal Java, package name matching where it needs
   to live in the target app. Add minimal same-signature stub `.java` files
   for any referenced classes the new code needs to compile against but
   that aren't independently available (matching the target app's real
   classes' signatures, read from their decompiled smali) — stubs are
   compile-only, never shipped.
2. `javac --release 8 -cp <android.jar> -d out <all .java files>`
3. `d8 --release --min-api 24 --lib <android.jar> --classpath out --output
   dex_dir <only the REAL target .class files, not the stubs>`
4. `zip` the resulting `classes.dex` alone into a bare zip (apktool needs a
   zip container, doesn't need a full valid APK/manifest for this step);
   `apktool d that.apk` → clean, compiler-correct `.smali` in `smali/...`.
5. Copy the relevant `.smali` files into the decompiled app's
   `smali_classes2/...` tree.
6. Wire it in with small, hand-edited, mechanical additions to existing
   files, closely mirroring an already-working analogous code path line by
   line (register reuse patterns, label/try-catch structure) rather than
   writing new logic freehand.

### Toolchain notes for this machine
- **No system Java.** Use Android Studio's bundled JBR:
  `JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"`.
- `apktool.jar` at project root. `zipalign`/`apksigner` at
  `~/Library/Android/sdk/build-tools/36.0.0/`. `debug.keystore` at project
  root (alias `androiddebugkey`, password `android` for both store and key).
- `android-37.0` platform jar at
  `~/Library/Android/sdk/platforms/android-37.0/android.jar` — used for
  compiling `DayMapProvider` against real `android.*`/`org.json.*`.
- Scratch build dirs at project root (kept, reusable for the next
  compile-splice addition): `daymap_provider_src/`, `daymap_provider_out/`,
  `daymap_provider_dex/`, `daymap_provider_smali/`,
  `daymap_provider.apk` (the bare-zip dex wrapper).
- A pile of `build_out*.apk` / `*-aligned.apk` / `*-signed.apk` iterations
  accumulated at project root across tonight's rebuild cycles — not cleaned
  up, could be pruned.
- `CLOUDS_URL`, `DAYMAP_ROOT_URL` (inside `DayMapProvider.java`'s compiled
  output), and `STORMS_URL` all still hardcode `192.168.1.8` (tonight's Mac
  LAN IP) — will need updating (regenerate `DayMapProvider` via the
  compile-splice steps above, or `sed` the literal in the smali directly)
  if the Mac's IP changes again, same caveat as before.

## SESSION UPDATE (2026-09-12) — read this first, several items below are now stale

This session found and fixed three independent bugs that were each silently
breaking the phone pipeline, got real satellite clouds rendering correctly
on the actual device for the first time, then went further into cloud-mask
and night-imagery quality. Details below; the pre-existing sections after
this one are the prior session's handoff and are mostly still accurate
except where noted.

### Bug #1 (the big one): cube face orientation was wrong — clouds mismatched geography

`pipeline/cubemap.py` mapped cube directions using `lat=asin(y)`,
`lon=atan2(x,-z)` (a Y-polar frame). But the app does **not** sample the
cloud cubemap with the raw sphere normal — `earth.vert` builds
`vCloudNormal = (vec4(a_normal,1.) * cloudTransformMatrix).xyz`, a
row-vector product (`transpose(M)·n`), and `EarthShaderProvider` builds
`M = identity.rotate(X,-90).rotate(Z,-90)`. Working through it,
`transpose(M) = Rz(+90)·Rx(+90)`, giving `d = (n.z, n.x, n.y)` — a
**Z-polar** frame: `lat=asin(d.z)`, `lon=atan2(d.y,d.x)`.

Fixed in `cubemap.py`'s `direction_to_lonlat()` (see the docstring there
for the full derivation). Confirmed three independent ways:
1. Decoded the app's own shipped `assets/earth/dayMap-*.ktx` (ETC2
   cubemap, via the `texture2ddecoder` pip package) and reprojected it
   back to equirect — old convention gave a mirrored world map, new gives
   a correct one (measured: land-mask correlation against a labeled
   reference, +0.69 correct vs +0.12 mirrored).
2. The shader-matrix derivation above.
3. The app's own `LightningStorms.smali` converts lat/lng to XYZ as
   `x=cos(lng)cos(lat), y=sin(lng)cos(lat), z=sin(lat)` — same Z-polar
   frame, and `stormsMap` is sampled with that same `vCloudNormal`.

End-to-end verification (simulating the full device path — DDS tiles →
DXT1 decode via `texture2ddecoder.decode_bc1` → 16-tile-per-face stitch →
cubemap → actual `earth.frag` sampling/blending — reproduced in
`pipeline/preview_globe.py`) gives **+0.98 correlation** between
reconstructed geography and the source cloud data, vs +0.45 mirrored.
All 12 cube edges checked pairwise for seam continuity (brute-force
edge/flip search) — seamless, differences of 8-19/255 (normal photo
noise), not the 5-10x-worse mismatch a wrong seam would show.

### Bug #2: `root.json`'s `baseUrl` said `localhost` — tiles never downloaded

The pipeline defaulted `--base-url` to `http://localhost:8765`. On the
phone, "localhost" resolves to the phone itself, not the Mac — so every
tile download silently went nowhere, every time, for the entire debugging
session, and the app fell back to its bundled stock texture. This is
almost certainly why earlier "it looks wrong" observations were partly
about stale cache (old fully-downloaded data from a prior session,
`adb install -r` never clears app data) rather than the orientation bug.

Fixed two ways:
- `pipeline/serve.py` (new): a small HTTP server that rewrites
  `root.json`'s `baseUrl` to match whatever `Host` header the request
  actually arrived on, so it self-corrects regardless of which network/IP
  the Mac is on. Run: `venv/bin/python3 pipeline/serve.py [--port 8765]
  [--dir output/latest]`. This replaces bare `python3 -m http.server` for
  serving `output/latest` — use `serve.py` from now on.
- Always pass a real reachable `--base-url` when generating (the phone's
  and Mac's LAN IP, found via `ipconfig getifaddr en0`/`en1` on macOS —
  **note the Mac's IP has changed 3+ times this session** between
  ethernet/wifi/hotspot; `serve.py`'s rewriting makes this a non-issue for
  serving, but the APK's hardcoded smali URL still needs matching if you
  rebuild it — see "APK state" below).

### Bug #3: date-selection used local time, not UTC

`date.today() - timedelta(days=1)` used the machine's local date (IST,
UTC+5:30). GIBS bins its daily mosaics by UTC day, so for part of every
night (00:00-05:30 IST) the code would silently request a UTC day that
hadn't started yet. Fixed: `datetime.now(timezone.utc).date() - timedelta(days=1)`.

**Tested and rejected**: the hypothesis that GIBS might publish/bin by US
Pacific time instead of UTC. Measured per-longitude coverage of a
partially-filled day (today, 9.2h into the UTC day) — data existed *only*
at 120-180°E, matching exactly where a sun-synchronous satellite (~13:30
local solar time) would be 1.5-5.5h into a UTC day; a PT-binned day would
show data somewhere else entirely, or (2.2h into a PT day) show almost no
data at all. Confirmed UTC binning, not PT.

**Also added**: a coverage-based fallback guard (`MIN_DAY_COVERAGE` /
`--min-coverage`, default 0.78) in `run_pipeline.py`. There is **no
"is this day complete" flag anywhere in the GIBS API** (checked: GetCapabilities
advertises the current date from the start of the UTC day; DescribeDomains
only returns temporal extents; `/std/`↔`/nrt/` reflects science-quality
versioning, not intra-day fill — verified against NASA's own docs). A
complete day composites to ~87% coverage (the rest is inherent — gaps
between orbit tracks, polar night); the guard fetches the chosen day,
and if coverage comes back well under that, retries one day earlier and
uses whichever is actually better. Tested both ways: forcing an
incomplete day correctly falls back (20.7%→rejected→used 88.2% day);
a normal complete-day run doesn't false-trigger.

### Phone verification — succeeded for the first time this session

After fixing all three bugs and getting the Mac/phone back on the same
LAN (`192.168.1.x`), the app successfully downloaded **all 96 tiles**
(confirmed via server access log — every `_2_col_row.dds` face 0-5, zero
non-200 responses), stitched them (no `"could not stitch"` error, which
is a real un-stubbed `Log.e` call so it would have shown up), and set the
globe as the active wallpaper (`mWallpaperComponent` confirmed via
`dumpsys wallpaper`).

Verified it was actually showing *our* clouds, not the stock bundled
texture, by: screenshotting the phone, isolating the globe disc, doing a
brute-force lat/lon orientation search against our cloud-mask equirect
(found a clear best match at lat+20/lon+80 = India, matching what was
visibly on screen), then comparing that same orientation against both
our data (**+0.192** correlation) and the stock bundled `clouds.ktx`
(**-0.107**, i.e. actively anti-correlated) — conclusive that it was
using our tiles.

**APK state**: smali URLs (`CLOUDS_URL`/`STORMS_URL` in
`CloudsProvider.smali`/`StormsProvider.smali`, plus the inline
`"http://...:8765/root.json"` constant in
`CloudUpdatingTask.smali`) currently point at whatever the Mac's IP was
at build time — **check/update this before the next phone test**, the Mac's
IP has moved between `192.168.1.5`, `10.227.199.169`, `192.168.1.8` and
back this session alone. My `CM_DEBUG` `Log.e` instrumentation (added
during this session's debugging, scattered through `CloudUpdatingTask.smali`
and `CloudsProvider.smali`) is **still in the last-built APK** and should
be stripped before any release build (harmless for further testing,
just noisy in logcat).

### Cloud mask quality: found a real problem, partially fixed, more work identified

User (correctly) flagged that the cloud mask looked "washed out" —
measured: bare Arabian desert scored a **higher** cloud-mask value (155/255)
than actual tropical storm cloud (151/255). The existing heuristic
(`brightness × (1-saturation)`) can't distinguish bright cloud from bright
sand/ice — same root cause as the Andes-snow-streak and Antarctic-cap
issues already known from the prior session.

Tried and **rejected**: using the app's own `dayMap` texture as a
"clear-sky reference" (call something cloud only if it's much brighter
than the known-bare surface at that point). Failed on user's specific
test case (real Sahel cloud on 2026-09-11): real cloud dropped from
201→51 (crushed) while desert dropped 137→0. The basemap is a different,
darker-exposed sensor than VIIRS, so the exposure-matching gain needed
pushed the bright-desert baseline high enough to eat real cloud along
with it. Don't reuse this approach without fixing the exposure mismatch
first.

Tried and **works better**: build the clear-sky reference from **VIIRS
itself** (darkest-per-pixel over a 12-day rolling window — same sensor,
no exposure-mismatch problem). Real Sahara cloud: 201→146 (mostly
preserved). Bare desert: 137→11 (mostly gone). Cloud/desert separation
more than doubled (65→135). Greenland/Antarctica/Andes-snow false
positives all dropped to near-zero. Still slightly grainier than Google's
stock mask, and still under-detects in permanently-cloudy zones (e.g.
Southern Ocean) where even a 12-day minimum is still cloudy so there's no
true bare-sky baseline. Histogram-matched the result onto Google's stock
mask's exact intensity distribution (median/p75/p90/p95 all within 1
point) to fix the "too strong" complaint — this part is validated but
**not yet wired into `run_pipeline.py`** (there's a half-finished
`pipeline/clearsky.py` + edits to `pipeline/cloud_mask.py`'s
`rgb_to_cloud_mask()` from before this got interrupted — has an
unresolved `TypeError` from a `str | None` type hint under this
environment's Python; trivial fix, just hasn't been redone since the
approach changed direction, see next item).

**Then a better idea emerged and testing pivoted to it**: instead of
deriving a synthetic cloud-density mask and painting white onto the
stylised basemap, use the real 3-satellite true-color composite **directly**
as the globe's day surface (edit `earth.frag` to sample the day texture
from our composite instead of `dayMap`, with our cloud cubemap no longer
needed as a separate whiteness-mask input). Rendered side-by-side via
`pipeline/preview_globe.py` — real clouds have actual photographic
texture/shading, no thresholding, no desert false-positives at all by
construction. Confirmed as clearly the best-looking option so far. **Not
yet implemented** (requires editing the plain-text `earth.frag` asset
inside the APK and re-thinking the pipeline's output format — currently
outputs a grayscale mask, would need to output the RGB composite instead
as DXT1/DDS tiles).

### Night-side imagery investigation (for the "direct satellite imagery" idea above)

Checked whether VIIRS-derived night imagery could replace/augment the
app's static `nightMap.ktx` the same way daytime imagery would replace
`dayMap.ktx`.

- **`VIIRS_Black_Marble`** (GIBS layer): advertised as daily-available in
  capabilities, but every daily date tested returns an empty 2KB image
  via the snapshot API — only the `2016-01-01` annual composite actually
  renders. Effectively static, not a live source. Rejected for that
  reason (also just a stylistic recolor of the same kind of static data
  the app already has, adds no live clouds).
- **`VIIRS_{NOAA20,NOAA21,SNPP}_DayNightBand`**: works daily (confirmed
  Sep 11, ~88% coverage), and — unlike Black Marble — actually shows
  moonlit cloud structure, not just city lights. But single-satellite
  images are badly swath/twilight-striped. Applying the same
  darkest-of-3 trick used for daytime clouds suppresses the striping
  well (visually confirmed).
- **Important physical fact, initially assumed otherwise**: the Day/Night
  Band is a single panchromatic sensor channel — there is no color
  information in raw VIIRS night data at all (measured: R=G=B=94.0
  exactly on the composite). Both the app's bundled `nightMap` and Black
  Marble's colors are artistic tints applied to greyscale data, not
  measurements. So any "colored" VIIRS night render is necessarily our
  own tinting choice, same as Google's.
- Built a prototype colorization: reuse the app's **existing** static
  `nightMap`/Black-Marble-2016 for city lights (they don't meaningfully
  change day to day, no reason to re-derive them), and add the **daily**
  DNB signal only as a separate cool-blue "moonlit cloud" layer on top —
  subtracting a static city-light mask from the daily DNB first so cities
  aren't double-counted as cloud. Rendered on the actual globe with real
  day+night+terminator: clouds visibly continue across the terminator
  from day into night, which neither the current app nor a same-approach
  using only the static bundled night texture can show.
- **User specifically tested this over China (the worst case — highest
  light pollution on Earth)**: found the city-subtraction is a real
  improvement (correlation of resulting "cloud" layer against city-light
  layer dropped from +0.561 raw to +0.240 after subtraction) but **not
  fully clean** — some city-glow bleed-through remains, visible as a soft
  glow tracing the coastal Chinese city belt. The layer does also carry
  genuine correlated daytime-cloud signal (+0.348 vs same-day true-color
  cloud), so it's a mix of real signal and residual contamination, not
  either one purely.
- **Proposed next step, not yet built ("option 2")**: stop trying to
  separate cloud-from-lights within the noisy night image entirely.
  Instead, take cloud positions already cleanly measured from the
  **daytime** true-color composite (unambiguous, no city-light
  confusion possible) and carry/reproject that same cloud pattern onto
  the night hemisphere, since clouds don't move fast enough to
  meaningfully change within the ~12h terminator crossing. Sidesteps the
  China contamination problem entirely since it never touches the DNB
  for cloud detection. Tradeoff: night clouds become a same-day daytime
  snapshot carried forward rather than an actual night-time measurement
  — very good for slow-moving systems, worse for anything that changed
  quickly in the intervening hours. Was about to prototype this when the
  session ended; not started.

### New/changed files this session
- `pipeline/cubemap.py` — the orientation fix (see docstring for full math)
- `pipeline/serve.py` — new, IP-independent root.json-rewriting server
- `pipeline/run_pipeline.py` — UTC date fix + coverage-guard fallback
- `pipeline/preview_globe.py` — new, renders the actual on-device look
  (full `earth.frag` day/night/terminator/atmosphere math) from a
  cloud-mask PNG, without needing a phone. Use this to judge any future
  mask/imagery change before rebuilding the APK.
- `pipeline/clearsky.py` — new, VIIRS-multi-day-minimum clear-sky
  reference builder. Half-wired into `cloud_mask.py`; direction changed
  before this was finished (see "direct imagery" above) — treat as
  reference/scratch, not production-ready.
- `pipeline/cloud_mask.py` — `rgb_to_cloud_mask()` got an optional
  `clear_sky=` parameter mid-session; has a known `TypeError` on this
  Python (3.9) from a `X | None` type hint used before checking the
  interpreter version — fix trivially by using `Optional[np.ndarray]` or
  just dropping the annotation, or finish the "direct imagery" pivot
  instead and revert this function.
- `decoded_wallpaper_test/AndroidManifest.xml` — briefly had
  `android:debuggable="true"` added then reverted; confirmed back to
  original (0 debuggable flags) as of end of session.

## Original goal

## Original goal
`decoded_wallpaper/` is a decompiled (apktool, smali) copy of Google's "Earth"
live wallpaper (package `com.breel.wallpapers18`, the "Marvelous Marble"
globe with real-time clouds + storms). Its two data sources are dead/frozen:

- Clouds: `CLOUDS_URL` in `CloudsProvider.smali` / `CloudUpdatingTask.smali`
  → `https://mw1.google.com/mw-weather/clouds-cubemap/root.json` — returns a
  `{"baseUrl": "..."}` pointer to 96 DDS tiles (6 cube faces × 4×4 grid of
  256×256 DXT1 tiles, named `{faceIndex}_2_{col}_{row}.dds`, faces in order
  px,nx,py,ny,pz,nz). Frozen since ~April 2026, updates only sporadically.
- Storms/lightning: `STORMS_URL`/`STORMS_URL_COMPRESSED` in
  `StormsProvider.smali` → `https://www.gstatic.com/pixel/livewallpaper/...`
  — a protobuf `StormLocations{repeated LatLng{lat_deg,lng_deg}}`. Frozen
  since **2021** (confirmed via `Last-Modified` header) despite looking
  plausible. Google's original source was almost certainly a paid global
  network (Vaisala GLD360 or similar) — no free equivalent exists.

Goal: replace both with our own live pipelines, then edit those two smali
URL constants to point at our hosted output, rebuild/sign/reinstall the APK.

## What's built (all in `cloud_pipeline/`)

### Clouds (`pipeline/run_pipeline.py`)
Source: NASA GIBS daily true-color mosaics from **3 VIIRS satellites**
(NOAA-21, NOAA-20, Suomi NPP — `pipeline/gibs.py`). Uses **yesterday's**
date (`date.today() - 1`) because "today" is incomplete until the UTC day
finishes (verified live: querying today at 03:51 UTC gave 97% black).

Pipeline: fetch 3 sources → **`darkest_of_n_composite`** (per-pixel
darkest-of-3 picks the least-hazy of 3 independent overpasses at each
pixel — this is what actually killed the granule-stripe/smoke-plume
artifacts that plagued every other attempt, see "Rejected approaches"
below) → cache-fill any residual gaps from last run → inpaint anything
still missing (`merge.py`) → `rgb_to_cloud_mask` (brightness×saturation
heuristic → grayscale cloud density, since Google's texture is a
cloud-only mask, not a color photo — confirmed by inspecting real
Google tiles) → `fix_antarctic_cap` (replaces the inpainted mid-gray mess
south of ~-60° lat with clean white — Antarctica's real satellite data
runs out and inpainting alone looked muddy) → `cubemap.py` reprojects to
6 faces (validated against real Google tiles: poles at py/ny, no
mirroring) → `dxt1.py` (hand-written BC1/DXT1 encoder, byte-verified
against real Google tile headers) → writes tiles + `root.json`.

**Versioning (just fixed)**: `root.json` lives at a stable URL; its
`baseUrl` points at a fresh `versions/<UTC-timestamp>/` folder each run
(keeps last 5, prunes older). This matters because `CloudUpdatingTask`
only re-downloads tiles if `baseUrl` differs from what it cached last
time — a static baseUrl would mean the app updates once and never again.

Run: `venv/bin/python3 pipeline/run_pipeline.py --width 4096`
(`--base-url` to set the real public URL prefix when hosted somewhere
other than `http://localhost:8765`).

**Correct cadence: once per day.** VIIRS is polar-orbiting; the GIBS daily
mosaic only changes once every 24h regardless of how often we query it —
confirmed by testing "today" (97% gap) vs "yesterday" (~0% gap after
cache-fill). Running more than daily doesn't get fresher data.

### Storms/lightning (`pipeline/run_storms.py`, `pipeline/storms.py`)
Three sources, since no single free source is global:
- **GOES-19 (East) + GOES-18 (West) GLM** (Geostationary Lightning
  Mapper): free, public, no-auth AWS S3 buckets
  (`noaa-goes19`/`noaa-goes18`), ~20-second file cadence, genuinely live.
  Covers Americas/Atlantic/Pacific.
- **EUMETSAT MTG-I2 Lightning Imager** (`li_afa` layer via
  `https://view.eumetsat.int/geoserver/wms`, EPSG:4326 direct): raster
  "Accumulated Flash Area" product; we threshold + connected-component
  centroid it into points. Covers Europe/Africa/Atlantic.
- **Xweather (Vaisala) free tier spot-checks**: the only source with any
  India/Asia-Pacific coverage — GOES/MTG's usable range only spans
  roughly 150°W→77°E and 148°E→62°W combined, leaving **~70°E–148°E dark**
  (India, Southeast Asia, China, Japan, Australia). Free tier is capped
  at single-point 100km-radius/5-minute queries (no bounding-box —
  that needs a paid Enterprise add-on), so we spot-check a curated list
  of 109 cities (`XWEATHER_SPOT_CITIES` in `storms.py`) covering India
  (45 cities), other South Asia, SE Asia, Japan, China, Australia/NZ.
  **Night-filtered**: only queries a city while it's local-solar
  nighttime there (`_is_local_night`, longitude-based, since lightning
  presumably only renders on the night side anyway) — this roughly
  doubles effective refresh frequency for the same monthly budget.
  Budget math: `cities × checks/day × 30 ≤ 15000`. At 109 cities with
  night-filtering, ~2.2h cadence keeps monthly calls ≈14,550 (simulated
  and confirmed, see conversation — bursty per-run 0-109, but correct
  total).
  Credentials: `secrets/xweather.env` (client_id + client_secret,
  `chmod 600`, gitignored-style — **do not commit or paste in chat**).

All points get grid-clustered (`cluster_points`, ~80-120km cells) to
match Google's original sparse look (~90 points globally) rather than
raw per-flash density (hundreds-thousands of points), then encoded via a
**hand-written protobuf encoder** (`encode_storm_locations` in
`storms.py`) matching the exact wire format — verified via manual
round-trip decode against real bytes downloaded from Google's frozen
endpoint.

**Not yet done for storms**: the same `root.json`-style version-rotation
fix that clouds just got. `StormsProvider` re-checks roughly **hourly**
(`TimeUtils.getTimeString()` returns `YYYYMMDDHH` in PST, compared to a
cached value — confirmed by reading the smali, this is finer-grained
than I originally guessed) but there's no "has content changed" gate
like clouds has, so a static URL is probably fine for storms — just
re-serve the freshest file at a fixed path. Worth double-checking before
wiring in.

Run: `venv/bin/python3 pipeline/run_storms.py --out output/latest/storm_locations --xweather-id <id> --xweather-secret <secret>`

## Rejected approaches (don't redo these)
- **Terra/Aqua MODIS** (instead of VIIRS): severe granule-to-granule
  striping baked into NASA's own daily mosaic (not our bug) — MODIS's
  narrower swath means more/harsher stitching seams than VIIRS.
- **Isotropic Gaussian blur destriping**: just acts as a local-contrast
  sharpen, doesn't isolate the stripe pattern, made things worse.
- **Along-track column-average destriping** (window=250, strength 0.6-1.3):
  reduces stripes but visibly washes out contrast/dynamic range; user
  rejected as "worse."
- **Per-column banded correction**: introduced new band-seam artifacts.
- **The actual fix**: switching to VIIRS (wider swath, much less
  striping) + darkest-of-3-satellite compositing, which suppresses what
  striping/haze/glint remained by construction rather than correcting
  for it after the fact. This is the one that actually worked.
- **Meteosat "Natural Colour RGB"** (`msg_fes:rgb_natural`) as a cloud
  source: rejected — it's a false-color diagnostic palette (ice/snow
  clouds render *cyan*, not white) plus its own tile-mosaic time-seam
  artifacts. `rgb_eview` also rejected (Europe-only HRV+IR composite,
  not true color either).

## Known open items / not yet done
1. **Smali edits**: haven't touched `CloudsProvider.smali` /
   `CloudUpdatingTask.smali` (`CLOUDS_URL`) or `StormsProvider.smali`
   (`STORMS_URL`/`STORMS_URL_COMPRESSED`) yet — still pointing at
   Google's dead endpoints. Need to repoint at wherever this pipeline's
   output ends up hosted, then rebuild/resign/reinstall the APK.
2. **No real hosting chosen yet** — everything currently serves from a
   local `python3 -m http.server 8765` over `output/latest/`
   (dies when the terminal session ends; restart with:
   `cd output/latest && python3 -m http.server 8765 &`). Need to pick
   somewhere real (own server, or a scheduled GitHub Action + static
   host) and set `--base-url` accordingly. Also need actual scheduling
   (cron / GitHub Actions) for daily clouds + ~2.2h storms runs — none
   set up yet, everything's been run manually so far.
3. **Considered but not built**: a from-scratch real-time-everywhere
   clouds rebuild using GOES-East + GOES-West + MTG + **INSAT-3S**
   (ISRO, via `https://mosdac.gov.in/live_data/wms/live3SL1BSTD4km/...` —
   confirmed working, EPSG:4326 direct, thermal IR `IMG_TIR1` channel,
   ~45-65min latency, real imagery verified). Combined, these 4
   geostationary satellites might tile the *entire* 360° of longitude
   with real overlap, which would eliminate VIIRS (and its once-daily
   staleness) entirely for clouds. Not started — would need: (a) INSAT
   "find latest available 30-min slot" search logic (files aren't
   published instantly), (b) converting IR brightness-temperature to a
   cloud-density mask comparable to GOES/MTG's visible-derived masks
   (colder cloud-top = denser cloud, standard technique but not yet
   implemented), (c) 4-way feathered merge. User wants this eventually
   but asked to finish/verify the existing VIIRS-based pipeline first.

   **Tried and reverted (2026-09-06)**: attempted a lighter-weight version
   of just this piece — overlaying live INSAT-3S TIR1 on top of the VIIRS
   base for India/Indian Ocean only (endpoints, calibration approach, and
   tone-matching-via-`match_histogram` all worked and are worth reusing),
   but the result showed a visible rectangular seam/tonal patch at full-
   globe zoom that the user rejected as "not correct." Reverted. If
   revisiting: the seam was a real tonal/texture mismatch between the two
   sensors surviving histogram-matching, not a bug in the feather math
   (verified the alpha ramp itself was smooth over ~600px) — would need
   either a much larger blend zone, a differently-shaped
   (non-rectangular/radial) mask, or accepting it can only ever be
   approximate and finding a presentation where that's not visible (e.g.
   only applying it within a tighter India-only box rather than the wider
   Indian-Ocean box tried this time).
4. Day/night terminator and lightning-glow rendering are **already
   correctly implemented in the original app** (`SolarPositionCalculator`
   / `TwilightCalculator.smali`, real-time astronomical math, no network
   dependency) — nothing to build there, confirmed by reading the smali.

## Useful reference commands
```bash
# regenerate clouds (once daily is correct cadence)
cd cloud_pipeline && ./venv/bin/python3 pipeline/run_pipeline.py --width 4096

# regenerate storms (~every 2.2h)
set -a; source secrets/xweather.env; set +a
./venv/bin/python3 pipeline/run_storms.py --out output/latest/storm_locations \
  --xweather-id "$XWEATHER_CLIENT_ID" --xweather-secret "$XWEATHER_CLIENT_SECRET"

# local test server (dies with the terminal — restart as needed)
cd output/latest && python3 -m http.server 8765 &

# verify what the app would see
curl -s http://localhost:8765/root.json
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8765/storm_locations
```
