"""Synthetic-data tests for pipeline/polar.py and the date-line fade (no network, no pytest needed).

Run: venv/bin/python3 -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
import polar  # noqa: E402
import run_pipeline  # noqa: E402
from cubemap import equirect_to_cubemap  # noqa: E402
from dxt1 import decode_dds_tile, encode_tile_to_dds  # noqa: E402
from merge import find_polar_hole, inpaint_remaining_gaps, soften_dateline  # noqa: E402

W, H = 1024, 512
QUIET = lambda *_: None  # noqa: E731


def _lat():
    return (90 - (np.arange(H) + 0.5) / H * 180)[:, None] * np.ones((1, W))


def synthetic_mask(south_hole_lat=-84.0, north_hole_lat=78.0, fringe=True, seed=0):
    """Cloud mask: textured clouds everywhere, bright ice south of -62, no data poleward of the hole latitudes,
    and a ring of dark half-lit pixels around the south hole (what real days have)."""
    rng = np.random.default_rng(seed)
    cm = np.clip(130 + 70 * (rng.random((H // 8, W // 8)).repeat(8, 0).repeat(8, 1) - 0.5), 0, 255)
    lat = _lat()
    cm = np.where(lat < -62, 240.0, cm)
    if fringe:
        cm = np.where((lat < south_hole_lat + 2) & (lat >= south_hole_lat), 120.0, cm)
    cm = np.where(lat < south_hole_lat, 0.0, cm)
    cm = np.where(lat > north_hole_lat, 0.0, cm)
    return cm.astype(np.uint8)


def faces_of(cm):
    return equirect_to_cubemap(np.stack([cm] * 3, -1), polar.FACE_SIZE)


def run(cm=None, **kw):
    cm = synthetic_mask() if cm is None else cm
    faces = faces_of(cm)
    out, rep = polar.apply_polar_fixes(faces, cm, log=QUIET, **kw)
    return faces, out, rep


class HoleMeasuringTests(unittest.TestCase):
    def test_south_hole_covers_dark_fringe_but_not_clean_ice(self):
        hole = polar.south_hole(synthetic_mask(south_hole_lat=-84))
        lat = _lat()
        self.assertTrue(hole[lat < -84].all(), "no-data region must be inside the mask")
        self.assertTrue(hole[(lat < -82.5) & (lat >= -83.5)].all(), "dark half-lit fringe must be inside the mask")
        self.assertFalse(hole[lat > -76].any(), "mask must not eat clean ice far from the hole")

    def test_north_hole_is_the_mirror_image(self):
        hole = polar.north_hole(synthetic_mask(north_hole_lat=78))
        lat = _lat()
        self.assertTrue(hole[lat > 78].all())
        self.assertFalse(hole[lat < 70].any())

    def test_no_hole_when_pole_is_covered(self):
        cm = synthetic_mask(south_hole_lat=-90, north_hole_lat=90, fringe=False)
        self.assertFalse(polar.south_hole(cm).any() or polar.north_hole(cm).any())

    def test_grows_with_hole_size(self):
        small = polar.south_hole(synthetic_mask(south_hole_lat=-86)).sum()
        big = polar.south_hole(synthetic_mask(south_hole_lat=-78)).sum()
        self.assertGreater(big, 1.5 * small)

    def test_measuring_scales_with_map_width(self):
        """Regression: the grow/margin pixel counts were tuned at width 4096; at width 1024 they were 4x too big
        relative to the map and swelled a 112 px hole to 211 px."""
        def radius(width):
            rng = np.random.default_rng(0)
            h = width // 2
            cm = np.clip(130 + 70 * (rng.random((h // 8, width // 8)).repeat(8, 0).repeat(8, 1) - 0.5), 0, 255)
            lat = (90 - (np.arange(h) + 0.5) / h * 180)[:, None] * np.ones((1, width))
            cm = np.where(lat > 80, 0, cm).astype(np.uint8)
            hole = polar._face(polar.north_hole(cm).astype(np.float32) * 255, "pz") / 255.0
            return polar._hole_radius(hole)
        small, big = radius(1024), radius(4096)
        self.assertLess(abs(small - big) / big, 0.15)


class DimBandTests(unittest.TestCase):
    """The dim terminator band around a winter hole varies (30 px at the Arctic in Dec 2025, 75 px at Antarctica
    in Jun 2026, none on Oct 23 2026). A fixed growth left a 40 px grey ring in June."""

    def _photo(self, r_hole, band):
        from scipy.ndimage import distance_transform_edt
        R = polar._radius_grid()
        hole = (R < r_hole).astype(np.float32)
        d = distance_transform_edt(hole < 0.5)
        photo = np.full(R.shape, 150.0, np.float32)
        photo[(d > 0) & (d <= band)] = 40.0
        photo[hole > 0.5] = 0.0
        return hole, photo

    def test_a_wide_band_is_covered(self):
        hole, photo = self._photo(300, 75)
        self.assertGreater(polar._hole_radius(polar.grow_face_hole(hole, photo)) - 300, 70)

    def test_no_band_means_no_growth(self):
        hole, photo = self._photo(220, 0)
        self.assertLess(polar._hole_radius(polar.grow_face_hole(hole, photo)) - 220, 4)

    def test_small_holes_are_never_grown(self):
        hole, photo = self._photo(120, 40)
        self.assertTrue(np.array_equal(polar.grow_face_hole(hole, photo), hole))


class FillTests(unittest.TestCase):
    def _south(self, hole_lat=-80):
        cm = synthetic_mask(south_hole_lat=hole_lat)
        Hm = polar.grow_face_hole(polar._face(polar.south_hole(cm).astype(np.float32) * 255, "nz") / 255.0)
        return faces_of(cm)["nz"][..., 0].astype(np.float32), Hm

    def test_fill_passes_the_sanity_check_with_and_without_the_texture(self):
        A, Hm = self._south()
        saved = polar.TEXTURE_PATH
        try:
            for path in (saved, Path("/nonexistent/texture.png")):
                polar.TEXTURE_PATH = path
                ok, why = polar.fill_looks_ok(polar.fill_hole(A, Hm, "nz"), Hm)
                self.assertTrue(ok, why)
        finally:
            polar.TEXTURE_PATH = saved

    def test_spread_takes_its_tone_from_the_nearest_edge(self):
        """Regression: a coarse-pyramid fill pulled a hole in the middle of white ice toward the global mean
        (the dark ocean far away): centre 207 vs 240 around it."""
        R = polar._radius_grid()
        photo = np.where(R < 330, 240.0, 40.0).astype(np.float32)
        hole = (R < 150).astype(np.float32)
        out = polar._spread_inward(np.where(hole > 0.5, 0, photo).astype(np.float32), 1 - hole)
        self.assertGreater(float(out[R < 100].mean()), 230.0)

    def test_spread_has_no_concentric_steps(self):
        R = polar._radius_grid()
        yy, xx = np.mgrid[0:polar.FACE_SIZE, 0:polar.FACE_SIZE]
        photo = (120 + 0.12 * xx).astype(np.float32)
        hole = (R < 300).astype(np.float32)
        out = polar._spread_inward(np.where(hole > 0.5, 0, photo), 1 - hole)
        rings = [out[(R > r) & (R < r + 6)].mean() for r in range(40, 280, 6)]
        self.assertLess(float(np.max(np.abs(np.diff(rings)))), 6.0)

    def test_spread_meets_the_ring_without_a_step(self):
        R = polar._radius_grid()
        n = polar.FACE_SIZE
        ang = np.arctan2(*(np.mgrid[0:n, 0:n] - (n - 1) / 2))
        sectors = np.where((((ang % (2 * np.pi)) / (2 * np.pi) * 12).astype(int)) % 2 == 0, 235.0, 70.0)
        out = polar._spread_inward(np.where(R >= 62, sectors, 0).astype(np.float32), (R >= 62).astype(np.float32))
        step = abs(out[(R >= 58) & (R < 62)].mean() - out[(R >= 62) & (R < 66)].mean())
        self.assertLess(step, 3.0)

    def test_texture_follows_the_contrast_of_the_surroundings(self):
        """The fixed texture must not paint dark blobs on flat white ice, and must give busy clouds some detail."""
        R = polar._radius_grid()
        hole = (R < 150).astype(np.float32)
        flat = np.where(hole > 0.5, 0, 240.0).astype(np.float32)
        rng = np.random.default_rng(2)
        busy = np.where(hole > 0.5, 0, np.clip(150 + 45 * rng.standard_normal(R.shape), 0, 255)).astype(np.float32)
        self.assertLess(float(polar.fill_hole(flat, hole, "nz")[R < 100].std()), 6.0)
        self.assertGreater(float(polar.fill_hole(busy, hole, "nz")[R < 100].std()), 5.0)

    def test_the_baked_texture_asset_is_there_and_valid(self):
        from PIL import Image
        img = Image.open(polar.TEXTURE_PATH)
        self.assertEqual(img.size, (polar.FACE_SIZE, polar.FACE_SIZE))
        tex = np.array(img.convert("L"))
        self.assertGreater(float(tex.std()), 20)                  # real cloud texture, not a flat image


class SanityTests(unittest.TestCase):
    def _hole(self, r=205):
        return (polar._radius_grid() < r).astype(np.float32)

    def test_a_dead_fill_is_caught(self):
        R, hole = polar._radius_grid(), self._hole()
        good = np.full(R.shape, 190, np.uint8)
        self.assertTrue(polar.fill_looks_ok(good, hole)[0])
        dead = good.copy()
        dead[R < 200] = 8
        self.assertFalse(polar.fill_looks_ok(dead, hole)[0])

    def test_a_hard_edge_is_caught(self):
        R, hole = polar._radius_grid(), self._hole()
        face = np.where(R < 205, 250, 60).astype(np.uint8)
        self.assertFalse(polar.fill_looks_ok(face, hole)[0])

    def test_a_good_fill_with_dark_patches_is_not_rejected(self):
        """Regression (Oct 23 2026): three good Arctic fills were rejected because 'the darkest 2% was below 25'
        -- clear Arctic water is dark -- and Google's blocky April tile was published instead for three days. A
        fill is dead when it is MOSTLY black, not when part of it is."""
        R, hole = polar._radius_grid(), self._hole()
        face = np.full(R.shape, 190, np.uint8)
        face[(R > 40) & (R < 120) & (np.abs(np.sin(R / 7.0)) > 0.985)] = 10      # thin dark arcs, a few percent
        frac = float((face[R < 0.9 * 205] < polar.DARK_LEVEL).mean())
        self.assertTrue(0.01 < frac < 0.08, frac)
        self.assertTrue(polar.fill_looks_ok(face, hole)[0])

    def test_tiny_holes_and_non_finite_pixels(self):
        R = polar._radius_grid()
        self.assertTrue(polar.fill_looks_ok(np.zeros(R.shape, np.uint8), (R < 5).astype(np.float32))[0])
        bad = np.full(R.shape, 190.0, np.float32)
        bad[100, 100] = np.nan
        self.assertFalse(polar.fill_looks_ok(bad, self._hole())[0])


class PreviousPublishedTests(unittest.TestCase):
    def test_dds_round_trip(self):
        rng = np.random.default_rng(0)
        g = np.clip(120 + 90 * np.sin(np.add.outer(np.arange(256) / 9.0, np.arange(256) / 13.0))
                    + rng.normal(0, 8, (256, 256)), 0, 255).astype(np.uint8)
        rgb = np.stack([g] * 3, -1)
        back = decode_dds_tile(encode_tile_to_dds(rgb))
        self.assertEqual(back.shape, rgb.shape)
        self.assertLess(np.abs(back.astype(int) - rgb.astype(int)).mean(), 8)      # DXT1 is lossy

    def _fake_site(self, faces):
        """A fake `requests.get` serving root.json and the 32 polar tiles for the given {pz,nz} 2-D faces."""
        class Resp:
            def __init__(self, content=b"", data=None):
                self.content, self._data = content, data

            def raise_for_status(self):
                pass

            def json(self):
                return self._data

        tiles = {}
        for name, idx in (("pz", 4), ("nz", 5)):
            for c in range(4):
                for r in range(4):
                    t = faces[name][r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
                    tiles[f"https://site/versions/v1/{idx}_2_{c}_{r}.dds"] = encode_tile_to_dds(np.stack([t] * 3, -1))

        def get(url, timeout=None):
            if url == "https://site/root.json":
                return Resp(data={"baseUrl": "https://site/versions/v1/"})
            return Resp(content=tiles[url])
        return get

    def test_reads_back_the_published_polar_faces(self):
        rng = np.random.default_rng(3)
        faces = {n: np.clip(128 + 60 * rng.standard_normal((1024, 1024)), 0, 255).astype(np.uint8) for n in ("pz", "nz")}
        got = polar.fetch_previous_faces("https://site", log=QUIET, get=self._fake_site(faces))
        self.assertEqual(set(got), {"pz", "nz"})
        self.assertLess(np.abs(got["pz"].astype(int) - faces["pz"].astype(int)).mean(), 40)

    def test_unreachable_or_broken_site_gives_nothing_and_never_raises(self):
        def down(url, timeout=None):
            raise ConnectionError("no route")

        class Junk:
            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("not json")
        self.assertEqual(polar.fetch_previous_faces("https://site", log=QUIET, get=down), {})
        self.assertEqual(polar.fetch_previous_faces("https://site", log=QUIET, get=lambda u, timeout=None: Junk()), {})


class DriverTests(unittest.TestCase):
    """The guarantees: the equator is never touched, and nothing here can raise."""

    EQUATOR = ("px", "nx", "py", "ny")

    def _assert_equator_untouched(self, faces, out):
        for n in self.EQUATOR:
            self.assertTrue(np.array_equal(out[n], faces[n]), n)

    def test_fills_both_poles_and_never_touches_the_equator(self):
        called = []
        faces, out, rep = run(get_previous=lambda: called.append(1) or {})
        self.assertEqual(rep["arctic"]["method"], "fixed texture fill")
        self.assertEqual(rep["antarctic"]["method"], "fixed texture fill")
        self.assertFalse(np.array_equal(out["pz"], faces["pz"]))
        self.assertFalse(np.array_equal(out["nz"], faces["nz"]))
        self._assert_equator_untouched(faces, out)
        self.assertEqual(called, [], "previous tiles are only fetched when a fill fails")

    def test_complete_pole_is_untouched(self):
        cm = synthetic_mask(south_hole_lat=-90, north_hole_lat=90, fringe=False)
        faces, out, rep = run(cm)
        self.assertFalse(rep["arctic"]["hole"] or rep["antarctic"]["hole"])
        self.assertTrue(np.array_equal(out["pz"], faces["pz"]) and np.array_equal(out["nz"], faces["nz"]))

    def test_a_broken_fill_reuses_the_last_published_tiles(self):
        previous = {"pz": np.full((1024, 1024), 200, np.uint8), "nz": np.full((1024, 1024), 210, np.uint8)}
        saved = polar.fill_hole
        polar.fill_hole = lambda A, Hm, face: np.zeros(A.shape, np.uint8)               # a dead fill
        try:
            faces, out, rep = run(get_previous=lambda: previous)
        finally:
            polar.fill_hole = saved
        self.assertEqual(rep["arctic"]["method"], "last published")
        self.assertEqual(rep["antarctic"]["method"], "last published")
        R = polar._radius_grid()
        self.assertGreater(float(out["pz"][..., 0][R < 40].mean()), 150)                 # yesterday's, not black
        self._assert_equator_untouched(faces, out)

    def test_a_broken_fill_and_no_previous_fades_to_empty(self):
        saved = polar.fill_hole
        polar.fill_hole = lambda A, Hm, face: np.zeros(A.shape, np.uint8)
        try:
            faces, out, rep = run(get_previous=lambda: {})
        finally:
            polar.fill_hole = saved
        self.assertEqual(rep["arctic"]["method"], "fade to empty")
        self._assert_equator_untouched(faces, out)

    def test_a_fill_that_raises_falls_through_to_the_last_published_tiles(self):
        def boom(*a, **k):
            raise RuntimeError("fill exploded")
        saved = polar.fill_hole
        polar.fill_hole = boom
        try:
            faces, out, rep = run(get_previous=lambda: {"pz": np.full((1024, 1024), 180, np.uint8),
                                                        "nz": np.full((1024, 1024), 180, np.uint8)})
        finally:
            polar.fill_hole = saved
        self.assertEqual(rep["arctic"]["method"], "last published")
        self.assertIn("fill exploded", rep["arctic"]["error"])

    def test_everything_failing_leaves_the_plain_faces_and_does_not_raise(self):
        def boom(*a, **k):
            raise RuntimeError("boom")
        saved = (polar.fill_hole, polar.fade_to_hole)
        polar.fill_hole = polar.fade_to_hole = boom
        try:
            faces, out, rep = run(get_previous=boom)
        finally:
            polar.fill_hole, polar.fade_to_hole = saved
        self.assertEqual(rep["arctic"]["method"], "plain (error)")
        self.assertEqual(rep["antarctic"]["method"], "plain (error)")
        for n in faces:
            self.assertTrue(np.array_equal(out[n], faces[n]), n)

    def test_one_pole_failing_does_not_stop_the_other(self):
        def boom(cm):
            raise RuntimeError("cannot measure")
        saved = polar.north_hole
        polar.north_hole = boom
        try:
            faces, out, rep = run()
        finally:
            polar.north_hole = saved
        self.assertEqual(rep["arctic"]["method"], "plain (error)")
        self.assertEqual(rep["antarctic"]["method"], "fixed texture fill")
        self.assertTrue(np.array_equal(out["pz"], faces["pz"]))

    def test_the_pipeline_wrapper_never_lets_a_polar_failure_through(self):
        saved = run_pipeline.apply_polar_fixes

        def boom(*a, **k):
            raise RuntimeError("polar module blew up")
        run_pipeline.apply_polar_fixes = boom
        try:
            faces = faces_of(synthetic_mask())
            report = {}
            out = run_pipeline.apply_polar_safely(faces, synthetic_mask(), "http://localhost:1", 40, report)
        finally:
            run_pipeline.apply_polar_fixes = saved
        for n in faces:
            self.assertTrue(np.array_equal(out[n], faces[n]), n)
        self.assertIn("polar module blew up", report["error"])
        json.dumps(report)                                                              # and it is writable as the report


class DatelineTests(unittest.TestCase):
    WIDTH = 4096                                   # the cross-fade band is 24 px here (it scales with the width)

    def _img(self):
        img = np.zeros((32, self.WIDTH), np.uint8)
        img[:, : self.WIDTH // 2] = 40
        img[:, self.WIDTH // 2:] = 200
        return np.roll(img, self.WIDTH // 2, axis=1)    # left edge 200, right edge 40: a hard date-line seam

    def test_seam_jump_removed_and_interior_untouched(self):
        img = self._img()
        out = soften_dateline(img)
        before = abs(int(img[0, 0]) - int(img[0, -1]))
        self.assertEqual(before, 160)
        self.assertLess(abs(float(out[0, 0]) - float(out[0, -1])), 0.1 * before)
        band = int(round(self.WIDTH * 24 / 4096))
        self.assertTrue(np.array_equal(out[:, band:-band], img[:, band:-band]))

    def test_gap_pixels_are_left_alone(self):
        img = self._img()
        gap = np.zeros(img.shape, np.float32)
        gap[10:20, :] = 1
        out = soften_dateline(img, gap)
        self.assertTrue(np.array_equal(out[10:20], img[10:20]))
        self.assertFalse(np.array_equal(out[25:30], img[25:30]))


class HoleSurvivesGapFillTests(unittest.TestCase):
    """Regression: with no cache the pipeline inpaints thin gaps that touch the polar hole; that smeared non-zero
    values into the hole, polar.south_hole then recognised only about half of it, and the Antarctic fill failed --
    only on GitHub, which has no cache."""

    def test_polar_hole_is_still_found_after_cache_free_gap_fill(self):
        cm = synthetic_mask(south_hole_lat=-84, fringe=False)
        still_gap = (cm == 0).astype(np.float32)
        still_gap[330:340, 100:600] = 1
        still_gap[(_lat() < -78) & (np.arange(W)[None, :] % 97 < 4)] = 1
        raw = np.where(still_gap > 0.5, 0, cm).astype(np.uint8)
        hole = find_polar_hole(still_gap)
        smeared = inpaint_remaining_gaps(np.stack([raw] * 3, -1), still_gap * (1.0 - hole))[..., 0]
        self.assertTrue((smeared[hole > 0.5] != 0).any(), "premise: inpainting does leak into the hole")
        restored = np.where(hole > 0.5, raw, smeared)
        self.assertTrue((restored[hole > 0.5] == 0).all())
        self.assertGreater(polar.south_hole(restored)[_lat() < -86].mean(), 0.99)


if __name__ == "__main__":
    unittest.main()
