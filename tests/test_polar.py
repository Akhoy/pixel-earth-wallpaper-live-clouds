"""Synthetic-data tests for pipeline/polar.py (no network, no pytest needed).

Run: venv/bin/python3 -m unittest discover -s tests -v
"""
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
import polar  # noqa: E402
from cubemap import equirect_to_cubemap  # noqa: E402

W, H = 1024, 512


def _lat():
    return (90 - (np.arange(H) + 0.5) / H * 180)[:, None] * np.ones((1, W))


def synthetic_mask(south_hole_lat=-84.0, north_hole_lat=72.0, fringe=True, seed=0):
    """Cloud mask: textured clouds everywhere, bright ice south of -62, no
    data poleward of the hole latitudes, and a ring of dark half-lit pixels
    around the south hole (what real days have)."""
    rng = np.random.default_rng(seed)
    cm = np.clip(130 + 70 * (rng.random((H // 8, W // 8)).repeat(8, 0).repeat(8, 1) - 0.5), 0, 255)
    lat = _lat()
    cm = np.where(lat < -62, 240.0, cm)
    if fringe:
        cm = np.where((lat < south_hole_lat + 3) & (lat >= south_hole_lat), 120.0, cm)
    cm = np.where(lat < south_hole_lat, 0.0, cm)
    cm = np.where(lat > north_hole_lat, 0.0, cm)
    return cm.astype(np.uint8)


def faces_of(cm):
    return equirect_to_cubemap(np.stack([cm] * 3, -1), polar.FACE_SIZE)


class SouthHoleTests(unittest.TestCase):
    def test_hole_covers_dark_fringe_but_not_clean_ice(self):
        cm = synthetic_mask(south_hole_lat=-84)
        hole = polar.south_hole(cm)
        lat = _lat()
        self.assertTrue(hole[lat < -84].all(), "no-data region must be inside the mask")
        self.assertTrue(hole[(lat < -82.5) & (lat >= -83.5)].all(), "dark half-lit fringe must be inside the mask")
        self.assertFalse(hole[lat > -76].any(), "mask must not eat clean ice far from the hole")

    def test_no_hole_when_pole_is_covered(self):
        cm = synthetic_mask(south_hole_lat=-90, fringe=False)
        self.assertFalse(polar.south_hole(cm).any())

    def test_grows_with_hole_size(self):
        small = polar.south_hole(synthetic_mask(south_hole_lat=-86)).sum()
        big = polar.south_hole(synthetic_mask(south_hole_lat=-78)).sum()
        self.assertGreater(big, 1.5 * small)

    def test_fill_has_no_ring_or_dark_pixels(self):
        cm = synthetic_mask(south_hole_lat=-80)
        Hm = polar._face(polar.south_hole(cm).astype(np.float32) * 255, "nz") / 255.0
        A = faces_of(cm)["nz"][..., 0].astype(np.float32)
        for donor in (True, False):
            out = polar.fill_hole(A, Hm, "nz" if donor else None)
            qa = polar.check_face(out, Hm, "south")
            self.assertTrue(qa["ok"], qa)


class DatelineTests(unittest.TestCase):
    WIDTH = 4096                          # real map width: the cross-fade band is 24 px here (it scales with width)

    def _img(self):
        img = np.zeros((32, self.WIDTH), np.uint8)
        img[:, : self.WIDTH // 2] = 40
        img[:, self.WIDTH // 2:] = 200
        return np.roll(img, self.WIDTH // 2, axis=1)    # left edge 200, right edge 40: a hard date-line seam

    def test_seam_jump_removed_and_interior_untouched(self):
        from merge import soften_dateline
        img = self._img()
        out = soften_dateline(img)
        before = abs(int(img[0, 0]) - int(img[0, -1]))
        self.assertEqual(before, 160)
        self.assertLess(abs(float(out[0, 0]) - float(out[0, -1])), 0.1 * before)    # >90% of the jump gone
        band = int(round(self.WIDTH * 24 / 4096))
        self.assertTrue(np.array_equal(out[:, band:-band], img[:, band:-band]))

    def test_gap_pixels_are_left_alone(self):
        from merge import soften_dateline
        img = self._img()
        gap = np.zeros(img.shape, np.float32)
        gap[10:20, :] = 1
        out = soften_dateline(img, gap)
        self.assertTrue(np.array_equal(out[10:20], img[10:20]))
        self.assertFalse(np.array_equal(out[25:30], img[25:30]))

    def test_infrared_seam_is_softened(self):
        ir = np.where(np.arange(self.WIDTH)[None, :] < self.WIDTH // 2, 80.0, 190.0) * np.ones((256, 1))
        ir = np.roll(ir, self.WIDTH // 2, axis=1).astype(np.float32)               # edges 190 | 80
        got = polar.fetch_polar_infrared(date(2026, 10, 13), self.WIDTH, lambda *a: ir, log=lambda *_: None)
        self.assertLess(abs(float(got[0][0, 0]) - float(got[0][0, -1])), 0.1 * 110)


class HoleSurvivesGapFillTests(unittest.TestCase):
    """Regression: with no cache the pipeline inpaints thin gaps that touch the polar hole; that smeared
    non-zero values into the hole, polar.south_hole then recognised only about half of it, and the
    Antarctic fill failed every check (a dark disc) -- only on GitHub, which has no cache."""

    def test_polar_hole_is_still_found_after_cache_free_gap_fill(self):
        from merge import find_polar_hole, fill_from_cache, inpaint_remaining_gaps
        cm = synthetic_mask(south_hole_lat=-84, fringe=False)
        still_gap = (cm == 0).astype(np.float32)
        still_gap[330:340, 100:600] = 1                       # thin swath gaps (a few rows tall)
        still_gap[(_lat() < -78) & (np.arange(W)[None, :] % 97 < 4)] = 1    # and slivers touching the hole's edge
        raw = np.where(still_gap > 0.5, 0, cm).astype(np.uint8)
        hole = find_polar_hole(still_gap)
        remaining = still_gap * (1.0 - hole)
        pseudo = inpaint_remaining_gaps(np.stack([raw] * 3, -1), remaining)
        smeared = pseudo[..., 0]
        self.assertTrue((smeared[hole > 0.5] != 0).any(), "premise: inpainting does leak into the hole")
        restored = np.where(hole > 0.5, raw, smeared)
        self.assertTrue((restored[hole > 0.5] == 0).all())
        covered = polar.south_hole(restored)[_lat() < -86].mean()
        self.assertGreater(covered, 0.99)


class QualityCheckTests(unittest.TestCase):
    def _hole(self, radius=100):
        R = polar._radius_grid()
        return (R < radius).astype(np.float32)

    def test_detects_dark_ring_and_disc(self):
        R = polar._radius_grid()
        hole = self._hole()
        good = np.full((1024, 1024), 240, np.uint8)
        self.assertTrue(polar.check_face(good, hole, "south")["ok"])
        ring = good.copy()
        ring[(R > 70) & (R < 95)] = 40                       # dark ring inside the hole (the Oct-5 failure)
        self.assertFalse(polar.check_face(ring, hole, "south")["ok"])
        disc = good.copy()
        disc[R < 90] = 20                                    # black disc
        self.assertFalse(polar.check_face(disc, hole, "south")["ok"])
        self.assertFalse(polar.check_face(disc, hole, "north")["ok"])

    def test_no_hole_is_ok(self):
        self.assertTrue(polar.check_face(np.zeros((1024, 1024), np.uint8), np.zeros((1024, 1024), np.float32), "south")["ok"])


class InfraredTests(unittest.TestCase):
    def _ir(self, mean=113.0, cap_black=0.0):
        ir = np.full((512, 1024), mean, np.float32)
        n = int(1024 * cap_black)
        if n:
            ir[: int(512 * 10 / 180), :n] = 0
        return ir

    def test_completeness(self):
        self.assertTrue(polar.ir_is_complete(self._ir()))
        self.assertFalse(polar.ir_is_complete(self._ir(mean=75)))          # Oct 13: partial day
        self.assertFalse(polar.ir_is_complete(self._ir(cap_black=0.10)))   # Oct 12: black wedge at the pole

    def test_falls_back_to_previous_complete_day(self):
        calls = []

        def fetch(sat, day, width):
            calls.append(day)
            return self._ir(mean=75) if day in ("2026-10-13", "2026-10-12") else self._ir()

        got = polar.fetch_polar_infrared(date(2026, 10, 13), 1024, fetch, log=lambda *_: None)
        self.assertIsNotNone(got)
        self.assertEqual(str(got[1]), "2026-10-11")

    def test_gives_up_when_no_day_is_complete(self):
        got = polar.fetch_polar_infrared(date(2026, 10, 13), 1024, lambda *a: self._ir(mean=50), log=lambda *_: None)
        self.assertIsNone(got)

    def test_survives_fetch_errors(self):
        def boom(*a):
            raise RuntimeError("GIBS down")

        self.assertIsNone(polar.fetch_polar_infrared(date(2026, 10, 13), 1024, boom, log=lambda *_: None))


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.cm = synthetic_mask(south_hole_lat=-82, north_hole_lat=74)
        self.faces = faces_of(self.cm)

    def _ir_fetch(self, sat, day, width):
        rng = np.random.default_rng(1)
        return (140 + 60 * rng.random((H // 8, W // 8)).repeat(8, 0).repeat(8, 1)).astype(np.float32)

    def test_fills_both_poles_and_reports_methods(self):
        out, rep = polar.apply_polar_fixes(self.faces, self.cm, date(2026, 10, 19), W, log=lambda *_: None,
                                           fetch=self._ir_fetch)
        self.assertEqual(rep["north"]["method"], "infrared overlay")   # regression: accepted fill must be used
        self.assertEqual(rep["south"]["method"], "ice fill + Google texture")
        self.assertFalse(np.array_equal(out["pz"], self.faces["pz"]))
        self.assertFalse(np.array_equal(out["nz"], self.faces["nz"]))
        for name in ("px", "nx", "py", "ny"):
            self.assertTrue(np.array_equal(out[name], self.faces[name]), name)

    def test_no_infrared_uses_cloud_fill_not_a_dark_disc(self):
        def boom(*a):
            raise RuntimeError("GIBS down")

        out, rep = polar.apply_polar_fixes(self.faces, self.cm, date(2026, 10, 19), W, log=lambda *_: None, fetch=boom)
        self.assertEqual(rep["north"]["method"], "cloud fill + Google texture")
        R = polar._radius_grid()
        self.assertGreater(out["pz"][..., 0][R < 40].mean(), 60)        # the hole is filled, not black/faded

    def test_older_infrared_day_is_tried_when_newest_fails_check(self):
        def fetch(sat, day, width):
            ir = self._ir_fetch(sat, day, width).copy()
            ir[0, 500] = 140.0 if day == "2026-10-19" else 150.0          # marks the newest day
            return ir

        orig = polar.arctic_overlay

        def fake(photo, hole, ir, cm, force_smooth=False):
            if abs(float(ir[0, 500]) - 140.0) < 0.01:                     # newest day's fill is broken
                return np.zeros(photo.shape, np.uint8), {"patch": False, "patch_score": None}
            return orig(photo, hole, ir, cm, force_smooth)

        polar.arctic_overlay = fake
        try:
            _, rep = polar.apply_polar_fixes(self.faces, self.cm, date(2026, 10, 19), W, log=lambda *_: None,
                                             fetch=fetch)
        finally:
            polar.arctic_overlay = orig
        self.assertEqual(rep["north"]["method"], "infrared overlay")
        self.assertEqual(rep["north"]["ir_day"], "2026-10-18")          # fell back to the previous day's infrared
        self.assertEqual(len(rep["north"]["rejected"]), 2)

    def test_infrared_disabled(self):
        _, rep = polar.apply_polar_fixes(self.faces, self.cm, date(2026, 10, 19), W, log=lambda *_: None, use_ir=False)
        self.assertEqual(rep["north"]["method"], "cloud fill + Google texture")
        self.assertEqual(rep["north"]["no_infrared"], "disabled")

    def test_arctic_fade_is_the_last_resort(self):
        saved = polar.QA["north"].copy()
        polar.QA["north"] = {"max_step": 0.0, "min_dark": 300.0}
        try:
            _, rep = polar.apply_polar_fixes(self.faces, self.cm, date(2026, 10, 19), W, log=lambda *_: None,
                                             use_ir=False)
        finally:
            polar.QA["north"] = saved
        self.assertEqual(rep["north"]["method"], "fade (fills failed check)")

    def test_failed_check_falls_back_to_fade(self):
        saved = polar.QA["south"].copy()
        polar.QA["south"] = {"max_step": 0.0, "min_dark": 300.0}      # nothing can pass
        try:
            _, rep = polar.apply_polar_fixes(self.faces, self.cm, date(2026, 10, 19), W, log=lambda *_: None,
                                             use_ir=False)
        finally:
            polar.QA["south"] = saved
        self.assertEqual(rep["south"]["method"], "fade (fills failed check)")
        self.assertEqual(len(rep["south"]["rejected"]), 3)      # two fills + the Google tile

    def test_google_tile_is_used_before_an_empty_disc(self):
        saved = polar.fill_hole
        polar.fill_hole = lambda A, Hm, donor=None: np.zeros(A.shape, np.uint8)   # both own-data fills "fail"
        try:
            out, rep = polar.apply_polar_fixes(self.faces, self.cm, date(2026, 10, 19), W, log=lambda *_: None,
                                               use_ir=False)
        finally:
            polar.fill_hole = saved
        self.assertEqual(rep["north"]["method"], "Google tile")
        self.assertEqual(rep["south"]["method"], "Google tile")
        R = polar._radius_grid()
        for face in ("pz", "nz"):                                  # filled with cloud, not black and not faded
            self.assertGreater(out[face][..., 0][R < 40].mean(), 60, face)

    def test_google_tile_missing_falls_through_to_fade(self):
        saved = (polar.fill_hole, polar.ASSETS)
        polar.fill_hole = lambda A, Hm, donor=None: np.zeros(A.shape, np.uint8)
        polar.ASSETS = Path("/nonexistent")
        try:
            _, rep = polar.apply_polar_fixes(self.faces, self.cm, date(2026, 10, 19), W, log=lambda *_: None,
                                             use_ir=False)
        finally:
            polar.fill_hole, polar.ASSETS = saved
        self.assertEqual(rep["north"]["method"], "fade (fills failed check)")
        self.assertEqual(rep["south"]["method"], "fade (fills failed check)")

    def test_complete_pole_is_untouched(self):
        cm = synthetic_mask(south_hole_lat=-90, north_hole_lat=90, fringe=False)
        faces = faces_of(cm)
        out, rep = polar.apply_polar_fixes(faces, cm, date(2026, 10, 19), W, log=lambda *_: None, use_ir=False)
        self.assertFalse(rep["north"]["hole"] or rep["south"]["hole"])
        self.assertTrue(np.array_equal(out["pz"], faces["pz"]) and np.array_equal(out["nz"], faces["nz"]))


if __name__ == "__main__":
    unittest.main()
