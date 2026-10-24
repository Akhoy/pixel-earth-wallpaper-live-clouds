"""Bake pipeline/assets/polar_texture.png, the fixed cloud texture polar.py puts in a polar no-data hole.

It is a real photograph of ordinary clouds (a mid-latitude Pacific cube face), used only for its fine texture:
polar.py takes the surrounding clouds' brightness from the live map and adds this texture's detail on top, scaled
to the contrast of the surroundings. It never changes, so nothing here is presented as current weather and
nothing can fail on a given day. (Google's own polar tile was tried first; it carries angular stitching seams.)

Usage: venv/bin/python3 scripts/make_polar_texture.py PATH/TO/_debug_face_ny.png
(the one in the repo came from the ny face of the 2026-10-23 run.)
"""
import sys
from pathlib import Path

from PIL import Image

OUT = Path(__file__).parent.parent / "pipeline" / "assets" / "polar_texture.png"

if __name__ == "__main__":
    img = Image.open(sys.argv[1]).convert("L")
    assert img.size == (1024, 1024), img.size
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"wrote {OUT}")
