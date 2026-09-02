#!/usr/bin/env python3
"""Elevation report for the generated heightmap.

Checks the things that are easy to break and hard to see in the image itself: that the
canvas is the size and encoding the rest of the project expects, and that the ground is
as level as it is meant to be.
"""
import os

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from generate_new_dem_12k import CANVAS_M, PLAYABLE_M, OFFSET_M, BASE_Z_M

FLAT_TOL_M = 0.01        # a clean canvas should be level to within this


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dem_path = os.path.join(script_dir, "dem_new_12k.png")
    if not os.path.exists(dem_path):
        print(f"Error: {dem_path} not found. Run generate_new_dem_12k.py first.")
        return

    img = Image.open(dem_path)
    raw = np.array(img, dtype=np.float32) / 100.0
    play = raw[OFFSET_M:OFFSET_M + PLAYABLE_M, OFFSET_M:OFFSET_M + PLAYABLE_M]

    print(f"=== Elevation report: {dem_path} ===")
    print(f"canvas   {raw.shape[1]}x{raw.shape[0]} px   "
          f"{raw.min():7.2f} .. {raw.max():7.2f} m")
    print(f"playable {play.shape[1]}x{play.shape[0]} m      "
          f"{play.min():7.2f} .. {play.max():7.2f} m   "
          f"(relief {play.max() - play.min():.2f} m)")

    # --- geometry and encoding
    print("\ngeometry:")
    ok_size = raw.shape == (CANVAS_M, CANVAS_M)
    print(f"   canvas {CANVAS_M}x{CANVAS_M} px      "
          + ("ok" if ok_size else f"   <-- got {raw.shape[1]}x{raw.shape[0]}"))
    print(f"   playable {PLAYABLE_M} m at offset {OFFSET_M} m   "
          + ("ok" if OFFSET_M * 2 + PLAYABLE_M == CANVAS_M
             else "   <-- playable area is not centred"))
    # Giants reads the PNG as 16-bit centimetres; a mode that is not integral, or a value
    # past 65535 cm, will not survive the import.
    print(f"   PIL mode {img.mode!r}                "
          + ("ok" if img.mode in ("I", "I;16") else "   <-- not a 16-bit integer image"))
    top = float(np.max(raw)) * 100.0
    print(f"   peak {top:.0f} cm                    "
          + ("ok" if top <= 65535.0 else "   <-- past the 16-bit ceiling"))

    # --- flatness
    spread = float(play.max() - play.min())
    off = float(np.abs(np.median(play) - BASE_Z_M))
    print("\nflatness (playable area):")
    print(f"   median {np.median(play):.2f} m, expected {BASE_Z_M:.2f} m"
          + ("" if off <= FLAT_TOL_M else "   <-- off the datum"))
    print(f"   spread {spread:.4f} m"
          + ("" if spread <= FLAT_TOL_M else "   <-- not level"))
    if spread > FLAT_TOL_M:
        peak = np.unravel_index(int(np.argmax(play)), play.shape)
        pit = np.unravel_index(int(np.argmin(play)), play.shape)
        print(f"   highest {play[peak]:.2f} m at X={peak[1]} Y={peak[0]}")
        print(f"   lowest  {play[pit]:.2f} m at X={pit[1]} Y={pit[0]}")

    # --- slope
    gy, gx = np.gradient(play)      # 1 px = 1 m, so the gradient is already a slope
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    print(f"\nslope deg  median {np.median(slope):.2f}   "
          f"p99 {np.percentile(slope, 99):.2f}   max {slope.max():.2f}")


if __name__ == "__main__":
    main()
