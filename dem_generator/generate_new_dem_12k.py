#!/usr/bin/env python3
"""FS25 heightmap generator - clean base canvas.

Builds the 12288x12288 m canvas (1 px = 1 m) the project has always used, with the
8192x8192 m playable area centred in it, and fills it with level ground at BASE_Z_M.

There is no landform here on purpose: no river, no lake, no settlement pads and no
reference imagery. This is the container - correct size, correct datum, correct encoding -
ready to be sculpted in Giants Editor.

Heights are stored as 16-bit centimetres (raw / 100 = metres), matching the rest of the
project and Giants Editor's import convention.
"""
import os
import time

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- canvas geometry -------------------------------------------------------------------
# 1 px = 1 m throughout. The canvas is larger than the playable area so the terrain does
# not end at the border the player can reach.
CANVAS_M = 12288
PLAYABLE_M = 8192
OFFSET_M = (CANVAS_M - PLAYABLE_M) // 2      # 2048 m of margin on every side

# --- datum -----------------------------------------------------------------------------
# The whole canvas sits at this height. Well clear of zero, so ground can still be cut
# down as well as raised once someone starts editing.
BASE_Z_M = 100.0
Z_MAX_CM = 62000.0                            # Giants' working ceiling, in centimetres


def main():
    t_start = time.time()
    n = CANVAS_M
    print(f"=== FS25 clean DEM generator ({n}x{n} m canvas, "
          f"{PLAYABLE_M} m playable) ===")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dem = os.path.join(script_dir, "dem_new_12k.png")
    out_vis = os.path.join(script_dir, "dem_new_visual_12k.png")
    out_detail = os.path.join(script_dir, "dem_new_visual_detail_12k.png")

    print(f"1. Flat ground at {BASE_Z_M:.2f} m...")
    terrain = np.full((n, n), BASE_Z_M, dtype=np.float32)

    raw = np.clip(terrain * 100.0, 0.0, Z_MAX_CM)

    print(f"2. Writing '{os.path.basename(out_dem)}'...")
    # uint16 straight out: Pillow writes that as a 16-bit greyscale PNG, which is what
    # Giants reads. The old mode="I" route is deprecated and drops out in Pillow 13.
    Image.fromarray(raw.astype(np.uint16)).save(out_dem)
    play = raw[OFFSET_M:OFFSET_M + PLAYABLE_M, OFFSET_M:OFFSET_M + PLAYABLE_M]
    print(f"   canvas   {raw.min()/100:.2f} .. {raw.max()/100:.2f} m")
    print(f"   playable {play.min()/100:.2f} .. {play.max()/100:.2f} m "
          f"(relief {(play.max()-play.min())/100:.1f} m)")

    print("3. Visualisations...")
    vis_scale = max(1, n // 1024)
    vis = raw[::vis_scale, ::vis_scale]

    def style(ax, title):
        ax.set_xlabel("X (East-West) [metres]", fontsize=11, fontweight='bold')
        ax.set_ylabel("Y (North-South) [metres]", fontsize=11, fontweight='bold')
        ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.5,
                alpha=0.35)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=14, color='white')

    def draw(ax, sub, extent):
        """Height itself, on a fixed scale centred on the datum.

        A hillshade of level ground carries no information and its normalisation is
        degenerate, so the raw height is shown instead. The window is fixed at the datum
        +/- 10 m: flat reads as one flat colour, and anything that is not stands out.
        """
        return ax.imshow(sub / 100.0, extent=extent, cmap='terrain',
                         vmin=BASE_Z_M - 10.0, vmax=BASE_Z_M + 10.0)

    # --- full canvas
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    im = draw(ax, vis, [0, n, n, 0])
    ax.set_xticks(np.arange(0, n + 1, 1024))
    ax.set_yticks(np.arange(0, n + 1, 1024))
    style(ax, f"Full DEM canvas ({n}x{n} px, 1 px = 1 m)")
    rect = plt.Rectangle((OFFSET_M, OFFSET_M), PLAYABLE_M, PLAYABLE_M,
                         fill=False, edgecolor='white', linewidth=2, linestyle='--',
                         label=f'Playable border ({PLAYABLE_M/1000:.1f} km)')
    ax.add_patch(rect)
    ax.legend(loc='upper right', facecolor='black', labelcolor='white', fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("height [m]", color='white')
    cb.ax.tick_params(colors='white')
    cb.outline.set_edgecolor('white')
    plt.savefig(out_vis, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"   {out_vis}")

    # --- playable area only
    p0 = OFFSET_M // vis_scale
    p1 = (OFFSET_M + PLAYABLE_M) // vis_scale
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    im = draw(ax, vis[p0:p1, p0:p1], [0, PLAYABLE_M, PLAYABLE_M, 0])
    ax.set_xticks(np.arange(0, PLAYABLE_M + 1, 1024))
    ax.set_yticks(np.arange(0, PLAYABLE_M + 1, 1024))
    style(ax, f"Playable area ({PLAYABLE_M/1000:.1f} x {PLAYABLE_M/1000:.1f} km)")
    ax.set_xlim(0, PLAYABLE_M)
    ax.set_ylim(PLAYABLE_M, 0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("height [m]", color='white')
    cb.ax.tick_params(colors='white')
    cb.outline.set_edgecolor('white')
    plt.savefig(out_detail, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"   {out_detail}")

    print(f"\n=== Done in {time.time() - t_start:.1f} s ===")


if __name__ == "__main__":
    main()
