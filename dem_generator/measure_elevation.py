#!/usr/bin/env python3
"""Conformance report for the generated heightmap.

This used to assert that the canvas was flat to within a centimetre, which was the
right check for a blank container and is guaranteed to fail now that there is a
landform on it. What replaces it is a layout-aware report: every number below is
checked against what `map_layout` says the terrain was supposed to be, so a constant
that drifts out of tune is caught here rather than in the editor.

It imports `map_layout`, not the generator - the generator drags in matplotlib, and
this is meant to be cheap enough to run as a gate.

    python3 measure_elevation.py      # -> report, exit 1 on any failure
"""
import math
import os
import sys

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_layout as ml
import map_geom as mg

DEM_NAME = "dem_new_12k.png"

# --- tolerances --------------------------------------------------------------------
RELIEF_SPAN_M = (18.0, 34.0)      # playable p1..p99
RELIEF_ABS_MAX_M = 40.0
FIELD_SLOPE_MEDIAN = (0.005, 0.035)
FIELD_SLOPE_P99 = 0.12
PAD_FLAT_TOL_M = 0.05
PAD_ERODE_M = 2.0
PAD_EDGE_MAX_SLOPE = 0.20
ROAD_GRADE_SLACK, RAIL_GRADE_SLACK = 0.005, 0.002
CROSS_FALL_MAX = 0.02
RIVER_RISE_TOL_M = 0.02           # 2x the 1 cm storage quantum
LAKE_SHORE_TOL_M, LAKE_DEPTH_TOL_M = 0.30, 0.60
BRIDGE_CLEAR_SLACK_M = 0.20


class Report:
    """The `   name ...  ok` / `   <-- reason` column style the old script used, plus a
    tally, so this can gate a build."""

    def __init__(self, indent="   "):
        self.indent, self.passed, self.failed = indent, 0, 0

    def check(self, name, ok, detail=""):
        if ok:
            self.passed += 1
            print(f"{self.indent}{name:<34} ok   {detail}")
        else:
            self.failed += 1
            print(f"{self.indent}{name:<34} FAIL {detail}")
        return ok

    def note(self, text):
        print(f"{self.indent}{'':<34}     {text}")


def _in_playable(xs, ys, inset=0.0):
    lo, hi = inset, ml.PLAYABLE_M - inset
    return (np.asarray(xs) >= lo) & (np.asarray(xs) <= hi) & \
           (np.asarray(ys) >= lo) & (np.asarray(ys) <= hi)


def _stations(poly, s_arr):
    """Points and unit tangents at given chainages, vectorised - see the twin of this
    in the generator."""
    P = np.asarray(poly, np.float64)
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(P[:, 0]),
                                                  np.diff(P[:, 1])))))
    x = np.interp(s_arr, s, P[:, 0])
    y = np.interp(s_arr, s, P[:, 1])
    dx, dy = np.gradient(x), np.gradient(y)
    L = np.hypot(dx, dy)
    L[L == 0.0] = 1.0
    return x, y, dx / L, dy / L


def _z(raw):
    return raw.astype(np.float32) / 100.0


def _sample(z, xs, ys):
    cx = np.clip(np.asarray(xs, np.float64) + ml.OFFSET_PX, 0, z.shape[1] - 1.001)
    cy = np.clip(np.asarray(ys, np.float64) + ml.OFFSET_PX, 0, z.shape[0] - 1.001)
    i0, j0 = cy.astype(np.int64), cx.astype(np.int64)
    fy, fx = cy - i0, cx - j0
    a = z[i0, j0] * (1 - fx) + z[i0, j0 + 1] * fx
    b = z[i0 + 1, j0] * (1 - fx) + z[i0 + 1, j0 + 1] * fx
    return a * (1 - fy) + b * fy


def _pad_window(z, pad, margin):
    x0, y0, x1, y1 = mg.ring_bbox(pad["ring"])
    i0 = max(0, int(y0 - margin + ml.OFFSET_PX))
    i1 = min(z.shape[0], int(y1 + margin + ml.OFFSET_PX) + 1)
    j0 = max(0, int(x0 - margin + ml.OFFSET_PX))
    j1 = min(z.shape[1], int(x1 + margin + ml.OFFSET_PX) + 1)
    X = np.arange(j0, j1, dtype=np.float32) - np.float32(ml.OFFSET_PX + pad["cx"])
    Y = (np.arange(i0, i1, dtype=np.float32)
         - np.float32(ml.OFFSET_PX + pad["cy"]))[:, None]
    a = math.radians(pad["angle_deg"])
    ca, sa = math.cos(a), math.sin(a)
    du = np.abs(X * ca + Y * sa) - pad["w"] / 2.0
    dv = np.abs(-X * sa + Y * ca) - pad["h"] / 2.0
    d_out = np.hypot(np.maximum(du, 0.0), np.maximum(dv, 0.0))
    d_in = np.maximum(du, dv)                     # negative inside
    return z[i0:i1, j0:j1], d_out, d_in


def run_checks(raw, L=None, indent="   "):
    L = L or ml.layout()
    rep = Report(indent)
    z = _z(raw)
    n = z.shape[0]

    # 1. container -------------------------------------------------------------------
    rep.check("canvas size", n == ml.CANVAS_PX and z.shape[1] == ml.CANVAS_PX,
              f"{n} x {z.shape[1]} px")
    rep.check("16-bit ceiling", int(raw.max()) <= 65535 and raw.dtype == np.uint16,
              f"peak {int(raw.max())} cm")

    play = z[ml.OFFSET_PX:ml.OFFSET_PX + ml.PLAYABLE_PX,
             ml.OFFSET_PX:ml.OFFSET_PX + ml.PLAYABLE_PX]

    # 2. relief ----------------------------------------------------------------------
    p1, p99 = np.percentile(play, 1), np.percentile(play, 99)
    span = float(p99 - p1)
    rep.check("playable relief p1..p99",
              RELIEF_SPAN_M[0] <= span <= RELIEF_SPAN_M[1],
              f"{span:.1f} m ({p1:.1f} .. {p99:.1f})")
    rep.check("playable relief absolute",
              float(play.max() - play.min()) <= RELIEF_ABS_MAX_M,
              f"{float(play.max()-play.min()):.1f} m "
              f"({play.min():.1f} .. {play.max():.1f})")

    # 3. field slope, away from everything that was deliberately shaped --------------
    step = 4
    sub = play[::step, ::step]
    gy, gx = np.gradient(sub.astype(np.float32), float(step))
    slope = np.hypot(gx, gy)
    keep = ~_shaped_mask(L, sub.shape[0], float(step))
    fs = slope[keep]
    med, p99s = float(np.median(fs)), float(np.percentile(fs, 99))
    rep.check("field slope median",
              FIELD_SLOPE_MEDIAN[0] <= med <= FIELD_SLOPE_MEDIAN[1],
              f"{100*med:.2f} %  (p95 {100*np.percentile(fs,95):.2f} %)")
    rep.check("field slope p99", p99s <= FIELD_SLOPE_P99, f"{100*p99s:.2f} %")

    # 4/5. platforms -----------------------------------------------------------------
    worst_flat, worst_edge = (-1.0, "-"), (-1.0, "-")
    for p in L["pads"]:
        win, d_out, d_in = _pad_window(z, p, p["blend_m"] + 6.0)
        inside = d_in <= -PAD_ERODE_M
        if inside.any():
            rng = float(win[inside].max() - win[inside].min())
            if rng > worst_flat[0]:
                worst_flat = (rng, p["name"])
        skirt = (d_out > 0.0) & (d_out < p["blend_m"])
        if skirt.any():
            gy2, gx2 = np.gradient(win.astype(np.float32))
            sl = np.hypot(gx2, gy2)[skirt]
            e = float(np.percentile(sl, 99.5))
            if e > worst_edge[0]:
                worst_edge = (e, p["name"])
    rep.check("pad flatness", worst_flat[0] <= PAD_FLAT_TOL_M,
              f"worst {worst_flat[0]*100:.1f} cm ({worst_flat[1]})")
    rep.check("pad edge integration", worst_edge[0] <= PAD_EDGE_MAX_SLOPE,
              f"worst {100*worst_edge[0]:.1f} % ({worst_edge[1]})")

    # 6/7/8. corridors ---------------------------------------------------------------
    for key, slack in (("road", ROAD_GRADE_SLACK), ("rail", RAIL_GRADE_SLACK)):
        line = L[key]
        ns = int(line["s"][-1] / 10.0) + 1
        ss = np.linspace(0.0, line["s"][-1], ns)
        px, py, tx, ty = _stations(line["centre"], ss)
        zc = _sample(z, px, py)
        # Only the part inside the playable square is checked. The alignment runs on
        # into the 2048 m margin and past the canvas edge, and sampling out there
        # clamps to the border - which reads as a cliff that is not in the terrain.
        live = _in_playable(px, py, inset=30.0)
        for b in [b for b in L["bridges"] if b["on"] == key]:
            live &= ~((ss > b["s0"] - 60.0) & (ss < b["s1"] + 60.0))
        seg_live = live[:-1] & live[1:]
        g = np.abs(np.diff(zc)) / (ss[1] - ss[0])
        g = np.where(seg_live, g, 0.0)
        i = int(np.argmax(g))
        rep.check(f"{key} gradient", g.max() <= line["max_grade"] + slack,
                  f"max {100*g.max():.2f} % at s={ss[i]/1000:.1f} km "
                  f"(limit {100*line['max_grade']:.1f} %)")
        h = line["half_w"]
        zl = _sample(z, px - ty * h, py + tx * h)
        zr = _sample(z, px + ty * h, py - tx * h)
        fall = np.abs(zl - zr)[live] / (2.0 * h)
        rep.check(f"{key} cross-fall", float(fall.max()) <= CROSS_FALL_MAX,
                  f"max {100*float(fall.max()):.2f} %")

    # 9. the river must never climb --------------------------------------------------
    riv = L["river"]
    lk = L["lake"]
    rx = np.array([p[0] for p in riv["centre"]])
    ry = np.array([p[1] for p in riv["centre"]])
    bed = _sample(z, rx, ry)
    live = _in_playable(rx, ry, inset=0.0)
    rs = np.asarray(riv["s"])
    live &= ~((rs > lk["s0"] - ml.LAKE_TAPER_M) & (rs < lk["s1"] + ml.LAKE_TAPER_M))
    idx = np.flatnonzero(live)
    viol, worst = 0, 0.0
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            continue
        rise = float(bed[b] - bed[a])
        if rise > RIVER_RISE_TOL_M:
            viol += 1
            worst = max(worst, rise)
    drop = float(bed[idx[0]] - bed[idx[-1]]) if len(idx) > 1 else 0.0
    rep.check("river monotonic descent", viol == 0,
              f"{viol} violation(s), worst +{worst*100:.0f} cm, drop across the "
              f"playable area {drop:.2f} m")

    # 10. the lake -------------------------------------------------------------------
    shore = _sample(z, [p[0] for p in lk["ring"]], [p[1] for p in lk["ring"]])
    dev = float(np.percentile(np.abs(shore - lk["z"]), 95))
    centre_z = float(_sample(z, [lk["cx"]], [lk["cy"]])[0])
    depth = lk["z"] - centre_z
    rep.check("lake shore level", dev <= LAKE_SHORE_TOL_M,
              f"worst {dev*100:.0f} cm off {lk['z']:.2f} m")
    rep.check("lake depth", abs(depth - lk["depth"]) <= LAKE_DEPTH_TOL_M,
              f"{depth:.2f} m (design {lk['depth']:.2f})")

    # 11. bridge clearance -----------------------------------------------------------
    ok_b = True
    for b in L["bridges"]:
        under = float(_sample(z, [b["x"]], [b["y"]])[0])
        clear = (b["water_z"] + b["deck_clear_m"]) - under
        if clear < b["deck_clear_m"] - BRIDGE_CLEAR_SLACK_M:
            ok_b = False
        rep.note(f"bridge on {b['on']:<4} ground {under:.2f} m, deck "
                 f"{b['water_z']+b['deck_clear_m']:.2f} m, clearance {clear:.2f} m")
    rep.check("bridge clearance", ok_b, f"{len(L['bridges'])} bridge(s)")

    print(f"{indent}{rep.passed} passed, {rep.failed} failed")
    return rep.failed == 0


def _shaped_mask(L, size_px, cell_m):
    """Everything the generator deliberately shaped, as a boolean raster.

    Drawn with PIL and grown with a distance transform rather than swept segment by
    segment: the sweep is O(segments x pixels) and it was, on its own, most of the
    runtime of this report.
    """
    img = Image.new("L", (size_px, size_px), 0)
    dr = ImageDraw.Draw(img)

    def line(poly, reach):
        pts = [((x) / cell_m, (y) / cell_m) for x, y in poly]
        dr.line(pts, fill=255, width=1)
        return reach

    reaches = []
    for poly, reach in ((L["road"]["centre"],
                         L["road"]["grade_half_m"] + L["road"]["blend_m"] + 60.0),
                        (L["rail"]["centre"],
                         L["rail"]["grade_half_m"] + L["rail"]["blend_m"] + 60.0),
                        (L["river"]["centre"], ml.VALLEY_INFLUENCE_M)):
        reaches.append(line(poly, reach))
    for t in L["tributaries"]:
        reaches.append(line(t["centre"], t["influence_m"]))
    # One transform, grown to the widest reach; the narrower features are simply
    # covered more generously than they need, which for an exclusion mask is fine.
    d = ndimage.distance_transform_edt(np.array(img) == 0) * cell_m
    mask = d < max(reaches)
    for p in L["pads"]:
        x0, y0, x1, y1 = mg.ring_bbox(p["ring"])
        m = p["blend_m"] + 40.0
        dr.rectangle([((x0 - m) / cell_m, (y0 - m) / cell_m),
                      ((x1 + m) / cell_m, (y1 + m) / cell_m)], fill=255)
    mask |= np.array(img) > 0
    return mask


def _rect_mask(xs, ys, pad, margin):
    a = math.radians(pad["angle_deg"])
    ca, sa = math.cos(a), math.sin(a)
    X, Y = xs - pad["cx"], ys - pad["cy"]
    du = np.abs(X * ca + Y * sa) - (pad["w"] / 2.0 + margin)
    dv = np.abs(-X * sa + Y * ca) - (pad["h"] / 2.0 + margin)
    return np.maximum(du, dv) <= 0.0


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, DEM_NAME)
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run generate_new_dem_12k.py first.")
        return 1
    img = Image.open(path)
    print(f"=== {DEM_NAME}: {img.size[0]}x{img.size[1]}, mode {img.mode} ===")
    if img.mode not in ("I", "I;16"):
        print(f"   PIL mode {img.mode!r} is not a 16-bit greyscale read")
    raw = np.array(img).astype(np.uint16)
    ok = run_checks(raw)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
