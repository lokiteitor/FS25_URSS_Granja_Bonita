#!/usr/bin/env python3
"""FS25 heightmap generator - the Ukrainian forest-steppe canvas, sculpted.

Builds the 12288x12288 m canvas (1 px = 1 m) the project has always used, with the
8192x8192 m playable area centred in it, and lays the landform on it:

  1. a rolling upland, tilted north-high, from fractal noise;
  2. the valley of the Bystra, carved as a rising envelope so the river always sits at
     the bottom of it, with an asymmetric bluff/slip-off pair of banks;
  3. five tributary gullies off the upland;
  4. the lake, a flat pool with a weir at its outlet;
  5. graded corridors for the main road and the railway, each with its own gradient
     limit, floating over the river on their bridges rather than damming it;
  6. thirty flat platforms - 3 villages, 7 farms, 20 industry pads - each blended into
     the ground around it instead of dropped on top of it.

Every one of those comes out of `map_layout.layout()`, which the OSM generator reads as
well: the two halves never derive the same thing twice, so the vectors and the terrain
cannot drift apart.

Heights are stored as 16-bit centimetres (raw / 100 = metres), matching the rest of the
project and Giants Editor's import convention.
"""
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_layout as ml
import map_geom as mg

CANVAS_M = ml.CANVAS_PX
PLAYABLE_M = ml.PLAYABLE_PX
OFFSET_M = ml.OFFSET_PX
BASE_Z_M = ml.BASE_Z_M
Z_MAX_CM = ml.Z_MAX_CM

# --- fractal relief ---------------------------------------------------------------
FBM_N = 3072                  # 4 m/px; the finest octave still spans 23 px per cycle
FBM_OCTAVES = 6
FBM_WAVELENGTH0_M = 3000.0
FBM_LACUNARITY, FBM_PERSISTENCE = 2.0, 0.5
UPSAMPLE_BLUR_PX = 2.5        # takes the 4 m blocks out of the nearest-neighbour lift

# --- how far each feature reaches -------------------------------------------------
VALLEY_R_M = ml.VALLEY_INFLUENCE_M
SMIN_VALLEY_M, SMIN_LAKE_M = 3.0, 2.0
CHANNEL_FORCE_M = 8.0         # the core that is written to the design bed outright
ANCHOR_RAMP_M = 500.0
PROFILE_STEP_M = 10.0
MAX_EARTHWORK_M = 12.0        # louder than a comment, quieter than a failure
Z_GUARD_M = 45.0              # a mistuned constant should warn, not wrap a uint16


# ================================================================= numeric helpers
def smootherstep(t):
    """C2 fade, clipped. Every blend in this file uses it rather than smoothstep: over
    a 100 m skirt the curvature break of smoothstep shows up as a faint ring in the
    hillshade, and this one has none."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def smin(a, b, k):
    """Smooth minimum. Exactly min(a, b) once |a - b| >= k, and rounds the contact over
    a k-metre band - which is what turns the join between a valley side and the natural
    hillside from a crease into a shoulder. Polynomial rather than logaddexp: exact
    outside the band, and no exp over 150 million pixels."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def _octave(rng, wavelength_m, n, span_m):
    """One octave: white noise at its own resolution, blurred, standardised, lifted.

    This is pf_generator/generate_soil.py:44-66's recipe - filter white noise, then
    `(x - mean) / (std + 1e-8)` - with the grid sized to the wavelength so the blur
    stays a couple of pixels wide. Blurring a full-size field at sigma 125 instead
    would cost more than the whole rest of the generator.
    """
    m = int(np.clip(round(span_m / (wavelength_m / 8.0)), 8, n))
    w = rng.standard_normal((m, m)).astype(np.float32)
    w = ndimage.gaussian_filter(w, 8.0 / 6.0, mode='wrap')
    w = (w - w.mean()) / (w.std() + 1e-8)
    if m != n:
        w = ndimage.zoom(w, n / m, order=3, mode='nearest').astype(np.float32)
        if w.shape[0] != n:
            w = np.pad(w[:n, :n], ((0, max(0, n - w.shape[0])),
                                   (0, max(0, n - w.shape[1]))), mode='edge')
    return w


def fbm(seed, n, span_m, wavelength0_m=FBM_WAVELENGTH0_M, octaves=FBM_OCTAVES,
        lacunarity=FBM_LACUNARITY, persistence=FBM_PERSISTENCE):
    """Fractal Brownian motion, standardised to zero mean and unit variance.

    The wavelengths form a geometric series and persistence is exactly 1 / lacunarity,
    so amplitude and wavelength halve together and every octave contributes the *same*
    slope. Six equal independent contributions combine in RMS, which is what lets the
    field slope be calculated in advance (1.5-3 % at FBM_SIGMA_M = 3 m) instead of
    being tuned by eye.
    """
    rng = np.random.default_rng(seed)
    specs, lam, amp = [], wavelength0_m, 1.0
    for _ in range(octaves):
        specs.append((lam, amp))
        lam /= lacunarity
        amp *= persistence
    out = np.zeros((n, n), np.float32)
    with ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as ex:
        futs = [ex.submit(_octave, np.random.default_rng(seed + 7 * i), lam, n, span_m)
                for i, (lam, _) in enumerate(specs)]
        for (lam, amp), f in zip(specs, futs):
            out += amp * f.result()
    return (out - out.mean()) / (out.std() + 1e-8)


def polyline_field(pts, cum_s, n, radius_m, want_side=False):
    """Distance to a polyline and chainage along it, as two float32 canvas rasters.

    Distance is initialised to radius_m, so every pixel outside every segment's window
    is left alone: the cost is proportional to the influenced area, not to n^2 times
    the segment count.
    """
    d = np.full((n, n), np.float32(radius_m))
    s = np.zeros((n, n), np.float32)
    side = np.zeros((n, n), np.int8) if want_side else None
    for i in range(len(pts) - 1):
        ax, ay = pts[i][0] + OFFSET_M, pts[i][1] + OFFSET_M
        bx, by = pts[i + 1][0] + OFFSET_M, pts[i + 1][1] + OFFSET_M
        x0 = max(0, int(math.floor(min(ax, bx) - radius_m)))
        x1 = min(n, int(math.ceil(max(ax, bx) + radius_m)) + 1)
        y0 = max(0, int(math.floor(min(ay, by) - radius_m)))
        y1 = min(n, int(math.ceil(max(ay, by) + radius_m)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        if L2 <= 0.0:
            continue
        px = np.arange(x0, x1, dtype=np.float32) - np.float32(ax)
        py = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(ay)
        t = np.clip((px * vx + py * vy) / L2, 0.0, 1.0)
        dd = np.hypot(px - t * vx, py - t * vy).astype(np.float32)
        sub_d = d[y0:y1, x0:x1]
        # The mask comes FIRST and both arrays are written through it. Doing the
        # distance with np.minimum(out=) and then deriving the mask silently leaves the
        # chainage describing a segment that is no longer the nearest one.
        m = dd < sub_d
        np.copyto(s[y0:y1, x0:x1],
                  (cum_s[i] + t * math.sqrt(L2)).astype(np.float32), where=m)
        if want_side:
            np.copyto(side[y0:y1, x0:x1],
                      np.where(px * vy - py * vx >= 0.0, 1, -1).astype(np.int8),
                      where=m)
        np.copyto(sub_d, dd, where=m)
    return d, s, side


def stations(poly, s_arr):
    """Points and unit tangents at given chainages, vectorised.

    Walking the polyline per station is O(stations x vertices) in Python and was the
    single slowest thing in this file; np.interp against the cumulative length is the
    same answer for a densified line and is O(n log n).
    """
    P = np.asarray(poly, np.float64)
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    s = np.concatenate(([0.0], np.cumsum(seg)))
    x = np.interp(s_arr, s, P[:, 0])
    y = np.interp(s_arr, s, P[:, 1])
    dx, dy = np.gradient(x), np.gradient(y)
    L = np.hypot(dx, dy)
    L[L == 0.0] = 1.0
    return x, y, dx / L, dy / L


def sample_bilinear(arr, xs_m, ys_m):
    """Height at arbitrary playable-metre coordinates. A few hundred points per call."""
    cx = np.clip(np.asarray(xs_m, np.float64) + OFFSET_M, 0.0, arr.shape[1] - 1.001)
    cy = np.clip(np.asarray(ys_m, np.float64) + OFFSET_M, 0.0, arr.shape[0] - 1.001)
    i0, j0 = cy.astype(np.int64), cx.astype(np.int64)
    fy, fx = cy - i0, cx - j0
    a = arr[i0, j0] * (1 - fx) + arr[i0, j0 + 1] * fx
    b = arr[i0 + 1, j0] * (1 - fx) + arr[i0 + 1, j0 + 1] * fx
    return a * (1 - fy) + b * fy


def moving_mean(z, ds, window_m):
    """Running mean along a profile, edges handled by replication."""
    k = max(1, int(round(window_m / ds)) | 1)
    pad = k // 2
    p = np.pad(z, pad, mode='edge')
    c = np.concatenate(([0.0], np.cumsum(p)))
    return (c[k:] - c[:-k]) / k


def enforce_grade(z, ds, max_grade, anchor_idx, anchor_z, iters=400):
    """Project a profile onto {|dz/ds| <= max_grade} while pinning the anchors.

    Gauss-Seidel: wherever a step exceeds the limit, move both ends half the excess
    towards each other, then re-impose the anchors. Both sets are convex, so this is
    alternating projection and it converges; 400 sweeps over ~900 stations is instant
    and lands well inside a millimetre of feasible.
    """
    z = np.asarray(z, np.float64).copy()
    lim = max_grade * ds
    for i, v in zip(anchor_idx, anchor_z):
        z[i] = v
    for _ in range(iters):
        d = np.diff(z)
        over = np.abs(d) - lim
        if over.max() <= 1e-6:
            break
        adj = np.sign(d) * np.maximum(over, 0.0) * 0.5
        z[:-1] += adj
        z[1:] -= adj
        for i, v in zip(anchor_idx, anchor_z):
            z[i] = v
    return z


def hat(t):
    """A smootherstep tent, 1 at the centre and 0 at |t| >= 1."""
    return 1.0 - smootherstep(np.abs(t))


# ==================================================================== sculpting stages
def stage_relief(n, L):
    """1. The rolling upland: the analytic macro landform plus fractal detail.

    `regional_z` lives in map_layout and is written with `cos` and `exp` injected, so
    the very same function the OSM half calls point by point is evaluated here over the
    whole canvas. One definition of the landform, two evaluators.
    """
    terrain = np.empty((n, n), np.float32)
    xs = (np.arange(n, dtype=np.float32) - np.float32(OFFSET_M))
    for r0 in range(0, n, 1024):
        r1 = min(n, r0 + 1024)
        ys = (np.arange(r0, r1, dtype=np.float32) - np.float32(OFFSET_M))[:, None]
        terrain[r0:r1] = ml.regional_z(xs, ys, cos=np.cos, exp=np.exp)
    detail = np.clip(fbm(ml.SEED + 101, FBM_N, float(CANVAS_M)),
                     -ml.FBM_CLIP_SIGMA, ml.FBM_CLIP_SIGMA) * ml.FBM_SIGMA_M
    k = n // FBM_N
    detail = np.repeat(np.repeat(detail, k, 0), k, 1)
    ndimage.gaussian_filter(detail, UPSAMPLE_BLUR_PX, output=detail)
    terrain += detail
    return terrain


def stage_valley(terrain, n, L):
    """2. The valley, as a rising envelope met with a smooth minimum.

    A valley is exactly "the terrain, but never higher than the valley surface", so the
    envelope is built to rise without bound past the floodplain and `smin` does the
    rest. Because smin never *raises* ground it cannot fight the noise - and because it
    never raises ground, the channel core has to be written outright afterwards or a
    chance low spot could sink the bed below its design level and break the descent.
    """
    riv = L["river"]
    d, s, side = polyline_field(riv["centre"], riv["s"], n, VALLEY_R_M, want_side=True)
    rs = np.asarray(riv["s"], np.float32)
    zw = np.interp(s, rs, np.asarray(riv["z"], np.float32)).astype(np.float32)
    W = np.interp(s, rs, np.asarray(riv["half_w"], np.float32)).astype(np.float32)
    D = np.interp(s, rs, np.asarray(riv["depth"], np.float32)).astype(np.float32)

    # One bank is a slip-off slope and the other a bluff, and they swap every ~2.6 km.
    # That is how a meandering river actually reads, and it costs one sine.
    asym = 1.0 + ml.VALLEY_ASYM * side * np.sin(2.0 * np.pi * s / ml.VALLEY_ASYM_LX)
    slope = np.float32(ml.VALLEY_SIDE_SLOPE) * asym.astype(np.float32)

    bed = zw - D * np.sqrt(np.clip(1.0 - (d / np.maximum(W, 1e-3)) ** 2, 0.0, 1.0))
    bank = zw + ml.FLOODPLAIN_RISE_M * smootherstep((d - W) / ml.BANK_M)
    outer = (zw + ml.FLOODPLAIN_RISE_M
             + slope * np.maximum(d - ml.FLOODPLAIN_HALF_M, 0.0))
    env = np.where(d <= W, bed, np.maximum(bank, outer)).astype(np.float32)
    del bed, bank, outer, asym, slope

    near = d < ml.FLOODPLAIN_HALF_M
    terrain[near] = np.minimum(terrain[near], env[near])
    terrain[:] = smin(terrain, env, SMIN_VALLEY_M)

    del env, near, zw, W, D, side
    force_channel(terrain, d, s, riv)
    return d, s


def force_channel(terrain, d, s, riv, skip=None):
    """Write the channel core to its design bed.

    Two things make this necessary rather than decorative. `smin` never raises ground,
    so a chance low spot in the noise could otherwise sink the bed below its design
    level; and the road and railway embankments, laid later, would otherwise dam the
    river where they approach their bridges. Running it again after the corridors is
    what turns "the river descends" from a hope into a guarantee - `river['z']` is
    asserted non-increasing and `depth` non-decreasing in map_layout, so the bed written
    here is strictly falling downstream.

    Only the pixels within a channel width of the centreline are touched, selected once
    into a flat index, so this costs a fraction of a full-canvas pass. `skip` is the
    lake's reach: the pool owns its own bed, and running a 20 cm channel through it
    would fill the basin back in.
    """
    rs = np.asarray(riv["s"], np.float32)
    hw = np.asarray(riv["half_w"], np.float32)
    core = d <= float(hw.max()) + CHANNEL_FORCE_M
    if skip is not None:
        core &= ~((s > skip[0]) & (s < skip[1]))
    dd, ss = d[core], s[core]
    W = np.interp(ss, rs, hw).astype(np.float32)
    D = np.interp(ss, rs, np.asarray(riv["depth"], np.float32)).astype(np.float32)
    Z = np.interp(ss, rs, np.asarray(riv["z"], np.float32)).astype(np.float32)
    w = 1.0 - smootherstep((dd - W) / CHANNEL_FORCE_M)
    t = terrain[core]
    terrain[core] = t + ((Z - D) - t) * w
    return int(core.sum())


def stage_tributaries(terrain, n, L):
    """2b. Five gullies off the upland. Without them the interfluves read as a dome."""
    for t in L["tributaries"]:
        d, s, _ = polyline_field(t["centre"], t["s"], n, t["influence_m"])
        zc = np.interp(s, np.asarray(t["s"], np.float32),
                       np.asarray(t["z"], np.float32)).astype(np.float32)
        env = (zc + t["depth"]
               + t["side_slope"] * np.maximum(d - t["half_w"], 0.0)).astype(np.float32)
        env = np.where(d <= t["half_w"], zc, env).astype(np.float32)
        terrain[:] = smin(terrain, env, 2.0)
        del d, s, zc, env
    return len(L["tributaries"])


def stage_lake(terrain, n, L):
    """3. The pool: flat water, a dish under it, and a shore that meets it exactly.

    The shoreline is not rasterised from the polygon. `lake_radius` generated that
    polygon, so inside-ness is analytic here - the vectors and the heightmap share one
    curve rather than two approximations of it.
    """
    lk = L["lake"]
    pad = 400.0
    x0 = max(0, int(lk["cx"] - lk["rx"] * 1.6 - pad + OFFSET_M))
    x1 = min(n, int(lk["cx"] + lk["rx"] * 1.6 + pad + OFFSET_M))
    y0 = max(0, int(lk["cy"] - lk["rx"] * 1.6 - pad + OFFSET_M))
    y1 = min(n, int(lk["cy"] + lk["rx"] * 1.6 + pad + OFFSET_M))
    X = np.arange(x0, x1, dtype=np.float32) - np.float32(OFFSET_M + lk["cx"])
    Y = (np.arange(y0, y1, dtype=np.float32) - np.float32(OFFSET_M + lk["cy"]))[:, None]
    a = math.radians(lk["angle_deg"])
    ca, sa = math.cos(a), math.sin(a)
    u = X * ca + Y * sa
    v = -X * sa + Y * ca
    theta = np.arctan2(v * (lk["rx"] / lk["ry"]), u)
    r = np.hypot(u / lk["rx"], v / lk["ry"]) / ml.lake_radius(theta, cos=np.cos)
    mean_r = (lk["rx"] + lk["ry"]) / 2.0
    bed = lk["z"] - lk["depth"] * np.sqrt(np.clip(1.0 - r * r, 0.0, 1.0))
    shore = lk["z"] + 0.09 * (r - 1.0) * mean_r
    env = np.where(r <= 1.0, bed, shore).astype(np.float32)
    win = terrain[y0:y1, x0:x1]
    win[:] = smin(win, env, SMIN_LAKE_M)
    # The basin is written out to the shoreline itself, where the dish meets LAKE_Z
    # exactly. Stopping short of it - at r = 0.985, say - leaves a metre-high step all
    # the way round the lake, because the dish is already a metre down by then.
    basin = r <= 1.0
    win[basin] = bed[basin]
    return lk["ha"]


def pad_target_z(terrain, pad):
    """Median of the natural surface under a footprint, snapped to 5 cm.

    Median rather than mean: one corner dropping into a gully must not drag the whole
    platform down with it, which is exactly what an average does.
    """
    x0, y0, x1, y1 = mg.ring_bbox(pad["ring"])
    i0 = max(0, int(y0 + OFFSET_M)); i1 = min(terrain.shape[0], int(y1 + OFFSET_M) + 1)
    j0 = max(0, int(x0 + OFFSET_M)); j1 = min(terrain.shape[1], int(x1 + OFFSET_M) + 1)
    w = pad_weight(pad, j0, j1, i0, i1)
    inside = w >= 0.999
    sub = terrain[i0:i1, j0:j1]
    vals = sub[inside] if inside.any() else sub
    return round(float(np.median(vals)) / 0.05) * 0.05


def pad_weight(pad, j0, j1, i0, i1):
    """1 inside the footprint, falling to 0 at blend_m from it, over the exact distance
    to the rotated rectangle."""
    X = np.arange(j0, j1, dtype=np.float32) - np.float32(OFFSET_M + pad["cx"])
    Y = (np.arange(i0, i1, dtype=np.float32) - np.float32(OFFSET_M + pad["cy"]))[:, None]
    a = math.radians(pad["angle_deg"])
    ca, sa = math.cos(a), math.sin(a)
    du = np.abs(X * ca + Y * sa) - pad["w"] / 2.0
    dv = np.abs(-X * sa + Y * ca) - pad["h"] / 2.0
    d_out = np.hypot(np.maximum(du, 0.0), np.maximum(dv, 0.0))
    return 1.0 - smootherstep(d_out / pad["blend_m"])


def apply_pad(terrain, pad, z):
    """Flatten a platform and blend its edge into the ground around it.

    Applied one pad at a time, which is only correct because `map_layout` keeps every
    pair at least blend_a + blend_b + 20 m apart: the skirts are therefore disjoint and
    no pad can tilt another's. The assert is the tripwire for that.
    """
    x0, y0, x1, y1 = mg.ring_bbox(pad["ring"])
    m = pad["blend_m"] + 4.0
    i0 = max(0, int(y0 - m + OFFSET_M)); i1 = min(terrain.shape[0], int(y1 + m + OFFSET_M) + 1)
    j0 = max(0, int(x0 - m + OFFSET_M)); j1 = min(terrain.shape[1], int(x1 + m + OFFSET_M) + 1)
    w = pad_weight(pad, j0, j1, i0, i1)
    sub = terrain[i0:i1, j0:j1]
    before = sub.copy()
    sub += (np.float32(z) - sub) * w
    cut = float((before - sub).max())
    fill = float((sub - before).max())
    return cut, fill


def line_profile(terrain, line, anchors, name):
    """Design a longitudinal profile over the ground the line actually crosses.

    Sample the sculpted terrain, run a moving mean over it, pull it onto the anchors
    with a tent each, then project onto the gradient limit. Anchors are the village
    platforms (at the platform's own height, so the road arrives level with the village
    instead of stepping into it) and the bridges (at the deck, so the corridor does not
    excavate the river it is supposed to span).
    """
    pts = line["centre"]
    total = line["s"][-1]
    ns = int(total / PROFILE_STEP_M) + 1
    s = np.linspace(0.0, total, ns)
    xs, ys, tx, ty = stations(pts, s)
    xy = list(zip(xs, ys))
    z_nat = sample_bilinear(terrain, xs, ys).astype(np.float64)
    ds = total / (ns - 1)
    z = moving_mean(z_nat, ds, line["smooth_m"])
    idx, vals = [], []
    for s_a, z_a in anchors:
        i = int(round(np.clip(s_a, 0.0, total) / ds))
        z += (z_a - z[i]) * hat((s - s[i]) / ANCHOR_RAMP_M)
        idx.append(i); vals.append(z_a)
    z = enforce_grade(z, ds, line["max_grade"], idx, vals)
    grade = np.abs(np.diff(z)) / ds
    work = np.abs(z - z_nat)
    return {"s": s, "z": z, "xy": xy, "ds": ds, "grade": grade, "work": work,
            "z_nat": z_nat, "name": name}


def apply_corridor(terrain, n, line, prof, bridges):
    """Flatten the corridor to its profile, and let go of it over the bridges.

    `bridge_suppress` is what stops the embankment filling the valley solid: over each
    declared span the corridor weight goes to zero, the ground stays as the river carved
    it, and the deck floats. The OSM half tags exactly these spans bridge=yes from the
    same list.
    """
    R = line["grade_half_m"] + line["blend_m"]
    d, s, _ = polyline_field(line["centre"], line["s"], n, R)
    z_c = np.interp(s, prof["s"], prof["z"]).astype(np.float32)
    w = 1.0 - smootherstep((d - line["grade_half_m"]) / line["blend_m"])
    for b in bridges:
        taper = 40.0
        w *= smootherstep((np.abs(s - 0.5 * (b["s0"] + b["s1"]))
                           - 0.5 * (b["s1"] - b["s0"])) / taper)
    terrain += (z_c - terrain) * w
    del d, s, z_c, w


# ============================================================================ output
def write_png(terrain, path):
    raw = np.clip(np.rint(terrain.astype(np.float64) * 100.0), 0.0, Z_MAX_CM)
    raw = raw.astype(np.uint16)
    # uint16 straight out: Pillow writes that as a 16-bit greyscale PNG, which is what
    # Giants reads. The old mode="I" route is deprecated and drops out in Pillow 13.
    Image.fromarray(raw).save(path)
    return raw


def visualise(terrain, L, out_vis, out_detail, out_prof, profiles):
    n = terrain.shape[0]
    step = max(1, n // 1024)
    vis = terrain[::step, ::step]
    lo, hi = np.percentile(vis, 0.5), np.percentile(vis, 99.5)
    ls = LightSource(azdeg=315, altdeg=45)

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
        """Height under a hillshade. The old note here said a hillshade of level ground
        carries no information - true of the blank canvas, and no longer true.

        The colourbar is driven by a standalone ScalarMappable rather than by the
        image: the image is an RGB blend, and handing that to colorbar draws a blank
        white strip.
        """
        rgb = ls.shade(sub, cmap=plt.get_cmap('terrain'), blend_mode='overlay',
                       vert_exag=1.5, vmin=lo, vmax=hi, dx=step, dy=step)
        ax.imshow(rgb, extent=extent)
        return plt.cm.ScalarMappable(norm=Normalize(lo, hi),
                                     cmap=plt.get_cmap('terrain'))

    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    fig.patch.set_facecolor('#111111'); ax.set_facecolor('#111111')
    im = draw(ax, vis, [0, n, n, 0])
    ax.set_xticks(np.arange(0, n + 1, 1024)); ax.set_yticks(np.arange(0, n + 1, 1024))
    style(ax, f"Full DEM canvas ({n}x{n} px, 1 px = 1 m)")
    ax.add_patch(plt.Rectangle((OFFSET_M, OFFSET_M), PLAYABLE_M, PLAYABLE_M, fill=False,
                               edgecolor='white', linewidth=2, linestyle='--',
                               label=f'Playable border ({PLAYABLE_M/1000:.1f} km)'))
    ax.legend(loc='upper right', facecolor='black', labelcolor='white', fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("height [m]", color='white'); cb.ax.tick_params(colors='white')
    cb.outline.set_edgecolor('white')
    plt.savefig(out_vis, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    p0, p1 = OFFSET_M // step, (OFFSET_M + PLAYABLE_M) // step
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    fig.patch.set_facecolor('#111111'); ax.set_facecolor('#111111')
    im = draw(ax, vis[p0:p1, p0:p1], [0, PLAYABLE_M, PLAYABLE_M, 0])
    style(ax, f"Playable area ({PLAYABLE_M/1000:.1f} x {PLAYABLE_M/1000:.1f} km) "
              "with the layout")
    ax.set_xticks(np.arange(0, PLAYABLE_M + 1, 1024))
    ax.set_yticks(np.arange(0, PLAYABLE_M + 1, 1024))
    r = L["river"]
    ax.plot(*zip(*r["centre"]), color='#38BDF8', lw=1.6)
    ax.plot(*zip(*L["lake"]["ring"]), color='#38BDF8', lw=1.4)
    ax.plot(*zip(*L["road"]["centre"]), color='#F59E0B', lw=2.2)
    ax.plot(*zip(*L["rail"]["centre"]), color='white', lw=1.4, ls='--')
    for p in L["pads"]:
        col = {'village': '#6366F1', 'farm': '#DB2777',
               'industry': '#F97316'}[p["kind"]]
        ax.add_patch(plt.Polygon(p["ring"], closed=True, fill=False, edgecolor=col,
                                 linewidth=1.2))
    ax.set_xlim(0, PLAYABLE_M); ax.set_ylim(PLAYABLE_M, 0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("height [m]", color='white'); cb.ax.tick_params(colors='white')
    cb.outline.set_edgecolor('white')
    plt.savefig(out_detail, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=140)
    fig.patch.set_facecolor('#111111')
    rs = np.asarray(L["river"]["s"]) / 1000.0
    axes[0].plot(rs, L["river"]["z"], color='#38BDF8')
    axes[0].set_title("River water surface", color='white')
    for ax, key in zip(axes[1:], ("road", "rail")):
        pr = profiles[key]
        ax.plot(pr["s"] / 1000.0, pr["z_nat"], color='#64748B', lw=0.8,
                label='ground')
        ax.plot(pr["s"] / 1000.0, pr["z"], color='#F59E0B', lw=1.6, label='design')
        ax.set_title(f"{L[key]['name']} profile "
                     f"(max grade {100*pr['grade'].max():.2f} %, limit "
                     f"{100*L[key]['max_grade']:.1f} %)", color='white')
        ax.legend(facecolor='#111111', labelcolor='white', fontsize=8)
    for ax in axes:
        ax.set_facecolor('#111111'); ax.tick_params(colors='white')
        ax.set_xlabel("chainage [km]", color='white')
        ax.set_ylabel("height [m]", color='white')
        for sp in ax.spines.values():
            sp.set_color('white')
    fig.tight_layout()
    plt.savefig(out_prof, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def main():
    t_start = time.time()
    n = CANVAS_M
    print(f"=== FS25 DEM generator ({n}x{n} m canvas, {PLAYABLE_M} m playable) ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dem = os.path.join(script_dir, "dem_new_12k.png")
    out_vis = os.path.join(script_dir, "dem_new_visual_12k.png")
    out_detail = os.path.join(script_dir, "dem_new_visual_detail_12k.png")
    out_prof = os.path.join(script_dir, "dem_new_profiles_12k.png")

    print("0. Layout...")
    L = ml.layout()
    print(f"   seed {L['seed']}, {len(L['pads'])} pads, river "
          f"{L['river']['s'][-1]/1000:.1f} km, {len(L['bridges'])} bridge(s)")

    print(f"1. Rolling relief (fbm sigma {ml.FBM_SIGMA_M:.1f} m, "
          f"{FBM_OCTAVES} octaves)...")
    t = time.time(); terrain = stage_relief(n, L)
    print(f"   {terrain.min():.1f} .. {terrain.max():.1f} m   [{time.time()-t:.1f} s]")

    print("2. River valley...")
    t = time.time(); river_d, river_s = stage_valley(terrain, n, L)
    print(f"   carved to {terrain.min():.1f} m   [{time.time()-t:.1f} s]")

    print("2b. Tributary gullies...")
    t = time.time(); k = stage_tributaries(terrain, n, L)
    print(f"   {k} gully(ies)   [{time.time()-t:.1f} s]")

    print("3. Lake basin...")
    t = time.time(); ha = stage_lake(terrain, n, L)
    print(f"   {ha:.1f} ha at {L['lake']['z']:.2f} m   [{time.time()-t:.1f} s]")

    # The village platforms are levelled to the ground as it stands now, and the road
    # and the railway are then anchored to those heights. That is what makes both of
    # them arrive at Bereh at exactly the village's height instead of stepping into it.
    village_z = {}
    for p in L["pads"]:
        if p["kind"] == "village":
            village_z[p["name"]] = pad_target_z(terrain, p)

    profiles = {}
    for key, label in (("road", "4. Road corridor"), ("rail", "5. Railway corridor")):
        line = L[key]
        anchors = []
        for p in L["pads"]:
            if p["kind"] != "village":
                continue
            d, _, _, s_at = mg.project_on_polyline((p["cx"], p["cy"]), line["centre"])
            if d < p["w"] / 2.0 + 30.0:
                anchors.append((s_at, village_z[p["name"]]))
        for b in L["bridges"]:
            if b["on"] == key:
                anchors.append((0.5 * (b["s0"] + b["s1"]),
                                b["water_z"] + b["deck_clear_m"]))
        anchors.sort()
        print(f"{label} (max grade {100*line['max_grade']:.1f} %)...")
        t = time.time()
        pr = line_profile(terrain, line, anchors, line["name"])
        profiles[key] = pr
        apply_corridor(terrain, n, line, pr, [b for b in L["bridges"]
                                              if b["on"] == key])
        i = int(np.argmax(pr["work"]))
        print(f"   {line['s'][-1]/1000:.1f} km, design grade 0.00-"
              f"{100*pr['grade'].max():.2f} %, {len(anchors)} anchor(s), "
              f"worst earthwork {pr['work'][i]:.1f} m at s={pr['s'][i]/1000:.1f} km"
              f"   [{time.time()-t:.1f} s]")
        if pr["work"].max() > MAX_EARTHWORK_M:
            print(f"   !  earthwork over {MAX_EARTHWORK_M:.0f} m - check the alignment")

    # The approach embankments cross the floodplain, so the channel is written back
    # through them: a bridge whose abutments dam the river is a lake, not a bridge.
    print("5b. Re-opening the channel under the bridges...")
    t = time.time()
    lk = L["lake"]
    px = force_channel(terrain, river_d, river_s, L["river"],
                       skip=(lk["s0"] - ml.LAKE_TAPER_M, lk["s1"] + ml.LAKE_TAPER_M))
    del river_d, river_s
    print(f"   {px/1e3:.0f}k px restored to the design bed   [{time.time()-t:.1f} s]")

    print("6. Pads...")
    t = time.time()
    worst = (0.0, "")
    for p in L["pads"]:
        z = village_z.get(p["name"]) if p["kind"] == "village" else None
        if z is None:
            z = pad_target_z(terrain, p)
        cut, fill = apply_pad(terrain, p, z)
        p["z_m"] = z
        if max(cut, fill) > worst[0]:
            worst = (max(cut, fill), p["name"])
    for kind in ("village", "farm", "industry"):
        sel = [p for p in L["pads"] if p["kind"] == kind]
        print(f"   {kind:<9} {len(sel):2d} pads, {sum(p['ha'] for p in sel):6.1f} ha, "
              f"z {min(p['z_m'] for p in sel):.2f} .. {max(p['z_m'] for p in sel):.2f} m")
    print(f"   worst cut/fill {worst[0]:.2f} m ({worst[1]})   [{time.time()-t:.1f} s]")

    lo, hi = BASE_Z_M - Z_GUARD_M, BASE_Z_M + Z_GUARD_M
    if terrain.min() < lo or terrain.max() > hi:
        print(f"   !  terrain {terrain.min():.1f}..{terrain.max():.1f} m leaves the "
              f"{lo:.0f}..{hi:.0f} m guard band - clamping")
    np.clip(terrain, lo, hi, out=terrain)

    print(f"7. Writing '{os.path.basename(out_dem)}'...")
    t = time.time(); raw = write_png(terrain, out_dem)
    play = raw[OFFSET_M:OFFSET_M + PLAYABLE_M, OFFSET_M:OFFSET_M + PLAYABLE_M]
    print(f"   canvas   {raw.min()/100:.2f} .. {raw.max()/100:.2f} m")
    print(f"   playable {play.min()/100:.2f} .. {play.max()/100:.2f} m "
          f"(relief {(int(play.max())-int(play.min()))/100:.1f} m)   "
          f"[{time.time()-t:.1f} s]")

    print("8. Self-check...")
    from measure_elevation import run_checks
    ok = run_checks(raw, L, indent="   ")

    print("9. Visualisations...")
    t = time.time()
    visualise(terrain, L, out_vis, out_detail, out_prof, profiles)
    for f in (out_vis, out_detail, out_prof):
        print(f"   {f}")
    print(f"   [{time.time()-t:.1f} s]")

    print(f"\n=== Done in {time.time() - t_start:.1f} s ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
