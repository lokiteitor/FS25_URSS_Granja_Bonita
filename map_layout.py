#!/usr/bin/env python3
"""The skeleton of the map: sizes, datum, and every piece of geometry both halves share.

This is the replacement for the missing `map_source.py`, in the same slot at the repo
root and reached with the same idiom the tree already uses:

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import map_layout as ml

The DEM generator and the OSM generator must agree *exactly* on where the river, the
lake, the road, the railway and every flat platform are, or the heightmap and the
vectors describe two different places. They agree because neither derives any of it:
both read `layout()`, which is built once from `SEED` and memoised.

Standard library only, on purpose. It is imported by both halves and by all three
readers, so putting numpy in this contract would spread it everywhere for no gain -
there is nothing here but polylines and rectangles.

Coordinates are playable metres, x east, y south from the north edge, the frame
`osm_generator/map_extent.py` defines. The layout is allowed to run outside the playable
square, from -OFFSET_M to PLAYABLE_M + OFFSET_M: the DEM sculpts all of it so the ground
does not stop dead at the border, and the OSM half clips to the playable part.

The fact that burns everyone, stated once:

    dem_array[row, col] == dem_array[int(y + OFFSET_M), int(x + OFFSET_M)]

y grows southwards, which is why it indexes the row.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "osm_generator"))
import map_extent as mx                                          # noqa: E402
import map_geom as mg                                            # noqa: E402

TWO_PI = 2.0 * math.pi

# --- re-exported from map_extent: the one source of truth for where the map is -----
LAT_CENTER, LON_CENTER = mx.LAT_CENTER, mx.LON_CENTER
PLAYABLE_M, HALF_M = mx.PLAYABLE_M, mx.HALF_M
M_PER_DEG, M_PER_DEG_LON = mx.M_PER_DEG, mx.M_PER_DEG_LON
local_to_global, global_to_local = mx.local_to_global, mx.global_to_local
bounds, polyline_length, ring_area_ha = mx.bounds, mx.polyline_length, mx.ring_area_ha

# --- canvas: 1 px = 1 m throughout -------------------------------------------------
CANVAS_M = 12288.0
OFFSET_M = (CANVAS_M - PLAYABLE_M) / 2.0          # 2048.0 m of margin on every side
# Integer twins. Every raster index in the DEM half goes through these, so "1 px = 1 m"
# is stated once instead of being re-derived at each slice - PLAYABLE_M is a float.
CANVAS_PX, PLAYABLE_PX, OFFSET_PX = int(CANVAS_M), int(PLAYABLE_M), int(OFFSET_M)
LAYOUT_MIN_M, LAYOUT_MAX_M = -OFFSET_M, PLAYABLE_M + OFFSET_M

# --- datum -------------------------------------------------------------------------
BASE_Z_M = 100.0
Z_MAX_CM = 62000.0                                # Giants' working ceiling, centimetres

# --- determinism -------------------------------------------------------------------
# One seed, per-purpose offsets (the `seed + 12345` idiom of generate_soil.py:84), so
# that adding a farm cannot move the river.
SEED = 2026
SEED_RIVER, SEED_LAKE, SEED_PADS = SEED + 11, SEED + 23, SEED + 37

# --- vertical design ---------------------------------------------------------------
UPLAND_Z_M = 105.0            # mean height of the rolling upland
TILT_N_M = 7.0                # how much higher the upland sits at the north edge
SWELL1_M, SWELL1_LX, SWELL1_LY = 3.0, 5200.0, 6100.0
SWELL2_M, SWELL2_LX = 1.8, 3100.0
VALLEY_TREND_M, VALLEY_TREND_W = 3.0, 1400.0

Z_RIVER_IN_M, Z_RIVER_OUT_M = 90.5, 84.5          # water surface, west edge -> east
RIVER_DEPTH_IN_M, RIVER_DEPTH_OUT_M = 2.0, 3.0    # bed below water, monotone rising
RIVER_HALF_IN_M, RIVER_HALF_OUT_M = 9.0, 16.0     # channel half-width
FLOODPLAIN_RISE_M, BANK_M, FLOODPLAIN_HALF_M = 2.5, 22.0, 150.0
VALLEY_SIDE_SLOPE, VALLEY_ASYM, VALLEY_ASYM_LX = 0.065, 0.35, 2600.0
VALLEY_INFLUENCE_M = 700.0
LAKE_DEPTH_M = 6.0
FBM_SIGMA_M, FBM_CLIP_SIGMA = 3.0, 2.5

# --- river plan --------------------------------------------------------------------
RIVER_Y0_M = 6200.0
RIVER_CTRL_STEP_M = 400.0
RIVER_MEANDER_M = 80.0
RIVER_STEP_M = 25.0
RIVER_MIN_RADIUS_M = 250.0
RIVER_NAME = "Bystra"

# --- lake --------------------------------------------------------------------------
LAKE_TARGET_X_M = 3300.0
LAKE_RX_M, LAKE_RY_M = 520.0, 260.0
LAKE_TAPER_M = 180.0
LAKE_NAME = "Bystre"

# --- road and railway --------------------------------------------------------------
CROSSING_XY = (4300.0, 3900.0)                    # road x rail == the main village

# The road is pinned to the two playable corners and to the crossing, and bows about
# 350 m off the chord in between, alternating sides. A single-sided bow - which is what
# monotone offsets give - reads as a straight line with a kink; the alternation is what
# makes it read as a road that follows the country.
ROAD_CTRL = [(-2250.0, -1650.0), (0.0, 0.0), (1350.0, 1650.0), (2850.0, 2300.0),
             CROSSING_XY, (5650.0, 5500.0), (7000.0, 6320.0), (8192.0, 8192.0),
             (10450.0, 10050.0)]
RAIL_CTRL = [(10450.0, -1750.0), (8500.0, 200.0), (6900.0, 1600.0), (5500.0, 2800.0),
             CROSSING_XY, (3100.0, 5050.0), (1900.0, 6250.0), (600.0, 7550.0),
             (-1850.0, 9500.0)]

ROAD_HALF_M, ROAD_GRADE_HALF_M, ROAD_BLEND_M = 6.0, 40.0, 70.0
RAIL_HALF_M, RAIL_GRADE_HALF_M, RAIL_BLEND_M = 8.0, 24.0, 90.0
ROAD_MAX_GRADE, RAIL_MAX_GRADE = 0.06, 0.015
ROAD_SMOOTH_M, RAIL_SMOOTH_M = 400.0, 1200.0
ROAD_MIN_RADIUS_M, RAIL_MIN_RADIUS_M = 400.0, 1200.0
BRIDGE_CLEAR_M = 4.0
BRIDGE_MARGIN_M = 70.0                            # deck reaches this far past the water

# --- platforms ---------------------------------------------------------------------
PAD_CHAMFER_M = 25.0
PAD_GAP_M = 20.0
PAD_EDGE_CLEAR_M = 120.0
PAD_RIVER_CLEAR_M = 60.0
PAD_LAKE_CLEAR_M = 80.0
PAD_MAX_CUT_M = 4.0           # bound the earthworks directly rather than the slope:
                              # with pads up to 600 m long a 3 % slope cap would still
                              # allow 9 m of cut at the ends
VILLAGE_BLEND_M, FARM_BLEND_M, INDUSTRY_BLEND_M = 110.0, 100.0, 70.0
INDUSTRY_MAX_HA = 5.05

VILLAGE_SPECS = [
    # name, anchor on the road, w, h
    ("Verkhivka", (1850.0, 1750.0), 380.0, 260.0),
    ("Bereh", CROSSING_XY, 560.0, 340.0),
    ("Nyzhne", (6600.0, 5900.0), 360.0, 240.0),
]

FARM_SPECS = [
    # sub, label, w, h, district box (x0, y0, x1, y1)
    ("cooperativa", "Granja Cooperativa", 600.0, 400.0, (4800.0, 1600.0, 6400.0, 2900.0)),
    ("granos",      "Granja Granos",      560.0, 375.0, (1900.0, 4400.0, 3400.0, 5400.0)),
    ("vacas",       "Granja Vacas",       520.0, 346.0, (5300.0, 5100.0, 7300.0, 6300.0)),
    ("cerdos",      "Granja Cerdos",      500.0, 300.0, (6500.0, 2500.0, 8000.0, 3900.0)),
    ("ovejas",      "Granja Ovejas",      450.0, 300.0, (3300.0, 6900.0, 5300.0, 7900.0)),
    ("invernaderos", "Granja Invernaderos", 600.0, 200.0, (900.0, 2400.0, 2700.0, 3500.0)),
    ("pollos",      "Granja Pollos",      420.0, 250.0, (5900.0, 6400.0, 7700.0, 7700.0)),
]

# Eight sidings in two rows along the railway south-west of Bereh, and twelve more
# scattered by district. Soviet-era industry sits on the railway, so it is laid out as
# an estate rather than sprinkled: one estate road then serves a whole row.
INDUSTRY_SIDES = [210.0, 215.0, 220.0, 224.0]     # 4.41 .. 5.02 ha
ESTATE_ROWS = ((380.0, 700.0), (860.0, 700.0))    # (perpendicular offset, start along)
ESTATE_SPACING_M = 500.0
ESTATE_N_PER_ROW = 4
INDUSTRY_DISTRICTS = [
    (900.0, 900.0, 2600.0, 2100.0), (5200.0, 700.0, 7200.0, 1900.0),
    (7000.0, 4300.0, 8000.0, 5600.0), (1000.0, 3900.0, 2200.0, 5200.0),
    (4900.0, 3000.0, 6300.0, 4300.0), (2500.0, 1200.0, 3900.0, 2300.0),
    (6400.0, 6600.0, 7800.0, 7900.0), (2300.0, 7000.0, 3600.0, 7900.0),
    (7100.0, 1900.0, 8000.0, 2900.0), (900.0, 5900.0, 2100.0, 7000.0),
    (3900.0, 900.0, 5200.0, 2000.0), (5100.0, 7000.0, 6400.0, 7900.0),
]

# ====================================================================== the landform
def river_axis_y(x, cos=math.cos):
    """The designed axis of the river valley, as a function of x.

    The channel meanders around this line by at most ~150 m, so the broad depression
    `regional_z` carves along it lands on the real water rather than beside it.
    """
    return (RIVER_Y0_M
            + 470.0 * cos(TWO_PI * (x - 1500.0) / 5400.0)
            + 165.0 * cos(TWO_PI * (x + 900.0) / 2200.0))


def regional_z(x, y, cos=math.cos, exp=math.exp):
    """The macro landform in metres: rolling upland, higher to the north, with a broad
    depression along the river axis.

    Pure arithmetic plus `cos` and `exp`, which are *passed in*. That is what lets this
    module stay standard library while the DEM half calls it as
    `regional_z(X, Y, cos=np.cos, exp=np.exp)` and gets the whole 12288 x 12288 canvas
    in one vectorised evaluation. One definition of the landform, two evaluators - so
    the vectors and the heightmap agree by construction rather than by coincidence.
    """
    z = UPLAND_Z_M + TILT_N_M * (0.5 - y / PLAYABLE_M)
    z = z + SWELL1_M * cos(TWO_PI * x / SWELL1_LX) * cos(TWO_PI * y / SWELL1_LY)
    z = z + SWELL2_M * cos(TWO_PI * (x + y) / SWELL2_LX)
    dy = y - river_axis_y(x, cos=cos)
    return z - VALLEY_TREND_M * exp(-(dy * dy) /
                                    (2.0 * VALLEY_TREND_W * VALLEY_TREND_W))


def slope_at(x, y, h=25.0):
    """|grad regional_z| by central differences. Analytic and free, which is how the
    OSM half puts woodland on steep ground without ever opening the heightmap."""
    dzdx = (regional_z(x + h, y) - regional_z(x - h, y)) / (2.0 * h)
    dzdy = (regional_z(x, y + h) - regional_z(x, y - h)) / (2.0 * h)
    return math.hypot(dzdx, dzdy)


def river_water_z(x):
    """Designed water surface at a given x. Linear and monotone falling west to east:
    6.0 m over 12.288 km is 0.049 %, a realistic lowland Ukrainian gradient. Monotone by
    construction, which is what makes the DEM's descent check trivial."""
    t = mg.clamp((x - LAYOUT_MIN_M) / CANVAS_M, 0.0, 1.0)
    return Z_RIVER_IN_M + (Z_RIVER_OUT_M - Z_RIVER_IN_M) * t


def to_canvas(x, y):
    return (x + OFFSET_M, y + OFFSET_M)


def to_canvas_px(x, y):
    return (int(x + OFFSET_M), int(y + OFFSET_M))


def to_playable(cx, cy):
    return (cx - OFFSET_M, cy - OFFSET_M)


# ====================================================================== the layout
_LAYOUT = None


def layout():
    """The full layout, built once and cached.

    Both halves call this. Returning the same object every time is not an optimisation:
    rebuilding it would re-run the seeded placement and could hand the DEM a different
    river from the one the OSM generator drew.
    """
    global _LAYOUT
    if _LAYOUT is None:
        _LAYOUT = _build()
        _validate(_LAYOUT)
    return _LAYOUT


# --- river --------------------------------------------------------------------------
def _build_river():
    rng = random.Random(SEED_RIVER)
    n = int((LAYOUT_MAX_M - LAYOUT_MIN_M) / RIVER_CTRL_STEP_M) + 1
    xs = [LAYOUT_MIN_M - 200.0 + i * RIVER_CTRL_STEP_M for i in range(n + 1)]
    # A correlated walk, not independent jitter: independent offsets at 400 m spacing
    # produce curvature a river cannot have, and the min-radius assert then fails.
    walk, off = [], 0.0
    for _ in xs:
        off = 0.62 * off + rng.uniform(-RIVER_MEANDER_M, RIVER_MEANDER_M)
        walk.append(off)

    amp = 1.0
    for _ in range(12):
        ctrl = []
        for x, w in zip(xs, walk):
            y = river_axis_y(x)
            dy = (river_axis_y(x + 1.0) - river_axis_y(x - 1.0)) / 2.0
            nx, ny = mg._unit(-dy, 1.0)
            ctrl.append((x + nx * w * amp, y + ny * w * amp))
        centre = mg.resample(mg.catmull_rom(ctrl, per_seg=40), RIVER_STEP_M)
        if mg.min_curve_radius(centre) >= RIVER_MIN_RADIUS_M:
            break
        amp *= 0.8

    s = mg.chainage(centre)
    total = s[-1] or 1.0
    half_w, depth, z = [], [], []
    for pt, si in zip(centre, s):
        t = si / total
        half_w.append(RIVER_HALF_IN_M + (RIVER_HALF_OUT_M - RIVER_HALF_IN_M) * t)
        depth.append(RIVER_DEPTH_IN_M + (RIVER_DEPTH_OUT_M - RIVER_DEPTH_IN_M) * t)
        z.append(river_water_z(pt[0]))
    # The meander leaves x very nearly monotone, but "very nearly" is not the same as
    # monotone, and the whole descent guarantee rests on this list never rising.
    for i in range(1, len(z)):
        z[i] = min(z[i], z[i - 1])
    return {"centre": centre, "half_w": half_w, "depth": depth, "z": z, "s": s,
            "name": RIVER_NAME}


# --- lake ---------------------------------------------------------------------------
def _lake_fourier():
    rng = random.Random(SEED_LAKE)
    return [(2, 0.10, rng.uniform(0.0, TWO_PI)),
            (3, 0.07, rng.uniform(0.0, TWO_PI)),
            (5, 0.045, rng.uniform(0.0, TWO_PI))]


LAKE_FOURIER = _lake_fourier()


def lake_radius(theta, cos=math.cos):
    """Shore radius at a bearing, as a multiple of the base ellipse.

    The OSM half samples this at 96 angles to build the ring and the DEM evaluates it
    per pixel, so the shoreline in the vectors and the shoreline in the heightmap are
    the same curve rather than two approximations of one.
    """
    r = 1.0
    for k, a, phi in LAKE_FOURIER:
        r = r + a * cos(k * theta + phi)
    return r


def _build_lake(river):
    centre, s = river["centre"], river["s"]
    i = min(range(len(centre)), key=lambda k: abs(centre[k][0] - LAKE_TARGET_X_M))
    cx, cy = centre[i]
    j = min(len(centre) - 1, i + 4)
    k = max(0, i - 4)
    angle = math.degrees(math.atan2(centre[j][1] - centre[k][1],
                                    centre[j][0] - centre[k][0]))
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    ring = []
    for a in range(96):
        th = TWO_PI * a / 96.0
        rr = lake_radius(th)
        u, v = LAKE_RX_M * rr * math.cos(th), LAKE_RY_M * rr * math.sin(th)
        ring.append((cx + u * ca - v * sa, cy + u * sa + v * ca))
    ring.append(ring[0])

    inside = [k for k in range(len(centre)) if mg.point_in_ring(centre[k], ring)]
    s0, s1 = (s[inside[0]], s[inside[-1]]) if inside else (s[i], s[i])
    z = river["z"][inside[0]] if inside else river["z"][i]
    # The pool is flat, and the drop it holds back reappears as a weir at the outlet.
    weir = z - (river["z"][inside[-1]] if inside else z)
    # The channel is faded out into the pool over LAKE_TAPER_M rather than switched off
    # at the shoreline: a hard cut leaves a step in the bed exactly where the river
    # meets the lake, and that step is what the descent check trips on.
    for k in range(len(centre)):
        if s[k] < s0 - LAKE_TAPER_M or s[k] > s1 + LAKE_TAPER_M:
            continue
        outside = max(s0 - s[k], s[k] - s1, 0.0)
        f = mg.smootherstep(outside / LAKE_TAPER_M)          # 0 in the pool, 1 clear
        river["half_w"][k] = 4.0 + (river["half_w"][k] - 4.0) * f
        river["depth"][k] = 0.2 + (river["depth"][k] - 0.2) * f
        if s0 <= s[k] <= s1:
            river["z"][k] = z
    return {"ring": ring, "cx": cx, "cy": cy, "rx": LAKE_RX_M, "ry": LAKE_RY_M,
            "angle_deg": angle, "fourier": LAKE_FOURIER, "z": z, "depth": LAKE_DEPTH_M,
            "s0": s0, "s1": s1, "weir_m": weir,
            "ha": mg.ring_area(ring) / 10000.0, "name": LAKE_NAME}


# --- road, railway and bridges ------------------------------------------------------
def _pin(poly, pt, snap_m=12.0):
    """Make `pt` an exact vertex of `poly`.

    `get_node` in the OSM writer keys on millimetre-rounded coordinates, so a micron of
    slop at the level crossing gives the road and the railway two different node ids and
    they are not joined at all.

    The nearest vertex is *moved* onto the point rather than a new one being inserted:
    the crossing is a control point of the spline, so that vertex is already within a
    few metres, and inserting instead would leave two vertices a hand's breadth apart -
    which shows up as a spike in `min_curve_radius` and trips the corridor assert.
    """
    out = [tuple(p) for p in poly]
    i = min(range(len(out)), key=lambda k: math.dist(out[k], pt))
    if math.dist(out[i], pt) <= snap_m:
        out[i] = (float(pt[0]), float(pt[1]))
        return out
    return mg.weave(out, [pt])


def _ctrl_span(ctrl):
    """Longest gap between consecutive control points - how fine the spline must be
    sampled for the polyline that comes out to be smooth at the resample step."""
    return max(math.dist(ctrl[i], ctrl[i + 1]) for i in range(len(ctrl) - 1))


def _build_line(ctrl, step, name, half, grade_half, blend, max_grade, smooth):
    # The spline is sampled finer than the resample step: sampling it coarsely and
    # then resampling only interpolates along the chords, and the kinks that leaves
    # read as a tight curve to min_curve_radius even though the design is straight.
    fine = mg.catmull_rom(ctrl, per_seg=max(12, int(_ctrl_span(ctrl) / step)))
    centre = _pin(mg.resample(fine, step), CROSSING_XY)
    return {"ctrl": [tuple(c) for c in ctrl], "centre": centre,
            "s": mg.chainage(centre), "half_w": half, "grade_half_m": grade_half,
            "blend_m": blend, "max_grade": max_grade, "smooth_m": smooth,
            "min_radius_m": mg.min_curve_radius(centre), "name": name}


def _find_bridges(line, which, river):
    """Where a line crosses the river, and how long its deck has to be."""
    out = []
    rc = river["centre"]
    for i in range(len(line["centre"]) - 1):
        a, b = line["centre"][i], line["centre"][i + 1]
        for j in range(len(rc) - 1):
            hit = mg.seg_intersect(a, b, rc[j], rc[j + 1])
            if hit is None:
                continue
            s_at = line["s"][i] + math.dist(a, hit)
            half = river["half_w"][j] + BRIDGE_MARGIN_M
            out.append({"on": which, "s0": s_at - half, "s1": s_at + half,
                        "x": hit[0], "y": hit[1], "deck_clear_m": BRIDGE_CLEAR_M,
                        "water_z": river["z"][j]})
            break
    return out


# --- platforms ----------------------------------------------------------------------
def _site_cut(cx, cy, w, h, angle_deg):
    """Worst |z - median| of the design surface under a footprint, on a 5 x 5 grid."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    zs = []
    for i in range(5):
        for j in range(5):
            u = (i / 4.0 - 0.5) * w
            v = (j / 4.0 - 0.5) * h
            zs.append(regional_z(cx + u * ca - v * sa, cy + u * sa + v * ca))
    zs.sort()
    med = zs[len(zs) // 2]
    return max(abs(zs[0] - med), abs(zs[-1] - med))


def _pad_ok(cand, placed, river, lake, road, rail):
    ring, blend = cand["ring"], cand["blend_m"]
    x0, y0, x1, y1 = mg.ring_bbox(ring)
    if x0 < PAD_EDGE_CLEAR_M or y0 < PAD_EDGE_CLEAR_M:
        return "edge"
    if x1 > PLAYABLE_M - PAD_EDGE_CLEAR_M or y1 > PLAYABLE_M - PAD_EDGE_CLEAR_M:
        return "edge"
    for other in placed:
        if mg.rect_dist(ring, other["ring"]) < blend + other["blend_m"] + PAD_GAP_M:
            return "pad %s" % other["name"]
    need = FLOODPLAIN_HALF_M + PAD_RIVER_CLEAR_M
    for p in ring:
        if mg.polyline_dist(p, river["centre"]) < need:
            return "river"
    for p in ring:
        if mg.point_in_ring(p, lake["ring"]):
            return "lake"
    if mg.rect_dist(ring, lake["ring"]) < PAD_LAKE_CLEAR_M:
        return "lake"
    if not cand.get("on_corridor"):
        for line, tag in ((road, "road"), (rail, "rail")):
            need = line["grade_half_m"] + line["blend_m"] + 20.0
            for p in ring:
                if mg.polyline_dist(p, line["centre"]) < need:
                    return tag
    if _site_cut(cand["cx"], cand["cy"], cand["w"], cand["h"],
                 cand["angle_deg"]) > PAD_MAX_CUT_M:
        return "cut"
    return None


def _make_pad(name, kind, sub, cx, cy, w, h, angle_deg, blend, z_ref=None,
              on_corridor=False):
    ring = mg.chamfer_rect(cx, cy, w, h, angle_deg, PAD_CHAMFER_M)
    return {"name": name, "kind": kind, "sub": sub, "cx": cx, "cy": cy, "w": w, "h": h,
            "angle_deg": angle_deg, "ring": ring, "ha": mg.ring_area(ring) / 10000.0,
            "blend_m": blend, "z_ref": z_ref, "on_corridor": on_corridor}


def _bearing_at(line, pt):
    _, _, i, s_at = mg.project_on_polyline(pt, line["centre"])
    a = line["centre"][max(0, i - 2)]
    b = line["centre"][min(len(line["centre"]) - 1, i + 3)]
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), s_at


def _build_pads(river, lake, road, rail):
    rng = random.Random(SEED_PADS)
    pads = []

    # Villages sit *on* the road - it runs through them, which is what "strung along the
    # road" means - so they are placed, not sampled, and exempted from road clearance.
    for name, anchor, w, h in VILLAGE_SPECS:
        _, pt, _, s_at = mg.project_on_polyline(anchor, road["centre"])
        ang, s_at = _bearing_at(road, anchor)
        pads.append(_make_pad("Village " + name, "village", name.lower(),
                              pt[0], pt[1], w, h, ang, VILLAGE_BLEND_M,
                              z_ref=("road", s_at), on_corridor=True))

    def place(cand_fn, tries, what):
        for _ in range(tries):
            cand = cand_fn()
            if cand is None:
                continue
            if _pad_ok(cand, pads, river, lake, road, rail) is None:
                pads.append(cand)
                return cand
        raise RuntimeError("could not site %s in %d tries" % (what, tries))

    for sub, label, w, h, box in FARM_SPECS:
        bx0, by0, bx1, by1 = box

        def cand(sub=sub, label=label, w=w, h=h, b=(bx0, by0, bx1, by1)):
            cx = rng.uniform(b[0] + w / 2.0, b[2] - w / 2.0)
            cy = rng.uniform(b[1] + h / 2.0, b[3] - h / 2.0)
            ang = rng.choice((0.0, 0.0, 90.0))
            ww, hh = (w, h) if ang == 0.0 else (h, w)
            return _make_pad(label, "farm", sub, cx, cy, ww, hh, 0.0, FARM_BLEND_M)
        place(cand, 6000, label)

    # Eight sidings on the railway south-west of Bereh, then twelve by district.
    n_ind = 0
    s_cross = mg.project_on_polyline(CROSSING_XY, rail["centre"])[3]
    for off, start in ESTATE_ROWS:
        for k in range(ESTATE_N_PER_ROW):
            n_ind += 1
            side = INDUSTRY_SIDES[(n_ind - 1) % len(INDUSTRY_SIDES)]
            s_at = s_cross + start + k * ESTATE_SPACING_M
            base, i, _ = mg.polyline_at(rail["centre"], s_at)
            a = rail["centre"][max(0, i - 1)]
            b = rail["centre"][min(len(rail["centre"]) - 1, i + 2)]
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            nx, ny = mg._unit(-(b[1] - a[1]), b[0] - a[0])
            cx, cy = base[0] + nx * off, base[1] + ny * off
            name = "Industry Pad %d" % n_ind

            def cand(cx=cx, cy=cy, side=side, ang=ang, name=name, nx=nx, ny=ny,
                     st=[0]):
                d = st[0] * 25.0
                st[0] += 1
                return _make_pad(name, "industry", "siding", cx + nx * d, cy + ny * d,
                                 side, side, ang, INDUSTRY_BLEND_M)
            place(cand, 200, name)

    for box in INDUSTRY_DISTRICTS:
        n_ind += 1
        side = INDUSTRY_SIDES[(n_ind - 1) % len(INDUSTRY_SIDES)]
        name = "Industry Pad %d" % n_ind

        def cand(box=box, side=side, name=name):
            cx = rng.uniform(box[0] + side / 2.0, box[2] - side / 2.0)
            cy = rng.uniform(box[1] + side / 2.0, box[3] - side / 2.0)
            return _make_pad(name, "industry", "yard", cx, cy, side, side, 0.0,
                             INDUSTRY_BLEND_M)
        place(cand, 6000, name)
    return pads


# --- assembly -----------------------------------------------------------------------
def _build():
    river = _build_river()
    lake = _build_lake(river)
    road = _build_line(ROAD_CTRL, 20.0, "Main Road", ROAD_HALF_M, ROAD_GRADE_HALF_M,
                       ROAD_BLEND_M, ROAD_MAX_GRADE, ROAD_SMOOTH_M)
    rail = _build_line(RAIL_CTRL, 20.0, "Main Line", RAIL_HALF_M, RAIL_GRADE_HALF_M,
                       RAIL_BLEND_M, RAIL_MAX_GRADE, RAIL_SMOOTH_M)
    bridges = _find_bridges(road, "road", river) + _find_bridges(rail, "rail", river)
    river["banks"] = (mg.offset_polyline(river["centre"], river["half_w"], 1),
                      mg.offset_polyline(river["centre"], river["half_w"], -1))
    pads = _build_pads(river, lake, road, rail)
    return {"seed": SEED, "crossing": CROSSING_XY, "river": river, "lake": lake,
            "road": road, "rail": rail, "bridges": bridges, "pads": pads}


def _validate(L):
    """Re-run every constraint over the finished layout and raise on the first failure.

    Each of these records a way the two halves can silently disagree, which is why they
    are assertions and not warnings.
    """
    n = 0
    river, lake, pads = L["river"], L["lake"], L["pads"]

    for i in range(1, len(river["z"])):
        assert river["z"][i] <= river["z"][i - 1] + 1e-9, "river climbs at vertex %d" % i
        n += 1
    # Depth is checked outside the lake only: the pool suppresses the channel on
    # purpose, so a trench is not ploughed across its bed, and it comes back after.
    for i in range(1, len(river["depth"])):
        if lake["s0"] - LAKE_TAPER_M - 60.0 <= river["s"][i] <= lake["s1"] + LAKE_TAPER_M + 60.0:
            continue
        if lake["s0"] - LAKE_TAPER_M - 60.0 <= river["s"][i - 1] <= lake["s1"] + LAKE_TAPER_M + 60.0:
            continue
        assert river["depth"][i] >= river["depth"][i - 1] - 1e-6, "depth jumps"
        n += 1
    assert len(river["half_w"]) == len(river["centre"]) == len(river["z"])
    n += 1

    for key, floor in (("road", ROAD_MIN_RADIUS_M), ("rail", RAIL_MIN_RADIUS_M)):
        r = L[key]["min_radius_m"]
        assert r >= floor, "%s min radius %.0f m < %.0f" % (key, r, floor)
        n += 1

    # The crossing must be the *same coordinate* on both lines, not merely nearby.
    for key in ("road", "rail"):
        d = min(math.dist(p, CROSSING_XY) for p in L[key]["centre"])
        assert d < 1e-6, "%s misses the crossing by %.3f m" % (key, d)
        n += 1

    kinds = {}
    for p in pads:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    assert kinds == {"village": 3, "farm": 7, "industry": 20}, kinds
    n += 1
    assert {p["sub"] for p in pads if p["kind"] == "farm"} == {s[0] for s in FARM_SPECS}
    n += 1

    for p in pads:
        if p["kind"] == "industry":
            assert abs(p["w"] - p["h"]) < 0.5, "%s is not square" % p["name"]
            assert p["ha"] <= INDUSTRY_MAX_HA, "%s is %.2f ha" % (p["name"], p["ha"])
            n += 2
        x0, y0, x1, y1 = mg.ring_bbox(p["ring"])
        assert x0 >= PAD_EDGE_CLEAR_M - 1e-6 and y0 >= PAD_EDGE_CLEAR_M - 1e-6
        assert x1 <= PLAYABLE_M - PAD_EDGE_CLEAR_M + 1e-6
        assert y1 <= PLAYABLE_M - PAD_EDGE_CLEAR_M + 1e-6
        n += 1

    for i in range(len(pads)):
        for j in range(i + 1, len(pads)):
            need = pads[i]["blend_m"] + pads[j]["blend_m"] + PAD_GAP_M
            d = mg.rect_dist(pads[i]["ring"], pads[j]["ring"])
            assert d >= need - 1e-6, "%s and %s are %.1f m apart, need %.1f" % (
                pads[i]["name"], pads[j]["name"], d, need)
            n += 1

    assert lake["ha"] > 20.0, "lake collapsed to %.1f ha" % lake["ha"]
    n += 1
    L["_checks"] = n
    return n


def tightest_pair(pads):
    best = (float("inf"), "", "")
    for i in range(len(pads)):
        for j in range(i + 1, len(pads)):
            need = pads[i]["blend_m"] + pads[j]["blend_m"] + PAD_GAP_M
            slack = mg.rect_dist(pads[i]["ring"], pads[j]["ring"]) - need
            if slack < best[0]:
                best = (slack, pads[i]["name"], pads[j]["name"])
    return best


def main():
    print("=== FS25 map layout (seed %d) ===" % SEED)
    L = layout()
    r, lk, rd, rl = L["river"], L["lake"], L["road"], L["rail"]
    print("1. River...    %.1f km, %.2f -> %.2f m (%.3f %% mean), min radius %.0f m"
          % (r["s"][-1] / 1000.0, r["z"][0], r["z"][-1],
             100.0 * (r["z"][0] - r["z"][-1]) / r["s"][-1],
             mg.min_curve_radius(r["centre"])))
    print("2. Lake...     %.1f ha at %.2f m, outlet weir %.2f m, %.0f m of channel"
          % (lk["ha"], lk["z"], lk["weir_m"], lk["s1"] - lk["s0"]))
    for key, line in (("3. Road", rd), ("4. Railway", rl)):
        br = [b for b in L["bridges"] if b["on"] == line["name"].split()[0].lower()]
        print("%s... %.1f km, min radius %.0f m" % (key, line["s"][-1] / 1000.0,
                                                    line["min_radius_m"]))
    for b in L["bridges"]:
        print("   bridge on %-4s at (%.0f, %.0f), water %.2f m, deck %.0f m"
              % (b["on"], b["x"], b["y"], b["water_z"], b["s1"] - b["s0"]))
    pads = L["pads"]
    for kind in ("village", "farm", "industry"):
        sel = [p for p in pads if p["kind"] == kind]
        print("5. %-9s %2d pads, %6.1f ha  (%.1f .. %.1f ha)"
              % (kind + "s...", len(sel), sum(p["ha"] for p in sel),
                 min(p["ha"] for p in sel), max(p["ha"] for p in sel)))
    slack, a, b = tightest_pair(pads)
    print("   tightest pair %.1f m of slack (%s <-> %s)" % (slack, a, b))
    worst = max(pads, key=lambda p: _site_cut(p["cx"], p["cy"], p["w"], p["h"],
                                              p["angle_deg"]))
    print("   worst site cut %.2f m (%s)"
          % (_site_cut(worst["cx"], worst["cy"], worst["w"], worst["h"],
                       worst["angle_deg"]), worst["name"]))
    print("[+] %d constraint(s) checked, all clear." % L["_checks"])


if __name__ == "__main__":
    main()
