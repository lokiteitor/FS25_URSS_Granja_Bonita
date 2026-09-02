#!/usr/bin/env python3
"""Fields, woodland and the headlands between them.

Two jobs live here, both raster-assisted and both driven by one shared question - "is
this ground already claimed?" - which `Occupancy` answers in constant time.

**Fields** are cut, not grown. Every parcel is born as a rectangle inside a horizontal
strip: its north and south edges *are* the strip's, and its east and west edges are
vertical cuts. There is no rotation anywhere in this file, so nothing can tilt a field -
which is the whole reason the guillotine is used instead of the Voronoi tessellation the
old bocage generator had. `carve_parcel` then trims each rectangle back to the ground
that is actually free, and that trimming is where the irregular edges come from: a field
the river or a wood edge really reaches loses that corner, and every other field stays a
clean rectangle.

**Woodland** is the Ukrainian forest-steppe pattern: a broken gallery forest along the
river, blocks on the steep ground of the valley sides, and - after the fields exist -
the shelterbelts, the *polezakhysni lisosmuhy* planted along the headlands between
parcels. A wood here is not a canopy; it is the block of ground trees get planted on by
hand in the editor, so the outlines are regularised into shapes with a workable interior
rather than left following a tree line.
"""
import math
import os
import sys

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_layout as ml
import map_geom as mg

# --- field sizing (mixed kolkhoz) --------------------------------------------------
MIN_FIELD_HA, MAX_FIELD_HA = 4.0, 102.0
SMALL_HA = (7.0, 16.0)          # inside a village halo
TYPICAL_HA = (22.0, 46.0)       # the block a band is cut towards. The median comes out
                                # a little under it: carving trims parcels back, and a
                                # band rarely divides into a whole number of them
BIG_HA = (85.0, 102.0)          # the plateau blocks
VILLAGE_HALO_M = 700.0
MAX_BIG_FIELDS = 10
MAX_FIELDS = 150                # ceiling on the field count. Fewer, larger blocks is a
                                # kolkhoz; it is also the size the cut targets are set
                                # from, so raising one without lowering the other breaks
                                # the check in generate_osm.verify
BIG_RIVER_CLEAR_M = 900.0
BIG_ASPECT = (1.2, 2.6)         # Ukrainian mega-fields are wide, not square
BIG_MAX_W_M, BIG_MAX_H_M = 1500.0, 1000.0

MIN_FIELD_WIDTH_M = 55.0        # 2*area/perimeter; drops the slivers a cut pinches off
MIN_RUN_M = 180.0
STRIP_H_M = (240.0, 700.0)
STRIP_TALL_M = 480.0
# The corridor between strips has to be wide enough to hold a shelterbelt outright: a
# belt planted in a nine-metre headland overlaps the fields either side of it, and two
# polygons on the same ground is not something Giants can make sense of.
STRIP_GAP_M = 28.0
HEADLAND_M = 9.0
FIELD_FILL_M = 2.0              # margin a finished parcel claims. Small on purpose: the
                                # cut geometry already keeps parcels apart, and a fat
                                # margin swallows the corridor the belts have to live in
WIDE_CUT_M = 28.0               # every third cut in a band, so cross belts fit too
WIDE_CUT_EVERY = 3
CARVE_STEP_M = 24.0
CARVE_MIN_OVERLAP_M = MIN_FIELD_WIDTH_M   # two neighbouring columns belong to the same
                                # parcel only if their free runs overlap by a workable
                                # width. Below that the "parcel" is two fields joined by
                                # a neck no machine can turn in - see carve_parcel
FIELD_SIMPLIFY_M = 6.0
SIMPLIFY_SLACK_M = 12.0         # every mask a parcel is cut against is grown by this
                                # first: the Douglas-Peucker pass that follows can pull
                                # an edge back across the boundary it was just cut to,
                                # which is how fields end up inside the woods
BAND_TOL_FRAC = 0.28            # of the strip height a column may be covered and still
                                # count - the partly-covered columns are exactly where
                                # carve_parcel bites in

# --- clearances: what a field may not grow into ------------------------------------
EDGE_MARGIN_M = 80.0
RIVER_BANK_M, LAKE_BANK_M = 30.0, 35.0     # open water margin: nothing at all here
# The bank proper: a field is trimmed back to its edge and the gallery forest fills it.
# The number is the DEM's own `FLOODPLAIN_HALF_M` rather than a second opinion - that is
# how far out the heightmap builds the flat terrace the river laid down, so the ground
# the vectors call floodplain and the ground the terrain calls floodplain are the same
# ground. Change it there, in map_layout.py, and both halves of the pipeline follow.
RIVER_FLOOD_M = ml.FLOODPLAIN_HALF_M
ROAD_CLEAR_M, SECONDARY_CLEAR_M, TRACK_CLEAR_M = 16.0, 11.0, 7.0
RAIL_CLEAR_M, PAD_CLEAR_M, WOOD_MARGIN_M = 22.0, 25.0, 12.0

# --- woodland ----------------------------------------------------------------------
WOOD_CELL_M = 16.0
MIN_WOOD_HA, MIN_BELT_HA = 1.5, 0.6
WOOD_SIMPLIFY_M = 20.0          # coarse on purpose: a dozen corners you can follow in
                                # the editor, not a two-metre-accurate tree line
WOOD_TARGET_FRAC = 0.09     # forest-steppe: 12-18 % wooded. This is the share of the
                            # *unclaimed* ground the slope blocks take, not the total -
                            # the gallery forest on the bank is drawn first and carries
                            # most of the woodland now, so the blocks ask for less
WOOD_CLEAR_M = 12.0
WOOD_CLOSE_M, WOOD_OPEN_M = 32.0, 24.0
WOOD_GROVE_M = 420.0     # scale at which the valley-side band breaks into groves
WOOD_GROVE_MIX = 0.85    # how much of the threshold is grove noise rather than slope
RIPARIAN_INNER_M = 8.0
RIPARIAN_EXTRA_M = 60.0         # how far past the bank the outer edge wanders. The band
                                # covers the bank at every station - that is the point of
                                # it - so all the shape has to come from the far edge
RIPARIAN_WAVE_M = 700.0
BELT_HALF_M = 11.0              # a 22 m shelterbelt: five to seven rows of poplar
BELT_MIN_L_M = 400.0
BELT_P = 0.75
RAIL_BELT_OFF_M, RAIL_BELT_HALF_M = 30.0, 10.0
WOOD_NOTCH_M = 96.            # firebreak cut to a platform the wood closed around.
                                # Generous on purpose: the chaikin-and-simplify pass
                                # that follows rounds a narrow bay straight back shut


# ===================================================================== the occupancy
class Occupancy:
    """Everything already claimed, on a 4 m raster.

    This is the one question the field cutter, the wood placer and the track layer all
    ask. Asking it against sixty polygons per point is what made the old generator reach
    for a spatial index; a boolean raster answers it in constant time, and the free-run
    scans every stage needs fall straight out of it as a column sum.

    Margins are applied by growing the ring before it is drawn rather than by dilating
    afterwards: for a rectangle that is exact, and it costs nothing.
    """

    CELL_M = 4.0

    def __init__(self, size_m=ml.PLAYABLE_M, cell_m=None):
        # The cell size is an instance attribute so a coarser copy can be built for a
        # different question: the road router asks about whole corridors, not field
        # edges, and a 4 m grid there is 36x the cells for no extra answer.
        self.CELL_M = float(cell_m) if cell_m else type(self).CELL_M
        self.size_m = float(size_m)
        self.n = int(round(size_m / self.CELL_M))
        self._img = Image.new("1", (self.n, self.n), 0)
        self._dr = ImageDraw.Draw(self._img)
        self._arr = None

    def _px(self, pts):
        c = self.CELL_M
        return [(x / c, y / c) for x, y in pts]

    def _dirty(self):
        self._arr = None

    def fill_ring(self, ring, margin=0.0):
        r = mg.grow_ring(ring, margin) if abs(margin) > 1e-9 else list(ring)
        self._dr.polygon(self._px(r), fill=1)
        self._dirty()

    def fill_polyline(self, pts, half):
        if len(pts) < 2:
            return
        w = max(1, int(round(2.0 * half / self.CELL_M)))
        self._dr.line(self._px(pts), fill=1, width=w, joint="curve")
        self._dirty()

    def fill_rect(self, x0, y0, x1, y1, margin=0.0):
        self._dr.rectangle(self._px([(x0 - margin, y0 - margin),
                                     (x1 + margin, y1 + margin)]), fill=1)
        self._dirty()

    def fill_border(self, margin):
        s = self.size_m
        for r in ((0, 0, s, margin), (0, s - margin, s, s),
                  (0, 0, margin, s), (s - margin, 0, s, s)):
            self.fill_rect(*r)

    @property
    def arr(self):
        if self._arr is None:
            self._arr = np.array(self._img, dtype=bool)
        return self._arr

    def coverage(self):
        return float(self.arr.mean())

    def covered(self, x, y):
        j, i = int(x / self.CELL_M), int(y / self.CELL_M)
        if not (0 <= i < self.n and 0 <= j < self.n):
            return True
        return bool(self.arr[i, j])

    def _cols(self, x0, x1):
        return (max(0, int(math.floor(x0 / self.CELL_M))),
                min(self.n, int(math.ceil(x1 / self.CELL_M))))

    def _rows(self, y0, y1):
        return (max(0, int(math.floor(y0 / self.CELL_M))),
                min(self.n, int(math.ceil(y1 / self.CELL_M))))

    @staticmethod
    def _runs(free, base, cell):
        """Maximal True runs of a boolean row, back in metres."""
        out, start = [], None
        for k, v in enumerate(free):
            if v and start is None:
                start = k
            elif not v and start is not None:
                out.append(((base + start) * cell, (base + k) * cell))
                start = None
        if start is not None:
            out.append(((base + start) * cell, (base + len(free)) * cell))
        return out

    def free_band(self, y0, y1, x0, x1, tol=0.0):
        """x runs where every row of the strip is free - or where at most `tol` metres
        of the strip height are covered. The road, the railway and the river cut a strip
        into bands here automatically, with no region finding anywhere."""
        # Rows strictly *inside* the band, not the rows it overlaps. For a 700 m field
        # strip the difference is nothing; for a 22 m belt corridor the two boundary
        # rows are half in the fields either side, and including them says every
        # corridor on the map is blocked.
        i0 = max(0, int(math.ceil(y0 / self.CELL_M)))
        i1 = min(self.n, int(math.floor(y1 / self.CELL_M)))
        j0, j1 = self._cols(x0, x1)
        if i0 >= i1 or j0 >= j1:
            return []
        cov = self.arr[i0:i1, j0:j1].sum(axis=0) * self.CELL_M
        return self._runs(cov <= tol, j0, self.CELL_M)

    def free_band_v(self, x0, x1, y0, y1, tol=0.0):
        """The transpose of free_band: y runs where every column strictly inside
        [x0, x1] is free. This is what a north-south headland corridor is scanned
        with."""
        j0 = max(0, int(math.ceil(x0 / self.CELL_M)))
        j1 = min(self.n, int(math.floor(x1 / self.CELL_M)))
        i0, i1 = self._rows(y0, y1)
        if i0 >= i1 or j0 >= j1:
            return []
        cov = self.arr[i0:i1, j0:j1].sum(axis=1) * self.CELL_M
        return self._runs(cov <= tol, i0, self.CELL_M)

    def free_run_y(self, x, y0, y1):
        """Longest free y interval in a column. Returns None if there is none worth
        having."""
        j = int(x / self.CELL_M)
        if not (0 <= j < self.n):
            return None
        i0, i1 = self._rows(y0, y1)
        if i0 >= i1:
            return None
        runs = self._runs(~self.arr[i0:i1, j], i0, self.CELL_M)
        if not runs:
            return None
        return max(runs, key=lambda r: r[1] - r[0])

    def rect_free(self, x0, y0, x1, y1):
        i0, i1 = self._rows(y0, y1)
        j0, j1 = self._cols(x0, x1)
        if i0 >= i1 or j0 >= j1:
            return False
        return not self.arr[i0:i1, j0:j1].any()


# ===================================================================== mask tracing
def _trace_rings(mask, cell_m, origin=(0.0, 0.0), min_cells=1):
    """Outer boundary of every 4-connected component of a boolean raster.

    The boundary is walked along the cracks between cells rather than through cell
    centres, so the ring is a closed rectilinear loop by construction and cannot cut a
    corner off the component. 4-connectivity is deliberate: with 8, two cells touching
    only at a corner join into one component whose boundary pinches to a point, and a
    pinched ring is not a polygon Giants will fill.
    """
    lbl, n = ndimage.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    slices = ndimage.find_objects(lbl)
    out = []
    for k in range(1, n + 1):
        sl = slices[k - 1]
        if sl is None:
            continue
        sub = (lbl[sl] == k)
        if sub.sum() < min_cells:
            continue
        r0, c0 = sl[0].start, sl[1].start
        p = np.pad(sub, 1)
        # Each boundary crack becomes one directed edge, oriented so the filled cell is
        # on the same hand throughout; the walk below then only ever has to follow them.
        up = p & ~np.roll(p, 1, 0)
        dn = p & ~np.roll(p, -1, 0)
        lf = p & ~np.roll(p, 1, 1)
        rt = p & ~np.roll(p, -1, 1)
        edges = {}
        for m, (da, db) in ((up, ((0, 0), (0, 1))), (dn, ((1, 1), (1, 0))),
                            (lf, ((1, 0), (0, 0))), (rt, ((0, 1), (1, 1)))):
            for i, j in np.argwhere(m):
                a = (int(i) + da[0], int(j) + da[1])
                b = (int(i) + db[0], int(j) + db[1])
                edges.setdefault(a, []).append(b)
        if not edges:
            continue
        loops = []
        while any(edges.values()):
            start = next(a for a, bs in edges.items() if bs)
            loop, cur = [start], start
            while True:
                nxts = edges.get(cur)
                if not nxts:
                    break
                cur = nxts.pop(0)
                loop.append(cur)
                if cur == start:
                    break
            if len(loop) > 4:
                loops.append(loop)
        if not loops:
            continue
        loop = max(loops, key=len)
        ring = [(origin[0] + (c - 1 + c0) * cell_m, origin[1] + (r - 1 + r0) * cell_m)
                for r, c in loop]
        if mg.ring_area(ring) > 0:
            out.append(ring)
    return out


def _disk(radius_cells):
    r = int(math.ceil(radius_cells))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= radius_cells * radius_cells


def _regularise(mask, cell_m):
    """Fill the narrow notches, cut the thin limbs, drop the specks.

    Port of what the bocage generator called `regularise`. The point is not fidelity:
    the polygon has to have an interior somebody can drive a planting pattern across in
    the editor, and a tree line traced honestly does not.
    """
    mask = ndimage.binary_closing(mask, _disk(WOOD_CLOSE_M / cell_m))
    mask = ndimage.binary_opening(mask, _disk(WOOD_OPEN_M / cell_m))
    return ndimage.binary_fill_holes(mask)


# ======================================================================= structural
def _slope_raster(n, cell_m):
    """|grad regional_z| over the playable area.

    `map_layout.regional_z` takes its `cos` and `exp` as arguments precisely so this can
    call it with numpy and get the whole grid at once - the same surface the DEM builds,
    so woodland lands on the slopes that actually exist.
    """
    xs = (np.arange(n, dtype=np.float64) + 0.5) * cell_m
    ys = xs[:, None]
    z = ml.regional_z(xs, ys, cos=np.cos, exp=np.exp)
    gy, gx = np.gradient(z, cell_m)
    return np.hypot(gx, gy)


def _break_enclosures(mask, pads, cell):
    """Cut a firebreak out to open ground from any platform the wood has closed around.

    `_trace_rings` returns outer boundaries and nothing else, so a wood that happens to
    surround a yard swallows it whole: the ring comes back with the platform inside it,
    and the editor would paint a forest over a flat industrial pad with a road running
    to it. A yard is swallowed exactly when the open ground it stands on cannot be
    walked to the edge of the map, which is a labelling question, and one notch to the
    nearest ground that can turns the enclosure into a bay the boundary walks round.
    """
    lbl, _ = ndimage.label(~mask)
    edge = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
    edge.discard(0)
    if not edge:
        return mask
    outside = np.isin(lbl, list(edge))
    _, idx = ndimage.distance_transform_edt(~outside, return_indices=True)
    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    dr = ImageDraw.Draw(img)
    n = mask.shape[0]
    for p in pads:
        j, i = int(p["cx"] / cell), int(p["cy"] / cell)
        if not (0 <= i < n and 0 <= j < n) or outside[i, j]:
            continue
        dr.line([(j, i), (int(idx[1][i, j]), int(idx[0][i, j]))], fill=0,
                width=max(2, int(round(WOOD_NOTCH_M / cell))))
    return np.array(img) > 0


def structural_woods(L, keepout, rng):
    """The gallery forest along the river and the blocks on the valley sides.

    The slope threshold is *derived*, not fixed: it is the quantile of the unclaimed
    ground that leaves the wanted share wooded. A constant would give either no
    woodland at all or four thousand hectares of it depending on how `regional_z` is
    scaled, and that constant lives in the other half of the pipeline.
    """
    cell = WOOD_CELL_M
    n = int(round(ml.PLAYABLE_M / cell))
    img = Image.new("1", (n, n), 0)
    dr = ImageDraw.Draw(img)

    riv = L["river"]
    s = riv["s"]
    total = s[-1] or 1.0
    phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(3)]
    # Continuous, deliberately. An earlier version broke the gallery wherever the wave
    # ran thin, which looked right from the air and left the bank bare in the gaps - and
    # bare bank is ground the field cutter would take. The band now reaches the bank
    # edge at every station and only its far side wanders.
    for side in (1, -1):
        outer = []
        for si in s:
            u = 0.0
            for k, ph in enumerate(phases):
                u += math.sin(2.0 * math.pi * si / (RIPARIAN_WAVE_M * (1 + 0.6 * k))
                              + ph + (0.0 if side > 0 else 1.7))
            outer.append(RIVER_FLOOD_M + RIPARIAN_EXTRA_M * (0.5 + u / 6.0))
        inner = [w + RIPARIAN_INNER_M for w in riv["half_w"]]
        a = mg.offset_polyline(riv["centre"], inner, side)
        b = mg.offset_polyline(riv["centre"], outer, side)
        ring = a + b[::-1]
        dr.polygon([(x / cell, y / cell) for x, y in ring], fill=1)
    # The lake is a widening of the same river, so its shore gets the same collar.
    dr.polygon([(x / cell, y / cell)
                for x, y in mg.grow_ring(L["lake"]["ring"], RIVER_FLOOD_M)], fill=1)
    riparian = np.array(img, dtype=bool)

    keep = ~_downsample(keepout.arr, keepout.CELL_M, cell, n)
    slope = _slope_raster(n, cell)
    # Slope alone puts a continuous ribbon of wood down both valley sides, which then
    # fuses with the gallery forest into one 400 ha amoeba nobody can plant by hand. A
    # coarse grove field is mixed into the score so the band breaks into copses - which
    # is what the valley sides of the forest-steppe actually look like from the air.
    grove = _grove_field(n, cell, rng)
    score = ((slope - slope.mean()) / (slope.std() + 1e-9)
             + WOOD_GROVE_MIX * grove)
    cand = score[keep]
    thr = float(np.quantile(cand, 1.0 - WOOD_TARGET_FRAC)) if cand.size else 1e9
    blocks = (score >= thr)

    # `& keep` again after regularising, not just before: closing bridges a notch and
    # `binary_fill_holes` fills a hole, and a platform punched out of the wood is
    # exactly a hole. Without this the forest closes back over the industry pads it was
    # told to leave alone, and no amount of firebreak afterwards can reopen them.
    mask = _break_enclosures(_regularise((riparian | blocks) & keep, cell) & keep,
                             L["pads"], cell)
    rings = []
    for ring in _trace_rings(mask, cell, min_cells=int(MIN_WOOD_HA * 1e4 / cell / cell)):
        ring = mg.simplify(mg.chaikin(ring, 2, closed=True), WOOD_SIMPLIFY_M)
        if len(ring) < 4:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        ha = mg.ring_area(ring) / 1e4
        if ha >= MIN_WOOD_HA:
            rings.append((ring, ha))
    rings.sort(key=lambda z: -z[1])
    return rings, thr


def _grove_field(n, cell_m, rng):
    """Standardised smooth noise at the grove scale, seeded off the layout."""
    m = max(4, int(round(n * cell_m / WOOD_GROVE_M)))
    g = np.random.default_rng(ml.SEED + 311).standard_normal((m, m))
    g = ndimage.zoom(g, n / m, order=3, mode='nearest')[:n, :n]
    if g.shape[0] < n:
        g = np.pad(g, ((0, n - g.shape[0]), (0, n - g.shape[1])), mode='edge')
    return (g - g.mean()) / (g.std() + 1e-9)


def _downsample(arr, src_cell, dst_cell, n):
    """Nearest-neighbour resample of a boolean raster onto another cell size."""
    idx = np.clip((np.arange(n) * dst_cell / src_cell).astype(int), 0, arr.shape[0] - 1)
    return arr[np.ix_(idx, idx)]


# =========================================================================== fields
def place_big_fields(occ, rng, river_centre, n_max=MAX_BIG_FIELDS):
    """The plateau blocks, placed before anything else is cut.

    They go first because a 100 ha rectangle cannot be assembled out of what the strips
    leave over - it has to have first refusal on the open ground, and the open ground is
    the flat plateau well away from the river.
    """
    cell = 128.0
    n = int(ml.PLAYABLE_M / cell)
    slope = _slope_raster(n, cell)
    cand = []
    for i in range(n):
        for j in range(n):
            x, y = (j + 0.5) * cell, (i + 0.5) * cell
            if occ.covered(x, y):
                continue
            d = mg.polyline_dist((x, y), river_centre)
            if d < BIG_RIVER_CLEAR_M:
                continue
            cand.append((slope[i, j], -d, x, y))
    cand.sort()
    out = []
    for _, _, x, y in cand:
        if len(out) >= n_max:
            break
        if occ.covered(x, y):
            continue
        target = rng.uniform(*BIG_HA) * 1e4
        x0, x1, y0, y1 = x - 300.0, x + 300.0, y - 200.0, y + 200.0
        if not occ.rect_free(x0 - HEADLAND_M, y0 - HEADLAND_M,
                             x1 + HEADLAND_M, y1 + HEADLAND_M):
            continue
        grew = True
        while grew and (x1 - x0) * (y1 - y0) < target:
            grew = False
            for dx0, dx1, dy0, dy1 in ((-60, 0, 0, 0), (0, 60, 0, 0),
                                       (0, 0, -40, 0), (0, 0, 0, 40)):
                nx0, nx1 = x0 + dx0, x1 + dx1
                ny0, ny1 = y0 + dy0, y1 + dy1
                w, h = nx1 - nx0, ny1 - ny0
                if w > BIG_MAX_W_M or h > BIG_MAX_H_M:
                    continue
                if not (BIG_ASPECT[0] <= w / h <= BIG_ASPECT[1]):
                    continue
                if not occ.rect_free(nx0 - HEADLAND_M, ny0 - HEADLAND_M,
                                     nx1 + HEADLAND_M, ny1 + HEADLAND_M):
                    continue
                x0, x1, y0, y1 = nx0, nx1, ny0, ny1
                grew = True
                if (x1 - x0) * (y1 - y0) >= target:
                    break
        ha = (x1 - x0) * (y1 - y0) / 1e4
        if ha < BIG_HA[0]:
            continue
        ring = mg.rect_ring((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)
        occ.fill_ring(ring, HEADLAND_M)
        out.append((ring, ha))
    return out


def carve_parcel(x0, x1, y0, y1, occ, step=CARVE_STEP_M):
    """Trim a rectangle to the ground that is actually free -> a list of rings.

    Columns are walked west to east and each contributes its longest free y-run inside
    [y0, y1]. A ring is then the north edge west to east, down the east side, and the
    south edge east to west. Because it is built from one interval per column it is
    y-monotone in x by construction: it cannot self-intersect, cannot enclose a hole and
    cannot split into two components - all three of which are ordinary outcomes of
    subtracting polygons, which is why that is not what happens here.

    What being y-monotone does *not* rule out is the parcel getting wrung out. Where a
    road crosses the strip on the diagonal it splits every column it touches in two, and
    the longest of the two halves is below the road on one side of it and above the road
    on the other. The ring that results is a legal simple polygon and a nonsense field:
    two lobes wrung together through a neck at the crossing. So the columns are cut into
    groups wherever consecutive free runs stop overlapping by a workable width, and each
    group becomes a field of its own - which is what the ground actually looks like: two
    fields, one either side of the road.

    A parcel whose columns all come back with the full [y0, y1] simplifies straight back
    to four corners, so most fields stay clean rectangles and only the ones the river or
    a wood edge really reaches pick up a wavy edge.
    """
    xs = list(np.arange(x0, x1 + 1e-6, step))
    if xs[-1] < x1 - 1e-6:
        xs.append(x1)
    groups, cur = [], []
    for x in xs:
        r = occ.free_run_y(min(x, x1 - 0.01), y0, y1)
        if r is None or (r[1] - r[0]) < MIN_FIELD_WIDTH_M:
            if cur:
                groups.append(cur)
                cur = []
            continue
        col = (x, max(r[0], y0), min(r[1], y1))
        if cur:
            _, pa, pb = cur[-1]
            if min(pb, col[2]) - max(pa, col[1]) < CARVE_MIN_OVERLAP_M:
                groups.append(cur)
                cur = []
        cur.append(col)
    if cur:
        groups.append(cur)

    out = []
    for g in groups:
        if len(g) < 2 or g[-1][0] - g[0][0] < MIN_FIELD_WIDTH_M:
            continue
        ring = ([(x, a) for x, a, _ in g]
                + [(x, b) for x, _, b in reversed(g)])
        ring.append(ring[0])
        ring = mg.simplify(ring, FIELD_SIMPLIFY_M)
        if len(ring) < 4:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if mg.ring_is_simple(ring):
            out.append(ring)
    return out


def _class_range(cx, cy, villages):
    for v in villages:
        if math.dist((cx, cy), (v["cx"], v["cy"])) < VILLAGE_HALO_M:
            return SMALL_HA
    return TYPICAL_HA


def cut_fields(occ, rng, villages):
    """Strips, bands, guillotine cuts, carve. Returns the parcels and the headlands."""
    fields, gaps = [], []
    lo, hi = EDGE_MARGIN_M, ml.PLAYABLE_M - EDGE_MARGIN_M
    y = lo
    strips = []
    while y < hi - STRIP_H_M[0]:
        h = rng.uniform(*STRIP_H_M)
        h = min(h, hi - y)
        if h < STRIP_H_M[0]:
            break
        strips.append((y, y + h))
        gaps.append({"axis": "x", "at": y + h + STRIP_GAP_M / 2.0,
                     "half": STRIP_GAP_M / 2.0, "a": lo, "b": hi, "use": None})
        y += h + STRIP_GAP_M

    for si, (y0, y1) in enumerate(strips):
        h = y1 - y0
        sub = [(y0, y1)]
        if h > STRIP_TALL_M:
            mid = y0 + h / 2.0
            sub = [(y0, mid - STRIP_GAP_M / 2.0), (mid + STRIP_GAP_M / 2.0, y1)]
        for a, b in sub:
            tol = BAND_TOL_FRAC * (b - a)
            for xa, xb in occ.free_band(a, b, lo, hi, tol=tol):
                if xb - xa < MIN_RUN_M:
                    continue
                band_ha = (xb - xa) * (b - a) / 1e4
                rlo, rhi = _class_range((xa + xb) / 2.0, (a + b) / 2.0, villages)
                target = rng.uniform(rlo, rhi)
                k = max(1, int(round(band_ha / target)))
                while k > 1 and ((xb - xa) - (k - 1) * HEADLAND_M) / k < MIN_FIELD_WIDTH_M:
                    k -= 1
                usable = (xb - xa) - (k - 1) * HEADLAND_M
                mean_w = usable / k
                cuts, widths_c = [xa], []
                pos = xa
                for c in range(k):
                    gap = (WIDE_CUT_M if (c + 1) % WIDE_CUT_EVERY == 0
                           else HEADLAND_M)
                    w = mean_w * (1.0 + (rng.uniform(-0.09, 0.09) if k > 1 else 0.0))
                    w = min(w, xb - pos)
                    cuts.append(pos + w)
                    widths_c.append(gap)
                    pos = cuts[-1] + gap
                for c in range(k):
                    px0, px1 = cuts[c], min(cuts[c + 1], xb)
                    if px1 - px0 < MIN_FIELD_WIDTH_M:
                        continue
                    for ring in carve_parcel(px0, px1, a, b, occ):
                        ha = mg.ring_area(ring) / 1e4
                        per = ml.polyline_length(ring)
                        if ha < MIN_FIELD_HA or ha > MAX_FIELD_HA:
                            continue
                        if per <= 0 or 2.0 * ha * 1e4 / per < MIN_FIELD_WIDTH_M:
                            continue
                        occ.fill_ring(ring, FIELD_FILL_M)
                        fields.append((ring, ha, si))
                    if c + 1 < k:
                        g = widths_c[c]
                        gaps.append({"axis": "y", "at": px1 + g / 2.0, "half": g / 2.0,
                                     "a": a, "b": b, "use": None})
    fields.sort(key=lambda f: (f[2], f[0][0][0]))
    return fields, gaps


# ===================================================================== shelterbelts
def shelterbelts(gaps, occ, rng, rail_centre):
    """The polezakhysni lisosmuhy - the belts planted along the headlands, plus the snow
    fence along the railway. Anything too narrow ever to become a field is handed to the
    woods here as well, which is the descendant of the bocage's `close_wood_gaps`: a
    strip nobody can plough, fence or build on is what you otherwise get."""
    out = []
    for g in gaps:
        if g["use"] is not None or g.get("half", 0.0) < BELT_HALF_M + 1.0:
            continue
        if rng.random() > BELT_P:
            continue
        half = BELT_HALF_M
        if g["axis"] == "x":
            for a, b in occ.free_band(g["at"] - half, g["at"] + half,
                                      g["a"], g["b"], tol=3.0):
                if b - a < BELT_MIN_L_M:
                    continue
                ring = mg.rect_ring((a + b) / 2.0, g["at"], b - a, 2 * half)
                out.append(ring)
                occ.fill_ring(ring, 1.0)
                g["use"] = "belt"
        else:
            a, b = g["a"], g["b"]
            if b - a < BELT_MIN_L_M:
                continue
            for ya, yb in occ.free_band_v(g["at"] - half, g["at"] + half, a, b,
                                          tol=3.0):
                if yb - ya < BELT_MIN_L_M:
                    continue
                ring = mg.rect_ring(g["at"], (ya + yb) / 2.0, 2 * half, yb - ya)
                out.append(ring)
                occ.fill_ring(ring, 1.0)
                g["use"] = "belt"

    inside = mg.clip_polyline_to_playable(rail_centre, EDGE_MARGIN_M,
                                          ml.PLAYABLE_M - EDGE_MARGIN_M)
    for piece in inside:
        centre = mg.offset_polyline(piece, RAIL_BELT_OFF_M + RAIL_BELT_HALF_M, side=1)
        cur = []
        for p in centre:
            if occ.covered(*p):
                if len(cur) > 4:
                    _emit_belt(cur, occ, out)
                cur = []
            else:
                cur.append(p)
        if len(cur) > 4:
            _emit_belt(cur, occ, out)

    rings = []
    for ring in out:
        ha = mg.ring_area(ring) / 1e4
        if ha >= MIN_BELT_HA:
            rings.append((ring, ha))
    return rings


def _emit_belt(pts, occ, out):
    if ml.polyline_length(pts) < BELT_MIN_L_M:
        return
    ring = mg.buffer_polyline(mg.simplify(pts, 8.0), RAIL_BELT_HALF_M)
    out.append(ring)
    occ.fill_ring(ring, 2.0)
