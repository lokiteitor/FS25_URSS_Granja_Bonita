#!/usr/bin/env python3
"""Geometry toolbox shared by the whole pipeline - polylines, rings and segments.

This is the replacement for the geometry half of the missing `map_source.py`. It is
deliberately **standard library only**: it is imported by `map_layout`, by the OSM
generator and by the DEM generator, and dragging numpy into that contract would buy
nothing. Everything here is lists of `(x, y)` tuples in playable metres, x east,
y south from the north edge - the frame `map_extent` defines.

Nothing in here knows about latitude, longitude or heightmaps. `polyline_length` and
`ring_area_ha` are not duplicated here either: they live in `osm_generator/map_extent.py`
and stay there.
"""
import math

__all__ = [
    "lerp", "clamp", "smoothstep", "smootherstep",
    "resample", "densify", "chainage", "chaikin", "simplify", "catmull_rom",
    "polyline_at", "offset_polyline", "project_on_polyline", "polyline_dist",
    "min_curve_radius", "clip_polyline_to_playable", "weave",
    "rect_ring", "chamfer_rect", "grow_ring", "point_in_ring", "centroid",
    "ring_area", "buffer_polyline", "ring_is_simple", "rect_dist", "ring_bbox",
    "point_seg_dist", "seg_intersect", "ray_hit", "seg_seg_dist",
]


# --- scalars ----------------------------------------------------------------------
def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def lerp(a, b, t):
    """Port of generate_osm_bocage.py:725."""
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def smoothstep(t):
    """C1 fade. Clipped, so it is safe to feed a raw ratio."""
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t):
    """C2 fade. Used for every terrain blend: over a 100 m skirt the curvature break
    of smoothstep shows up as a faint ring in the hillshade, and this one has none."""
    t = clamp(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# --- polylines --------------------------------------------------------------------
def chainage(pts):
    """Cumulative distance along a polyline. len(result) == len(pts)."""
    s = [0.0]
    for i in range(1, len(pts)):
        s.append(s[-1] + math.dist(pts[i - 1], pts[i]))
    return s


def densify(pts, max_seg):
    """Split any segment longer than max_seg. Vertices are preserved exactly."""
    if len(pts) < 2:
        return list(pts)
    out = [tuple(pts[0])]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        n = max(1, int(math.ceil(math.dist(a, b) / max_seg)))
        for k in range(1, n + 1):
            out.append(lerp(a, b, k / n))
    return out


def resample(pts, step):
    """Evenly spaced vertices along a polyline. Both endpoints are kept exactly, so
    the last interval is short rather than the line being cut off."""
    if len(pts) < 2:
        return list(pts)
    s = chainage(pts)
    total = s[-1]
    if total <= 0.0:
        return [tuple(pts[0])]
    out, j = [], 0
    n = max(1, int(round(total / step)))
    for k in range(n + 1):
        t = total * k / n
        while j + 2 < len(pts) and s[j + 1] < t:
            j += 1
        seg = s[j + 1] - s[j]
        u = 0.0 if seg <= 0.0 else (t - s[j]) / seg
        out.append(lerp(pts[j], pts[j + 1], u))
    return out


def chaikin(pts, iters=2, closed=False):
    """Corner cutting. For an open line the two endpoints come out untouched, which
    matters: a junction node is placed on an endpoint and smoothing it afterwards
    would move the road a metre away from the way it was supposed to join."""
    out = [tuple(p) for p in pts]
    for _ in range(iters):
        if len(out) < 3:
            break
        nxt = []
        if not closed:
            nxt.append(out[0])
        rng = range(len(out)) if closed else range(len(out) - 1)
        for i in rng:
            a, b = out[i], out[(i + 1) % len(out)]
            nxt.append(lerp(a, b, 0.25))
            nxt.append(lerp(a, b, 0.75))
        if not closed:
            nxt.append(out[-1])
        out = nxt
    return out


def simplify(pts, tol):
    """Douglas-Peucker, iterative rather than recursive so a 20 000-point river
    cannot blow the stack."""
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a, b = pts[i0], pts[i1]
        best, best_i = -1.0, -1
        for i in range(i0 + 1, i1):
            d = point_seg_dist(pts[i], a, b)
            if d > best:
                best, best_i = d, i
        if best > tol:
            keep[best_i] = True
            stack.append((i0, best_i))
            stack.append((best_i, i1))
    return [tuple(p) for p, k in zip(pts, keep) if k]


def catmull_rom(ctrl, per_seg=16, alpha=0.5):
    """Centripetal Catmull-Rom through the control points.

    Centripetal (alpha = 0.5) rather than uniform: uniform Catmull-Rom loops back on
    itself wherever the control points are unevenly spaced, which is exactly what a
    hand-placed road alignment is. Endpoints are duplicated so the curve starts and
    ends on the first and last control point.
    """
    p = [tuple(ctrl[0])] + [tuple(c) for c in ctrl] + [tuple(ctrl[-1])]
    out = []
    for i in range(len(p) - 3):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]

        def tnext(ta, a, b):
            d = math.dist(a, b)
            return ta + (d ** alpha if d > 0.0 else 1e-6)

        t0 = 0.0
        t1 = tnext(t0, p0, p1)
        t2 = tnext(t1, p1, p2)
        t3 = tnext(t2, p2, p3)
        for k in range(per_seg):
            t = t1 + (t2 - t1) * k / per_seg
            a1 = _mix(p0, p1, (t1 - t) / (t1 - t0), (t - t0) / (t1 - t0))
            a2 = _mix(p1, p2, (t2 - t) / (t2 - t1), (t - t1) / (t2 - t1))
            a3 = _mix(p2, p3, (t3 - t) / (t3 - t2), (t - t2) / (t3 - t2))
            b1 = _mix(a1, a2, (t2 - t) / (t2 - t0), (t - t0) / (t2 - t0))
            b2 = _mix(a2, a3, (t3 - t) / (t3 - t1), (t - t1) / (t3 - t1))
            out.append(_mix(b1, b2, (t2 - t) / (t2 - t1), (t - t1) / (t2 - t1)))
    out.append(tuple(ctrl[-1]))
    return out


def _mix(a, b, wa, wb):
    return (a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb)


def polyline_at(poly, t):
    """Point at distance t along a polyline, plus (segment index, fraction).
    Port of generate_osm_bocage.py:808."""
    s = chainage(poly)
    if t <= 0.0:
        return poly[0], 0, 0.0
    if t >= s[-1]:
        return poly[-1], len(poly) - 2, 1.0
    for i in range(len(poly) - 1):
        if s[i + 1] >= t:
            seg = s[i + 1] - s[i]
            u = 0.0 if seg <= 0.0 else (t - s[i]) / seg
            return lerp(poly[i], poly[i + 1], u), i, u
    return poly[-1], len(poly) - 2, 1.0


def project_on_polyline(pt, poly):
    """-> (distance, closest point, segment index, chainage). The chainage is what
    lets the DEM look up the water level or the road profile at a pixel."""
    s = chainage(poly)
    best = (float("inf"), poly[0], 0, 0.0)
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        vx, vy = b[0] - a[0], b[1] - a[1]
        L2 = vx * vx + vy * vy
        u = 0.0 if L2 <= 0.0 else clamp(((pt[0] - a[0]) * vx + (pt[1] - a[1]) * vy) / L2,
                                        0.0, 1.0)
        q = (a[0] + vx * u, a[1] + vy * u)
        d = math.dist(pt, q)
        if d < best[0]:
            best = (d, q, i, s[i] + u * math.sqrt(L2))
    return best


def polyline_dist(pt, poly):
    return project_on_polyline(pt, poly)[0]


def offset_polyline(pts, half_w, side=1):
    """Offset by half_w to one side, using the averaged vertex normal with a miter
    clamp. side = +1 is the left of the direction of travel in screen coordinates."""
    n = len(pts)
    if n < 2:
        return [tuple(p) for p in pts]
    if not isinstance(half_w, (list, tuple)):
        half_w = [float(half_w)] * n
    normals = []
    for i in range(n):
        if i == 0:
            dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        elif i == n - 1:
            dx, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
        else:
            dx = pts[i + 1][0] - pts[i - 1][0]
            dy = pts[i + 1][1] - pts[i - 1][1]
        L = math.hypot(dx, dy) or 1.0
        normals.append((-dy / L * side, dx / L * side))
    out = []
    for i in range(n):
        nx, ny = normals[i]
        out.append((pts[i][0] + nx * half_w[i], pts[i][1] + ny * half_w[i]))
    return out


def min_curve_radius(pts):
    """Smallest circumradius over consecutive triples. This is the number that keeps
    nearest-segment chainage single-valued inside a corridor: where the radius drops
    below the corridor's influence width the medial axis enters the band, and the
    profile lookup jumps across it."""
    best = float("inf")
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        if area2 < 1e-9:
            continue
        best = min(best, ab * bc * ca / (2.0 * area2))
    return best


def clip_polyline_to_playable(pts, lo=0.0, hi=8192.0):
    """Split a polyline at the playable border, keeping only the inside pieces. The
    layout runs out to the canvas edge so the DEM has ground beyond the border; the
    OSM half emits only what the player can reach."""
    pieces, cur = [], []
    inside = lambda p: lo <= p[0] <= hi and lo <= p[1] <= hi
    for i in range(len(pts)):
        p = pts[i]
        if inside(p):
            cur.append(tuple(p))
        else:
            if cur:
                q = _border_cross(cur[-1], p, lo, hi)
                if q:
                    cur.append(q)
                pieces.append(cur)
                cur = []
        if not inside(p) and i + 1 < len(pts) and inside(pts[i + 1]):
            q = _border_cross(pts[i + 1], p, lo, hi)
            if q:
                cur.append(q)
    if cur:
        pieces.append(cur)
    return [p for p in pieces if len(p) >= 2]


def _border_cross(inside_pt, outside_pt, lo, hi):
    """Bisection to the border. Cheap, exact to a millimetre, and immune to the
    corner cases an analytic clip has when a segment leaves through two edges."""
    a, b = inside_pt, outside_pt
    ok = lambda p: lo <= p[0] <= hi and lo <= p[1] <= hi
    if not ok(a):
        return None
    for _ in range(40):
        m = lerp(a, b, 0.5)
        if ok(m):
            a = m
        else:
            b = m
    return a


def weave(poly, extra):
    """Rebuild a polyline so the given points become real vertices of it.

    Port of generate_osm_bocage.py:822. Without this a spur that merely touches a road
    is not joined to it: `get_node` shares a node only when both ways carry the exact
    same coordinate.
    """
    if not extra:
        return [tuple(p) for p in poly]
    hits = []
    for q in extra:
        d, pt, i, _ = project_on_polyline(q, poly)
        seg = math.dist(poly[i], poly[i + 1]) or 1.0
        u = math.dist(poly[i], pt) / seg
        hits.append((i, u, pt))
    hits.sort()
    out, k = [], 0
    for i in range(len(poly) - 1):
        out.append(tuple(poly[i]))
        while k < len(hits) and hits[k][0] == i:
            if math.dist(hits[k][2], poly[i]) > 1e-6:
                out.append(hits[k][2])
            k += 1
    out.append(tuple(poly[-1]))
    ded = [out[0]]
    for p in out[1:]:
        if math.dist(p, ded[-1]) > 1e-6:
            ded.append(p)
    return ded


# --- rings ------------------------------------------------------------------------
def rect_ring(cx, cy, w, h, angle_deg=0.0):
    """Closed 5-point ring of a rectangle, optionally rotated about its centre."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    hw, hh = w / 2.0, h / 2.0
    pts = []
    for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        pts.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    pts.append(pts[0])
    return pts


def chamfer_rect(cx, cy, w, h, angle_deg=0.0, c=25.0):
    """Rectangle with the corners cut off - nine points closed. A platform reads as a
    deliberate pad rather than a raw box, and the DEM's skirt gets a corner to work
    with instead of a right angle."""
    c = min(c, w / 2.0 - 1.0, h / 2.0 - 1.0)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    hw, hh = w / 2.0, h / 2.0
    local = [(-hw + c, -hh), (hw - c, -hh), (hw, -hh + c), (hw, hh - c),
             (hw - c, hh), (-hw + c, hh), (-hw, hh - c), (-hw, -hh + c)]
    pts = [(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca) for dx, dy in local]
    pts.append(pts[0])
    return pts


def ring_bbox(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def centroid(ring):
    """Port of generate_osm_bocage.py:354 - skips the closing point."""
    pts = ring[:-1] if len(ring) > 2 and math.dist(ring[0], ring[-1]) < 1e-9 else ring
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def ring_area(ring):
    """Shoelace area in square metres. The ring may be open or closed."""
    pts = ring[:-1] if len(ring) > 2 and math.dist(ring[0], ring[-1]) < 1e-9 else ring
    if len(pts) < 3:
        return 0.0
    twice = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] -
                pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    return abs(twice) / 2.0


def point_in_ring(pt, ring):
    """Ray cast. Port of generate_osm_bocage.py:601."""
    x, y = pt
    pts = ring[:-1] if len(ring) > 2 and math.dist(ring[0], ring[-1]) < 1e-9 else ring
    inside = False
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xx = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if xx > x:
                inside = not inside
    return inside


def grow_ring(ring, m):
    """Offset a simple ring outwards (m > 0) or inwards (m < 0), miter joins clamped
    at 3x. Exact for a rectangle, which is what most calls here are."""
    if abs(m) < 1e-9:
        return [tuple(p) for p in ring]
    pts = ring[:-1] if len(ring) > 2 and math.dist(ring[0], ring[-1]) < 1e-9 else ring
    n = len(pts)
    if n < 3:
        return [tuple(p) for p in ring]
    # Screen coordinates have y southwards, so a ring whose shoelace sum is positive
    # runs clockwise on screen; the outward normal flips with it.
    twice = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                for i in range(n))
    sign = 1.0 if twice > 0 else -1.0
    out = []
    for i in range(n):
        p, nxt, prv = pts[i], pts[(i + 1) % n], pts[i - 1]
        e1 = _unit(p[0] - prv[0], p[1] - prv[1])
        e2 = _unit(nxt[0] - p[0], nxt[1] - p[1])
        n1 = (e1[1] * sign, -e1[0] * sign)
        n2 = (e2[1] * sign, -e2[0] * sign)
        bx, by = n1[0] + n2[0], n1[1] + n2[1]
        L = math.hypot(bx, by)
        if L < 1e-9:
            bx, by, L = n2[0], n2[1], 1.0
        bx, by = bx / L, by / L
        cosr = bx * n2[0] + by * n2[1]
        scale = min(3.0, 1.0 / cosr) if cosr > 1e-6 else 3.0
        out.append((p[0] + bx * m * scale, p[1] + by * m * scale))
    out.append(out[0])
    return out


def _unit(dx, dy):
    L = math.hypot(dx, dy) or 1.0
    return (dx / L, dy / L)


def buffer_polyline(pts, half):
    """Polyline -> one closed ring: the left offset, then the right offset reversed.
    `half` may be a scalar or one value per vertex (a river that widens downstream)."""
    left = offset_polyline(pts, half, side=1)
    right = offset_polyline(pts, half, side=-1)
    ring = left + right[::-1]
    ring.append(ring[0])
    return ring


def ring_is_simple(ring):
    """True when no two non-adjacent edges cross. A field ring that fails this is a
    bow-tie and Giants will fill it wrong."""
    pts = ring[:-1] if len(ring) > 2 and math.dist(ring[0], ring[-1]) < 1e-9 else ring
    n = len(pts)
    if n < 4:
        return True
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if seg_intersect(a, b, c, d) is not None:
                return False
    return True


def rect_dist(r1, r2):
    """Minimum distance between two convex rings. 0 when they touch or overlap."""
    for p in r1:
        if point_in_ring(p, r2):
            return 0.0
    for p in r2:
        if point_in_ring(p, r1):
            return 0.0
    a = r1[:-1] if math.dist(r1[0], r1[-1]) < 1e-9 else r1
    b = r2[:-1] if math.dist(r2[0], r2[-1]) < 1e-9 else r2
    best = float("inf")
    for i in range(len(a)):
        for j in range(len(b)):
            best = min(best, seg_seg_dist(a[i], a[(i + 1) % len(a)],
                                          b[j], b[(j + 1) % len(b)]))
    return best


# --- segments ---------------------------------------------------------------------
def point_seg_dist(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy
    if L2 <= 0.0:
        return math.dist(p, a)
    u = clamp(((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2, 0.0, 1.0)
    return math.dist(p, (a[0] + vx * u, a[1] + vy * u))


def seg_intersect(a, b, c, d):
    """-> the crossing point, or None. Proper crossings only: touching endpoints do
    not count, or every ring would report itself as self-intersecting."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den
    if 1e-9 < t < 1.0 - 1e-9 and 1e-9 < u < 1.0 - 1e-9:
        return (a[0] + r[0] * t, a[1] + r[1] * t)
    return None


def seg_seg_dist(a, b, c, d):
    if seg_intersect(a, b, c, d) is not None:
        return 0.0
    return min(point_seg_dist(a, c, d), point_seg_dist(b, c, d),
               point_seg_dist(c, a, b), point_seg_dist(d, a, b))


def ray_hit(origin, direction, poly):
    """First intersection of a ray with a polyline -> (point, segment index) or None.
    Port of generate_osm_bocage.py:839."""
    ox, oy = origin
    dx, dy = _unit(*direction)
    best = None
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ex, ey = b[0] - a[0], b[1] - a[1]
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = ((a[0] - ox) * ey - (a[1] - oy) * ex) / den
        u = ((a[0] - ox) * dy - (a[1] - oy) * dx) / den
        if t > 1e-6 and -1e-9 <= u <= 1.0 + 1e-9:
            if best is None or t < best[0]:
                best = (t, (ox + dx * t, oy + dy * t), i)
    return None if best is None else (best[1], best[2])
