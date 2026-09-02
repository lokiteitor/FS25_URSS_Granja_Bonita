#!/usr/bin/env python3
"""Access roads: how every yard on the map reaches the main road.

Two promises are made here, and both are the kind that a "nearest point on the network"
spur cannot keep:

  * **Every farm and every industry platform is connected to the main road.** Not to
    *something* - to the main road, along a chain of ways that share their nodes. Aiming
    each yard at whatever way happens to be closest builds spur trees that never touch
    the road at all: eight industry pads hanging off an estate road that hangs off
    nothing.
  * **No road crosses a platform.** A yard is a place a road arrives at, not a place it
    drives through, so a link stops on the fence line and every other platform on the
    map is an obstacle to be passed on one side.

Both fall out of one construction. The playable area is rasterised at `CELL_M` into a
cost grid where the platforms, the river and the lake are impassable and the railway is
merely expensive; a multi-source Dijkstra flood then measures, for every cell, the
cheapest route back to the road network. Yards are attached one at a time, cheapest
first, and each accepted link is pushed back into the flood as a new source at zero
cost - so the next yard routes to the road *or to a link already joined to it*, and the
whole thing grows as one tree rooted on the main road. That is Prim's algorithm with
a geodesic distance, and connectivity is a property of it rather than something to test
for afterwards.

The estate rows on the railway need no special case any more: the four pads of a row
chain onto each other's links and the chain reaches the road once, which is exactly the
estate road the previous version had to lay out by hand.
"""
import heapq
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_layout as ml
import map_geom as mg
import parcels as pc

CELL_M = 24.0            # routing grid. Fine enough that a 20 m gap is still a gap,
                         # coarse enough that the flood is a second of work
PAD_CLEAR_M = 20.0       # how far a link road stays off a platform edge
RIVER_CLEAR_M = 25.0
LAKE_CLEAR_M = 35.0
EDGE_CLEAR_M = 40.0
RAIL_HALF_M = 34.0
WOOD_COST = 1.5          # a road through a wood is fine, it just costs the trees
RAIL_COST = 9.0          # a level crossing is allowed, at the price of ~600 m of detour
GATE_BAND_M = 1.6 * CELL_M
SIMPLIFY_M = 14.0        # under PAD_CLEAR_M, so straightening can never cut a corner
                         # far enough to reach the platform it was routed around
PULL_LOOKAHEAD = 80      # vertices the taut-string pass may skip in one go
INF = float("inf")


def weld(way, pt, get_node):
    """Make `pt` a real vertex of `way`, so a way ending there shares its node.

    This is the step that separates a junction from a coincidence. `get_node` keys on
    millimetre-rounded coordinates, so two ways are joined only when both *carry* the
    same point; a spur whose endpoint merely lies on a road segment is, in the file, a
    way that touches nothing.
    """
    if any(math.dist(pt, q) < 1e-3 for q in way['coords']):
        return False
    way['coords'] = mg.weave(way['coords'], [pt])
    way['node_refs'] = [get_node(x, y) for x, y in way['coords']]
    return True


def _dedupe(pts, tol=1e-3):
    out = [tuple(pts[0])]
    for p in pts[1:]:
        if math.dist(p, out[-1]) > tol:
            out.append(tuple(p))
    return out


class Router:
    """The cost grid, the flood, and the growing tree of link roads on top of it."""

    def __init__(self, L, woods=()):
        self.cell = CELL_M
        self.n = n = int(round(ml.PLAYABLE_M / CELL_M))

        blocked = pc.Occupancy(cell_m=CELL_M)
        blocked.fill_border(EDGE_CLEAR_M)
        for p in L["pads"]:
            blocked.fill_ring(p["ring"], PAD_CLEAR_M)
        blocked.fill_polyline(L["river"]["centre"],
                              max(L["river"]["half_w"]) + RIVER_CLEAR_M)
        blocked.fill_ring(L["lake"]["ring"], LAKE_CLEAR_M)

        # Expensive, not impassable: the railway can be crossed at grade, and the wood
        # can be felled. Making either one a wall strands the yards behind it.
        rail = pc.Occupancy(cell_m=CELL_M)
        rail.fill_polyline(L["rail"]["centre"], RAIL_HALF_M)
        wood = pc.Occupancy(cell_m=CELL_M)
        for ring, _ in woods:
            wood.fill_ring(ring)

        blk, rl, wd = blocked.arr, rail.arr, wood.arr
        self.blk = blk.reshape(-1).tolist()
        cost = []
        for i in range(n):
            for j in range(n):
                if blk[i, j]:
                    cost.append(INF)
                elif rl[i, j]:
                    cost.append(RAIL_COST)
                elif wd[i, j]:
                    cost.append(WOOD_COST)
                else:
                    cost.append(1.0)
        self.cost = cost

        self.dist = [INF] * (n * n)
        self.par = [-1] * (n * n)
        self.owner = [-1] * (n * n)
        self.heap = []
        self.ways = []
        self._nbr = [(dr, dc, CELL_M * math.hypot(dr, dc))
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                     if (dr, dc) != (0, 0)]

    # --- grid helpers ---------------------------------------------------------------
    def _centre(self, i):
        r, c = divmod(i, self.n)
        return ((c + 0.5) * self.cell, (r + 0.5) * self.cell)

    def _index(self, x, y):
        c, r = int(x / self.cell), int(y / self.cell)
        if not (0 <= r < self.n and 0 <= c < self.n):
            return -1
        return r * self.n + c

    def _blocked_at(self, x, y):
        i = self._index(x, y)
        return True if i < 0 else self.blk[i]

    def _visible(self, a, b):
        """True when the straight line a-b stays on passable ground."""
        L = math.dist(a, b)
        steps = max(1, int(L / (self.cell * 0.4)))
        for k in range(steps + 1):
            t = k / steps
            if self._blocked_at(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t):
                return False
        return True

    def _cells_along(self, pts):
        out = []
        for a, b in zip(pts, pts[1:]):
            L = math.dist(a, b)
            for k in range(max(1, int(L / (self.cell * 0.5))) + 1):
                t = k / max(1, int(L / (self.cell * 0.5)))
                i = self._index(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                if i >= 0:
                    out.append(i)
        return out

    # --- the flood ------------------------------------------------------------------
    def add_source_way(self, way):
        """Enter a way into the flood as free ground: everything routed from now on may
        arrive here instead of walking all the way back to the road."""
        k = len(self.ways)
        self.ways.append(way)
        for i in self._cells_along(way['coords']):
            if self.cost[i] < INF and self.dist[i] > 0.0:
                self.dist[i] = 0.0
                self.par[i] = -1
                self.owner[i] = k
                heapq.heappush(self.heap, (0.0, i))
        return k

    def relax(self):
        dist, par, cost, heap, n, nbr = (self.dist, self.par, self.cost, self.heap,
                                         self.n, self._nbr)
        while heap:
            d, i = heapq.heappop(heap)
            if d > dist[i]:
                continue
            r, c = divmod(i, n)
            for dr, dc, w in nbr:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < n and 0 <= cc < n):
                    continue
                j = rr * n + cc
                cj = cost[j]
                if cj == INF:
                    continue
                nd = d + w * cj
                if nd < dist[j]:
                    dist[j] = nd
                    par[j] = i
                    heapq.heappush(heap, (nd, j))

    def _trace(self, i):
        path = []
        while True:
            path.append(i)
            if self.par[i] == -1:
                return path[::-1], self.owner[i]
            i = self.par[i]

    # --- attaching a yard -----------------------------------------------------------
    def _gate_cells(self, pad):
        """The passable cells in the collar just outside a platform - where a road may
        legitimately stop."""
        ring = mg.grow_ring(pad["ring"], PAD_CLEAR_M + GATE_BAND_M)
        x0, y0, x1, y1 = mg.ring_bbox(ring)
        c = self.cell
        out = []
        for r in range(max(0, int(y0 / c)), min(self.n, int(y1 / c) + 1)):
            for k in range(max(0, int(x0 / c)), min(self.n, int(x1 / c) + 1)):
                i = r * self.n + k
                if self.cost[i] == INF:
                    continue
                if mg.point_in_ring(self._centre(i), ring):
                    out.append(i)
        return out

    def _best_gate(self, pad):
        best = None
        for i in self._gate_cells(pad):
            d = self.dist[i]
            if d < INF and (best is None or d < best[0]):
                best = (d, i)
        return best

    def _tidy(self, pts):
        """Turn the grid trace into a road: straight where it can be, bent where the
        ground it was routed around demands it.

        Two passes, and both are needed. Douglas-Peucker alone leaves the staircase an
        eight-connected flood produces wherever two routes tie on cost - a right-angled
        zigzag no amount of tolerance under `SIMPLIFY_M` can collapse. Pulling the
        string taut alone is too slow on a two-hundred-cell trace. So: simplify to the
        corners, restore any detour the shortcut cut through, then pull the corners out
        against the same obstacle mask the route was found on, which is what makes the
        result a line a grader could actually have built.
        """
        if len(pts) < 3:
            return list(pts)
        s = mg.simplify(pts, SIMPLIFY_M)
        keep = [s[0]]
        for k in range(1, len(s)):
            if self._visible(keep[-1], s[k]):
                keep.append(s[k])
            else:
                i0, i1 = pts.index(s[k - 1]), pts.index(s[k])
                keep.extend(pts[i0 + 1:i1 + 1])
        out, i = [keep[0]], 0
        while i < len(keep) - 1:
            j = min(len(keep) - 1, i + PULL_LOOKAHEAD)
            while j > i + 1 and not self._visible(keep[i], keep[j]):
                j -= 1
            out.append(keep[j])
            i = j
        return out

    def link_pads(self, add_way, get_node, pads, tags_for):
        """Attach every pad, cheapest first, growing the network as we go.

        Cheapest first is what makes a row of pads chain onto one another instead of
        each driving its own kilometre back to the road: by the time the second pad of
        a row is considered, the first one's link is already zero-cost ground.
        """
        self.relax()
        pending = list(pads)
        done, orphans = [], []
        while pending:
            best = None
            for p in pending:
                g = self._best_gate(p)
                if g is not None and (best is None or g[0] < best[0]):
                    best = (g[0], p, g[1])
            if best is None:
                orphans = list(pending)
                break
            _, pad, cell = best
            pending.remove(pad)

            path, owner = self._trace(cell)
            pts = self._tidy([self._centre(i) for i in path])

            # The road end: the exact point on the way we are leaving, welded into it,
            # so the two ways share a node rather than merely meeting on the render.
            target = self.ways[owner]
            _, q, _, _ = mg.project_on_polyline(pts[0], target['coords'])
            pts[0] = q
            # The yard end: the gate on the fence line. The link stops there - driving
            # it into the middle of the yard is the crossing this module exists to
            # avoid.
            hit = mg.ray_hit(pts[-1], (pad["cx"] - pts[-1][0], pad["cy"] - pts[-1][1]),
                             pad["ring"])
            if hit is not None:
                pts.append(hit[0])
            pts = _dedupe(pts)
            if len(pts) < 2:
                orphans.append(pad)
                continue

            way = add_way(pts, tags_for(pad))
            weld(target, q, get_node)
            self.add_source_way(way)
            self.relax()
            done.append((pad, way))
        return done, orphans
