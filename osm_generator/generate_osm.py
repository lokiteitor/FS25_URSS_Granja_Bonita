#!/usr/bin/env python3
"""Write `map.osm`: the Ukrainian forest-steppe layout for the 8192 x 8192 m playable area.

Everything structural - the river, the lake, the road, the railway and the thirty flat
platforms - comes from `map_layout.layout()`, which the DEM generator reads as well.
Neither half re-derives the other's geometry, so the heightmap and the vectors describe
the same place by construction rather than by agreement.

What this file adds on top of that skeleton is the countryside: woodland, fields, the
link roads and farm tracks between them, and the shelterbelts along the headlands. The
field cutter and the wood shaper live in `parcels.py`.

Tag vocabulary, unchanged from the previous map so `visualize_osm.py` and
`check_forest_nodes.py` keep working untouched:

    landuse=farmland                                 fields
    landuse=farmyard                                 villages, farms, industry pads
    natural=wood + landuse=farmyard + leaf_type      woodland and shelterbelts
    natural=water + waterway=riverbank | water=lake  the river and the lake
    highway=primary / secondary / tertiary           road hierarchy
    railway=rail                                     the line (new: visualize_osm.py has
                                                     always drawn it, but no generator
                                                     had ever emitted one)
    bridge=yes + layer=1                             river crossings

Two rules the old bocage generator learned the hard way, restated because everything
here depends on them:

  * `get_node` keys on millimetre-rounded coordinates, and that IS the junction
    mechanism. Two ways carrying the same coordinate share a node; two ways that merely
    cross at a coordinate neither carries are not joined at all.
  * Smooth first, weave second, emit third. Aiming a spur at an unsmoothed centreline
    and then emitting the smoothed one leaves the two a metre apart: close enough to
    look joined, far enough not to be.
  * Every yard reaches the main road, and no road drives through a yard. That is
    `roads.py`, and `verify` measures both on the written file rather than trusting the
    intention behind them.
"""
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_layout as ml
import map_geom as mg
import map_extent as mx
import parcels as pc
import roads as rd

OUT_NAME = "map.osm"
SEED = ml.SEED

# --- link roads ---------------------------------------------------------------------
TRACK_MIN_L_M = 320.0
TRACK_P = 0.65                 # share of the free headland corridors a track may claim.
                               # The rest are left for the shelterbelts: tracks and belts
                               # want the same ground, and a track pass that takes every
                               # corridor it can reach leaves the map with no belts at all
TRACK_JOIN_M = 45.0            # how close a track must come to count as joined. Wider
                               # than it looks it should be: a free run stops at the
                               # road *corridor*, not at the road, so the nearest a
                               # candidate can get is the corridor half-width
STUB_MIN_M = 200.0
LEAF_TYPES = ("broadleaved", "broadleaved", "mixed")


# ========================================================================== the pools
def make_pools():
    """Node and way pools as closures over dicts - the idiom from
    generate_osm_bocage.py:156-179, kept because the coordinate-keyed node table is the
    only thing that makes two ways share a junction."""
    nodes, node_coords, node_tags = {}, {}, {}
    ways = []
    next_node_id, next_way_id = [1], [1]

    def get_node(x, y):
        key = (round(float(x), 3), round(float(y), 3))
        if key not in nodes:
            nodes[key] = next_node_id[0]
            node_coords[next_node_id[0]] = ml.local_to_global(*key)
            next_node_id[0] += 1
        return nodes[key]

    def add_way(coords, tags):
        pts = [(float(x), float(y)) for x, y in coords]
        ways.append({'id': next_way_id[0], 'coords': pts, 'tags': dict(tags),
                     'node_refs': [get_node(x, y) for x, y in pts]})
        next_way_id[0] += 1
        return ways[-1]

    return get_node, add_way, ways, node_coords, node_tags


def write_osm(path, ways, node_coords, node_tags):
    """Port of generate_osm_bocage.py:1279, plus optional node tags.

    A level crossing is a tag on the *node*, not on either way, so the writer has to be
    able to carry one. Neither reader parses node tags, so adding them is free.
    """
    minlat, minlon, maxlat, maxlon = mx.bounds()
    osm = ET.Element('osm', version='0.6', generator='FS25 map pipeline')
    osm.append(ET.Comment(
        f"\n       Playable area: {mx.PLAYABLE_M:.0f} x {mx.PLAYABLE_M:.0f} m, "
        f"centre {mx.LAT_CENTER:.4f}, {mx.LON_CENTER:.4f}\n"
        "       (Cherkasy oblast forest-steppe - deep chernozem, the highest-yielding\n"
        "       arable belt in Ukraine).\n"
        "       Local coordinates are playable metres, x east, y south from the north\n"
        f"       edge, so the centre of the map is ({mx.HALF_M:.0f}, {mx.HALF_M:.0f}).\n"
        f"       Built from map_layout.py, seed {ml.SEED}; the DEM generator reads the\n"
        "       same layout, so terrain and vectors agree.\n  "))
    ET.SubElement(osm, 'bounds', {
        'minlat': f"{minlat:.10f}", 'minlon': f"{minlon:.10f}",
        'maxlat': f"{maxlat:.10f}", 'maxlon': f"{maxlon:.10f}"})

    stamp = {'version': '1', 'timestamp': '2026-09-01T12:00:00Z',
             'changeset': '1', 'uid': '1', 'user': 'generator'}
    for nid in sorted(node_coords):
        lat, lon = node_coords[nid]
        n = ET.SubElement(osm, 'node', {'id': str(nid), 'lat': f"{lat:.10f}",
                                        'lon': f"{lon:.10f}", **stamp})
        for k, v in node_tags.get(nid, {}).items():
            ET.SubElement(n, 'tag', {'k': k, 'v': v})
    for w in ways:
        el = ET.SubElement(osm, 'way', {'id': str(w['id']), **stamp})
        for ref in w['node_refs']:
            ET.SubElement(el, 'nd', {'ref': str(ref)})
        for k, v in w['tags'].items():
            ET.SubElement(el, 'tag', {'k': k, 'v': str(v)})

    pretty = minidom.parseString(ET.tostring(osm, encoding='utf-8')).toprettyxml(
        indent='  ', encoding='utf-8')
    with open(path, "wb") as fh:
        fh.write(pretty)


# ======================================================================= linear work
def split_line(line, bridges):
    """Cut a centreline into (points, is_bridge) runs using its own chainage.

    The spans come straight from `layout()['bridges']`, the same list the DEM uses to
    stop its corridor filling the valley - so the way tagged bridge=yes here is exactly
    the stretch the terrain leaves open underneath.
    """
    pts, s = line["centre"], line["s"]

    def on_bridge(si):
        return any(b["s0"] <= si <= b["s1"] for b in bridges)

    pieces, cur, cur_b = [], [], on_bridge(s[0])
    for p, si in zip(pts, s):
        b = on_bridge(si)
        if b != cur_b and cur:
            cur.append(p)
            pieces.append((cur, cur_b))
            cur, cur_b = [p], b
        else:
            cur.append(p)
    if cur:
        pieces.append((cur, cur_b))
    return pieces


def emit_line(add_way, line, bridges, tags, name, keep=()):
    """Emit a centreline as playable-area ways, bridges tagged separately.

    `keep` is woven back in after the simplify pass. Without it the level crossing
    disappears: it sits on a nearly straight stretch of both alignments, so
    Douglas-Peucker drops it from each, and the node the two lines were supposed to
    share ends up referenced by neither.
    """
    out = []
    for pts, is_bridge in split_line(line, bridges):
        for piece in mg.clip_polyline_to_playable(pts, 0.0, ml.PLAYABLE_M):
            if ml.polyline_length(piece) < 5.0:
                continue
            piece = mg.simplify(piece, 2.0)
            for k in keep:
                if mg.polyline_dist(k, piece) < 3.0:
                    piece = mg.weave(piece, [k])
                    i = min(range(len(piece)), key=lambda z: math.dist(piece[z], k))
                    piece[i] = (float(k[0]), float(k[1]))
            t = dict(tags)
            t['name'] = name
            if is_bridge:
                t['bridge'] = 'yes'
                t['layer'] = '1'
            out.append(add_way(piece, t))
    return out


def touches(a, b, tol=TRACK_JOIN_M):
    """True when two polylines cross or come within tol of each other."""
    for i in range(len(a) - 1):
        for j in range(len(b) - 1):
            if mg.seg_intersect(a[i], a[i + 1], b[j], b[j + 1]) is not None:
                return True
            if mg.seg_seg_dist(a[i], a[i + 1], b[j], b[j + 1]) <= tol:
                return True
    return False


def connect_crossings(ways, get_node):
    """Give every at-grade crossing a shared node.

    Port of generate_osm_bocage.py:1213, generalised to the railway. Ways on different
    layers are skipped: splicing a bridge to the way it flies over produces a road that
    dives into the river, which is the kind of thing you only notice in game.
    """
    linear = [w for w in ways if 'highway' in w['tags'] or 'railway' in w['tags']]
    inserts = {id(w): [] for w in linear}
    for a in range(len(linear)):
        wa = linear[a]
        for b in range(a + 1, len(linear)):
            wb = linear[b]
            if wa['tags'].get('bridge') or wb['tags'].get('bridge'):
                continue
            if wa['tags'].get('layer') != wb['tags'].get('layer'):
                continue
            ax0 = min(p[0] for p in wa['coords']); ax1 = max(p[0] for p in wa['coords'])
            ay0 = min(p[1] for p in wa['coords']); ay1 = max(p[1] for p in wa['coords'])
            bx0 = min(p[0] for p in wb['coords']); bx1 = max(p[0] for p in wb['coords'])
            by0 = min(p[1] for p in wb['coords']); by1 = max(p[1] for p in wb['coords'])
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            for i in range(len(wa['coords']) - 1):
                for j in range(len(wb['coords']) - 1):
                    hit = mg.seg_intersect(wa['coords'][i], wa['coords'][i + 1],
                                           wb['coords'][j], wb['coords'][j + 1])
                    if hit is not None:
                        inserts[id(wa)].append(hit)
                        inserts[id(wb)].append(hit)
    n = 0
    for w in linear:
        extra = inserts[id(w)]
        if not extra:
            continue
        w['coords'] = mg.weave(w['coords'], extra)
        w['node_refs'] = [get_node(x, y) for x, y in w['coords']]
        n += len(extra)
    return n


def prune_stubs(chains, min_len):
    """Drop dead-end chains shorter than min_len. Port of bocage:660, simplified: the
    tracks here are straight segments, so a stub is just a short chain that touches the
    network at one end only."""
    keep = []
    for c in chains:
        if ml.polyline_length(c) >= min_len:
            keep.append(c)
    return keep


# ============================================================================ stages
def stage_water(add_way, L):
    riv, lk = L["river"], L["lake"]
    n = 0
    # Cut into reaches so no single way carries hundreds of nodes.
    step = max(2, len(riv["centre"]) // 3)
    for a in range(0, len(riv["centre"]) - 1, step):
        b = min(len(riv["centre"]), a + step + 1)
        pts = riv["centre"][a:b]
        hw = riv["half_w"][a:b]
        ring = mg.buffer_polyline(pts, hw)
        ring = [p for p in ring]
        add_way(mg.simplify(ring, 3.0),
                {'natural': 'water', 'waterway': 'riverbank',
                 'name': f"Rio {riv['name']}"})
        n += 1
    add_way(mg.simplify(lk["ring"], 4.0),
            {'natural': 'water', 'water': 'lake', 'name': f"Lago {lk['name']}",
             'area': 'yes'})
    return n + 1, mg.ring_area(lk["ring"]) / 1e4


def stage_pads(add_way, L):
    counts = {}
    for p in L["pads"]:
        tags = {'landuse': 'farmyard', 'name': f"{p['name']} ({p['ha']:.1f} ha)"}
        if p["kind"] == "village":
            tags['place'] = 'village'
        add_way(p["ring"], tags)
        counts[p["kind"]] = counts.get(p["kind"], 0) + 1
    return counts


def stage_woods(add_way, woods, belts):
    for i, (ring, ha) in enumerate(woods, 1):
        add_way(ring, {'natural': 'wood', 'landuse': 'farmyard',
                       'leaf_type': LEAF_TYPES[i % len(LEAF_TYPES)],
                       'name': f"Wood {i} ({ha:.1f} ha)"})
    for i, (ring, ha) in enumerate(belts, 1):
        add_way(ring, {'natural': 'wood', 'landuse': 'farmyard',
                       'leaf_type': 'broadleaved',
                       'name': f"Shelterbelt {i} ({ha:.1f} ha)"})


def village_streets(add_way, L):
    """A back lane and two or three cross streets inside each village.

    Far less machinery than the bocage's `village_streets`: the main road already runs
    the length of the pad - that is what "strung along the road" means - so what is
    missing is only the block either side of it.
    """
    out = []
    for p in L["pads"]:
        if p["kind"] != "village":
            continue
        a = math.radians(p["angle_deg"])
        ca, sa = math.cos(a), math.sin(a)

        def loc(u, v):
            return (p["cx"] + u * ca - v * sa, p["cy"] + u * sa + v * ca)

        hw, hh = p["w"] / 2.0, p["h"] / 2.0
        for v in (-hh * 0.55, hh * 0.55):
            lane = [loc(-hw * 0.82, v), loc(hw * 0.82, v)]
            out.append(add_way(lane, {'highway': 'tertiary',
                                      'name': 'Village Lane'}))
        for u in (-hw * 0.55, 0.0, hw * 0.55):
            st = [loc(u, -hh * 0.8), loc(u, hh * 0.8)]
            out.append(add_way(st, {'highway': 'secondary',
                                    'name': 'Village Street'}))
    return out


def farm_tracks(add_way, get_node, rng, gaps, occ, network, river_centre):
    """Service tracks along the headlands the fields left.

    The headlands already *are* a rectilinear corridor grid, so no path search is
    needed; what is needed is a guarantee that every track can be reached. Candidates
    are grown outwards from the ways that already exist, one pass at a time, and
    whatever never attaches is dropped rather than emitted as an unreachable way.

    Tracks never bridge the river: a dirt track with a bridge every 300 m is not
    credible, and each one would cost the DEM a span to hold open.
    """
    cand = []
    for g in gaps:
        if g["use"] is not None or rng.random() >= TRACK_P:
            continue
        h = pc.TRACK_CLEAR_M
        if g["axis"] == "x":
            for a, b in occ.free_band(g["at"] - h, g["at"] + h, g["a"], g["b"], tol=3.0):
                if b - a >= TRACK_MIN_L_M:
                    cand.append((g, [(a, g["at"]), (b, g["at"])]))
        else:
            for a, b in occ.free_band_v(g["at"] - h, g["at"] + h, g["a"], g["b"],
                                        tol=3.0):
                if b - a >= TRACK_MIN_L_M:
                    cand.append((g, [(g["at"], a), (g["at"], b)]))
    cand = [(g, c) for g, c in cand
            if min(mg.polyline_dist(p, river_centre) for p in c) > 60.0]

    accepted, changed = [], True
    while changed:
        changed = False
        for item in list(cand):
            g, c = item
            reach = [w for w in network if touches(c, w['coords'])]
            reach += [w for _, w in accepted if touches(c, w['coords'])]
            if not reach:
                continue
            # Snap the loose ends onto whatever they reached, and weld the snapped point
            # into that way. A track that stops the corridor's half-width short of the
            # road looks joined on the render and is not joined in the file, and that is
            # the one mistake the whole node-sharing scheme exists to avoid.
            p0, w0 = _snap(c[0], reach)
            p1, w1 = _snap(c[-1], reach)
            cand.remove(item)
            changed = True
            if (w0 is None and w1 is None) or ml.polyline_length([p0, p1]) < STUB_MIN_M:
                continue
            way = add_way([p0, p1], {'highway': 'tertiary', 'name': 'Farm Track'})
            for pt, tw in ((p0, w0), (p1, w1)):
                if tw is not None:
                    rd.weld(tw, pt, get_node)
            occ.fill_polyline([p0, p1], pc.TRACK_CLEAR_M)
            g["use"] = "track"
            accepted.append((g, way))
    return (len(accepted),
            sum(ml.polyline_length(w['coords']) for _, w in accepted))


def _snap(pt, reach):
    """Pull an endpoint onto the nearest way it is close to -> (point, that way)."""
    best = None
    for w in reach:
        d, q, _, _ = mg.project_on_polyline(pt, w['coords'])
        if best is None or d < best[0]:
            best = (d, q, w)
    if best is None or best[0] > TRACK_JOIN_M * 1.5:
        return pt, None
    return best[1], best[2]


# ============================================================================== main
def main():
    t0 = time.time()
    print("=== Generating OSM data for the Ukraine map ===")
    print(f"   centre {mx.LAT_CENTER:.4f}, {mx.LON_CENTER:.4f} - Cherkasy oblast "
          "forest-steppe")
    rng = random.Random(SEED)
    get_node, add_way, ways, node_coords, node_tags = make_pools()

    print("0. Layout contract...")
    L = ml.layout()
    kinds = {}
    for p in L["pads"]:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    assert kinds == {"village": 3, "farm": 7, "industry": 20}, kinds
    print(f"   seed {L['seed']}, {len(L['pads'])} pads "
          f"({kinds['village']}/{kinds['farm']}/{kinds['industry']}), "
          f"{len(L['bridges'])} bridge(s), crossing at "
          f"{L['crossing'][0]:.0f}, {L['crossing'][1]:.0f}")

    print("1. River and lake...")
    n_water, lake_ha = stage_water(add_way, L)
    print(f"   {n_water} way(s), lake {lake_ha:.1f} ha")

    occ = pc.Occupancy()
    occ.fill_border(pc.EDGE_MARGIN_M)
    occ.fill_ring(L["lake"]["ring"], pc.LAKE_BANK_M + pc.SIMPLIFY_SLACK_M)
    occ.fill_polyline(L["river"]["centre"],
                      max(L["river"]["half_w"]) + pc.RIVER_BANK_M + pc.SIMPLIFY_SLACK_M)
    occ.fill_polyline(L["road"]["centre"], pc.ROAD_CLEAR_M + pc.SIMPLIFY_SLACK_M)
    occ.fill_polyline(L["rail"]["centre"], pc.RAIL_CLEAR_M + pc.SIMPLIFY_SLACK_M)
    for p in L["pads"]:
        occ.fill_ring(p["ring"], pc.PAD_CLEAR_M + pc.SIMPLIFY_SLACK_M)

    print("2. Structural woodland...")
    woods, thr = pc.structural_woods(L, occ, rng)
    for ring, ha in woods:
        occ.fill_ring(ring, pc.WOOD_MARGIN_M + pc.SIMPLIFY_SLACK_M)
    print(f"   {len(woods)} block(s), {sum(w[1] for w in woods):.0f} ha "
          f"(slope/grove threshold {thr:.3f}, derived)")

    print("3. Villages, farms and industry pads...")
    counts = stage_pads(add_way, L)
    for kind in ("village", "farm", "industry"):
        sel = [p for p in L["pads"] if p["kind"] == kind]
        print(f"   {kind:<9} {len(sel):2d}, {sum(p['ha'] for p in sel):6.1f} ha "
              f"({min(p['ha'] for p in sel):.1f} .. {max(p['ha'] for p in sel):.1f})")

    print("4. Main road and railway...")
    road_ways = emit_line(add_way, L["road"],
                          [b for b in L["bridges"] if b["on"] == "road"],
                          {'highway': 'primary'}, "Main Road",
                          keep=[L["crossing"]])
    rail_ways = emit_line(add_way, L["rail"],
                          [b for b in L["bridges"] if b["on"] == "rail"],
                          {'railway': 'rail', 'usage': 'main', 'gauge': '1520'},
                          "Railway", keep=[L["crossing"]])
    cross_id = get_node(*L["crossing"])
    node_tags[cross_id] = {'railway': 'level_crossing'}
    km = lambda ws: sum(ml.polyline_length(w['coords']) for w in ws) / 1000.0
    print(f"   primary {km(road_ways):.1f} km in {len(road_ways)} way(s), "
          f"railway {km(rail_ways):.1f} km in {len(rail_ways)} way(s), "
          f"level crossing node {cross_id}")

    print("5. Link roads...")
    streets = village_streets(add_way, L)
    router = rd.Router(L, woods)
    for w in road_ways:
        # Not the bridge decks: a spur joining one mid-span has nothing to stand on.
        if w['tags'].get('bridge') != 'yes':
            router.add_source_way(w)
    links, orphans = router.link_pads(
        add_way, get_node,
        [p for p in L["pads"] if p["kind"] in ("farm", "industry")],
        lambda p: {'highway': 'secondary',
                   'name': (f"{p['name']} Road" if p["kind"] == "farm"
                            else f"{p['name']} Access")})
    for w in streets + [w for _, w in links]:
        occ.fill_polyline(w['coords'], pc.SECONDARY_CLEAR_M + pc.SIMPLIFY_SLACK_M)
    link_km = sum(ml.polyline_length(w['coords']) for _, w in links) / 1000.0
    print(f"   {len(streets)} village street(s), {len(links)} yard link(s), "
          f"{link_km:.1f} km")
    for p in orphans:
        print(f"   !  {p['name']} could not be routed to the main road")

    print("6. Big plateau fields...")
    big = pc.place_big_fields(occ, rng, L["river"]["centre"])
    print(f"   {len(big)} field(s), {min(b[1] for b in big):.1f}-"
          f"{max(b[1] for b in big):.1f} ha, {sum(b[1] for b in big):.0f} ha")

    print("7. Field strips...")
    villages = [p for p in L["pads"] if p["kind"] == "village"]
    strip_fields, gaps = pc.cut_fields(occ, rng, villages)
    all_fields = [(r, h) for r, h in big] + [(r, h) for r, h, _ in strip_fields]
    all_fields.sort(key=lambda f: (round(mg.centroid(f[0])[1] / 250.0),
                                   mg.centroid(f[0])[0]))
    for i, (ring, ha) in enumerate(all_fields, 1):
        add_way(ring, {'landuse': 'farmland', 'name': f"Field {i} ({ha:.1f} ha)"})
    areas = sorted(h for _, h in all_fields)
    played = mx.PLAYABLE_M ** 2 / 1e4
    print(f"   {len(all_fields)} field(s): {areas[0]:.1f} / "
          f"{areas[len(areas)//2]:.1f} / {areas[-1]:.1f} ha (min/median/max), "
          f"{sum(areas):.0f} ha farmed ({100*sum(areas)/played:.1f} %)")

    print("8. Farm tracks along the headlands...")
    network = road_ways + streets + [w for _, w in links]
    n_tracks, track_km = farm_tracks(add_way, get_node, rng, gaps, occ, network,
                                     L["river"]["centre"])
    print(f"   {n_tracks} track(s), {track_km/1000.0:.1f} km")

    print("9. Shelterbelts...")
    belts = pc.shelterbelts(gaps, occ, rng, L["rail"]["centre"])
    stage_woods(add_way, woods, belts)
    wood_ha = sum(w[1] for w in woods) + sum(b[1] for b in belts)
    print(f"   {len(belts)} belt(s), {sum(b[1] for b in belts):.0f} ha; "
          f"woodland total {wood_ha:.0f} ha ({100*wood_ha/played:.1f} %)")

    print("10. Splicing shared nodes at crossings...")
    n_spliced = connect_crossings(ways, get_node)
    print(f"   {n_spliced} node(s) spliced")

    print("11. Writing map.osm...")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)
    write_osm(out, ways, node_coords, node_tags)
    ok = verify(out, L, cross_id)
    print(f"[+] Wrote the Ukraine layout to '{out}'  [{time.time()-t0:.1f} s]")
    return 0 if ok else 1


def verify(path, L, cross_id):
    """Read the file back and measure it, rather than trusting the numbers just written.

    The projection round-trip is the check the blank generator already had: a sign slip
    is invisible in the raw degrees and obvious here. The rest are the promises this
    map makes to whoever opens it in the editor.
    """
    root = ET.parse(path).getroot()
    b = root.find('bounds').attrib
    sw = mx.global_to_local(float(b['minlat']), float(b['minlon']))
    ne = mx.global_to_local(float(b['maxlat']), float(b['maxlon']))
    nodes = {int(n.get('id')): mx.global_to_local(float(n.get('lat')),
                                                  float(n.get('lon')))
             for n in root.findall('node')}
    print(f"   extent {abs(ne[0]-sw[0]):.3f} x {abs(sw[1]-ne[1]):.3f} m, "
          f"{len(nodes)} nodes, {len(root.findall('way'))} ways")

    fails = []
    counts, cross_uses = {}, set()
    used, read = {}, []
    for w in root.findall('way'):
        tags = {t.get('k'): t.get('v') for t in w.findall('tag')}
        refs = [int(nd.get('ref')) for nd in w.findall('nd')]
        coords = [nodes[r] for r in refs]
        read.append({'tags': tags, 'refs': refs, 'coords': coords})
        for r in refs:
            used[r] = used.get(r, 0) + 1
        if cross_id in refs:
            cross_uses.add(tags.get('highway') or tags.get('railway'))
        if len(refs) < 2:
            fails.append(f"way {w.get('id')} has {len(refs)} node(s)")
        name = tags.get('name', '')
        if tags.get('natural') == 'wood':
            key = 'wood'
        elif tags.get('natural') == 'water':
            key = 'water'
        elif tags.get('landuse') in ('farmland', 'farmyard'):
            key = tags['landuse']
        elif 'railway' in tags:
            key = 'railway'
        elif 'highway' in tags:
            key = 'highway'
        else:
            key = 'other'
        counts[key] = counts.get(key, 0) + 1
        if key in ('farmland', 'farmyard', 'wood', 'water'):
            if refs[0] != refs[-1]:
                fails.append(f"{name or key} way {w.get('id')} is not closed")
            ha = mx.ring_area_ha(coords)
            if key == 'farmland' and not (pc.MIN_FIELD_HA - 0.5 <= ha
                                          <= pc.MAX_FIELD_HA + 0.5):
                fails.append(f"{name} is {ha:.1f} ha")
            if name.startswith('Industry Pad'):
                if ha > ml.INDUSTRY_MAX_HA:
                    fails.append(f"{name} is {ha:.2f} ha, over the 5 ha limit")
                xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
                w_m, h_m = max(xs) - min(xs), max(ys) - min(ys)
                if abs(w_m - h_m) > 0.5:
                    fails.append(f"{name} is not square ({w_m:.1f} x {h_m:.1f} m)")

    # A platform is punched out of the woodland mask, but the morphology that follows
    # can close back over it, so the promise is checked on the file: no wood polygon may
    # cover a yard.
    woods = [w for w in read if w['tags'].get('natural') == 'wood']
    for p in L['pads']:
        px0, py0, px1, py1 = mg.ring_bbox(p['ring'])
        for w in woods:
            wx0, wy0, wx1, wy1 = mg.ring_bbox(w['coords'])
            if wx1 < px0 or px1 < wx0 or wy1 < py0 or py1 < wy0:
                continue
            if any(mg.point_in_ring(c, w['coords']) for c in p['ring']):
                fails.append(f"{w['tags'].get('name', 'a wood')} covers {p['name']}")
                break

    network_checks(read, L, fails)
    if cross_uses != {'primary', 'rail'}:
        fails.append(f"the level crossing node is used by {sorted(cross_uses)}, "
                     "not by the primary road and the railway")
    shared = sum(1 for v in used.values() if v > 1)
    if shared < 100:
        fails.append(f"only {shared} shared node(s): the network is in pieces")

    print("   " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"   shared nodes (junctions) {shared}")
    for f in fails[:8]:
        print(f"   !  {f}")
    if fails:
        print(f"   !  {len(fails)} check(s) failed")
    return not fails


def network_checks(read, L, fails):
    """Reachability and trespass, measured on the file rather than on the intention.

    The union-find runs on node ids, so it sees the joins that are actually in the XML:
    a spur whose endpoint merely lies on a road segment is a separate component here,
    which is precisely the failure it is meant to catch.
    """
    hw = [w for w in read if 'highway' in w['tags']]
    par = {}

    def find(a):
        par.setdefault(a, a)
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for w in hw:
        for r in w['refs'][1:]:
            ra, rb = find(w['refs'][0]), find(r)
            if ra != rb:
                par[ra] = rb

    main = {find(w['refs'][0]) for w in hw if w['tags'].get('highway') == 'primary'}
    if len(main) != 1:
        fails.append(f"the main road is in {len(main)} piece(s) rather than one")
    stranded = [w for w in hw if find(w['refs'][0]) not in main]
    if stranded:
        names = sorted({w['tags'].get('name', '?') for w in stranded})
        fails.append(f"{len(stranded)} way(s) never reach the main road: "
                     + ", ".join(names[:4]))

    for p in L['pads']:
        if p['kind'] == 'village':
            continue
        collar = mg.grow_ring(p['ring'], 3.0)
        if not any(find(w['refs'][0]) in main
                   and any(mg.point_in_ring(c, collar) for c in w['coords'])
                   for w in hw):
            fails.append(f"{p['name']} has no road connected to the main road")

    # A yard is a place a road arrives at, not one it drives through. The village pads
    # are the deliberate exception: the main road runs the length of them, and their own
    # streets are inside them by definition.
    for p in L['pads']:
        inner = mg.grow_ring(p['ring'], -2.0)
        x0, y0, x1, y1 = mg.ring_bbox(p['ring'])
        for w in hw:
            name = w['tags'].get('name', '')
            if p['kind'] == 'village' and (name == 'Main Road'
                                           or name.startswith('Village')):
                continue
            for a, b in zip(w['coords'], w['coords'][1:]):
                if max(a[0], b[0]) < x0 or min(a[0], b[0]) > x1:
                    continue
                if max(a[1], b[1]) < y0 or min(a[1], b[1]) > y1:
                    continue
                if mg.point_in_ring(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0), inner):
                    fails.append(f"{name or 'a road'} drives through {p['name']}")
                    break


if __name__ == '__main__':
    sys.exit(main())
