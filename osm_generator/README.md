# osm_generator

`generate_osm.py` writes `map.osm` for the 8192 x 8192 m playable area.

- `generate_osm.py`       the pipeline and the writer
- `parcels.py`            the field cutter and the wood shaper
- `roads.py`              the router that links every yard to the main road
- `map_extent.py`         centre, size and projection - the one source of truth for *where*
- `visualize_osm.py`      render map.osm to map_osm_visual.png
- `check_forest_nodes.py` feature inventory (counts, areas, road and rail network)

```
python3 generate_osm.py        # -> map.osm, ~2300 nodes, ~335 ways
python3 check_forest_nodes.py
python3 visualize_osm.py       # -> map_osm_visual.png
```

Everything structural - river, lake, road, railway, the thirty platforms - is read from
`../map_layout.py`, which the DEM generator reads too. This half adds the countryside on
top: woodland, fields, link roads, farm tracks and shelterbelts.

`generate_osm.py`, `parcels.py` and `roads.py` use numpy, scipy and Pillow. The two reader scripts
and `map_extent.py` are standard library plus matplotlib for the render.

## Map centre

    LAT_CENTER = 49.1000
    LON_CENTER = 31.3000

Cherkasy oblast forest-steppe: deep chernozem, the highest-yielding arable belt in
Ukraine. Round coordinates, and the 8 x 8 km square lands entirely on farmland - clear of
the city and well west of the Dnipro reservoir, which sits at roughly 32.3 E at this
latitude. To move the map, change those two numbers and re-run both generators.

## Extent

Local coordinates are playable metres, x east, y south from the north edge, so the centre
of the map sits at (4096, 4096). Projection: equirectangular about the centre, 111111.0 m
per degree of latitude and 111111.0 * cos(LAT_CENTER) m per degree of longitude.

    lat = LAT_CENTER - (y - 4096) / 111111.0
    lon = LON_CENTER + (x - 4096) / (111111.0 * cos(radians(LAT_CENTER)))

Which puts the corners of the playable area at:

    minlat  49.0631359631      south edge, y = 8192
    maxlat  49.1368640369      north edge, y = 0
    minlon  31.2436967483      west edge,  x = 0
    maxlon  31.3563032517      east edge,  x = 8192

These are the four values in the `<bounds>` element of `map.osm`.

## Tag vocabulary

    landuse=farmland                                 fields
    landuse=farmyard                                 villages, farms, industry pads
    natural=wood + landuse=farmyard + leaf_type      woodland and shelterbelts
    natural=water + waterway=riverbank | water=lake  the river and the lake
    highway=primary / secondary / tertiary           road hierarchy
    railway=rail                                     the line
    bridge=yes + layer=1                             river crossings
    railway=level_crossing                           on the node where road meets rail

Two rules everything else depends on:

* **`get_node` keys on millimetre-rounded coordinates, and that is the junction
  mechanism.** Two ways carrying the same coordinate share a node; two ways that merely
  cross at a coordinate neither carries are not joined at all. `connect_crossings` is the
  safety net that splices the rest.
* **Smooth first, weave second, emit third.** Aiming a spur at an unsmoothed centreline
  and emitting the smoothed one leaves the two a metre apart - close enough to look
  joined, far enough not to be. The same trap is why the level crossing is woven back in
  after the Douglas-Peucker pass, which would otherwise drop it from both lines.
* **Every yard reaches the main road, and no road drives through a yard.** `roads.py`
  floods a cost grid outwards from the road network - platforms, river and lake
  impassable, railway merely expensive - and attaches the farms and the industry
  platforms one at a time, cheapest first, each accepted link becoming free ground for
  the next. So the links grow as one tree rooted on the main road, and reachability is a
  property of the construction rather than something to hope for. `verify` re-reads the
  written file and checks both promises with a union-find on the node ids: a link that
  merely ends *on* a road segment without carrying its coordinate shows up here as a
  separate component.

## Reference

`generate_osm_bocage.py` is the previous English bocage generator, centred on
52.0620, -1.3400. It is not part of the build - it needs the long-gone `map_source.py` -
and it is kept because several of its routines were ported from it: the node and way
pools, `write_osm`, `connect_road_crossings`, `weave`, `point_in_ring` and the raster
morphology that regularises a wood outline. Its `connect_pads` - aim each yard at the
nearest point of whatever network exists - is the one routine deliberately *not* ported:
on this layout it left twenty-one of the twenty-seven yards on spur trees that never
reached the road.
