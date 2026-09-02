# osm_generator

`generate_osm.py` writes `map.osm` for the 8192 x 8192 m playable area.

- `map_extent.py`         centre, size and projection - the one source of truth
- `generate_osm.py`       write map.osm: the extent, and nothing else
- `visualize_osm.py`      render map.osm to map_osm_visual.png
- `check_forest_nodes.py` feature inventory (counts, areas, road network)

Standard library plus matplotlib for the render; no numpy, scipy or Pillow, and no
dependency on `map_source.py` (shared with the DEM generator, currently missing from
the tree).

    python3 generate_osm.py        # -> map.osm, 0 nodes, 0 ways
    python3 check_forest_nodes.py  # -> every feature class reports "none"
    python3 visualize_osm.py       # -> map_osm_visual.png, the empty frame

## Map centre

    LAT_CENTER = 49.1000
    LON_CENTER = 31.3000

These live in `map_extent.py`, and everything else on the map is derived from them: to
move the map, change those two numbers and re-run `generate_osm.py`. If `map_source.py`
is ever restored, its LAT_CENTER / LON_CENTER must be set to match, or the DEM and the
vectors will be built for different places.

### Why this centre

Cherkasy oblast forest-steppe: deep chernozem, the highest-yielding arable belt in
Ukraine. Round coordinates, and the 8 x 8 km square lands entirely on farmland - clear
of the city and well west of the Dnipro reservoir, which sits at roughly 32.3 E at this
latitude.

## Extent

The playable area is 8192 x 8192 m. Local coordinates are playable metres, x east,
y south from the north edge, so the centre of the map sits at (4096, 4096).

Projection: equirectangular about the centre, 111111.0 m per degree of latitude and
111111.0 * cos(LAT_CENTER) m per degree of longitude.

    lat = LAT_CENTER - (y - 4096) / 111111.0
    lon = LON_CENTER + (x - 4096) / (111111.0 * cos(radians(LAT_CENTER)))

Which puts the corners of the playable area at:

    minlat  49.0631359631      south edge, y = 8192
    maxlat  49.1368640369      north edge, y = 0
    minlon  31.2436967483      west edge,  x = 0
    maxlon  31.3563032517      east edge,  x = 8192

These are the four values in the `<bounds>` element of `map.osm`.

## State

`map.osm` carries the extent only - no nodes, no ways. Every feature the English layout
had (fields, woods, farmyards, roads, river and lake) is gone, and `generate_osm.py`
now writes that empty file rather than rebuilding them, so the result stays clean.

The tag vocabulary is unchanged and both reader scripts still understand it, so
features can be laid back on top without touching either:

    landuse=farmland                              fields
    landuse=farmyard                              village, industry pads, yards
    natural=wood + landuse=farmyard + leaf_type   woodland
    natural=water                                 the river
    highway=primary / secondary / tertiary        road hierarchy

Two files still hold the old English layout centred on 52.0620, -1.3400:

- `generate_osm_bocage.py` - the previous generator, kept because it is the only copy
  and nothing here is in git history. Not part of the build: it needs the missing
  `map_source.py`, and it writes to `map.osm`, so running it would overwrite the clean
  file.
- `custom_osm.osm` - a JOSM re-save of an earlier version of the map.
