# FS25_URSS_Granja_Bonita

Asset pipeline for a Farming Simulator 25 map: a Ukrainian forest-steppe landscape in
Cherkasy oblast (49.1000 N, 31.3000 E), on the deep-chernozem belt. Nothing here is game
code - the four generators produce files that are imported into Giants Editor by hand.

```
map_geom.py        geometry toolbox (stdlib): polylines, rings, segments
map_layout.py      THE SKELETON - sizes, datum, seed, river, lake, road, railway, pads
osm_generator/     vectors    -> map.osm, map_osm_visual.png
dem_generator/     heightmap  -> dem_new_12k.png (+ three visualisations)
pf_generator/      Precision Farming soil -> soilMap.png
visualizer/        Three.js preview -> dem_viewer_3d.html
```

## The one thing to know

`map_layout.py` is the single source of truth for **where everything is**. The DEM
generator and the OSM generator both read `map_layout.layout()`; neither re-derives the
other's geometry. That is what keeps the heightmap and the vectors describing the same
place - the river in `map.osm` falls in the valley in `dem_new_12k.png` because both came
out of the same list of points.

It is standard library only and takes no arguments: one `SEED = 2026` drives every
random choice, so the map is reproducible. `python3 map_layout.py` prints the layout and
re-checks its own constraints without writing anything.

`map_layout` also owns the vertical design. `regional_z(x, y, cos=..., exp=...)` takes
its maths functions as arguments, so the OSM half calls it point by point with `math` and
the DEM half calls it with `numpy` over the whole 12288 x 12288 canvas - one definition of
the landform, two evaluators.

## The map

| | |
|---|---|
| Playable area | 8192 x 8192 m, on a 12288 m canvas (2048 m of margin per side) |
| Relief | 87-114 m over the playable area (datum 100 m); field slopes 1-3 % |
| Water | the Bystra, ~11 km west to east, falling 90.5 -> 84.5 m, with a 43 ha lake |
| Main road | NW to SE corner to corner, gently curving, max grade 6 % |
| Railway | NE to SW, perpendicular, max grade 1.5 %, crossing the road at the main village |
| Villages | 3, strung along the main road: Verkhivka, Bereh (the crossing), Nyzhne |
| Farms | 7: cooperativa, granos, vacas, cerdos, ovejas, invernaderos, pollos |
| Industry | 20 square platforms of ~5 ha, eight of them sidings on the railway |
| Fields | ~164, east-west aligned, median 18 ha, five of them near 100 ha; 54 % of the area |
| Woodland | ~13 %: gallery forest on the river, valley-side blocks, shelterbelts on the headlands |

## Running it

```bash
python3 map_layout.py                       # layout report, writes nothing

cd osm_generator
python3 generate_osm.py                     # -> map.osm            (~2 s)
python3 check_forest_nodes.py               # feature inventory
python3 visualize_osm.py                    # -> map_osm_visual.png

cd ../dem_generator
python3 generate_new_dem_12k.py             # -> dem_new_12k.png    (~70 s)
python3 measure_elevation.py                # conformance report, exit 1 on failure

cd ../visualizer
python3 create_3d_viewer.py                 # -> dem_viewer_3d.html
```

Both generators verify their own output and both checkers exit non-zero on a failure, so
the whole thing can be run as a gate.

Requires `numpy`, `scipy`, `Pillow` and `matplotlib`. `map_geom.py` and `map_layout.py`
have no dependencies at all.
