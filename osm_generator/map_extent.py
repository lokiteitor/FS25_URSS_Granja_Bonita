#!/usr/bin/env python3
"""Centre, size and projection of the playable area - the one source of truth.

The map is a clean 8192 x 8192 m square centred on the Cherkasy oblast forest-steppe,
the deep-chernozem belt that is the highest-yielding arable land in Ukraine. Nothing is
laid out on it: `generate_osm.py` writes the extent and nothing else.

Local coordinates are playable metres, x east, y south from the north edge, so the
centre of the map sits at (PLAYABLE_M / 2, PLAYABLE_M / 2).

This module deliberately has no dependencies beyond the standard library, so the three
scripts in this folder run without numpy, scipy or Pillow, and without `map_source.py`
(shared with the DEM generator, and currently missing from the tree). If map_source is
ever restored, its LAT_CENTER / LON_CENTER must be set to the values below, or the two
halves of the pipeline will build terrain and vectors for different places.
"""
import math

# --- where the map is -------------------------------------------------------------
# Cherkasy oblast forest-steppe. Round coordinates, and the 8 x 8 km square lands
# entirely on farmland: clear of the city and well west of the Dnipro reservoir, which
# sits at roughly 32.3 E at this latitude.
LAT_CENTER = 49.1000
LON_CENTER = 31.3000

# --- how big it is ----------------------------------------------------------------
PLAYABLE_M = 8192.0
HALF_M = PLAYABLE_M / 2.0

# --- how the two relate -----------------------------------------------------------
# Equirectangular about the centre. 111111.0 m per degree is the constant the rest of
# the pipeline was built with (1e7 m from the equator to the pole, over 90 degrees);
# keeping it means a local metre here is the same metre the DEM uses.
M_PER_DEG = 111111.0
M_PER_DEG_LON = M_PER_DEG * math.cos(math.radians(LAT_CENTER))


def local_to_global(x, y):
    """Playable metres -> (lat, lon). y grows southwards, so it subtracts."""
    return (LAT_CENTER - (y - HALF_M) / M_PER_DEG,
            LON_CENTER + (x - HALF_M) / M_PER_DEG_LON)


def global_to_local(lat, lon):
    """(lat, lon) -> playable metres. The inverse of local_to_global."""
    return (HALF_M + (lon - LON_CENTER) * M_PER_DEG_LON,
            HALF_M - (lat - LAT_CENTER) * M_PER_DEG)


def bounds():
    """The four values of the OSM `<bounds>` element, as (minlat, minlon, maxlat,
    maxlon). The south-west corner is local (0, PLAYABLE_M); the north-east is
    (PLAYABLE_M, 0)."""
    minlat, minlon = local_to_global(0.0, PLAYABLE_M)
    maxlat, maxlon = local_to_global(PLAYABLE_M, 0.0)
    return minlat, minlon, maxlat, maxlon


def polyline_length(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def ring_area_ha(ring):
    """Shoelace area in hectares. The ring may be given open or closed."""
    pts = ring[:-1] if len(ring) > 2 and math.dist(ring[0], ring[-1]) < 1e-9 else ring
    if len(pts) < 3:
        return 0.0
    twice = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] -
                pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    return abs(twice) / 2.0 / 10000.0
