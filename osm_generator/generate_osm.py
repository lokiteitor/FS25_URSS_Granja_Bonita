#!/usr/bin/env python3
"""Write `map.osm`: the extent of the playable area, and nothing else.

The map is a blank canvas - no fields, woods, farmyards, roads or water. What goes in
the file is the technical description of the ground and nothing more: an 8192 x 8192 m
square centred on the Cherkasy oblast forest-steppe (see `map_extent.py`), which is
what the rest of the pipeline and the Giants Editor import need in order to agree on
where the map is and how big it is.

The tag vocabulary the map used to carry is unchanged and still understood by
`visualize_osm.py` and `check_forest_nodes.py`, so features can be laid back on top
without touching either:
    landuse=farmland                              fields
    landuse=farmyard                              village, industry pads, yards
    natural=wood + landuse=farmyard + leaf_type   woodland
    natural=water                                 the river
    highway=primary / secondary / tertiary        road hierarchy

The English bocage layout that used to be generated here is kept, unused, in
`generate_osm_bocage.py`.
"""
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

import map_extent as mx

OUT_NAME = "map.osm"


def build_osm():
    minlat, minlon, maxlat, maxlon = mx.bounds()

    osm = ET.Element('osm', version='0.6', generator='FS25 map pipeline')
    osm.append(ET.Comment(
        f"\n       Playable area: {mx.PLAYABLE_M:.0f} x {mx.PLAYABLE_M:.0f} m, "
        f"centre {mx.LAT_CENTER:.4f}, {mx.LON_CENTER:.4f}\n"
        "       (Cherkasy oblast forest-steppe - deep chernozem, the highest-yielding\n"
        "       arable belt in Ukraine).\n"
        "       Local coordinates are playable metres, x east, y south from the north\n"
        f"       edge, so the centre of the map is ({mx.HALF_M:.0f}, {mx.HALF_M:.0f}).\n"
        f"       Projection: equirectangular about the centre, {mx.M_PER_DEG:.1f} m per\n"
        f"       degree of latitude and {mx.M_PER_DEG:.1f} * cos(LAT_CENTER) m per\n"
        "       degree of longitude.\n"
        "       No features: this file carries the extent only.\n  "))
    ET.SubElement(osm, 'bounds', {
        'minlat': f"{minlat:.10f}", 'minlon': f"{minlon:.10f}",
        'maxlat': f"{maxlat:.10f}", 'maxlon': f"{maxlon:.10f}"})
    return osm


def main():
    print("=== Generating OSM data for the Ukraine map ===")
    print(f"   centre {mx.LAT_CENTER:.4f}, {mx.LON_CENTER:.4f} - Cherkasy oblast "
          "forest-steppe")

    osm = build_osm()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)
    pretty = minidom.parseString(ET.tostring(osm, encoding='utf-8')).toprettyxml(
        indent='  ', encoding='utf-8')
    with open(out, "wb") as fh:
        fh.write(pretty)

    # Read the file back and measure it, rather than trusting the numbers just written:
    # a sign slip in the projection is invisible in the raw degrees and obvious here.
    root = ET.parse(out).getroot()
    b = root.find('bounds').attrib
    sw = mx.global_to_local(float(b['minlat']), float(b['minlon']))
    ne = mx.global_to_local(float(b['maxlat']), float(b['maxlon']))
    print(f"   extent {abs(ne[0] - sw[0]):.3f} x {abs(sw[1] - ne[1]):.3f} m, "
          f"{len(root.findall('node'))} nodes, {len(root.findall('way'))} ways")
    print(f"[+] Wrote the clean extent to '{out}'.")


if __name__ == '__main__':
    main()
