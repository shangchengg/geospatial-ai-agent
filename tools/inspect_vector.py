
from pathlib import Path
from pprint import pprint

import geopandas as gpd
import pyogrio


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "data" / "vector"


def inspect_layer(path, layer=None):
    print("\n" + "=" * 60)
    print("File:", path.name)
    print("Layer:", layer or "Default")

    gdf = gpd.read_file(path, layer=layer)

    print("Feature count:", len(gdf))
    print("CRS:", gdf.crs)
    print("Geometry types:", gdf.geom_type.unique().tolist())
    print("Columns:", gdf.columns.tolist())

    print("\nSample attributes:")
    pprint(
        gdf.drop(columns="geometry", errors="ignore")
        .head(3)
        .to_dict(orient="records")
    )


def main():
    geojson_path = VECTOR_DIR / "biomass.geojson"
    gpkg_path = VECTOR_DIR / "transition_biomass.gpkg"

    if geojson_path.exists():
        inspect_layer(geojson_path)
    else:
        print("Missing:", geojson_path)

    if gpkg_path.exists():
        layers = pyogrio.list_layers(gpkg_path)

        print("\nGeoPackage Layers:")
        pprint(layers.tolist())

        for layer_name, geometry_type in layers:
            inspect_layer(gpkg_path, layer=layer_name)
    else:
        print("Missing:", gpkg_path)


if __name__ == "__main__":
    main()
