
from pathlib import Path
import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parent.parent

BIOMASS_PATH = BASE_DIR / "data/vector/biomass.geojson"
CESIUM_PATH = BASE_DIR / "data/test/forest_stands.geojson"


def inspect_file(path):
    gdf = gpd.read_file(path)

    print(f"\n===== {path.name} =====")
    print("Feature count:", len(gdf))
    print("CRS:", gdf.crs)
    print("Geometry types:", gdf.geom_type.unique().tolist())
    print("Columns:", gdf.columns.tolist())

    print("\nSample attributes:")
    print(
        gdf.drop(columns="geometry", errors="ignore")
        .head(3)
        .to_string(index=False)
    )

    return gdf


def compare_files():
    biomass = inspect_file(BIOMASS_PATH)
    cesium = inspect_file(CESIUM_PATH)

    print("\n===== COMPARISON =====")

    print(
        "Same feature count:",
        len(biomass) == len(cesium)
    )

    print(
        "Same CRS:",
        biomass.crs == cesium.crs
    )

    biomass_columns = set(biomass.columns)
    cesium_columns = set(cesium.columns)

    print(
        "Common columns:",
        sorted(biomass_columns & cesium_columns)
    )

    print(
        "Only in biomass:",
        sorted(biomass_columns - cesium_columns)
    )

    print(
        "Only in Cesium:",
        sorted(cesium_columns - biomass_columns)
    )

    if "POLYGON_ID" in biomass.columns and "POLYGON_ID" in cesium.columns:
        biomass_ids = set(
            biomass["POLYGON_ID"].dropna().astype(str)
        )

        cesium_ids = set(
            cesium["POLYGON_ID"].dropna().astype(str)
        )

        print("Same POLYGON_ID sets:", biomass_ids == cesium_ids)
        print("Missing in Cesium:", sorted(biomass_ids - cesium_ids))
        print("Extra in Cesium:", sorted(cesium_ids - biomass_ids))

        print(
            "Duplicate IDs in biomass:",
            biomass["POLYGON_ID"].duplicated().any()
        )
        print(
            "Duplicate IDs in Cesium:",
            cesium["POLYGON_ID"].duplicated().any()
        )
        
    # Compare FEATURE_ID between datasets
    print("\n===== FEATURE_ID CHECK =====")

    biomass_ids = set(
        biomass["FEATURE_ID"].dropna().astype(int)
    )

    cesium_ids = set(
        cesium["FEATURE_ID"].dropna().astype(int)
    )

    print("Same FEATURE_ID sets:", biomass_ids == cesium_ids)

    print(
        "Missing in Cesium:",
        sorted(biomass_ids - cesium_ids)
    )

    print(
        "Extra in Cesium:",
        sorted(cesium_ids - biomass_ids)
    )

    print(
        "Biomass duplicate FEATURE_ID:",
        biomass["FEATURE_ID"].duplicated().any()
    )

    print(
        "Cesium duplicate FEATURE_ID:",
        cesium["FEATURE_ID"].duplicated().any()
    )

    print("\n===== ID MAPPING PREVIEW =====")

    mapping = biomass[
        ["FEATURE_ID", "POLYGON_ID"]
    ].merge(
        cesium[["FEATURE_ID", "stand_id"]],
        on="FEATURE_ID",
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    print(mapping.to_string(index=False))



if __name__ == "__main__":
    compare_files()

