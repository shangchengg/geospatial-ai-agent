
from pathlib import Path
import json

import geopandas as gpd
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "data" / "vector"

BIOMASS_PATH = VECTOR_DIR / "biomass.geojson"
TRANSITION_PATH = VECTOR_DIR / "transition_biomass.gpkg"


def load_stands():
    """Load biomass and join land-cover transition attributes."""

    biomass = gpd.read_file(BIOMASS_PATH)

    transition = gpd.read_file(
        TRANSITION_PATH,
        layer="stands_transition_group"
    )

    # Select only required attributes
    transition = transition[
        ["POLYGON_ID", "dom_group", "dom_prop", "dom_code"]
    ].copy()

    # Validate identifiers before joining
    for df in (biomass, transition):
        if df["POLYGON_ID"].isna().any():
            raise ValueError("Missing POLYGON_ID values")

        if df["POLYGON_ID"].duplicated().any():
            raise ValueError("Duplicate POLYGON_ID values")

    # Left join using the stand identifier
    merged = biomass.merge(
        transition,
        on="POLYGON_ID",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    # Do not silently accept unmatched stands
    if (merged["_merge"] != "both").any():
        raise ValueError(
            "Some biomass stands have no transition match"
        )

    merged = merged.drop(columns="_merge")

    return merged


def query_biomass_loss():
    """Find stands with negative biomass change."""

    gdf = load_stands()

    selected = gdf[gdf["biomass_change"] < 0].copy()

    selected = selected.sort_values("biomass_change")

    columns = [
        "POLYGON_ID",
        "FEATURE_ID",
        "WHOLE_STEM_BIOMASS_PER_HA",
        "biomass_change",
        "dom_group",
        "dom_prop"
    ]

    records = selected[columns].to_dict(orient="records")

    return {
        "query_type": "biomass_loss",
        "count": len(records),
        "stands": records
    }


def query_stable_forest(min_proportion=0.8):
    """Find stands where stable forest proportion exceeds a threshold."""

    if not 0 <= min_proportion <= 1:
        raise ValueError("min_proportion must be between 0 and 1")

    gdf = load_stands()

    selected = gdf[
        (gdf["dom_group"] == "Stable_forest")
        & (gdf["dom_prop"] >= min_proportion)
    ].copy()

    selected = selected.sort_values(
        "dom_prop",
        ascending=False
    )

    columns = [
        "POLYGON_ID",
        "FEATURE_ID",
        "WHOLE_STEM_BIOMASS_PER_HA",
        "biomass_change",
        "dom_group",
        "dom_prop"
    ]

    records = selected[columns].to_dict(orient="records")

    return {
        "query_type": "stable_forest",
        "threshold": min_proportion,
        "count": len(records),
        "stands": records
    }


if __name__ == "__main__":

    print("\n===== BIOMASS LOSS =====")
    print(
        json.dumps(
            query_biomass_loss(),
            indent=2,
            ensure_ascii=False
        )
    )

    print("\n===== STABLE FOREST =====")
    print(
        json.dumps(
            query_stable_forest(0.8),
            indent=2,
            ensure_ascii=False
        )
    )
