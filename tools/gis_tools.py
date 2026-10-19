
import json

from langchain_core.tools import tool

from tools.geojson_query import (
    query_biomass_loss,
    query_stable_forest,
)


@tool
def get_biomass_loss() -> str:
    """
    Find all forest stands with negative biomass_change.

    Use this tool when the user asks which stands lost
    biomass, experienced AGB decrease, or had negative
    biomass change.

    Returns stand IDs, biomass change values and
    dominant land-cover transition information.
    """
    result = query_biomass_loss()

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    )


@tool
def get_stable_forest(min_proportion: float = 0.8) -> str:
    """
    Find stands where the dominant transition group is
    Stable_forest and its proportion is at least the
    specified threshold.

    min_proportion must be between 0 and 1.
    For example, 80 percent corresponds to 0.8.
    """
    result = query_stable_forest(
        min_proportion=min_proportion
    )

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    )
