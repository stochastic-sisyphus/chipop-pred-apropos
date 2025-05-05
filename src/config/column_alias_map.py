# src/config/column_alias_map.py

from typing import Dict, List, Optional, TypedDict, Union
from enum import Enum


# Type Definitions
class ColumnType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    BOOLEAN = "boolean"
    COORDINATE = "coordinate"


class ColumnMetadata(TypedDict, total=False):
    type: ColumnType
    description: str
    format: Optional[str]
    range: Optional[List[Union[int, float]]]
    example: Union[str, int, float]
    validation: Optional[str]


ColumnAliasMap = Dict[str, List[str]]
ColumnCategoryMap = Dict[str, ColumnAliasMap]
ColumnMetadataMap = Dict[str, ColumnMetadata]

# Column Categories and Aliases
column_aliases_by_category: ColumnCategoryMap = {
    "Identifiers": {
        "zip_code": ["zip", "zipcode", "zip code tabulation area", "zcta"],
        "year": ["fiscal_year", "reporting_year", "yr"],
    },
    "Modeling": {
        "total_population": ["population", "pop_total", "pop"],
        "median_household_income": ["hh_income", "med_house_inc", "median_income"],
        "median_home_value": ["home_value", "house_value", "median_property_value"],
        "labor_force": ["labor_force_population"],
    },
    "Permits": {
        "residential_permits": ["res_permits", "permits_residential"],
        "commercial_permits": ["comm_permits", "permits_commercial"],
        "retail_permits": ["ret_permits", "permits_retail"],
        "total_permits": ["permits_total"],
    },
    "Economics": {
        "personal_income": ["per_capita_income", "income_personal", "pers_inc"],
    },
    "Coordinates": {
        "xcoordinate": ["x_coord", "x"],
        "ycoordinate": ["y_coord", "y"],
        "latitude": ["lat"],
        "longitude": ["lon", "lng", "long"],
    },
    "Dates": {
        "issue_date": ["permit_issued_date", "issued_date"],
        "issue_date_year": ["permit_year", "issued_year"],
        "issue_date_month": ["permit_month", "issued_month"],
    },
    "Licensing & Businesses": {
        "total_licenses": ["licenses_total"],
        "active_licenses": ["licenses_active"],
        "unique_businesses": ["distinct_businesses", "business_count"],
    },
    "Construction": {
        "residential_construction_cost": ["res_cost", "res_const_cost"],
        "commercial_construction_cost": ["comm_cost", "comm_const_cost"],
        "retail_construction_cost": ["ret_cost", "ret_const_cost"],
        "retail_space": ["ret_sqft", "retail_sqft"],
        "housing_units": ["housing_stock", "units"],
        "housing_density": ["housing_dens"],
    },
    "Misc": {
        "permit_id": ["id_permit", "permit_code", "id"],
        "street_number": ["street_no"],
        "ward": ["city_ward"],
        "census_tract": ["tract"],
        "community_area": ["community"],
    },
}

# Column Metadata
column_metadata: ColumnMetadataMap = {
    "zip_code": {
        "type": ColumnType.STRING,
        "description": "US ZIP code",
        "format": r"^\d{5}(-\d{4})?$",
        "example": "60601",
        "validation": "zip_code_validator",
    },
    "year": {
        "type": ColumnType.INTEGER,
        "description": "Calendar or fiscal year",
        "range": [1900, 2100],
        "example": 2024,
    },
    "total_population": {
        "type": ColumnType.INTEGER,
        "description": "Total population count",
        "range": [0, 1000000000],
        "example": 2700000,
    },
    # Add more metadata as needed
}


# Helper Functions
def get_flat_aliases() -> Dict[str, str]:
    """
    Creates a flattened mapping of aliases to canonical names.

    Returns:
        Dict[str, str]: Mapping of each alias to its canonical name
    """
    flat_aliases = {}
    for category in column_aliases_by_category.values():
        for canonical, aliases in category.items():
            for alias in aliases:
                flat_aliases[alias] = canonical
            flat_aliases[canonical] = canonical
    return flat_aliases


def get_canonical_name(column_name: str) -> str:
    """
    Returns the canonical name for a given column alias.

    Args:
        column_name: The column name to standardize

    Returns:
        str: The canonical column name if found, otherwise the original name
    """
    flat_aliases = get_flat_aliases()
    return flat_aliases.get(column_name.lower().strip(), column_name)


def get_category(column_name: str) -> Optional[str]:
    """
    Returns the category of a given column name.

    Args:
        column_name: The column name to look up

    Returns:
        Optional[str]: The category name if found, None otherwise
    """
    canonical = get_canonical_name(column_name)
    return next(
        (
            category
            for category, columns in column_aliases_by_category.items()
            if canonical in columns
        ),
        None,
    )


def get_metadata(column_name: str) -> Optional[ColumnMetadata]:
    """
    Returns metadata for a given column name.

    Args:
        column_name: The column name to look up

    Returns:
        Optional[ColumnMetadata]: Column metadata if found, None otherwise
    """
    canonical = get_canonical_name(column_name)
    return column_metadata.get(canonical)


def get_columns_by_category(category: str) -> List[str]:
    """
    Returns all canonical column names in a given category.

    Args:
        category: The category name

    Returns:
        List[str]: List of canonical column names
    """
    return list(column_aliases_by_category.get(category, {}).keys())


def validate_column_value(column_name: str, value: any) -> bool:
    """
    Validates a value against the column's metadata rules.

    Args:
        column_name: The column name
        value: The value to validate

    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    metadata = get_metadata(column_name)
    if not metadata:
        return True

    col_type = metadata.get("type")
    if not col_type:
        return True

    try:
        if col_type == ColumnType.INTEGER:
            value = int(value)
            if "range" in metadata:
                min_val, max_val = metadata["range"]
                if not min_val <= value <= max_val:
                    raise ValueError(f"Value out of range for {column_name}: {value}")
        elif col_type == ColumnType.FLOAT:
            value = float(value)
            if "range" in metadata:
                min_val, max_val = metadata["range"]
                if not min_val <= value <= max_val:
                    raise ValueError(f"Value out of range for {column_name}: {value}")
        elif col_type == ColumnType.STRING and "format" in metadata:
            import re

            if not re.match(metadata["format"], str(value)):
                raise ValueError(f"Invalid format for {column_name}: {value}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid value for {column_name}: {value}") from e

    return True


# Flatten aliases for backward compatibility
column_aliases = {
    canonical: aliases
    for category in column_aliases_by_category.values()
    for canonical, aliases in category.items()
}


# Validate structure on import
def validate_structure():
    """Validates the column alias and metadata structure."""
    flat_aliases = get_flat_aliases()

    # Check for duplicate aliases
    seen_aliases = set()
    for alias in flat_aliases:
        if alias in seen_aliases:
            raise ValueError(f"Duplicate alias found: {alias}")
        seen_aliases.add(alias)

    # Check metadata consistency
    for canonical in column_aliases:
        if canonical in column_metadata and not isinstance(column_metadata[canonical], dict):
            raise ValueError(f"Invalid metadata format for {canonical}")


validate_structure()
