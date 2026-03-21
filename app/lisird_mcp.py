"""
lisird_mcp.py — MCP server exposing the LASP Interactive Solar IRradiance
Datacenter (LISIRD) via the LaTiS data access API.

LaTiS underlies both LISIRD and the Space Weather Data Portal. It provides
access to over 130 solar datasets from LASP, NASA, NOAA, and NSO through a
"Functional Data Model" that allows SQL-like querying directly in the URL
(projecting variables, slicing time ranges, choosing output formats).

API reference: https://lasp.colorado.edu/lisird/latis/dap2/

Tools exposed:
    query_solar_irradiance  — Query solar irradiance / space weather datasets
    list_lisird_datasets    — List all datasets available through LISIRD

Environment variables:
    LISIRD_BASE_URL   (default: https://lasp.colorado.edu/lisird/latis/dap2)
    LISIRD_TIMEOUT    (default: 30 seconds)

Run as a standalone MCP server:
    python lisird_mcp.py
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

load_dotenv()

LISIRD_BASE_URL: str = os.getenv(
    "LISIRD_BASE_URL",
    "https://lasp.colorado.edu/lisird/latis/dap2",
).rstrip("/")
LISIRD_TIMEOUT: float = float(os.getenv("LISIRD_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "LISIRD",
    instructions=(
        "Tools for querying solar irradiance and space weather datasets through "
        "the LASP Interactive Solar IRradiance Datacenter (LISIRD) via the LaTiS "
        "data access API. Use list_lisird_datasets to discover available datasets "
        "and query_solar_irradiance to retrieve measurements from a specific dataset."
    ),
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_latis_url(
    dataset_id: str,
    format: str,
    variables: str | None,
    start_date: str | None,
    end_date: str | None,
) -> str:
    """Build a LaTiS DAP2 query URL with optional variable projection and time constraints.

    LaTiS uses a non-standard query syntax where constraints are expressions like
    ``time>=2020-01-01`` rather than key=value pairs, so the query string is
    assembled manually and passed as a pre-built URL (not via httpx params).
    """
    base = f"{LISIRD_BASE_URL}/{dataset_id}.{format}"
    parts: list[str] = []
    if variables:
        # Comma-separated projection comes first in the query string
        parts.append(",".join(v.strip() for v in variables.split(",") if v.strip()))
    if start_date:
        parts.append(f"time>={start_date}")
    if end_date:
        parts.append(f"time<={end_date}")
    return f"{base}?{'&'.join(parts)}" if parts else base


def _latis_get(url: str) -> Any:
    """Perform a GET request against the LaTiS API and return parsed JSON."""
    with httpx.Client(timeout=LISIRD_TIMEOUT) as client:
        response = client.get(url)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
def query_solar_irradiance(
    dataset_id: Annotated[
        str,
        Field(
            description=(
                "LaTiS dataset identifier, e.g. 'sorce_tsi_24hr' (SORCE TSI daily "
                "averages) or 'nrlssi2' (NRL Solar Spectral Irradiance model). "
                "Use list_lisird_datasets to browse all available identifiers."
            ),
        ),
    ],
    start_date: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Start of the time range (inclusive). "
                "Accepts ISO 8601 dates or timestamps, e.g. '2003-01-01' or "
                "'2003-01-01T00:00:00.000Z'."
            ),
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "End of the time range (inclusive). "
                "Accepts ISO 8601 dates or timestamps, e.g. '2003-12-31' or "
                "'2003-12-31T23:59:59.999Z'."
            ),
        ),
    ] = None,
    variables: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Comma-separated list of variable names to return (SQL-like "
                "projection). Leave blank to return all variables in the dataset. "
                "Example: 'time,tsi' to fetch only time and total solar irradiance."
            ),
        ),
    ] = None,
    output_format: Annotated[
        str,
        Field(
            default="json",
            description=(
                "Response format — 'json' (default) or 'csv'. "
                "Use 'csv' for tabular data that is easier to parse as a table."
            ),
        ),
    ] = "json",
) -> dict[str, Any]:
    """Query a LISIRD solar dataset via the LaTiS API.

    Retrieves measurements from any of the 130+ solar datasets available through
    the LASP Interactive Solar IRradiance Datacenter (LISIRD). Supports time-range
    filtering and variable selection to keep responses compact, avoiding the need
    to download and parse large NetCDF files.

    Returns the queried data as a dictionary, or an error dict on failure.

    Example — fetch daily total solar irradiance (TSI) from SORCE for 2020-01:
        query_solar_irradiance(
            dataset_id="sorce_tsi_24hr",
            start_date="2020-01-01",
            end_date="2020-01-31",
            variables="time,tsi",
        )
    """
    url = _build_latis_url(dataset_id, output_format, variables, start_date, end_date)
    try:
        return _latis_get(url)
    except httpx.HTTPStatusError as exc:
        return {
            "error": f"LISIRD returned HTTP {exc.response.status_code}",
            "detail": exc.response.text,
            "url": url,
        }
    except httpx.RequestError as exc:
        return {"error": f"Request to LISIRD failed: {exc}", "url": url}


@mcp.tool()
def list_lisird_datasets() -> dict[str, Any]:
    """List all solar datasets available through LISIRD.

    Queries the LaTiS catalog endpoint and returns metadata for every dataset
    accessible via the LISIRD public API, including dataset identifiers, titles,
    and descriptions. Use the returned identifiers as the ``dataset_id`` parameter
    for query_solar_irradiance.

    Example:
        list_lisird_datasets()
    """
    url = f"{LISIRD_BASE_URL}/catalog.json"
    try:
        return _latis_get(url)
    except httpx.HTTPStatusError as exc:
        return {
            "error": f"LISIRD returned HTTP {exc.response.status_code}",
            "detail": exc.response.text,
        }
    except httpx.RequestError as exc:
        return {"error": f"Request to LISIRD failed: {exc}"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
