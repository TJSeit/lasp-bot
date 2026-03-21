"""
lasp_mcp.py — Consolidated MCP server exposing multiple LASP data APIs.

Combines LISIRD solar irradiance, MMS Science Data Center, AIM CIPS
mesospheric cloud, and HAPI heliophysics time-series tools into a single
MCP server that any MCP-compatible client can connect to.

API references:
    LISIRD: https://lasp.colorado.edu/lisird/latis/dap2/
    MMS SDC: https://lasp.colorado.edu/mms/sdc/public/about/how-to/
    AIM CIPS: https://lasp.colorado.edu/aim/latis/dap2/
    HAPI spec: https://hapi-server.org/

Tools exposed:
    LISIRD (solar irradiance / space weather):
        list_lisird_datasets    — List all datasets available through LISIRD
        query_solar_irradiance  — Query solar irradiance / space weather datasets

    MMS SDC (magnetospheric multiscale):
        list_mms_files          — List available MMS science data files
        get_mms_file_urls       — Retrieve download URLs for MMS science data files

    AIM CIPS (mesospheric clouds):
        query_mesospheric_data  — Query AIM CIPS noctilucent cloud datasets

    HAPI (heliophysics time-series):
        hapi_time_series_stream — Stream time-series data from any HAPI server

Environment variables:
    LISIRD_BASE_URL   (default: https://lasp.colorado.edu/lisird/latis/dap2)
    LISIRD_TIMEOUT    (default: 30 seconds)
    MMS_SDC_BASE_URL  (default: https://lasp.colorado.edu/mms/sdc/public/files/api/v1)
    MMS_SDC_TIMEOUT   (default: 30 seconds)
    AIM_BASE_URL      (default: https://lasp.colorado.edu/aim/latis/dap2)
    AIM_TIMEOUT       (default: 30 seconds)
    HAPI_TIMEOUT      (default: 30 seconds)

Run as a standalone MCP server:
    python lasp_mcp.py
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LISIRD_BASE_URL: str = os.getenv(
    "LISIRD_BASE_URL",
    "https://lasp.colorado.edu/lisird/latis/dap2",
).rstrip("/")
LISIRD_TIMEOUT: float = float(os.getenv("LISIRD_TIMEOUT", "30"))

MMS_SDC_BASE_URL: str = os.getenv(
    "MMS_SDC_BASE_URL",
    "https://lasp.colorado.edu/mms/sdc/public/files/api/v1",
).rstrip("/")
MMS_SDC_TIMEOUT: float = float(os.getenv("MMS_SDC_TIMEOUT", "30"))

AIM_BASE_URL: str = os.getenv(
    "AIM_BASE_URL",
    "https://lasp.colorado.edu/aim/latis/dap2",
).rstrip("/")
AIM_TIMEOUT: float = float(os.getenv("AIM_TIMEOUT", "30"))

HAPI_TIMEOUT: float = float(os.getenv("HAPI_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# MMS SDC — shared constants
# ---------------------------------------------------------------------------

_VALID_SC_IDS = {"mms1", "mms2", "mms3", "mms4"}
_VALID_INSTRUMENTS = {
    "afg", "aspoc", "dsp", "edi", "edp", "epd-eis",
    "feeps", "fgm", "fpi", "hpca", "mec", "scm", "sdp", "ulf",
}
_VALID_DATA_RATES = {"brst", "fast", "slow", "srvy"}
_VALID_DATA_LEVELS = {"l1a", "l1b", "l2", "l2pre", "l3", "ql"}

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "LASP",
    instructions=(
        "Tools for querying LASP (Laboratory for Atmospheric and Space Physics) "
        "data APIs. Available tool groups:\n"
        "- Solar irradiance (LISIRD): list_lisird_datasets, query_solar_irradiance\n"
        "- Magnetospheric data (MMS SDC): list_mms_files, get_mms_file_urls\n"
        "- Mesospheric clouds (AIM CIPS): query_mesospheric_data\n"
        "- Heliophysics time-series (HAPI): hapi_time_series_stream"
    ),
)

# ---------------------------------------------------------------------------
# Internal helpers — LaTiS (LISIRD & AIM CIPS)
# ---------------------------------------------------------------------------


async def _latis_get(url: str, timeout: float) -> Any:
    """Perform a GET request against a LaTiS DAP2 API and return parsed JSON."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
    response.raise_for_status()
    return response.json()


def _build_lisird_url(
    dataset_id: str,
    output_format: str,
    variables: str | None,
    start_date: str | None,
    end_date: str | None,
) -> str:
    """Build a LaTiS DAP2 query URL for LISIRD with optional variable projection
    and time constraints.

    LaTiS uses a non-standard query syntax where constraints are expressions like
    ``time>=2020-01-01`` rather than key=value pairs, so the query string is
    assembled manually.
    """
    base = f"{LISIRD_BASE_URL}/{dataset_id}.{output_format}"
    parts: list[str] = []
    if variables:
        parts.append(",".join(v.strip() for v in variables.split(",") if v.strip()))
    if start_date:
        parts.append(f"time>={start_date}")
    if end_date:
        parts.append(f"time<={end_date}")
    return f"{base}?{'&'.join(parts)}" if parts else base


def _build_aim_url(
    dataset_id: str,
    output_format: str,
    variable_constraints: str | None,
) -> str:
    """Build a LaTiS DAP2 query URL for the AIM CIPS API.

    AIM CIPS uses the same LaTiS base as LISIRD but accepts raw constraint
    expressions (e.g. ``time>=2022-305&time<2023-001``) as a pre-formatted
    query string.
    """
    url = f"{AIM_BASE_URL}/{dataset_id}.{output_format}"
    if variable_constraints:
        return f"{url}?{variable_constraints}"
    return url


# ---------------------------------------------------------------------------
# Internal helpers — MMS SDC
# ---------------------------------------------------------------------------


def _build_sdc_params(
    sc_id: str | None,
    instrument_id: str | None,
    data_rate_mode: str | None,
    data_level: str | None,
    start_date: str | None,
    end_date: str | None,
    version: str | None,
) -> dict[str, str]:
    """Assemble MMS SDC query parameters, omitting any that are None or empty."""
    params: dict[str, str] = {}
    if sc_id:
        params["sc_id"] = sc_id
    if instrument_id:
        params["instrument_id"] = instrument_id
    if data_rate_mode:
        params["data_rate_mode"] = data_rate_mode
    if data_level:
        params["data_level"] = data_level
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if version:
        params["version"] = version
    return params


async def _sdc_get(path: str, params: dict[str, str]) -> Any:
    """Perform a GET request against the MMS SDC API and return parsed JSON."""
    url = f"{MMS_SDC_BASE_URL}/{path}"
    async with httpx.AsyncClient(timeout=MMS_SDC_TIMEOUT) as client:
        response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Internal helpers — HAPI
# ---------------------------------------------------------------------------


async def _hapi_get(url: str, params: dict[str, str]) -> Any:
    """Perform a GET request against a HAPI server and return parsed JSON."""
    async with httpx.AsyncClient(timeout=HAPI_TIMEOUT) as client:
        response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# MCP tools — LISIRD (solar irradiance)
# ---------------------------------------------------------------------------


@mcp.tool()
async def query_solar_irradiance(
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
    url = _build_lisird_url(dataset_id, output_format, variables, start_date, end_date)
    try:
        return await _latis_get(url, LISIRD_TIMEOUT)
    except httpx.HTTPStatusError as exc:
        return {
            "error": f"LISIRD returned HTTP {exc.response.status_code}",
            "detail": exc.response.text,
            "url": url,
        }
    except httpx.RequestError as exc:
        return {"error": f"Request to LISIRD failed: {exc}", "url": url}


@mcp.tool()
async def list_lisird_datasets() -> dict[str, Any]:
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
        return await _latis_get(url, LISIRD_TIMEOUT)
    except httpx.HTTPStatusError as exc:
        return {
            "error": f"LISIRD returned HTTP {exc.response.status_code}",
            "detail": exc.response.text,
        }
    except httpx.RequestError as exc:
        return {"error": f"Request to LISIRD failed: {exc}"}


# ---------------------------------------------------------------------------
# MCP tools — MMS SDC (magnetospheric multiscale)
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_mms_files(
    sc_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Spacecraft ID — one or more of mms1, mms2, mms3, mms4 "
                "(comma-separated). Leave blank to include all spacecraft."
            ),
        ),
    ] = None,
    instrument_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Instrument identifier — e.g. fgm, fpi, edp, hpca, mec, scm, "
                "edi, feeps, epd-eis, aspoc, dsp, ulf, afg, sdp "
                "(comma-separated). Leave blank to include all instruments."
            ),
        ),
    ] = None,
    data_rate_mode: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Data rate mode — one or more of srvy, brst, fast, slow "
                "(comma-separated). Leave blank to include all data rates."
            ),
        ),
    ] = None,
    data_level: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Data processing level — one or more of l1a, l1b, l2, l2pre, "
                "l3, ql (comma-separated). Leave blank to include all levels."
            ),
        ),
    ] = None,
    start_date: Annotated[
        str | None,
        Field(
            default=None,
            description="Start of the time range in YYYY-MM-DD format.",
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            default=None,
            description="End of the time range in YYYY-MM-DD format.",
        ),
    ] = None,
    version: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "File version string (e.g. '3.3.0'). Leave blank to return all "
                "versions."
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    """List available MMS science data files matching the supplied filters.

    Returns a dictionary containing file metadata (filenames, time tags, sizes,
    checksums, etc.) as returned by the MMS SDC ``file_info/science`` endpoint.
    At least one filter parameter (sc_id, instrument_id, data_rate_mode,
    data_level, start_date, or end_date) should be supplied to narrow results.

    Example — list all survey-rate FGM level-2 files for MMS1 in March 2016:
        list_mms_files(
            sc_id="mms1",
            instrument_id="fgm",
            data_rate_mode="srvy",
            data_level="l2",
            start_date="2016-03-01",
            end_date="2016-03-31",
        )
    """
    params = _build_sdc_params(
        sc_id, instrument_id, data_rate_mode, data_level, start_date, end_date, version
    )
    try:
        return await _sdc_get("file_info/science", params)
    except httpx.HTTPStatusError as exc:
        return {
            "error": f"MMS SDC returned HTTP {exc.response.status_code}",
            "detail": exc.response.text,
        }
    except httpx.RequestError as exc:
        return {"error": f"Request to MMS SDC failed: {exc}"}


@mcp.tool()
async def get_mms_file_urls(
    sc_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Spacecraft ID — one or more of mms1, mms2, mms3, mms4 "
                "(comma-separated). Leave blank to include all spacecraft."
            ),
        ),
    ] = None,
    instrument_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Instrument identifier — e.g. fgm, fpi, edp, hpca, mec, scm, "
                "edi, feeps, epd-eis, aspoc, dsp, ulf, afg, sdp "
                "(comma-separated). Leave blank to include all instruments."
            ),
        ),
    ] = None,
    data_rate_mode: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Data rate mode — one or more of srvy, brst, fast, slow "
                "(comma-separated). Leave blank to include all data rates."
            ),
        ),
    ] = None,
    data_level: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Data processing level — one or more of l1a, l1b, l2, l2pre, "
                "l3, ql (comma-separated). Leave blank to include all levels."
            ),
        ),
    ] = None,
    start_date: Annotated[
        str | None,
        Field(
            default=None,
            description="Start of the time range in YYYY-MM-DD format.",
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            default=None,
            description="End of the time range in YYYY-MM-DD format.",
        ),
    ] = None,
    version: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "File version string (e.g. '3.3.0'). Leave blank to return all "
                "versions."
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    """Retrieve download URLs for MMS science data files.

    Calls the MMS SDC ``file_names/science`` endpoint and constructs a list of
    direct HTTPS download URLs for the matching files. Use this tool when you
    need the actual download links rather than full metadata.

    Example — get download URLs for burst-mode FPI level-2 files from MMS2:
        get_mms_file_urls(
            sc_id="mms2",
            instrument_id="fpi",
            data_rate_mode="brst",
            data_level="l2",
            start_date="2016-10-16",
            end_date="2016-10-16",
        )
    """
    params = _build_sdc_params(
        sc_id, instrument_id, data_rate_mode, data_level, start_date, end_date, version
    )
    try:
        data = await _sdc_get("file_names/science", params)
    except httpx.HTTPStatusError as exc:
        return {
            "error": f"MMS SDC returned HTTP {exc.response.status_code}",
            "detail": exc.response.text,
        }
    except httpx.RequestError as exc:
        return {"error": f"Request to MMS SDC failed: {exc}"}

    # The file_names endpoint returns a newline-separated list of relative paths.
    # Build fully-qualified HTTPS download URLs.
    base = "https://lasp.colorado.edu/mms/sdc/public"
    if isinstance(data, str):
        names = [n.strip() for n in data.splitlines() if n.strip()]
        urls = [f"{base}{n}" if n.startswith("/") else f"{base}/{n}" for n in names]
        return {"urls": urls, "count": len(urls)}

    # Newer API versions may return JSON directly; pass it through unchanged.
    return data if isinstance(data, dict) else {"result": data}


# ---------------------------------------------------------------------------
# MCP tools — AIM CIPS (mesospheric clouds)
# ---------------------------------------------------------------------------


@mcp.tool()
async def query_mesospheric_data(
    dataset_id: Annotated[
        str,
        Field(
            description=(
                "AIM CIPS LaTiS dataset identifier, e.g. 'aim_cips_anc_na' "
                "(CIPS Northern-hemisphere ancillary data). Use the AIM LaTiS "
                "catalog at https://lasp.colorado.edu/aim/latis/dap2/catalog.json "
                "to browse available datasets."
            ),
        ),
    ],
    variable_constraints: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "LaTiS constraint expression appended verbatim as the URL query "
                "string. Supports variable projection, time slices (using day-of-year "
                "format), and comparison operators. "
                "Example: 'time>=2022-305&time<2023-001' retrieves data from "
                "day 305 of 2022 through the start of 2023. "
                "Example with projection: 'time,albedo&time>=2022-001'. "
                "Leave blank to retrieve the entire dataset (may be very large)."
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
    """Query an AIM CIPS dataset via the LaTiS DAP2 API.

    The Aeronomy of Ice in the Mesosphere (AIM) mission monitors Earth's
    noctilucent clouds (NLCs) using the Cloud Imaging and Particle Size (CIPS)
    instrument. This tool provides constraint-based access to CIPS datasets
    through the AIM-specific LaTiS instance at LASP.

    Time values in CIPS datasets often use a ``YYYY-DDD`` day-of-year format
    (e.g. ``2022-305`` for day 305 of 2022) rather than calendar dates.

    Returns the queried data as a dictionary, or an error dict on failure.

    Example — retrieve CIPS Northern-hemisphere data for a date range:
        query_mesospheric_data(
            dataset_id="aim_cips_anc_na",
            variable_constraints="time>=2022-305&time<2023-001",
        )

    Example — project specific variables alongside a time constraint:
        query_mesospheric_data(
            dataset_id="aim_cips_anc_na",
            variable_constraints="time,albedo&time>=2022-001&time<2022-100",
        )
    """
    url = _build_aim_url(dataset_id, output_format, variable_constraints)
    try:
        return await _latis_get(url, AIM_TIMEOUT)
    except httpx.HTTPStatusError as exc:
        return {
            "error": f"AIM CIPS returned HTTP {exc.response.status_code}",
            "detail": exc.response.text,
            "url": url,
        }
    except httpx.RequestError as exc:
        return {"error": f"Request to AIM CIPS failed: {exc}", "url": url}


# ---------------------------------------------------------------------------
# MCP tools — HAPI (heliophysics time-series)
# ---------------------------------------------------------------------------


@mcp.tool()
async def hapi_time_series_stream(
    server_url: Annotated[
        str,
        Field(
            description=(
                "Base URL of the HAPI server (without trailing slash), e.g. "
                "'https://lasp.colorado.edu/lisird/hapi' (LASP/LISIRD), "
                "'https://cdaweb.gsfc.nasa.gov/hapi' (NASA Goddard CDAWeb), or "
                "'https://amda.irap.omp.eu/service/hapi' (European Space Agency). "
                "Any HAPI-compliant server can be used."
            ),
        ),
    ],
    dataset: Annotated[
        str,
        Field(
            description=(
                "Dataset (catalog entry) ID on the target HAPI server, e.g. "
                "'LISIRD3/composite_lya' or 'AC_K0_SWE'. "
                "Query the server's /catalog endpoint to discover available IDs."
            ),
        ),
    ],
    start: Annotated[
        str,
        Field(
            description=(
                "Start of the requested time range in ISO 8601 format "
                "(e.g. '2020-01-01T00:00:00Z'). Must be within the dataset's "
                "coverage interval."
            ),
        ),
    ],
    stop: Annotated[
        str,
        Field(
            description=(
                "End of the requested time range (exclusive) in ISO 8601 format "
                "(e.g. '2020-01-31T23:59:59Z'). Must be after ``start``."
            ),
        ),
    ],
    parameters: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Comma-separated list of parameter names to include in the response. "
                "Leave blank to return all parameters in the dataset. "
                "Example: 'Time,B_GSM' to fetch only the time column and the "
                "magnetic field in GSM coordinates."
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    """Stream time-series data from any HAPI-compliant heliophysics server.

    The Heliophysics Application Programmer's Interface (HAPI) is a standardized
    REST API for accessing time-series space-physics data. Because HAPI is a
    cross-agency standard, this single tool can pull data from LASP, NASA Goddard
    (CDAWeb), the European Space Agency (AMDA), and any other HAPI-compliant node
    using identical parameters.

    The tool calls the HAPI ``/data`` endpoint and returns the JSON response,
    which includes a ``parameters`` metadata block and a ``data`` array of
    time-ordered records.

    Returns the HAPI JSON response as a dictionary, or an error dict on failure.

    Example — fetch composite Lyman-alpha from LASP LISIRD HAPI:
        hapi_time_series_stream(
            server_url="https://lasp.colorado.edu/lisird/hapi",
            dataset="LISIRD3/composite_lya",
            start="2020-01-01T00:00:00Z",
            stop="2020-01-31T23:59:59Z",
            parameters="Time,irradiance",
        )

    Example — fetch solar wind data from NASA CDAWeb:
        hapi_time_series_stream(
            server_url="https://cdaweb.gsfc.nasa.gov/hapi",
            dataset="AC_K0_SWE",
            start="2020-06-01T00:00:00Z",
            stop="2020-06-02T00:00:00Z",
        )
    """
    base = server_url.rstrip("/")
    endpoint = f"{base}/data"
    params: dict[str, str] = {
        "id": dataset,
        "time.min": start,
        "time.max": stop,
        "format": "json",
    }
    if parameters:
        params["parameters"] = parameters

    try:
        return await _hapi_get(endpoint, params)
    except httpx.HTTPStatusError as exc:
        return {
            "error": f"HAPI server returned HTTP {exc.response.status_code}",
            "detail": exc.response.text,
            "server_url": server_url,
            "dataset": dataset,
        }
    except httpx.RequestError as exc:
        return {
            "error": f"Request to HAPI server failed: {exc}",
            "server_url": server_url,
            "dataset": dataset,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
