"""
mms_sdc_mcp.py — MCP server exposing the MMS Science Data Center public API.

The Magnetospheric Multiscale (MMS) Science Data Center (SDC) at LASP provides
a REST API for discovering and downloading MMS science data files.

API reference: https://lasp.colorado.edu/mms/sdc/public/about/how-to/

Tools exposed:
    list_mms_files     — Query available MMS science data files (returns metadata)
    get_mms_file_urls  — Retrieve download URLs for MMS science data files

Environment variables:
    MMS_SDC_BASE_URL   (default: https://lasp.colorado.edu/mms/sdc/public/files/api/v1)
    MMS_SDC_TIMEOUT    (default: 30 seconds)

Run as a standalone MCP server:
    python mms_sdc_mcp.py
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

load_dotenv()

MMS_SDC_BASE_URL: str = os.getenv(
    "MMS_SDC_BASE_URL",
    "https://lasp.colorado.edu/mms/sdc/public/files/api/v1",
).rstrip("/")
MMS_SDC_TIMEOUT: float = float(os.getenv("MMS_SDC_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# Shared constants — valid parameter values from the MMS SDC documentation
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
    "MMS SDC",
    instructions=(
        "Tools for querying the Magnetospheric Multiscale (MMS) Science Data "
        "Center (SDC) at LASP. Use list_mms_files to discover available science "
        "data files and get_mms_file_urls to obtain download URLs."
    ),
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_params(
    sc_id: str | None,
    instrument_id: str | None,
    data_rate_mode: str | None,
    data_level: str | None,
    start_date: str | None,
    end_date: str | None,
    version: str | None,
) -> dict[str, str]:
    """Assemble query parameters, omitting any that are None or empty."""
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


def _sdc_get(path: str, params: dict[str, str]) -> Any:
    """Perform a GET request against the MMS SDC API and return parsed JSON."""
    url = f"{MMS_SDC_BASE_URL}/{path}"
    with httpx.Client(timeout=MMS_SDC_TIMEOUT) as client:
        response = client.get(url, params=params)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_mms_files(
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
    params = _build_params(
        sc_id, instrument_id, data_rate_mode, data_level, start_date, end_date, version
    )
    try:
        return _sdc_get("file_info/science", params)
    except httpx.HTTPStatusError as exc:
        return {
            "error": f"MMS SDC returned HTTP {exc.response.status_code}",
            "detail": exc.response.text,
        }
    except httpx.RequestError as exc:
        return {"error": f"Request to MMS SDC failed: {exc}"}


@mcp.tool()
def get_mms_file_urls(
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
    params = _build_params(
        sc_id, instrument_id, data_rate_mode, data_level, start_date, end_date, version
    )
    try:
        data = _sdc_get("file_names/science", params)
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
