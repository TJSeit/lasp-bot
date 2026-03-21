"""
Tests for the consolidated LASP MCP server (lasp_mcp.py).

All network I/O is mocked so the tests run without any internet access or
running LASP servers. Existing LISIRD and MMS SDC tool logic is already
covered by test_lisird_mcp.py and test_mms_sdc_mcp.py; this file focuses
on the consolidated module integration and the two new tools:
  - query_mesospheric_data  (AIM CIPS LaTiS API)
  - hapi_time_series_stream (HAPI standard API)
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_module():
    """Remove the cached lasp_mcp module so each test starts clean."""
    for mod in list(sys.modules.keys()):
        if mod == "lasp_mcp":
            del sys.modules[mod]


def _mock_httpx_response(json_data=None, text_data: str = "", status_code: int = 200):
    """Return a minimal mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text_data
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.return_value = {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _mock_async_httpx_client(mock_response):
    """Return a mock httpx.AsyncClient usable as an async context manager."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)
    return mock_client


# ---------------------------------------------------------------------------
# Module-level sanity checks
# ---------------------------------------------------------------------------


class TestModuleExportsAllTools:
    """The consolidated module exposes all expected tool functions."""

    def test_all_tools_present(self):
        _fresh_module()
        import lasp_mcp as mod

        expected = [
            "list_lisird_datasets",
            "query_solar_irradiance",
            "list_mms_files",
            "get_mms_file_urls",
            "query_mesospheric_data",
            "hapi_time_series_stream",
        ]
        for name in expected:
            assert hasattr(mod, name), f"Missing tool: {name}"

    def test_all_helpers_present(self):
        _fresh_module()
        import lasp_mcp as mod

        expected_helpers = [
            "_latis_get",
            "_build_lisird_url",
            "_build_aim_url",
            "_build_sdc_params",
            "_sdc_get",
            "_hapi_get",
        ]
        for name in expected_helpers:
            assert hasattr(mod, name), f"Missing helper: {name}"

    def test_default_config_values(self):
        _fresh_module()
        import lasp_mcp as mod

        assert "lisird" in mod.LISIRD_BASE_URL
        assert "mms" in mod.MMS_SDC_BASE_URL
        assert "aim" in mod.AIM_BASE_URL
        assert mod.LISIRD_TIMEOUT > 0
        assert mod.MMS_SDC_TIMEOUT > 0
        assert mod.AIM_TIMEOUT > 0
        assert mod.HAPI_TIMEOUT > 0


# ---------------------------------------------------------------------------
# _build_aim_url
# ---------------------------------------------------------------------------


class TestBuildAimUrl:
    """_build_aim_url constructs the correct AIM CIPS LaTiS query URL."""

    def test_no_constraints_returns_bare_url(self):
        _fresh_module()
        import lasp_mcp as mod

        url = mod._build_aim_url("aim_cips_anc_na", "json", None)
        assert url == f"{mod.AIM_BASE_URL}/aim_cips_anc_na.json"
        assert "?" not in url

    def test_constraints_appended_as_query_string(self):
        _fresh_module()
        import lasp_mcp as mod

        url = mod._build_aim_url(
            "aim_cips_anc_na", "json", "time>=2022-305&time<2023-001"
        )
        assert url == (
            f"{mod.AIM_BASE_URL}/aim_cips_anc_na.json"
            "?time>=2022-305&time<2023-001"
        )

    def test_csv_format_in_url(self):
        _fresh_module()
        import lasp_mcp as mod

        url = mod._build_aim_url("aim_cips_anc_na", "csv", None)
        assert "/aim_cips_anc_na.csv" in url

    def test_projection_with_time_constraint(self):
        _fresh_module()
        import lasp_mcp as mod

        constraints = "time,albedo&time>=2022-001&time<2022-100"
        url = mod._build_aim_url("aim_cips_anc_na", "json", constraints)
        assert "?" in url
        assert "time,albedo" in url
        assert "time>=2022-001" in url

    def test_empty_string_constraint_treated_as_no_constraint(self):
        _fresh_module()
        import lasp_mcp as mod

        url = mod._build_aim_url("aim_cips_anc_na", "json", "")
        assert "?" not in url


# ---------------------------------------------------------------------------
# query_mesospheric_data
# ---------------------------------------------------------------------------


class TestQueryMesosphericData:
    """query_mesospheric_data calls the AIM CIPS LaTiS API and returns data."""

    def test_returns_json_on_success(self):
        _fresh_module()
        import lasp_mcp as mod

        fake_data = {
            "aim_cips_anc_na": {
                "parameters": ["time", "albedo"],
                "data": [[1667433600000, 0.82], [1667520000000, 0.79]],
            }
        }

        with patch("lasp_mcp._latis_get", new_callable=AsyncMock, return_value=fake_data):
            result = asyncio.run(mod.query_mesospheric_data(
                dataset_id="aim_cips_anc_na",
                variable_constraints="time>=2022-305&time<2023-001",
            ))

        assert result == fake_data

    def test_passes_correct_url_to_latis_get(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list[tuple] = []

        async def fake_latis_get(url, timeout):
            captured.append((url, timeout))
            return {}

        with patch("lasp_mcp._latis_get", side_effect=fake_latis_get):
            asyncio.run(mod.query_mesospheric_data(
                dataset_id="aim_cips_anc_na",
                variable_constraints="time>=2022-305&time<2023-001",
            ))

        assert len(captured) == 1
        url, timeout = captured[0]
        assert "aim_cips_anc_na.json" in url
        assert "time>=2022-305" in url
        assert "time<2023-001" in url
        assert timeout == mod.AIM_TIMEOUT

    def test_no_constraints_returns_full_dataset_url(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list[str] = []

        async def fake_latis_get(url, timeout):
            captured.append(url)
            return {}

        with patch("lasp_mcp._latis_get", side_effect=fake_latis_get):
            asyncio.run(mod.query_mesospheric_data(dataset_id="aim_cips_anc_na"))

        assert "?" not in captured[0]

    def test_csv_format_used_when_specified(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list[str] = []

        async def fake_latis_get(url, timeout):
            captured.append(url)
            return {}

        with patch("lasp_mcp._latis_get", side_effect=fake_latis_get):
            asyncio.run(mod.query_mesospheric_data(
                dataset_id="aim_cips_anc_na",
                output_format="csv",
            ))

        assert ".csv" in captured[0]

    def test_default_format_is_json(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list[str] = []

        async def fake_latis_get(url, timeout):
            captured.append(url)
            return {}

        with patch("lasp_mcp._latis_get", side_effect=fake_latis_get):
            asyncio.run(mod.query_mesospheric_data(dataset_id="aim_cips_anc_na"))

        assert ".json" in captured[0]

    def test_http_error_returns_error_dict(self):
        _fresh_module()
        import lasp_mcp as mod

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch(
            "lasp_mcp._latis_get",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=mock_response,
            ),
        ):
            result = asyncio.run(mod.query_mesospheric_data(dataset_id="nonexistent_dataset"))

        assert "error" in result
        assert "404" in result["error"]

    def test_http_error_includes_url_in_result(self):
        _fresh_module()
        import lasp_mcp as mod

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch(
            "lasp_mcp._latis_get",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "400 Bad Request",
                request=MagicMock(),
                response=mock_response,
            ),
        ):
            result = asyncio.run(mod.query_mesospheric_data(
                dataset_id="aim_cips_anc_na",
                variable_constraints="time>=2022-001",
            ))

        assert "url" in result

    def test_network_error_returns_error_dict(self):
        _fresh_module()
        import lasp_mcp as mod

        with patch(
            "lasp_mcp._latis_get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            result = asyncio.run(mod.query_mesospheric_data(dataset_id="aim_cips_anc_na"))

        assert "error" in result
        assert "failed" in result["error"].lower()

    def test_network_error_includes_url_in_result(self):
        _fresh_module()
        import lasp_mcp as mod

        with patch(
            "lasp_mcp._latis_get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("Timed out"),
        ):
            result = asyncio.run(mod.query_mesospheric_data(dataset_id="aim_cips_anc_na"))

        assert "url" in result


# ---------------------------------------------------------------------------
# hapi_time_series_stream
# ---------------------------------------------------------------------------


class TestHapiTimeSeriesStream:
    """hapi_time_series_stream calls the HAPI /data endpoint and returns data."""

    def test_returns_json_on_success(self):
        _fresh_module()
        import lasp_mcp as mod

        fake_data = {
            "HAPI": "3.1",
            "status": {"code": 1200, "message": "OK"},
            "parameters": [
                {"name": "Time", "type": "isotime"},
                {"name": "irradiance", "type": "double", "units": "W/m^2/nm"},
            ],
            "data": [
                ["2020-01-01T00:00:00.000Z", 0.00528],
                ["2020-01-02T00:00:00.000Z", 0.00527],
            ],
        }

        with patch("lasp_mcp._hapi_get", new_callable=AsyncMock, return_value=fake_data):
            result = asyncio.run(mod.hapi_time_series_stream(
                server_url="https://lasp.colorado.edu/lisird/hapi",
                dataset="LISIRD3/composite_lya",
                start="2020-01-01T00:00:00Z",
                stop="2020-01-31T23:59:59Z",
                parameters="Time,irradiance",
            ))

        assert result == fake_data

    def test_passes_correct_endpoint_and_params(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list[tuple] = []

        async def fake_hapi_get(url, params):
            captured.append((url, params))
            return {}

        with patch("lasp_mcp._hapi_get", side_effect=fake_hapi_get):
            asyncio.run(mod.hapi_time_series_stream(
                server_url="https://lasp.colorado.edu/lisird/hapi",
                dataset="LISIRD3/composite_lya",
                start="2020-01-01T00:00:00Z",
                stop="2020-01-31T23:59:59Z",
                parameters="Time,irradiance",
            ))

        assert len(captured) == 1
        url, params = captured[0]
        assert url == "https://lasp.colorado.edu/lisird/hapi/data"
        assert params["id"] == "LISIRD3/composite_lya"
        assert params["time.min"] == "2020-01-01T00:00:00Z"
        assert params["time.max"] == "2020-01-31T23:59:59Z"
        assert params["parameters"] == "Time,irradiance"
        assert params["format"] == "json"

    def test_trailing_slash_stripped_from_server_url(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list[str] = []

        async def fake_hapi_get(url, params):
            captured.append(url)
            return {}

        with patch("lasp_mcp._hapi_get", side_effect=fake_hapi_get):
            asyncio.run(mod.hapi_time_series_stream(
                server_url="https://lasp.colorado.edu/lisird/hapi/",
                dataset="LISIRD3/composite_lya",
                start="2020-01-01T00:00:00Z",
                stop="2020-01-31T23:59:59Z",
            ))

        assert captured[0] == "https://lasp.colorado.edu/lisird/hapi/data"

    def test_parameters_omitted_when_none(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list[dict] = []

        async def fake_hapi_get(url, params):
            captured.append(params)
            return {}

        with patch("lasp_mcp._hapi_get", side_effect=fake_hapi_get):
            asyncio.run(mod.hapi_time_series_stream(
                server_url="https://cdaweb.gsfc.nasa.gov/hapi",
                dataset="AC_K0_SWE",
                start="2020-06-01T00:00:00Z",
                stop="2020-06-02T00:00:00Z",
            ))

        assert "parameters" not in captured[0]

    def test_format_always_set_to_json(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list[dict] = []

        async def fake_hapi_get(url, params):
            captured.append(params)
            return {}

        with patch("lasp_mcp._hapi_get", side_effect=fake_hapi_get):
            asyncio.run(mod.hapi_time_series_stream(
                server_url="https://lasp.colorado.edu/lisird/hapi",
                dataset="LISIRD3/composite_lya",
                start="2020-01-01T00:00:00Z",
                stop="2020-01-31T23:59:59Z",
            ))

        assert captured[0]["format"] == "json"

    def test_works_with_nasa_cdaweb_server(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list[tuple] = []

        async def fake_hapi_get(url, params):
            captured.append((url, params))
            return {}

        with patch("lasp_mcp._hapi_get", side_effect=fake_hapi_get):
            asyncio.run(mod.hapi_time_series_stream(
                server_url="https://cdaweb.gsfc.nasa.gov/hapi",
                dataset="AC_K0_SWE",
                start="2020-06-01T00:00:00Z",
                stop="2020-06-02T00:00:00Z",
                parameters="Time,Np",
            ))

        url, params = captured[0]
        assert url == "https://cdaweb.gsfc.nasa.gov/hapi/data"
        assert params["id"] == "AC_K0_SWE"

    def test_http_error_returns_error_dict_with_metadata(self):
        _fresh_module()
        import lasp_mcp as mod

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch(
            "lasp_mcp._hapi_get",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=mock_response,
            ),
        ):
            result = asyncio.run(mod.hapi_time_series_stream(
                server_url="https://lasp.colorado.edu/lisird/hapi",
                dataset="nonexistent",
                start="2020-01-01T00:00:00Z",
                stop="2020-01-31T23:59:59Z",
            ))

        assert "error" in result
        assert "404" in result["error"]
        assert result["server_url"] == "https://lasp.colorado.edu/lisird/hapi"
        assert result["dataset"] == "nonexistent"

    def test_network_error_returns_error_dict_with_metadata(self):
        _fresh_module()
        import lasp_mcp as mod

        with patch(
            "lasp_mcp._hapi_get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            result = asyncio.run(mod.hapi_time_series_stream(
                server_url="https://lasp.colorado.edu/lisird/hapi",
                dataset="LISIRD3/composite_lya",
                start="2020-01-01T00:00:00Z",
                stop="2020-01-31T23:59:59Z",
            ))

        assert "error" in result
        assert "failed" in result["error"].lower()
        assert "server_url" in result
        assert "dataset" in result

    def test_timeout_error_returns_error_dict(self):
        _fresh_module()
        import lasp_mcp as mod

        with patch(
            "lasp_mcp._hapi_get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("Timed out"),
        ):
            result = asyncio.run(mod.hapi_time_series_stream(
                server_url="https://lasp.colorado.edu/lisird/hapi",
                dataset="LISIRD3/composite_lya",
                start="2020-01-01T00:00:00Z",
                stop="2020-01-31T23:59:59Z",
            ))

        assert "error" in result


# ---------------------------------------------------------------------------
# _hapi_get (integration-style, mocking httpx.AsyncClient)
# ---------------------------------------------------------------------------


class TestHapiGet:
    """_hapi_get calls the supplied URL with params and returns parsed JSON."""

    def test_calls_correct_url_with_params(self):
        _fresh_module()
        import lasp_mcp as mod

        fake_json = {
            "HAPI": "3.1",
            "status": {"code": 1200, "message": "OK"},
            "data": [["2020-01-01T00:00:00.000Z", 0.00528]],
        }
        mock_response = _mock_httpx_response(json_data=fake_json)
        mock_client = _mock_async_httpx_client(mock_response)

        endpoint = "https://lasp.colorado.edu/lisird/hapi/data"
        params = {
            "id": "LISIRD3/composite_lya",
            "time.min": "2020-01-01T00:00:00Z",
            "time.max": "2020-01-31T23:59:59Z",
            "format": "json",
        }

        with patch("lasp_mcp.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(mod._hapi_get(endpoint, params))

        mock_client.get.assert_awaited_once_with(endpoint, params=params)
        assert result == fake_json

    def test_raises_on_http_error(self):
        _fresh_module()
        import lasp_mcp as mod

        mock_response = _mock_httpx_response(status_code=404, text_data="Not Found")
        mock_client = _mock_async_httpx_client(mock_response)

        with patch("lasp_mcp.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(mod._hapi_get(
                    "https://lasp.colorado.edu/lisird/hapi/data",
                    {"id": "bad_dataset", "time.min": "2020-01-01", "time.max": "2020-01-31", "format": "json"},
                ))


# ---------------------------------------------------------------------------
# _latis_get (consolidated module — signature includes timeout)
# ---------------------------------------------------------------------------


class TestLatisGetConsolidated:
    """_latis_get in the consolidated module accepts a timeout parameter."""

    def test_calls_correct_url(self):
        _fresh_module()
        import lasp_mcp as mod

        fake_json = {"aim_cips_anc_na": {"data": []}}
        mock_response = _mock_httpx_response(json_data=fake_json)
        mock_client = _mock_async_httpx_client(mock_response)

        test_url = (
            "https://lasp.colorado.edu/aim/latis/dap2/aim_cips_anc_na.json"
            "?time>=2022-305&time<2023-001"
        )

        with patch("lasp_mcp.httpx.AsyncClient", return_value=mock_client):
            result = asyncio.run(mod._latis_get(test_url, 30.0))

        mock_client.get.assert_awaited_once_with(test_url)
        assert result == fake_json

    def test_uses_supplied_timeout(self):
        _fresh_module()
        import lasp_mcp as mod

        mock_response = _mock_httpx_response(json_data={})
        mock_client = _mock_async_httpx_client(mock_response)

        custom_timeout = 60.0

        with patch("lasp_mcp.httpx.AsyncClient", return_value=mock_client) as mock_cls:
            asyncio.run(mod._latis_get("https://example.com/data.json", custom_timeout))

        mock_cls.assert_called_once_with(timeout=custom_timeout)

    def test_raises_on_http_error(self):
        _fresh_module()
        import lasp_mcp as mod

        mock_response = _mock_httpx_response(status_code=500, text_data="Server Error")
        mock_client = _mock_async_httpx_client(mock_response)

        with patch("lasp_mcp.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(mod._latis_get("https://example.com/bad.json", 30.0))


# ---------------------------------------------------------------------------
# Existing tools — smoke tests in the consolidated module
# ---------------------------------------------------------------------------


class TestExistingToolsInConsolidatedModule:
    """Smoke tests confirming LISIRD and MMS tools still work in lasp_mcp."""

    def test_query_solar_irradiance_returns_data(self):
        _fresh_module()
        import lasp_mcp as mod

        fake_data = {"sorce_tsi_24hr": {"data": [[1577836800000, 1360.5]]}}
        with patch("lasp_mcp._latis_get", new_callable=AsyncMock, return_value=fake_data):
            result = asyncio.run(mod.query_solar_irradiance(
                dataset_id="sorce_tsi_24hr",
                start_date="2020-01-01",
                end_date="2020-01-31",
            ))
        assert result == fake_data

    def test_list_lisird_datasets_calls_catalog(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list[str] = []

        async def fake_latis_get(url, timeout):
            captured.append(url)
            return {}

        with patch("lasp_mcp._latis_get", side_effect=fake_latis_get):
            asyncio.run(mod.list_lisird_datasets())

        assert captured[0].endswith("/catalog.json")

    def test_list_mms_files_calls_file_info_endpoint(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list = []

        async def fake_sdc_get(path, params):
            captured.append(path)
            return {}

        with patch("lasp_mcp._sdc_get", side_effect=fake_sdc_get):
            asyncio.run(mod.list_mms_files(sc_id="mms1"))

        assert captured[0] == "file_info/science"

    def test_get_mms_file_urls_calls_file_names_endpoint(self):
        _fresh_module()
        import lasp_mcp as mod

        captured: list = []

        async def fake_sdc_get(path, params):
            captured.append(path)
            return ""

        with patch("lasp_mcp._sdc_get", side_effect=fake_sdc_get):
            asyncio.run(mod.get_mms_file_urls(sc_id="mms1"))

        assert captured[0] == "file_names/science"
