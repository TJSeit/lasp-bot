"""
Tests for the LISIRD MCP server (lisird_mcp.py).

All network I/O is mocked so the tests run without any internet access or
running LISIRD server.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_module():
    """Remove the cached lisird_mcp module so each test starts clean."""
    for mod in list(sys.modules.keys()):
        if mod == "lisird_mcp":
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


# ---------------------------------------------------------------------------
# _build_latis_url
# ---------------------------------------------------------------------------


class TestBuildLatisUrl:
    """_build_latis_url assembles the correct LaTiS query URL."""

    def test_no_filters_returns_base_url(self):
        _fresh_module()
        import lisird_mcp as mod

        url = mod._build_latis_url("sorce_tsi_24hr", "json", None, None, None)
        assert url == f"{mod.LISIRD_BASE_URL}/sorce_tsi_24hr.json"
        assert "?" not in url

    def test_start_date_included_in_query(self):
        _fresh_module()
        import lisird_mcp as mod

        url = mod._build_latis_url("sorce_tsi_24hr", "json", None, "2020-01-01", None)
        assert "time>=2020-01-01" in url

    def test_end_date_included_in_query(self):
        _fresh_module()
        import lisird_mcp as mod

        url = mod._build_latis_url("sorce_tsi_24hr", "json", None, None, "2020-12-31")
        assert "time<=2020-12-31" in url

    def test_both_dates_joined_with_ampersand(self):
        _fresh_module()
        import lisird_mcp as mod

        url = mod._build_latis_url(
            "sorce_tsi_24hr", "json", None, "2020-01-01", "2020-12-31"
        )
        assert "time>=2020-01-01" in url
        assert "time<=2020-12-31" in url
        assert "&" in url

    def test_variables_added_before_time_constraints(self):
        _fresh_module()
        import lisird_mcp as mod

        url = mod._build_latis_url(
            "sorce_tsi_24hr", "json", "time,tsi", "2020-01-01", None
        )
        query = url.split("?", 1)[1]
        # variable projection should appear before the time constraint
        assert query.startswith("time,tsi")

    def test_csv_format_in_url(self):
        _fresh_module()
        import lisird_mcp as mod

        url = mod._build_latis_url("sorce_tsi_24hr", "csv", None, None, None)
        assert "/sorce_tsi_24hr.csv" in url

    def test_variables_with_spaces_are_stripped(self):
        _fresh_module()
        import lisird_mcp as mod

        url = mod._build_latis_url(
            "sorce_tsi_24hr", "json", "time, tsi , flux", None, None
        )
        assert "time,tsi,flux" in url

    def test_iso_timestamp_allowed_for_dates(self):
        _fresh_module()
        import lisird_mcp as mod

        url = mod._build_latis_url(
            "sorce_tsi_24hr",
            "json",
            None,
            "2020-01-01T00:00:00.000Z",
            "2020-12-31T23:59:59.999Z",
        )
        assert "time>=2020-01-01T00:00:00.000Z" in url
        assert "time<=2020-12-31T23:59:59.999Z" in url


# ---------------------------------------------------------------------------
# query_solar_irradiance
# ---------------------------------------------------------------------------


class TestQuerySolarIrradiance:
    """query_solar_irradiance builds the URL and returns JSON data."""

    def test_returns_json_on_success(self):
        _fresh_module()
        import lisird_mcp as mod

        fake_data = {
            "sorce_tsi_24hr": {
                "parameters": ["time", "tsi"],
                "data": [[1577836800000, 1360.5], [1577923200000, 1360.6]],
            }
        }

        with patch("lisird_mcp._latis_get", return_value=fake_data):
            result = mod.query_solar_irradiance(
                dataset_id="sorce_tsi_24hr",
                start_date="2020-01-01",
                end_date="2020-01-31",
            )

        assert result == fake_data

    def test_passes_correct_url_to_latis_get(self):
        _fresh_module()
        import lisird_mcp as mod

        captured: list[str] = []

        def fake_latis_get(url):
            captured.append(url)
            return {}

        with patch("lisird_mcp._latis_get", side_effect=fake_latis_get):
            mod.query_solar_irradiance(
                dataset_id="sorce_tsi_24hr",
                start_date="2020-01-01",
                end_date="2020-01-31",
            )

        assert len(captured) == 1
        url = captured[0]
        assert "sorce_tsi_24hr.json" in url
        assert "time>=2020-01-01" in url
        assert "time<=2020-01-31" in url

    def test_variables_projection_included_in_url(self):
        _fresh_module()
        import lisird_mcp as mod

        captured: list[str] = []

        def fake_latis_get(url):
            captured.append(url)
            return {}

        with patch("lisird_mcp._latis_get", side_effect=fake_latis_get):
            mod.query_solar_irradiance(
                dataset_id="sorce_tsi_24hr",
                variables="time,tsi",
            )

        assert "time,tsi" in captured[0]

    def test_csv_format_used_when_specified(self):
        _fresh_module()
        import lisird_mcp as mod

        captured: list[str] = []

        def fake_latis_get(url):
            captured.append(url)
            return {}

        with patch("lisird_mcp._latis_get", side_effect=fake_latis_get):
            mod.query_solar_irradiance(
                dataset_id="sorce_tsi_24hr",
                output_format="csv",
            )

        assert ".csv" in captured[0]

    def test_no_dates_returns_full_dataset_url(self):
        _fresh_module()
        import lisird_mcp as mod

        captured: list[str] = []

        def fake_latis_get(url):
            captured.append(url)
            return {}

        with patch("lisird_mcp._latis_get", side_effect=fake_latis_get):
            mod.query_solar_irradiance(dataset_id="sorce_tsi_24hr")

        url = captured[0]
        assert "time>=" not in url
        assert "time<=" not in url
        assert "?" not in url

    def test_http_error_returns_error_dict(self):
        _fresh_module()
        import lisird_mcp as mod

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch(
            "lisird_mcp._latis_get",
            side_effect=httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=mock_response,
            ),
        ):
            result = mod.query_solar_irradiance(dataset_id="nonexistent_dataset")

        assert "error" in result
        assert "404" in result["error"]

    def test_network_error_returns_error_dict(self):
        _fresh_module()
        import lisird_mcp as mod

        with patch(
            "lisird_mcp._latis_get",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            result = mod.query_solar_irradiance(dataset_id="sorce_tsi_24hr")

        assert "error" in result
        assert "failed" in result["error"].lower()

    def test_error_dict_includes_url(self):
        _fresh_module()
        import lisird_mcp as mod

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch(
            "lisird_mcp._latis_get",
            side_effect=httpx.HTTPStatusError(
                "400 Bad Request",
                request=MagicMock(),
                response=mock_response,
            ),
        ):
            result = mod.query_solar_irradiance(
                dataset_id="sorce_tsi_24hr",
                start_date="2020-01-01",
            )

        assert "url" in result

    def test_default_format_is_json(self):
        _fresh_module()
        import lisird_mcp as mod

        captured: list[str] = []

        def fake_latis_get(url):
            captured.append(url)
            return {}

        with patch("lisird_mcp._latis_get", side_effect=fake_latis_get):
            mod.query_solar_irradiance(dataset_id="sorce_tsi_24hr")

        assert ".json" in captured[0]


# ---------------------------------------------------------------------------
# list_lisird_datasets
# ---------------------------------------------------------------------------


class TestListLisirdDatasets:
    """list_lisird_datasets calls the catalog endpoint and returns the payload."""

    def test_returns_catalog_on_success(self):
        _fresh_module()
        import lisird_mcp as mod

        fake_catalog = {
            "datasets": [
                {"id": "sorce_tsi_24hr", "title": "SORCE TSI 24-Hour Averages"},
                {"id": "nrlssi2", "title": "NRL Solar Spectral Irradiance Model"},
            ]
        }

        with patch("lisird_mcp._latis_get", return_value=fake_catalog):
            result = mod.list_lisird_datasets()

        assert result == fake_catalog

    def test_calls_catalog_endpoint(self):
        _fresh_module()
        import lisird_mcp as mod

        captured: list[str] = []

        def fake_latis_get(url):
            captured.append(url)
            return {}

        with patch("lisird_mcp._latis_get", side_effect=fake_latis_get):
            mod.list_lisird_datasets()

        assert len(captured) == 1
        assert captured[0].endswith("/catalog.json")

    def test_http_error_returns_error_dict(self):
        _fresh_module()
        import lisird_mcp as mod

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        with patch(
            "lisird_mcp._latis_get",
            side_effect=httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=MagicMock(),
                response=mock_response,
            ),
        ):
            result = mod.list_lisird_datasets()

        assert "error" in result
        assert "503" in result["error"]

    def test_network_error_returns_error_dict(self):
        _fresh_module()
        import lisird_mcp as mod

        with patch(
            "lisird_mcp._latis_get",
            side_effect=httpx.TimeoutException("Timed out"),
        ):
            result = mod.list_lisird_datasets()

        assert "error" in result


# ---------------------------------------------------------------------------
# _latis_get (integration-style, mocking httpx.Client)
# ---------------------------------------------------------------------------


class TestLatisGet:
    """_latis_get calls the supplied URL and returns parsed JSON."""

    def test_calls_correct_url(self):
        _fresh_module()
        import lisird_mcp as mod

        fake_json = {"sorce_tsi_24hr": {"data": [[1577836800000, 1360.5]]}}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_response = _mock_httpx_response(json_data=fake_json)
        mock_client.get.return_value = mock_response

        test_url = (
            "https://lasp.colorado.edu/lisird/latis/dap2/sorce_tsi_24hr.json"
            "?time>=2020-01-01&time<=2020-01-31"
        )

        with patch("lisird_mcp.httpx.Client", return_value=mock_client):
            result = mod._latis_get(test_url)

        mock_client.get.assert_called_once_with(test_url)
        assert result == fake_json

    def test_raises_on_http_error(self):
        _fresh_module()
        import lisird_mcp as mod

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_response = _mock_httpx_response(status_code=404, text_data="Not Found")
        mock_client.get.return_value = mock_response

        with patch("lisird_mcp.httpx.Client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                mod._latis_get(
                    "https://lasp.colorado.edu/lisird/latis/dap2/bad_dataset.json"
                )
