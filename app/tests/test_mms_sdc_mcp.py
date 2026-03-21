"""
Tests for the MMS SDC MCP server (mms_sdc_mcp.py).

All network I/O is mocked so the tests run without any internet access or
running MMS SDC server.
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
    """Remove the cached mms_sdc_mcp module so each test starts clean."""
    for mod in list(sys.modules.keys()):
        if mod == "mms_sdc_mcp":
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
# _build_params
# ---------------------------------------------------------------------------


class TestBuildParams:
    """_build_params omits None and empty values, keeps non-empty ones."""

    def test_all_none_returns_empty_dict(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        result = mod._build_params(None, None, None, None, None, None, None)
        assert result == {}

    def test_populated_fields_are_included(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        result = mod._build_params("mms1", "fgm", "srvy", "l2", "2016-01-01", "2016-01-31", None)
        assert result == {
            "sc_id": "mms1",
            "instrument_id": "fgm",
            "data_rate_mode": "srvy",
            "data_level": "l2",
            "start_date": "2016-01-01",
            "end_date": "2016-01-31",
        }
        assert "version" not in result

    def test_version_included_when_provided(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        result = mod._build_params("mms2", None, None, None, None, None, "3.3.0")
        assert result["version"] == "3.3.0"

    def test_empty_string_omitted(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        result = mod._build_params("", "fgm", "", None, None, None, None)
        assert "sc_id" not in result
        assert result["instrument_id"] == "fgm"
        assert "data_rate_mode" not in result

    def test_comma_separated_values_passed_through(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        result = mod._build_params("mms1,mms2", "fgm,fpi", "srvy,brst", "l2", None, None, None)
        assert result["sc_id"] == "mms1,mms2"
        assert result["instrument_id"] == "fgm,fpi"
        assert result["data_rate_mode"] == "srvy,brst"


# ---------------------------------------------------------------------------
# list_mms_files
# ---------------------------------------------------------------------------


class TestListMmsFiles:
    """list_mms_files queries file_info/science and returns the JSON payload."""

    def test_returns_json_on_success(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        fake_data = {
            "file_id": [1, 2],
            "filename": [
                "mms1_fgm_srvy_l2_20160101_v3.3.0.cdf",
                "mms1_fgm_srvy_l2_20160102_v3.3.0.cdf",
            ],
        }

        with patch("mms_sdc_mcp._sdc_get", return_value=fake_data):
            result = mod.list_mms_files(
                sc_id="mms1",
                instrument_id="fgm",
                data_rate_mode="srvy",
                data_level="l2",
                start_date="2016-01-01",
                end_date="2016-01-31",
            )

        assert result == fake_data
        assert "filename" in result
        assert len(result["filename"]) == 2

    def test_passes_correct_endpoint_and_params(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        captured: list[tuple] = []

        def fake_sdc_get(path, params):
            captured.append((path, params))
            return {}

        with patch("mms_sdc_mcp._sdc_get", side_effect=fake_sdc_get):
            mod.list_mms_files(
                sc_id="mms3",
                instrument_id="fpi",
                data_rate_mode="brst",
                data_level="l2",
                start_date="2016-10-16",
                end_date="2016-10-16",
            )

        assert len(captured) == 1
        path, params = captured[0]
        assert path == "file_info/science"
        assert params["sc_id"] == "mms3"
        assert params["instrument_id"] == "fpi"
        assert params["data_rate_mode"] == "brst"
        assert params["data_level"] == "l2"
        assert params["start_date"] == "2016-10-16"
        assert params["end_date"] == "2016-10-16"

    def test_omits_none_params(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        captured: list[dict] = []

        def fake_sdc_get(path, params):
            captured.append(params)
            return {}

        with patch("mms_sdc_mcp._sdc_get", side_effect=fake_sdc_get):
            mod.list_mms_files(sc_id="mms1")

        params = captured[0]
        assert "instrument_id" not in params
        assert "data_rate_mode" not in params
        assert "data_level" not in params

    def test_http_error_returns_error_dict(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch(
            "mms_sdc_mcp._sdc_get",
            side_effect=httpx.HTTPStatusError(
                "400 Bad Request",
                request=MagicMock(),
                response=mock_response,
            ),
        ):
            result = mod.list_mms_files(sc_id="mms1")

        assert "error" in result
        assert "400" in result["error"]

    def test_network_error_returns_error_dict(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        with patch(
            "mms_sdc_mcp._sdc_get",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            result = mod.list_mms_files(sc_id="mms1")

        assert "error" in result
        assert "failed" in result["error"].lower()

    def test_no_params_still_calls_api(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        captured: list = []

        def fake_sdc_get(path, params):
            captured.append((path, params))
            return {}

        with patch("mms_sdc_mcp._sdc_get", side_effect=fake_sdc_get):
            mod.list_mms_files()

        assert len(captured) == 1
        assert captured[0][0] == "file_info/science"
        assert captured[0][1] == {}


# ---------------------------------------------------------------------------
# get_mms_file_urls
# ---------------------------------------------------------------------------


class TestGetMmsFileUrls:
    """get_mms_file_urls queries file_names/science and returns download URLs."""

    def test_returns_urls_from_newline_text(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        file_list = (
            "/mms1/fgm/srvy/l2/2016/01/mms1_fgm_srvy_l2_20160101_v3.3.0.cdf\n"
            "/mms1/fgm/srvy/l2/2016/01/mms1_fgm_srvy_l2_20160102_v3.3.0.cdf\n"
        )

        with patch("mms_sdc_mcp._sdc_get", return_value=file_list):
            result = mod.get_mms_file_urls(
                sc_id="mms1",
                instrument_id="fgm",
                data_rate_mode="srvy",
                data_level="l2",
                start_date="2016-01-01",
                end_date="2016-01-02",
            )

        assert "urls" in result
        assert result["count"] == 2
        for url in result["urls"]:
            assert url.startswith("https://lasp.colorado.edu/mms/sdc/public")
            assert "mms1_fgm_srvy_l2" in url

    def test_passes_correct_endpoint(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        captured: list = []

        def fake_sdc_get(path, params):
            captured.append(path)
            return ""

        with patch("mms_sdc_mcp._sdc_get", side_effect=fake_sdc_get):
            mod.get_mms_file_urls(sc_id="mms1")

        assert captured[0] == "file_names/science"

    def test_urls_are_fully_qualified_https(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        file_list = "/mms2/fpi/brst/l2/2016/10/mms2_fpi_brst_l2_20161016_v3.cdf\n"

        with patch("mms_sdc_mcp._sdc_get", return_value=file_list):
            result = mod.get_mms_file_urls(sc_id="mms2")

        assert result["urls"][0].startswith("https://")

    def test_empty_response_returns_zero_urls(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        with patch("mms_sdc_mcp._sdc_get", return_value=""):
            result = mod.get_mms_file_urls(sc_id="mms1")

        assert result["urls"] == []
        assert result["count"] == 0

    def test_json_response_passed_through(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        json_data = {"files": ["file1.cdf", "file2.cdf"]}

        with patch("mms_sdc_mcp._sdc_get", return_value=json_data):
            result = mod.get_mms_file_urls(sc_id="mms1")

        assert result == json_data

    def test_http_error_returns_error_dict(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch(
            "mms_sdc_mcp._sdc_get",
            side_effect=httpx.HTTPStatusError(
                "500 Error",
                request=MagicMock(),
                response=mock_response,
            ),
        ):
            result = mod.get_mms_file_urls(sc_id="mms1")

        assert "error" in result
        assert "500" in result["error"]

    def test_network_error_returns_error_dict(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        with patch(
            "mms_sdc_mcp._sdc_get",
            side_effect=httpx.TimeoutException("Timed out"),
        ):
            result = mod.get_mms_file_urls(sc_id="mms1")

        assert "error" in result

    def test_path_without_leading_slash_still_valid_url(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        # Some API responses may omit the leading slash
        file_list = "mms1/fgm/srvy/l2/2016/01/mms1_fgm_srvy_l2_20160101_v3.3.0.cdf\n"

        with patch("mms_sdc_mcp._sdc_get", return_value=file_list):
            result = mod.get_mms_file_urls(sc_id="mms1")

        assert result["urls"][0].startswith("https://lasp.colorado.edu/mms/sdc/public/")


# ---------------------------------------------------------------------------
# _sdc_get (integration-style, mocking httpx.Client)
# ---------------------------------------------------------------------------


class TestSdcGet:
    """_sdc_get builds the correct URL and returns parsed JSON."""

    def test_calls_correct_url_with_params(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        fake_json = {"filename": ["test.cdf"]}
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_response = _mock_httpx_response(json_data=fake_json)
        mock_client.get.return_value = mock_response

        with patch("mms_sdc_mcp.httpx.Client", return_value=mock_client):
            result = mod._sdc_get("file_info/science", {"sc_id": "mms1"})

        mock_client.get.assert_called_once_with(
            f"{mod.MMS_SDC_BASE_URL}/file_info/science",
            params={"sc_id": "mms1"},
        )
        assert result == fake_json

    def test_raises_on_http_error(self):
        _fresh_module()
        import mms_sdc_mcp as mod

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_response = _mock_httpx_response(status_code=404, text_data="Not Found")
        mock_client.get.return_value = mock_response

        with patch("mms_sdc_mcp.httpx.Client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                mod._sdc_get("file_info/science", {})
