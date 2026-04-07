"""
Tests for the sitemap methods added to LaspCorpusBuilder.

All network I/O is mocked so the tests run without any internet access.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# build_corpus is importable because conftest.py adds the indexer directory to
# sys.path before pytest collects any test modules.
from build_corpus import LaspCorpusBuilder


def _make_builder(tmp_path):
    """Return a LaspCorpusBuilder that writes to a temporary directory."""
    return LaspCorpusBuilder(download_dir=str(tmp_path))


def _mock_response(content: bytes, status: int = 200):
    """Return a minimal mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status} Error"
        )
    return resp


# ---------------------------------------------------------------------------
# fetch_sitemap_urls
# ---------------------------------------------------------------------------

URLSET_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://lasp.colorado.edu/missions/maven/</loc></url>
  <url><loc>https://lasp.colorado.edu/missions/mms/</loc></url>
  <url><loc>https://lasp.colorado.edu/about/</loc></url>
</urlset>"""

SITEMAPINDEX_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://lasp.colorado.edu/sitemap-missions.xml</loc></sitemap>
  <sitemap><loc>https://lasp.colorado.edu/sitemap-news.xml</loc></sitemap>
</sitemapindex>"""

CHILD_SITEMAP_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://lasp.colorado.edu/news/article1/</loc></url>
  <url><loc>https://lasp.colorado.edu/news/article2/</loc></url>
</urlset>"""


class TestFetchSitemapUrls:
    """LaspCorpusBuilder.fetch_sitemap_urls parses sitemap XML correctly."""

    def test_urlset_returns_all_loc_urls(self, tmp_path):
        builder = _make_builder(tmp_path)
        builder.session.get = MagicMock(return_value=_mock_response(URLSET_XML))

        urls = builder.fetch_sitemap_urls("https://lasp.colorado.edu/sitemap.xml")

        assert urls == [
            "https://lasp.colorado.edu/missions/maven/",
            "https://lasp.colorado.edu/missions/mms/",
            "https://lasp.colorado.edu/about/",
        ]

    def test_sitemapindex_recurses_into_child_sitemaps(self, tmp_path):
        builder = _make_builder(tmp_path)

        missions_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://lasp.colorado.edu/missions/maven/</loc></url>
</urlset>"""

        def fake_get(url, **kwargs):
            if "sitemap-missions" in url:
                return _mock_response(missions_xml)
            if "sitemap-news" in url:
                return _mock_response(CHILD_SITEMAP_XML)
            return _mock_response(SITEMAPINDEX_XML)

        builder.session.get = MagicMock(side_effect=fake_get)

        urls = builder.fetch_sitemap_urls("https://lasp.colorado.edu/sitemap.xml")

        assert "https://lasp.colorado.edu/missions/maven/" in urls
        assert "https://lasp.colorado.edu/news/article1/" in urls
        assert "https://lasp.colorado.edu/news/article2/" in urls
        assert len(urls) == 3

    def test_http_error_returns_empty_list(self, tmp_path):
        builder = _make_builder(tmp_path)
        builder.session.get = MagicMock(
            side_effect=requests.exceptions.ConnectionError("timeout")
        )

        urls = builder.fetch_sitemap_urls("https://lasp.colorado.edu/sitemap.xml")

        assert urls == []

    def test_non_200_response_returns_empty_list(self, tmp_path):
        builder = _make_builder(tmp_path)
        builder.session.get = MagicMock(return_value=_mock_response(b"", status=404))

        urls = builder.fetch_sitemap_urls("https://lasp.colorado.edu/sitemap.xml")

        assert urls == []

    def test_invalid_xml_returns_empty_list(self, tmp_path):
        builder = _make_builder(tmp_path)
        builder.session.get = MagicMock(
            return_value=_mock_response(b"<not valid xml <<>>")
        )

        urls = builder.fetch_sitemap_urls("https://lasp.colorado.edu/sitemap.xml")

        assert urls == []

    def test_unknown_root_element_returns_empty_list(self, tmp_path):
        builder = _make_builder(tmp_path)
        builder.session.get = MagicMock(
            return_value=_mock_response(b"<unknown><item>x</item></unknown>")
        )

        urls = builder.fetch_sitemap_urls("https://lasp.colorado.edu/sitemap.xml")

        assert urls == []

    def test_no_namespace_urlset_also_parsed(self, tmp_path):
        """Sitemaps without an explicit XML namespace should still be parsed."""
        xml = b"""<?xml version="1.0"?>
<urlset>
  <url><loc>https://lasp.colorado.edu/page/</loc></url>
</urlset>"""
        builder = _make_builder(tmp_path)
        builder.session.get = MagicMock(return_value=_mock_response(xml))

        urls = builder.fetch_sitemap_urls("https://lasp.colorado.edu/sitemap.xml")

        assert urls == ["https://lasp.colorado.edu/page/"]


# ---------------------------------------------------------------------------
# crawl_from_sitemap
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _record_source
# ---------------------------------------------------------------------------


class TestRecordSource:
    """_record_source writes manifest keys that match actual file paths."""

    def test_pdf_key_uses_actual_subdirectory_name(self, tmp_path):
        """Manifest key for PDFs must use 'pdfs/', not 'pdf/'."""
        import json

        builder = _make_builder(tmp_path)
        builder._record_source("pdf", "document.pdf", "https://lasp.colorado.edu/doc.pdf")

        with open(tmp_path / "source_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "pdfs/document.pdf" in manifest
        assert manifest["pdfs/document.pdf"] == "https://lasp.colorado.edu/doc.pdf"

    def test_html_text_key_uses_actual_subdirectory_name(self, tmp_path):
        """Manifest key for web text must use 'web_text/', not 'html_text/'."""
        import json

        builder = _make_builder(tmp_path)
        builder._record_source("html_text", "page.txt", "https://lasp.colorado.edu/page")

        with open(tmp_path / "source_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "web_text/page.txt" in manifest

    def test_github_key_uses_actual_subdirectory_name(self, tmp_path):
        """Manifest key for GitHub docs must use 'github_docs/', not 'github/'."""
        import json

        builder = _make_builder(tmp_path)
        builder._record_source(
            "github",
            "repo_README.md",
            "https://raw.githubusercontent.com/lasp/repo/main/README.md",
        )

        with open(tmp_path / "source_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "github_docs/repo_README.md" in manifest

    def test_pds_data_key_uses_actual_subdirectory_name(self, tmp_path):
        """Manifest key for PDS data must use 'pds_metadata/', not 'pds_data/'."""
        import json

        builder = _make_builder(tmp_path)
        builder._record_source("pds_data", "label.xml", "https://lasp.colorado.edu/label.xml")

        with open(tmp_path / "source_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "pds_metadata/label.xml" in manifest

    def test_no_manifest_entry_when_source_url_is_empty(self, tmp_path):
        """_record_source must not write a manifest when source_url is empty."""
        builder = _make_builder(tmp_path)
        builder._record_source("pdf", "document.pdf", "")

        assert not (tmp_path / "source_manifest.json").exists()


# ---------------------------------------------------------------------------
# crawl_from_sitemap
# ---------------------------------------------------------------------------


class TestCrawlFromSitemap:
    """LaspCorpusBuilder.crawl_from_sitemap filters and scrapes valid URLs."""

    def test_scrapes_valid_lasp_urls(self, tmp_path):
        builder = _make_builder(tmp_path)
        builder.fetch_sitemap_urls = MagicMock(
            return_value=[
                "https://lasp.colorado.edu/missions/maven/",
                "https://lasp.colorado.edu/about/",
            ]
        )
        builder.scrape_web_and_pds = MagicMock()

        with patch("time.sleep"):
            queued = builder.crawl_from_sitemap(
                "https://lasp.colorado.edu/sitemap.xml"
            )

        assert queued == 2
        assert builder.scrape_web_and_pds.call_count == 2

    def test_includes_people_urls(self, tmp_path):
        builder = _make_builder(tmp_path)
        builder.fetch_sitemap_urls = MagicMock(
            return_value=[
                "https://lasp.colorado.edu/missions/maven/",
                "https://lasp.colorado.edu/people/john-doe/",
            ]
        )
        builder.scrape_web_and_pds = MagicMock()

        with patch("time.sleep"):
            queued = builder.crawl_from_sitemap(
                "https://lasp.colorado.edu/sitemap.xml"
            )

        assert queued == 2
        scraped_urls = [call[0][0] for call in builder.scrape_web_and_pds.call_args_list]
        assert any("missions/maven" in u for u in scraped_urls)
        assert any("people/john-doe" in u for u in scraped_urls)

    def test_excludes_foreign_domain_urls(self, tmp_path):
        builder = _make_builder(tmp_path)
        builder.fetch_sitemap_urls = MagicMock(
            return_value=[
                "https://lasp.colorado.edu/missions/",
                "https://external.example.com/page/",
            ]
        )
        builder.scrape_web_and_pds = MagicMock()

        with patch("time.sleep"):
            queued = builder.crawl_from_sitemap(
                "https://lasp.colorado.edu/sitemap.xml"
            )

        assert queued == 1

    def test_skips_already_visited_urls(self, tmp_path):
        builder = _make_builder(tmp_path)
        builder.fetch_sitemap_urls = MagicMock(
            return_value=["https://lasp.colorado.edu/missions/maven/"]
        )
        # Use _normalize_url to ensure the stored URL matches the normalised
        # form that crawl_from_sitemap compares against.
        builder.visited_urls.add(
            builder._normalize_url("https://lasp.colorado.edu/missions/maven/")
        )
        builder.scrape_web_and_pds = MagicMock()

        with patch("time.sleep"):
            queued = builder.crawl_from_sitemap(
                "https://lasp.colorado.edu/sitemap.xml"
            )

        assert queued == 0
        builder.scrape_web_and_pds.assert_not_called()

    def test_returns_zero_when_sitemap_is_empty(self, tmp_path):
        builder = _make_builder(tmp_path)
        builder.fetch_sitemap_urls = MagicMock(return_value=[])
        builder.scrape_web_and_pds = MagicMock()

        queued = builder.crawl_from_sitemap("https://lasp.colorado.edu/sitemap.xml")

        assert queued == 0
        builder.scrape_web_and_pds.assert_not_called()

    def test_uses_max_depth_one_for_each_page(self, tmp_path):
        """Each sitemap page is scraped with max_depth=1 to capture linked binaries."""
        builder = _make_builder(tmp_path)
        builder.fetch_sitemap_urls = MagicMock(
            return_value=["https://lasp.colorado.edu/missions/maven/"]
        )
        builder.scrape_web_and_pds = MagicMock()

        with patch("time.sleep"):
            builder.crawl_from_sitemap("https://lasp.colorado.edu/sitemap.xml")

        builder.scrape_web_and_pds.assert_called_once()
        _, kwargs = builder.scrape_web_and_pds.call_args
        assert kwargs.get("depth") == 0
        assert kwargs.get("max_depth") == 1
