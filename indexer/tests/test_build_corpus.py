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

    def test_excludes_people_urls(self, tmp_path):
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

        assert queued == 1
        scraped_url = builder.scrape_web_and_pds.call_args[0][0]
        assert "missions/maven" in scraped_url

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
