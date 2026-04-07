"""
Tests for the sitemap methods added to LaspCorpusBuilder.

All network I/O is mocked so the tests run without any internet access.
"""

import os
from urllib.parse import urlparse
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


# ---------------------------------------------------------------------------
# scrape_paginated
# ---------------------------------------------------------------------------

PUB_BASE = "https://lasp.colorado.edu/our-expertise/science/scientific-publications"
PUB_BASE_PATH = urlparse(PUB_BASE).path


def _html_page(page_num, total_pages, papers_per_page=5, sequential=False):
    """Return mock HTML for one page of the publications listing.

    When *sequential* is True, only a "Next" link to the immediately following
    page is included (simulating prev/next-only pagination).  Otherwise, links
    to every page number are emitted (numbered pagination).

    Page 1 navigation links use the bare base URL (no ``/page/1/`` suffix),
    matching standard WordPress pagination behaviour.
    """
    pdfs = "\n".join(
        f'<a href="{PUB_BASE}/paper-p{page_num}-{i}.pdf">Paper {page_num}-{i}</a>'
        for i in range(papers_per_page)
    )

    def _page_href(p):
        return PUB_BASE + "/" if p == 1 else f"{PUB_BASE}/page/{p}/"

    if sequential:
        if page_num < total_pages:
            nav = f'<a href="{_page_href(page_num + 1)}">Next</a>'
        else:
            nav = ""
    else:
        nav = "\n".join(
            f'<a href="{_page_href(p)}">{p}</a>'
            for p in range(1, total_pages + 1)
            if p != page_num
        )

    return (
        f"<html><body><main>{pdfs}</main><nav>{nav}</nav></body></html>"
    ).encode()


def _pub_session_mock(total_pages, papers_per_page=5, sequential=False):
    """Return a mock session whose get() simulates paginated publications."""
    import re

    def fake_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        parsed_path = urlparse(url).path

        if parsed_path.lower().endswith(".pdf"):
            resp.status_code = 200
            resp.headers = {"Content-Type": "application/pdf"}
            resp.iter_content = MagicMock(return_value=iter([b"%PDF fake"]))
            return resp

        if PUB_BASE_PATH in parsed_path:
            m = re.search(r"/page/(\d+)/?$", parsed_path)
            page_num = int(m.group(1)) if m else 1
            html = _html_page(page_num, total_pages, papers_per_page, sequential)
            resp.status_code = 200
            resp.headers = {"Content-Type": "text/html; charset=utf-8"}
            resp.text = html.decode()
            resp.content = html
            return resp

        resp.status_code = 404
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        return resp

    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=fake_get)
    return mock_session


class TestScrapePaginated:
    """LaspCorpusBuilder.scrape_paginated covers all pages of a paginated listing."""

    def test_sequential_pagination_visits_all_pages(self, tmp_path):
        """All pages must be visited even when each page only has a 'Next' link.

        With max_depth=2 the old scrape_web_and_pds approach would stop after
        page 3.  scrape_paginated must visit every page regardless.
        """
        TOTAL_PAGES = 10
        builder = _make_builder(tmp_path)
        builder.session = _pub_session_mock(TOTAL_PAGES, sequential=True)

        with patch("time.sleep"):
            pages_visited = builder.scrape_paginated(PUB_BASE + "/")

        assert pages_visited == TOTAL_PAGES

        # Every page URL must be in visited_urls
        for p in range(2, TOTAL_PAGES + 1):
            page_url = builder._normalize_url(f"{PUB_BASE}/page/{p}/")
            assert page_url in builder.visited_urls, (
                f"Page {p} was not visited (sequential pagination)"
            )

    def test_numbered_pagination_visits_all_pages(self, tmp_path):
        """All pages must be visited when every page number is visible on the index."""
        TOTAL_PAGES = 8
        builder = _make_builder(tmp_path)
        builder.session = _pub_session_mock(TOTAL_PAGES, sequential=False)

        with patch("time.sleep"):
            pages_visited = builder.scrape_paginated(PUB_BASE + "/")

        assert pages_visited == TOTAL_PAGES

        for p in range(2, TOTAL_PAGES + 1):
            page_url = builder._normalize_url(f"{PUB_BASE}/page/{p}/")
            assert page_url in builder.visited_urls, (
                f"Page {p} was not visited (numbered pagination)"
            )

    def test_all_pdfs_downloaded_from_every_page(self, tmp_path):
        """PDFs linked on any listing page must be saved to the pdf directory."""
        TOTAL_PAGES = 4
        PAPERS_PER_PAGE = 3
        builder = _make_builder(tmp_path)
        builder.session = _pub_session_mock(
            TOTAL_PAGES, papers_per_page=PAPERS_PER_PAGE, sequential=True
        )

        with patch("time.sleep"):
            builder.scrape_paginated(PUB_BASE + "/")

        pdf_dir = tmp_path / "pdfs"
        saved_pdfs = list(pdf_dir.glob("*.pdf"))
        expected = TOTAL_PAGES * PAPERS_PER_PAGE
        assert len(saved_pdfs) == expected, (
            f"Expected {expected} PDFs but found {len(saved_pdfs)}: "
            f"{[p.name for p in saved_pdfs]}"
        )

    def test_links_outside_base_path_not_queued(self, tmp_path):
        """Links whose paths do not start with the publications base path must
        not be added to the pagination queue."""
        html = (
            b"<html><body>"
            b'<a href="https://lasp.colorado.edu/missions/">Missions</a>'
            b'<a href="https://external.example.com/paper.html">External</a>'
            b'<a href="https://lasp.colorado.edu/our-expertise/science/'
            b'scientific-publications/paper1.pdf">PDF</a>'
            b"</body></html>"
        )

        def fake_get(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if url.lower().endswith(".pdf"):
                resp.status_code = 200
                resp.headers = {"Content-Type": "application/pdf"}
                resp.iter_content = MagicMock(return_value=iter([b"%PDF"]))
                return resp
            resp.status_code = 200
            resp.headers = {"Content-Type": "text/html"}
            resp.text = html.decode()
            resp.content = html
            return resp

        builder = _make_builder(tmp_path)
        builder.session.get = MagicMock(side_effect=fake_get)

        with patch("time.sleep"):
            pages_visited = builder.scrape_paginated(PUB_BASE + "/")

        # Only the start URL itself should be a listing page
        assert pages_visited == 1

        visited = builder.visited_urls
        assert not any("missions" in u for u in visited), (
            "Off-path /missions/ URL must not be visited"
        )
        assert not any(urlparse(u).netloc == "external.example.com" for u in visited), (
            "External domain URL must not be visited"
        )

    def test_already_visited_pages_are_skipped(self, tmp_path):
        """A page already in visited_urls must not be re-fetched."""
        builder = _make_builder(tmp_path)
        builder.session = _pub_session_mock(total_pages=3, sequential=False)

        # Pre-mark page 2 as visited
        page2 = builder._normalize_url(f"{PUB_BASE}/page/2/")
        builder.visited_urls.add(page2)

        with patch("time.sleep"):
            pages_visited = builder.scrape_paginated(PUB_BASE + "/")

        # Page 2 was skipped; pages 1 and 3 were visited
        assert pages_visited == 2
        assert page2 in builder.visited_urls  # still present (pre-existing)

    def test_returns_pages_visited_count(self, tmp_path):
        """Return value must equal the exact number of listing pages visited."""
        TOTAL_PAGES = 6
        builder = _make_builder(tmp_path)
        builder.session = _pub_session_mock(TOTAL_PAGES, sequential=True)

        with patch("time.sleep"):
            result = builder.scrape_paginated(PUB_BASE + "/")

        assert result == TOTAL_PAGES

    def test_http_error_on_page_does_not_stop_crawl(self, tmp_path):
        """A network error on one page must not prevent subsequent pages
        from being visited."""
        import re

        def fake_get(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            parsed_path = urlparse(url).path

            # Page 2 always fails
            if re.search(r"/page/2/?$", parsed_path):
                resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                    "503"
                )
                return resp

            if PUB_BASE_PATH in parsed_path:
                m = re.search(r"/page/(\d+)/?$", parsed_path)
                page_num = int(m.group(1)) if m else 1
                # Numbered pagination: all page links visible on index
                html = _html_page(page_num, 4, 2, sequential=False)
                resp.status_code = 200
                resp.headers = {"Content-Type": "text/html"}
                resp.text = html.decode()
                resp.content = html
                return resp

            resp.status_code = 404
            resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
            return resp

        builder = _make_builder(tmp_path)
        builder.session.get = MagicMock(side_effect=fake_get)

        with patch("time.sleep"):
            pages_visited = builder.scrape_paginated(PUB_BASE + "/")

        # Pages 1, 3, and 4 are successfully visited; page 2 failed but didn't abort
        assert pages_visited == 3
        assert builder._normalize_url(f"{PUB_BASE}/page/3/") in builder.visited_urls
        assert builder._normalize_url(f"{PUB_BASE}/page/4/") in builder.visited_urls

