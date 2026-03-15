"""
Tests for build_index.load_documents.

All file I/O is performed against a temporary directory so the tests run
without any network access or GPU.
"""

import os
import sys

import pytest

# conftest.py adds the indexer directory to sys.path already, but also add
# it here so the test file is importable when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from build_index import load_documents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path, text: str):
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    """load_documents correctly loads supported file types."""

    def test_loads_md_files_without_spacy(self, tmp_path):
        """Markdown files must load via TextLoader (no spaCy / en_core_web_sm)."""
        md_file = tmp_path / "README.md"
        _write(md_file, "# Hello\n\nThis is a markdown document.")

        docs = load_documents(str(tmp_path))

        assert len(docs) == 1
        assert "Hello" in docs[0].page_content

    def test_md_doc_has_source_file_metadata(self, tmp_path):
        """Each loaded document must carry a source_file metadata key."""
        _write(tmp_path / "notes.md", "Some content.")

        docs = load_documents(str(tmp_path))

        assert docs[0].metadata.get("source_file") == "notes.md"

    def test_loads_txt_files(self, tmp_path):
        """Plain text files should be loaded."""
        _write(tmp_path / "data.txt", "Plain text content.")

        docs = load_documents(str(tmp_path))

        assert len(docs) == 1
        assert "Plain text" in docs[0].page_content

    def test_skips_unsupported_extensions(self, tmp_path):
        """Files with unsupported extensions must be silently skipped."""
        _write(tmp_path / "image.png", "not real png")
        _write(tmp_path / "archive.zip", "not real zip")

        docs = load_documents(str(tmp_path))

        assert docs == []

    def test_loads_multiple_file_types(self, tmp_path):
        """Multiple supported file types in the same directory should all load."""
        _write(tmp_path / "readme.md", "# Project")
        _write(tmp_path / "notes.txt", "Some notes.")

        docs = load_documents(str(tmp_path))

        assert len(docs) == 2

    def test_missing_corpus_dir_returns_empty(self, tmp_path):
        """A non-existent directory should return an empty list without raising."""
        docs = load_documents(str(tmp_path / "does_not_exist"))

        assert docs == []

    def test_source_url_populated_from_manifest(self, tmp_path):
        """source_url metadata is set when a source_manifest.json entry exists."""
        import json

        _write(tmp_path / "page.md", "# Page content")
        manifest = {"page.md": "https://example.com/page"}
        (tmp_path / "source_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        docs = load_documents(str(tmp_path))

        assert docs[0].metadata.get("source_url") == "https://example.com/page"
