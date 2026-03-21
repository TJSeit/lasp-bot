"""
Tests for build_index.load_documents.

All file I/O is performed against a temporary directory so the tests run
without any network access or GPU.
"""

import logging
import os
import sys
from unittest.mock import patch, MagicMock

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

    def test_source_url_populated_from_manifest_in_subdir(self, tmp_path):
        """source_url is resolved correctly for files in sub-directories.

        On Windows, Path.relative_to() uses backslash separators.  The
        manifest keys use forward slashes (e.g. ``subdir/file.md``).  The old
        code called ``.replace('\\\\', '/')`` which, at runtime, only matched
        the two-character sequence ``\\`` (double-backslash); it never matched
        the single backslash ``\\`` Windows path separator, so the manifest
        lookup silently returned an empty string.  The fix changes the pattern
        to ``.replace('\\', '/')`` so individual Windows separators are
        normalised to forward slashes before the lookup.
        """
        import json

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        _write(subdir / "doc.md", "# Sub document")
        manifest = {"subdir/doc.md": "https://example.com/subdoc"}
        (tmp_path / "source_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        docs = load_documents(str(tmp_path))

        assert docs[0].metadata.get("source_url") == "https://example.com/subdoc"


class TestPypdfWarningsSuppressed:
    """pypdf 'wrong pointing object' warnings must not reach the console."""

    def test_pypdf_logger_level_is_error(self):
        """The pypdf logger must be configured at ERROR level so that
        'Ignoring wrong pointing object' warnings are silenced.

        build_index is imported at module level above, so its logger
        configuration code has already executed by the time this test runs.
        """
        pypdf_logger = logging.getLogger("pypdf")
        assert pypdf_logger.level == logging.ERROR


class TestBuildIndexDeviceSelection:
    """build_index selects the correct device depending on CUDA availability."""

    def _run_build_index(self, tmp_path, cuda_available: bool):
        """Helper: run build_index with a single .txt document and the given
        CUDA availability, mocking out HuggingFaceEmbeddings and FAISS so that
        no real GPU or model download is required.

        Returns (all_hf_calls, mock_faiss) so callers can inspect which
        HuggingFaceEmbeddings instances were created and which FAISS factory
        method was invoked.
        """
        import build_index as bi

        (tmp_path / "doc.txt").write_text("Hello world.", encoding="utf-8")

        all_hf_calls: list[dict] = []

        def fake_hf_embeddings(**kwargs):
            all_hf_calls.append(kwargs)
            mock_emb = MagicMock()
            # embed_documents must return a list of vectors (one per document).
            mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3]]
            return mock_emb

        mock_vector_db = MagicMock()

        with patch("build_index.torch.cuda.is_available", return_value=cuda_available), \
             patch("build_index.HuggingFaceEmbeddings", side_effect=fake_hf_embeddings), \
             patch("build_index.FAISS") as mock_faiss:
            mock_faiss.from_documents.return_value = mock_vector_db
            mock_faiss.from_embeddings.return_value = mock_vector_db
            bi.build_index(str(tmp_path), output_dir=str(tmp_path / "index"))

        return all_hf_calls, mock_faiss

    def test_uses_cuda_when_available(self, tmp_path):
        """When CUDA is available, the first embedding model is placed on 'cuda'."""
        all_hf_calls, _ = self._run_build_index(tmp_path, cuda_available=True)
        # The first HuggingFaceEmbeddings call must use the GPU for fast embedding.
        assert all_hf_calls[0].get("model_kwargs", {}).get("device") == "cuda"

    def test_cuda_path_uses_cpu_embeddings_for_index(self, tmp_path):
        """When CUDA is available, the FAISS index is built with a CPU embeddings
        wrapper to guarantee cross-platform (Mac/CPU) serialization compatibility."""
        all_hf_calls, mock_faiss = self._run_build_index(tmp_path, cuda_available=True)
        # A second HuggingFaceEmbeddings instance on 'cpu' must be created to
        # wrap the index so that save_local() writes a CPU-compatible file.
        assert len(all_hf_calls) == 2
        assert all_hf_calls[1].get("model_kwargs", {}).get("device") == "cpu"
        # FAISS.from_embeddings (not from_documents) is used in the CUDA path.
        mock_faiss.from_embeddings.assert_called_once()
        mock_faiss.from_documents.assert_not_called()

    def test_falls_back_to_cpu_when_cuda_unavailable(self, tmp_path):
        """When CUDA is unavailable, the embedding model falls back to 'cpu'."""
        all_hf_calls, mock_faiss = self._run_build_index(tmp_path, cuda_available=False)
        assert all_hf_calls[0].get("model_kwargs", {}).get("device") == "cpu"
        # CPU path uses the standard FAISS.from_documents path.
        mock_faiss.from_documents.assert_called_once()
        mock_faiss.from_embeddings.assert_not_called()
