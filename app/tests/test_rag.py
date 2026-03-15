"""
Tests for the LASP bot RAG pipeline and FastAPI endpoints.

All external I/O (Azure Blob Storage, Azure AI Foundry, FAISS) is mocked so
the tests run without any cloud credentials or GPU.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import rag
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_fake_doc(content: str, source: str = "lasp_doc.pdf", page: int = 1):
    """Return a minimal langchain Document-like object."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page}
    return doc


def _make_fake_llm_response(text: str):
    """Return a minimal azure-ai-inference response-like object."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# rag.py unit tests
# ---------------------------------------------------------------------------


class TestDownloadFaissIndex:
    """_download_faiss_index downloads two blobs and writes them to disk."""

    def test_downloads_both_files(self, tmp_path):
        mock_blob_service = MagicMock()
        mock_container = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container
        mock_container.download_blob.return_value.readall.return_value = b"data"

        with patch("rag.BlobServiceClient") as MockBS:
            MockBS.from_connection_string.return_value = mock_blob_service
            rag._download_faiss_index("conn", "container", "prefix", str(tmp_path))

        assert mock_container.download_blob.call_count == 2
        called_blobs = {
            call.args[0] for call in mock_container.download_blob.call_args_list
        }
        assert called_blobs == {"prefix/index.faiss", "prefix/index.pkl"}
        assert (tmp_path / "index.faiss").read_bytes() == b"data"
        assert (tmp_path / "index.pkl").read_bytes() == b"data"


class TestAnswerQuery:
    """answer_query retrieves docs and calls the LLM."""

    def test_returns_answer_and_sources(self):
        docs = [
            _make_fake_doc("LASP studies space weather.", source="overview.pdf", page=3),
            _make_fake_doc("Solar wind data is measured at L1.", source="solar.pdf", page=7),
        ]
        retriever = MagicMock()
        retriever.invoke.return_value = docs

        llm_client = MagicMock()
        llm_client.complete.return_value = _make_fake_llm_response(
            "LASP studies space weather and solar wind."
        )

        result = rag.answer_query(retriever, llm_client, "What does LASP study?")

        assert result["answer"] == "LASP studies space weather and solar wind."
        assert len(result["sources"]) == 2
        assert result["sources"][0]["source"] == "overview.pdf"
        assert result["sources"][0]["page"] == 3
        assert result["sources"][1]["source"] == "solar.pdf"

    def test_passes_context_to_llm(self):
        docs = [_make_fake_doc("Important context.")]
        retriever = MagicMock()
        retriever.invoke.return_value = docs

        llm_client = MagicMock()
        llm_client.complete.return_value = _make_fake_llm_response("Answer.")

        rag.answer_query(retriever, llm_client, "My question?")

        call_kwargs = llm_client.complete.call_args
        messages = call_kwargs.kwargs["messages"]
        # The user message should contain both context and the question
        user_msg_content = messages[-1].content
        assert "Important context." in user_msg_content
        assert "My question?" in user_msg_content

    def test_empty_sources_when_no_metadata(self):
        doc = MagicMock()
        doc.page_content = "Some text."
        doc.metadata = {}  # no source / page keys
        retriever = MagicMock()
        retriever.invoke.return_value = [doc]

        llm_client = MagicMock()
        llm_client.complete.return_value = _make_fake_llm_response("OK.")

        result = rag.answer_query(retriever, llm_client, "q?")
        assert result["sources"][0]["source"] == ""
        assert result["sources"][0]["page"] == ""


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------


def _fresh_modules():
    """Remove cached main/rag modules so each test gets a clean import."""
    import sys

    for mod in list(sys.modules.keys()):
        if mod in ("main", "rag"):
            del sys.modules[mod]


@pytest.fixture()
def app_client():
    """
    Pytest fixture that yields a (TestClient, fake_retriever, fake_llm) tuple.
    Uses TestClient as a context manager so the FastAPI lifespan is triggered.
    """
    _fresh_modules()

    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = [
        _make_fake_doc("LASP is located in Boulder, CO.", source="about.pdf", page=1)
    ]
    fake_llm = MagicMock()
    fake_llm.complete.return_value = _make_fake_llm_response(
        "LASP is located in Boulder, Colorado."
    )

    with patch("rag.build_rag_chain", return_value=(fake_retriever, fake_llm)):
        import main as app_module

        with TestClient(app_module.app) as client:
            yield client, fake_retriever, fake_llm


class TestHealthEndpoint:
    def test_health_ok(self, app_client):
        client, _, _ = app_client
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root_ok(self, app_client):
        client, _, _ = app_client
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["service"] == "lasp-bot"


class TestQueryEndpoint:
    def test_successful_query(self, app_client):
        client, _, _ = app_client
        resp = client.post("/query", json={"question": "Where is LASP located?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "Boulder" in body["answer"]

    def test_empty_question_rejected(self, app_client):
        client, _, _ = app_client
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422  # Pydantic validation error

    def test_missing_question_rejected(self, app_client):
        client, _, _ = app_client
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_llm_error_returns_500(self):
        _fresh_modules()

        fake_retriever = MagicMock()
        fake_retriever.invoke.return_value = [_make_fake_doc("ctx")]
        fake_llm = MagicMock()
        fake_llm.complete.side_effect = RuntimeError("LLM unavailable")

        with patch("rag.build_rag_chain", return_value=(fake_retriever, fake_llm)):
            import main as app_module

            with TestClient(app_module.app, raise_server_exceptions=False) as client:
                resp = client.post("/query", json={"question": "test?"})
        assert resp.status_code == 500
