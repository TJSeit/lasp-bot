"""
Tests for the LASP bot RAG pipeline and FastAPI endpoints.

All external I/O (FAISS, Ollama) is mocked so the tests run without any
running services or GPU.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import rag
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_fake_doc(content: str, source: str = "lasp_doc.pdf", page: int = 1, source_url: str = ""):
    """Return a minimal langchain Document-like object."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page, "source_url": source_url}
    return doc


def _make_fake_llm_response(text: str):
    """Return a minimal ollama chat response-like object."""
    response = MagicMock()
    response.message.content = text
    return response


# ---------------------------------------------------------------------------
# rag.py unit tests
# ---------------------------------------------------------------------------


class TestBuildRagChain:
    """build_rag_chain loads the FAISS index from a local directory."""

    def test_loads_vectorstore_from_local_dir(self):
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever

        with patch("rag._load_vectorstore", return_value=mock_vectorstore) as mock_load, \
             patch("rag.ollama.Client") as MockClient:
            retriever, llm_client = rag.build_rag_chain()

        mock_load.assert_called_once_with(rag.FAISS_INDEX_DIR, rag.EMBEDDING_MODEL)
        mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": rag.TOP_K})
        MockClient.assert_called_once_with(host=rag.OLLAMA_BASE_URL)
        assert retriever is mock_retriever


class TestAnswerQuery:
    """answer_query retrieves docs and calls the LLM."""

    def test_returns_answer_and_sources(self):
        docs = [
            _make_fake_doc(
                "LASP studies space weather.",
                source="overview.pdf",
                page=3,
                source_url="https://lasp.colorado.edu/overview",
            ),
            _make_fake_doc(
                "Solar wind data is measured at L1.",
                source="solar.pdf",
                page=7,
                source_url="https://lasp.colorado.edu/solar",
            ),
        ]
        retriever = MagicMock()
        retriever.invoke.return_value = docs

        llm_client = MagicMock()
        llm_client.chat.return_value = _make_fake_llm_response(
            "LASP studies space weather and solar wind."
        )

        result = rag.answer_query(retriever, llm_client, "What does LASP study?")

        assert result["answer"] == "LASP studies space weather and solar wind."
        assert len(result["sources"]) == 2
        assert result["sources"][0]["source"] == "overview.pdf"
        assert result["sources"][0]["page"] == 3
        assert result["sources"][0]["source_url"] == "https://lasp.colorado.edu/overview"
        assert result["sources"][1]["source"] == "solar.pdf"

    def test_passes_context_to_llm(self):
        docs = [_make_fake_doc("Important context.")]
        retriever = MagicMock()
        retriever.invoke.return_value = docs

        llm_client = MagicMock()
        llm_client.chat.return_value = _make_fake_llm_response("Answer.")

        rag.answer_query(retriever, llm_client, "My question?")

        call_kwargs = llm_client.chat.call_args
        messages = call_kwargs.kwargs["messages"]
        # The user message should contain both context and the question
        user_msg_content = messages[-1]["content"]
        assert "Important context." in user_msg_content
        assert "My question?" in user_msg_content

    def test_system_prompt_instructs_technical_answers(self):
        docs = [_make_fake_doc("Some context.")]
        retriever = MagicMock()
        retriever.invoke.return_value = docs

        llm_client = MagicMock()
        llm_client.chat.return_value = _make_fake_llm_response("Answer.")

        rag.answer_query(retriever, llm_client, "A question?")

        call_kwargs = llm_client.chat.call_args
        messages = call_kwargs.kwargs["messages"]
        system_msg_content = messages[0]["content"]
        # System prompt should guide the LLM toward technical responses
        assert "technically" in system_msg_content.lower()
        assert messages[0]["role"] == "system"

    def test_empty_sources_when_no_metadata(self):
        doc = MagicMock()
        doc.page_content = "Some text."
        doc.metadata = {}  # no source / page keys
        retriever = MagicMock()
        retriever.invoke.return_value = [doc]

        llm_client = MagicMock()
        llm_client.chat.return_value = _make_fake_llm_response("OK.")

        result = rag.answer_query(retriever, llm_client, "q?")
        assert result["sources"][0]["source"] == ""
        assert result["sources"][0]["page"] == ""
        assert result["sources"][0]["source_url"] == ""


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
    fake_llm.chat.return_value = _make_fake_llm_response(
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
        fake_llm.chat.side_effect = RuntimeError("LLM unavailable")

        with patch("rag.build_rag_chain", return_value=(fake_retriever, fake_llm)):
            import main as app_module

            with TestClient(app_module.app, raise_server_exceptions=False) as client:
                resp = client.post("/query", json={"question": "test?"})
        assert resp.status_code == 500
