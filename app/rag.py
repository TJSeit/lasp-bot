"""
rag.py — RAG pipeline for the LASP bot.

On startup:
  1. Load the FAISS index from a local directory.
  2. Build a hybrid retriever combining dense FAISS search with sparse BM25
     keyword search via LangChain's EnsembleRetriever.
  3. Initialise an Ollama client for LLM inference.

On each query:
  1. Route the question: if it is purely conversational (e.g. a greeting),
     send it directly to Ollama without touching the retriever.
  2. Otherwise retrieve the top-K most relevant document chunks using the
     hybrid retriever and call the local Ollama server with the context.

Environment variables (see ../.env.example):
    FAISS_INDEX_DIR   (default: lasp_faiss_index)
    EMBEDDING_MODEL   (default: all-MiniLM-L6-v2)
    OLLAMA_BASE_URL   (default: http://localhost:11434)
    OLLAMA_MODEL      (default: llama3)
    TOP_K             (default: 5)
"""

import os
import re
from typing import Any

import ollama
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "lasp_faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TOP_K = int(os.getenv("TOP_K", "5"))

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the "
    "Laboratory for Atmospheric and Space Physics (LASP). "
    "Users are often technically minded, so prefer precise, technical language "
    "and include relevant details such as units, methodologies, and instrument "
    "names when they appear in the context. "
    "Use only the provided context to answer. "
    "If the answer cannot be found in the context, say so clearly."
)

# ---------------------------------------------------------------------------
# Document relevance router
# ---------------------------------------------------------------------------

# Phrases that identify purely conversational messages that do not require
# any LASP document retrieval.  The pattern is intentionally conservative
# so that short but legitimate technical questions are never skipped.
_CONVERSATIONAL_RE = re.compile(
    r"^\s*(hi|hello|hey|howdy|how are you|good morning|good afternoon|"
    r"good evening|thanks|thank you|bye|goodbye|what'?s up|sup|yo)\b",
    re.IGNORECASE,
)
# If the question contains more than this many words it is unlikely to be
# purely conversational, regardless of how it starts.
_CONVERSATIONAL_MAX_WORDS = 8


def _is_conversational(question: str) -> bool:
    """Return True if the question appears to be casual conversation.

    When True, the retriever is bypassed and the question is forwarded
    directly to Ollama, saving compute cycles on the embedding and FAISS
    search steps.
    """
    if len(question.split()) > _CONVERSATIONAL_MAX_WORDS:
        return False
    return bool(_CONVERSATIONAL_RE.match(question))


def _load_vectorstore(index_dir: str, embedding_model: str) -> FAISS:
    """Load the FAISS vector store from disk using CPU embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
    )
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_rag_chain() -> tuple[Any, ollama.Client]:
    """
    Load the FAISS index, build a hybrid retriever, and initialise an Ollama client.

    The hybrid retriever combines:
    - Dense FAISS search (semantic similarity, good for conceptual questions)
    - Sparse BM25 keyword search (exact term matching, good for acronyms and
      specific identifiers like instrument names or file version numbers)

    Results from both retrievers are merged using Reciprocal Rank Fusion (RRF)
    via LangChain's EnsembleRetriever.

    Returns a (retriever, llm_client) tuple ready to handle queries.
    """
    vectorstore = _load_vectorstore(FAISS_INDEX_DIR, EMBEDDING_MODEL)

    # Dense retriever backed by FAISS.
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Sparse BM25 retriever built from the documents already stored in the
    # FAISS docstore (no extra disk reads required).
    docs = list(vectorstore.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(docs, k=TOP_K)

    # Hybrid retriever: equal weight to semantic and keyword search.
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )

    llm_client = ollama.Client(host=OLLAMA_BASE_URL)
    return retriever, llm_client


def answer_query(
    retriever: Any,
    llm_client: ollama.Client,
    question: str,
    history: list[dict] | None = None,
) -> dict:
    """
    Route the question, optionally retrieve relevant chunks, and call Ollama.

    Conversational questions (e.g. greetings) are forwarded directly to Ollama
    without touching the retriever, cutting latency in half for simple queries.
    All other questions go through the hybrid retriever before LLM inference.

    Args:
        retriever:   LangChain retriever backed by the hybrid FAISS + BM25 index.
        llm_client:  Ollama client used for LLM inference.
        question:    The user's current question.
        history:     Optional list of prior conversation turns, each a dict
                     with ``role`` ("user" or "assistant") and ``content`` keys.
                     When provided these are inserted between the system prompt
                     and the current user message so the model can refer back
                     to earlier exchanges.

    Returns a dict with:
        answer  (str)  — model-generated answer
        sources (list) — list of {source, page, source_url} dicts for cited
                         chunks (empty list when the router bypasses retrieval)
    """
    # --- Document relevance router -------------------------------------------
    # Bypass the retriever entirely for purely conversational messages to avoid
    # wasting compute on embedding + FAISS search for simple exchanges.
    if _is_conversational(question):
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *(history or []),
            {"role": "user", "content": question},
        ]
        response = llm_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.0, "num_predict": 1024},
        )
        return {"answer": response.message.content, "sources": []}

    # --- Standard RAG path ---------------------------------------------------
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = [
        {
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", ""),
            "source_url": doc.metadata.get("source_url", ""),
        }
        for doc in docs
    ]

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    response = llm_client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.0, "num_predict": 1024},
    )
    answer = response.message.content
    return {"answer": answer, "sources": sources}
