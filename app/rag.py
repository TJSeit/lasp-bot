"""
rag.py — RAG pipeline for the LASP bot.

On startup:
  1. Load the FAISS index from a local directory.
  2. Initialise an Ollama client for LLM inference.

On each query:
  1. Retrieve the top-K most relevant document chunks.
  2. Call the local Ollama server with the retrieved context and return the
     generated answer.

Environment variables (see ../.env.example):
    FAISS_INDEX_DIR   (default: lasp_faiss_index)
    EMBEDDING_MODEL   (default: all-MiniLM-L6-v2)
    OLLAMA_BASE_URL   (default: http://localhost:11434)
    OLLAMA_MODEL      (default: llama3)
    TOP_K             (default: 5)
"""

import os
from typing import Any

import ollama
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
    Load the FAISS index from a local directory and initialise an Ollama client.

    Returns a (retriever, llm_client) tuple ready to handle queries.
    """
    vectorstore = _load_vectorstore(FAISS_INDEX_DIR, EMBEDDING_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm_client = ollama.Client(host=OLLAMA_BASE_URL)
    return retriever, llm_client


def answer_query(
    retriever: Any,
    llm_client: ollama.Client,
    question: str,
    history: list[dict] | None = None,
) -> dict:
    """
    Retrieve relevant chunks and call the local Ollama model.

    Args:
        retriever:   LangChain retriever backed by the FAISS index.
        llm_client:  Ollama client used for LLM inference.
        question:    The user's current question.
        history:     Optional list of prior conversation turns, each a dict
                     with ``role`` ("user" or "assistant") and ``content`` keys.
                     When provided these are inserted between the system prompt
                     and the current user message so the model can refer back
                     to earlier exchanges.

    Returns a dict with:
        answer  (str)  — model-generated answer
        sources (list) — list of {source, page} dicts for cited chunks
    """
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

    messages: list[dict] = [
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
