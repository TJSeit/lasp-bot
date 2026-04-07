"""
main.py — FastAPI application for the LASP bot.

Endpoints:
    GET  /          — health check
    GET  /health    — health check (JSON)
    POST /query     — accept a question, return answer + source citations

The RAG chain (FAISS retriever + Ollama LLM client) is initialised
once on startup via a lifespan context manager so it is shared across requests.
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lasp_mcp import run_in_background
from rag import answer_query, build_rag_chain

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index from local storage and initialise the RAG chain."""
    if os.getenv("MCP_ENABLED", "true").lower() != "false":
        run_in_background()
    retriever, llm_client = build_rag_chain()
    _state["retriever"] = retriever
    _state["llm_client"] = llm_client
    yield
    _state.clear()


app = FastAPI(
    title="LASP Bot",
    description=(
        "A RAG application that answers questions about the "
        "Laboratory for Atmospheric and Space Physics (LASP) using a local FAISS "
        "vector index and a locally-hosted Ollama LLM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(description="Role of the message author")
    content: str = Field(min_length=1, description="Message content")


class QueryRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=2000, description="Question to ask the LASP bot")]
    history: list[ConversationTurn] = Field(
        default=[],
        description="Prior conversation turns to include for multi-turn dialogue",
    )


class SourceReference(BaseModel):
    source: str = Field(description="PDF filename or path")
    page: str | int = Field(description="Page number within the PDF")
    source_url: str = Field(default="", description="Original URL for the source chunk when available")


class QueryResponse(BaseModel):
    answer: str = Field(description="Model-generated answer")
    sources: list[SourceReference] = Field(description="Source chunks used to generate the answer")


@app.get("/", include_in_schema=False)
async def root():
    return {"status": "ok", "service": "lasp-bot"}


@app.get("/health", summary="Health check")
async def health():
    """Returns 200 when the service is running and the RAG chain is loaded."""
    if "retriever" not in _state:
        raise HTTPException(status_code=503, detail="RAG chain not yet initialised.")
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, summary="Query the LASP documentation")
async def query(request: QueryRequest):
    """
    Submit a natural-language question.  The service retrieves relevant chunks
    from the FAISS index and generates an answer using the locally-hosted
    Ollama LLM.
    """
    if "retriever" not in _state:
        raise HTTPException(status_code=503, detail="RAG chain not yet initialised.")
    try:
        history = [{"role": t.role, "content": t.content} for t in request.history]
        result = answer_query(_state["retriever"], _state["llm_client"], request.question, history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(**result)
