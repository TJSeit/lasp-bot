"""
rag.py — RAG pipeline for the LASP bot.

On startup:
  1. Download the FAISS index from Azure Blob Storage.
  2. Load it into a LangChain FAISS vector store (CPU embeddings).

On each query:
  1. Retrieve the top-K most relevant document chunks.
  2. Call the Azure AI Foundry Serverless API (Llama 3 / GPT-4o-mini) with the
     retrieved context and return the generated answer.

Environment variables (see ../.env.example):
    AZURE_STORAGE_CONNECTION_STRING
    AZURE_STORAGE_CONTAINER_NAME   (default: lasp-index)
    INDEX_BLOB_PREFIX              (default: faiss_index)
    EMBEDDING_MODEL                (default: all-MiniLM-L6-v2)
    AZURE_AI_ENDPOINT
    AZURE_AI_API_KEY
    AZURE_AI_MODEL                 (default: Meta-Llama-3-8B-Instruct)
    TOP_K                          (default: 5)
"""

import os
import tempfile
from pathlib import Path
from typing import Any

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "lasp-index")
INDEX_BLOB_PREFIX = os.getenv("INDEX_BLOB_PREFIX", "faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
AZURE_AI_ENDPOINT = os.environ.get("AZURE_AI_ENDPOINT", "")
AZURE_AI_API_KEY = os.environ.get("AZURE_AI_API_KEY", "")
AZURE_AI_MODEL = os.getenv("AZURE_AI_MODEL", "Meta-Llama-3-8B-Instruct")
TOP_K = int(os.getenv("TOP_K", "5"))

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the "
    "Laboratory for Atmospheric and Space Physics (LASP). "
    "Use only the provided context to answer. "
    "If the answer cannot be found in the context, say so clearly."
)


def _download_faiss_index(
    connection_string: str,
    container_name: str,
    prefix: str,
    dest_dir: str,
) -> None:
    """Download index.faiss and index.pkl from Azure Blob Storage."""
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)
    for filename in ["index.faiss", "index.pkl"]:
        blob_name = f"{prefix}/{filename}"
        local_path = Path(dest_dir) / filename
        blob_data = container_client.download_blob(blob_name).readall()
        local_path.write_bytes(blob_data)


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


def build_rag_chain() -> tuple[Any, ChatCompletionsClient]:
    """
    Download the FAISS index from Azure Blob Storage, load it into memory, and
    return a (retriever, llm_client) tuple ready to handle queries.

    The temporary directory used to stage the index files is cleaned up
    immediately after the vector store is loaded into memory.
    """
    with tempfile.TemporaryDirectory(prefix="lasp_faiss_") as tmp_dir:
        _download_faiss_index(
            AZURE_STORAGE_CONNECTION_STRING,
            AZURE_STORAGE_CONTAINER_NAME,
            INDEX_BLOB_PREFIX,
            tmp_dir,
        )
        vectorstore = _load_vectorstore(tmp_dir, EMBEDDING_MODEL)

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm_client = ChatCompletionsClient(
        endpoint=AZURE_AI_ENDPOINT,
        credential=AzureKeyCredential(AZURE_AI_API_KEY),
    )
    return retriever, llm_client


def answer_query(
    retriever: Any,
    llm_client: ChatCompletionsClient,
    question: str,
) -> dict:
    """
    Retrieve relevant chunks and call the Azure AI Foundry model.

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

    response = llm_client.complete(
        model=AZURE_AI_MODEL,
        messages=[
            SystemMessage(content=_SYSTEM_PROMPT),
            UserMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content
    return {"answer": answer, "sources": sources}
