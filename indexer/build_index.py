"""
build_index.py — Build a FAISS vector index from a directory of LASP PDFs and
upload it to Azure Blob Storage.

Run this script locally (NVIDIA GPU recommended) before deploying the FastAPI app.

Usage:
    python build_index.py /path/to/lasp/pdfs

Environment variables (see ../.env.example):
    AZURE_STORAGE_CONNECTION_STRING
    AZURE_STORAGE_CONTAINER_NAME   (default: lasp-index)
    INDEX_BLOB_PREFIX              (default: faiss_index)
    EMBEDDING_MODEL                (default: all-MiniLM-L6-v2)
    CHUNK_SIZE                     (default: 512)
    CHUNK_OVERLAP                  (default: 64)
"""

import argparse
import os
import tempfile
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "lasp-index")
INDEX_BLOB_PREFIX = os.getenv("INDEX_BLOB_PREFIX", "faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))


def load_documents(pdf_dir: str):
    """Load all PDFs from the given directory using langchain's PDF loader."""
    print(f"Loading PDFs from: {pdf_dir}")
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from PDF files.")
    return documents


def split_documents(documents):
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s).")
    return chunks


def build_faiss_index(chunks, embedding_model: str) -> FAISS:
    """Generate embeddings (GPU) and build a FAISS vector store."""
    print(f"Generating embeddings with model '{embedding_model}' (device=cuda)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"},  # use local NVIDIA GPU
        encode_kwargs={"batch_size": 64},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("FAISS index built.")
    return vectorstore


def upload_index_to_azure(
    vectorstore: FAISS,
    connection_string: str,
    container_name: str,
    prefix: str,
) -> None:
    """Save the FAISS index to a temp directory and upload both files to Azure Blob Storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vectorstore.save_local(tmp_dir)

        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service.get_container_client(container_name)

        try:
            container_client.create_container()
            print(f"Created blob container '{container_name}'.")
        except Exception:
            pass  # container already exists

        for filename in ["index.faiss", "index.pkl"]:
            local_path = Path(tmp_dir) / filename
            blob_name = f"{prefix}/{filename}"
            print(f"Uploading '{filename}' → blob '{blob_name}' ...")
            with open(local_path, "rb") as fh:
                container_client.upload_blob(name=blob_name, data=fh, overwrite=True)

    print(
        f"Index successfully uploaded to container '{container_name}' "
        f"under prefix '{prefix}'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from LASP PDFs and upload to Azure Blob Storage."
    )
    parser.add_argument(
        "pdf_dir",
        help="Path to the directory containing LASP PDF files.",
    )
    args = parser.parse_args()

    documents = load_documents(args.pdf_dir)
    chunks = split_documents(documents)
    vectorstore = build_faiss_index(chunks, EMBEDDING_MODEL)
    upload_index_to_azure(
        vectorstore,
        AZURE_STORAGE_CONNECTION_STRING,
        AZURE_STORAGE_CONTAINER_NAME,
        INDEX_BLOB_PREFIX,
    )


if __name__ == "__main__":
    main()
