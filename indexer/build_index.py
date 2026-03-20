"""
build_index.py — Build a FAISS vector index from a directory of LASP PDFs and
save it to a local directory.

Run this script locally (NVIDIA GPU recommended) before running the FastAPI app.

Usage:
    python build_index.py /path/to/lasp/pdfs

Environment variables (see ../.env.example):
    EMBEDDING_MODEL   (default: all-MiniLM-L6-v2)
    CHUNK_SIZE        (default: 512)
    CHUNK_OVERLAP     (default: 64)
"""

import os
import argparse
import logging
import json
from pathlib import Path


import torch

from dotenv import load_dotenv
load_dotenv()

# LangChain Imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredXMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# Suppress noisy pypdf warnings about malformed PDF internal references
# (e.g. "Ignoring wrong pointing object") — pypdf handles these gracefully.
logging.getLogger("pypdf").setLevel(logging.ERROR)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))


def _load_source_manifest(corpus_path: Path) -> dict[str, str]:
    manifest_path = corpus_path / "source_manifest.json"
    if not manifest_path.exists():
        logging.warning("No source manifest found; source URLs will be missing for some chunks.")
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        logging.warning(f"Failed to read source manifest: {e}")
    return {}


def _manifest_key_for_path(corpus_path: Path, filepath: Path) -> str:
    return str(filepath.relative_to(corpus_path)).replace('\\', '/')

def load_documents(corpus_dir):
    """Recursively load documents from the corpus directory based on file type."""
    documents = []
    corpus_path = Path(corpus_dir)
    source_manifest = _load_source_manifest(corpus_path)
    
    if not corpus_path.exists():
        logging.error(f"Corpus directory '{corpus_dir}' does not exist.")
        return documents

    # Define how to load specific extensions
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': TextLoader,
        '.xml': UnstructuredXMLLoader,
        '.lbl': TextLoader # Treat NASA PDS label files as plain text
    }

    logging.info(f"Scanning {corpus_dir} for documents...")
    
    for filepath in corpus_path.rglob('*'):
        if filepath.is_file():
            ext = filepath.suffix.lower()
            if ext in loaders:
                try:
                    logging.info(f"Loading: {filepath.name}")
                    loader_class = loaders[ext]
                    # TextLoader requires explicit encoding to prevent crash on weird characters
                    if loader_class == TextLoader:
                        loader = loader_class(str(filepath), autodetect_encoding=True)
                    else:
                        loader = loader_class(str(filepath))

                    loaded_docs = loader.load()
                    source_url = source_manifest.get(_manifest_key_for_path(corpus_path, filepath), "")
                    for doc in loaded_docs:
                        doc.metadata["source_file"] = filepath.name
                        if source_url:
                            doc.metadata["source_url"] = source_url
                    documents.extend(loaded_docs)
                except Exception as e:
                    logging.warning(f"Failed to load {filepath.name}: {e}")
            else:
                logging.debug(f"Skipping unsupported file type: {filepath.name}")
                
    logging.info(f"Successfully loaded {len(documents)} document pages/sections.")
    return documents

def build_index(corpus_dir, output_dir="lasp_faiss_index"):
    # 1. Load Documents
    docs = load_documents(corpus_dir)
    if not docs:
        logging.error("No documents loaded. Exiting.")
        return

    # 2. Split Documents into Chunks
    # LASP docs are highly technical; larger overlap ensures context isn't lost between pages
    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    logging.info(f"Created {len(chunks)} text chunks.")

    # 3. Initialize Embeddings (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        logging.warning(
            "CUDA is not available. If you have an NVIDIA GPU, ensure you have "
            "installed the CUDA-enabled PyTorch wheel (the CPU-only wheel is "
            "installed by default when torch is resolved from PyPI). "
            "Re-run: pip install -r requirements.txt  — the requirements file "
            "includes --extra-index-url https://download.pytorch.org/whl/cu124 "
            "which provides the correct CUDA build. Falling back to CPU "
            "(indexing will be significantly slower)."
        )
    logging.info(f"Initializing HuggingFace Embeddings (model={EMBEDDING_MODEL}) on {device.upper()}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True} # Better for cosine similarity
    )

    # 4. Build and Save FAISS Index
    logging.info("Building FAISS vector index (this may take a few minutes on your GPU)...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(output_dir, exist_ok=True)
    vector_db.save_local(output_dir)
    logging.info(f"SUCCESS: FAISS index saved locally to '{output_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS vector index from the LASP corpus.")
    parser.add_argument("corpus_dir", help="Path to the directory containing scraped LASP data (e.g., lasp_corpus)")
    parser.add_argument("--output", default="lasp_faiss_index", help="Output directory for the FAISS index files")
    
    args = parser.parse_args()
    build_index(args.corpus_dir, args.output)