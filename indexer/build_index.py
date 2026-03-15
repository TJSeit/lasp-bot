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

import os
import argparse
import logging
from pathlib import Path

# LangChain Imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_documents(corpus_dir):
    """Recursively load documents from the corpus directory based on file type."""
    documents = []
    corpus_path = Path(corpus_dir)
    
    if not corpus_path.exists():
        logging.error(f"Corpus directory '{corpus_dir}' does not exist.")
        return documents

    # Define how to load specific extensions
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
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
                        
                    documents.extend(loader.load())
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
        chunk_size=750,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    logging.info(f"Created {len(chunks)} text chunks.")

    # 3. Initialize GPU Embeddings
    # BGE-Small is heavily optimized for technical RAG and runs great on local CUDA
    logging.info("Initializing HuggingFace Embeddings on CUDA...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cuda'},
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