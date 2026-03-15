# LASP Bot — Local RAG Application

A retrieval-augmented generation (RAG) chatbot that answers questions about the
**Laboratory for Atmospheric and Space Physics (LASP)** documentation.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Local machine (NVIDIA GPU recommended)                      │
│                                                              │
│  indexer/build_index.py                                      │
│    ├─ Loads PDFs with LangChain PyPDFDirectoryLoader         │
│    ├─ Splits text into overlapping chunks                    │
│    ├─ Generates embeddings with sentence-transformers (GPU)  │
│    └─ Saves FAISS index to local directory                   │
└─────────────────────┬────────────────────────────────────────┘
                      │ FAISS index (index.faiss + index.pkl)
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Local machine                                               │
│                                                              │
│  app/main.py  FastAPI                                        │
│    POST /query                                               │
│      ├─ Embed question (sentence-transformers, CPU)          │
│      ├─ Retrieve top-K chunks from FAISS                     │
│      └─ Call local Ollama server (llama3, mistral, etc.)     │
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

| Tool | Purpose |
|------|---------|
| Python 3.11+ | All scripts and the app |
| NVIDIA GPU + CUDA | Fast embedding generation (indexer only; CPU fallback available) |
| [Ollama](https://ollama.com) | Local LLM inference server |
| Docker (optional) | Build & run the app image |

---

## Quick Start

### 1 — Install and start Ollama

```bash
# Install Ollama: https://ollama.com/download
# Pull a model (e.g. llama3, mistral, gemma3)
ollama pull llama3
# Ollama starts automatically; verify with:
ollama list
```

### 2 — Configure environment variables

```bash
cp .env.example .env
# Edit .env if you want non-default values
```

| Variable | Description |
|----------|-------------|
| `FAISS_INDEX_DIR` | Local path to the FAISS index directory (default: `lasp_faiss_index`) |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | Model name as shown by `ollama list` (default: `llama3`) |
| `EMBEDDING_MODEL` | sentence-transformers model (default: `all-MiniLM-L6-v2`) |
| `TOP_K` | Chunks to retrieve per query (default: `5`) |
| `CHUNK_SIZE` | Token chunk size for indexer (default: `512`) |
| `CHUNK_OVERLAP` | Chunk overlap for indexer (default: `64`) |

### 3 — Build the FAISS index (local, GPU)

```bash
cd indexer
pip install -r requirements.txt
# For CUDA 12 GPU support:
# pip install faiss-gpu-cu12
python build_index.py /path/to/lasp/pdfs
```

The script loads every `.pdf` in the given directory, splits pages into chunks,
generates embeddings on the local GPU, and saves a FAISS index to
`lasp_faiss_index/` (or the path specified with `--output`).

### 4 — Run the FastAPI app locally

```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

Example query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What instruments does LASP operate?"}'
```

### 5 — Build & run the Docker image (optional)

```bash
cd app
docker build -t lasp-bot:latest .
docker run -p 8000:8000 \
  -v /path/to/lasp_faiss_index:/app/lasp_faiss_index \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  lasp-bot:latest
```

---

## API Reference

### `POST /query`

Request body:

```json
{ "question": "What does the MAVEN mission study?" }
```

Response:

```json
{
  "answer": "MAVEN studies the Martian upper atmosphere and its interaction with the solar wind.",
  "sources": [
    { "source": "maven_overview.pdf", "page": 4 }
  ]
}
```

### `GET /health`

Returns `{"status": "ok"}` once the RAG chain is initialised.

---

## Running Tests

```bash
cd app
pip install -r requirements.txt pytest httpx
pytest tests/ -v
```

---

## Repository Structure

```
lasp-bot/
├── indexer/
│   ├── build_index.py   # Local GPU script: PDFs → FAISS (saved locally)
│   └── requirements.txt
├── app/
│   ├── main.py          # FastAPI application
│   ├── rag.py           # RAG pipeline (local FAISS + Ollama)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       └── test_rag.py
├── .env.example
└── README.md
```
