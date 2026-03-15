# LASP Bot — Cost-Optimized RAG Application

A retrieval-augmented generation (RAG) chatbot that answers questions about the
**Laboratory for Atmospheric and Space Physics (LASP)** documentation.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Local machine (NVIDIA GPU)                                  │
│                                                              │
│  indexer/build_index.py                                      │
│    ├─ Loads PDFs with LangChain PyPDFDirectoryLoader         │
│    ├─ Splits text into overlapping chunks                    │
│    ├─ Generates embeddings with sentence-transformers (GPU)  │
│    └─ Builds FAISS index → uploads to Azure Blob Storage     │
└─────────────────────┬────────────────────────────────────────┘
                      │ FAISS index (index.faiss + index.pkl)
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Azure Blob Storage  (lasp-index container)                  │
└─────────────────────┬────────────────────────────────────────┘
                      │ download on startup
                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Azure Container Apps  (Serverless)                          │
│                                                              │
│  app/main.py  FastAPI                                        │
│    POST /query                                               │
│      ├─ Embed question (sentence-transformers, CPU)          │
│      ├─ Retrieve top-K chunks from FAISS                     │
│      └─ Call Azure AI Foundry Serverless API                 │
│           (Llama 3 or GPT-4o-mini)                           │
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

| Tool | Purpose |
|------|---------|
| Python 3.11+ | All scripts and the app |
| NVIDIA GPU + CUDA | Fast embedding generation (indexer only) |
| Azure Storage Account | Stores the FAISS index |
| Azure AI Foundry deployment | Serverless LLM endpoint |
| Docker | Build & push the app image |
| Azure Container Apps | Host the FastAPI app |

---

## Quick Start

### 1 — Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in every value
```

| Variable | Description |
|----------|-------------|
| `AZURE_STORAGE_CONNECTION_STRING` | Connection string for the Azure Storage account |
| `AZURE_STORAGE_CONTAINER_NAME` | Blob container name (default: `lasp-index`) |
| `INDEX_BLOB_PREFIX` | Blob path prefix for the FAISS files (default: `faiss_index`) |
| `AZURE_AI_ENDPOINT` | Azure AI Foundry Serverless endpoint URL |
| `AZURE_AI_API_KEY` | Azure AI Foundry API key |
| `AZURE_AI_MODEL` | Model name, e.g. `Meta-Llama-3-8B-Instruct` or `gpt-4o-mini` |
| `EMBEDDING_MODEL` | sentence-transformers model (default: `all-MiniLM-L6-v2`) |
| `TOP_K` | Chunks to retrieve per query (default: `5`) |
| `CHUNK_SIZE` | Token chunk size for indexer (default: `512`) |
| `CHUNK_OVERLAP` | Chunk overlap for indexer (default: `64`) |

### 2 — Build the FAISS index (local, GPU)

```bash
cd indexer
pip install -r requirements.txt
# For CUDA 12 GPU support:
# pip install faiss-gpu-cu12
python build_index.py /path/to/lasp/pdfs
```

The script loads every `.pdf` in the given directory, splits pages into chunks,
generates embeddings on the local GPU, builds a FAISS index, and uploads
`index.faiss` + `index.pkl` to Azure Blob Storage.

### 3 — Run the FastAPI app locally

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

### 4 — Build & push the Docker image

```bash
cd app
docker build -t lasp-bot:latest .

# Push to Azure Container Registry (replace with your registry)
az acr login --name <your-registry>
docker tag lasp-bot:latest <your-registry>.azurecr.io/lasp-bot:latest
docker push <your-registry>.azurecr.io/lasp-bot:latest
```

### 5 — Deploy to Azure Container Apps

```bash
az containerapp create \
  --name lasp-bot \
  --resource-group <rg> \
  --environment <env-name> \
  --image <your-registry>.azurecr.io/lasp-bot:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 5 \
  --env-vars \
      AZURE_STORAGE_CONNECTION_STRING=secretref:storage-conn \
      AZURE_STORAGE_CONTAINER_NAME=lasp-index \
      INDEX_BLOB_PREFIX=faiss_index \
      AZURE_AI_ENDPOINT=secretref:ai-endpoint \
      AZURE_AI_API_KEY=secretref:ai-key \
      AZURE_AI_MODEL=Meta-Llama-3-8B-Instruct \
      EMBEDDING_MODEL=all-MiniLM-L6-v2 \
      TOP_K=5
```

> **Tip:** Store sensitive values as Container Apps secrets with `az containerapp secret set`.

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
│   ├── build_index.py   # Local GPU script: PDFs → FAISS → Azure Blob
│   └── requirements.txt
├── app/
│   ├── main.py          # FastAPI application
│   ├── rag.py           # RAG pipeline (FAISS + Azure AI Foundry)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       └── test_rag.py
├── .env.example
└── README.md
```