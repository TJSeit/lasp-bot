# LASP Bot — Local RAG Application

<p align="center">
  <img src="https://github.com/user-attachments/assets/7c1cc5f0-2df9-48ec-b50e-c2f5fe8b307f" alt="lasp-bot logo" width="300"/>
</p>

A retrieval-augmented generation (RAG) chatbot that answers questions about the
**Laboratory for Atmospheric and Space Physics (LASP)** documentation.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Local machine (NVIDIA GPU recommended)                      │
│                                                              │
│  indexer/build_index.py                                      │
│    ├─ Loads PDFs, text, XML, and label files (multi-format)  │
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
│                                                              │
│  app/discord_bot.py  Discord bot                             │
│    !ask / !lasp <question>                                   │
│      ├─ Embed question (sentence-transformers, CPU)          │
│      ├─ Retrieve top-K chunks from FAISS                     │
│      └─ Call local Ollama server (llama3, mistral, etc.)     │
│                                                              │
│  app/lasp_mcp.py  MCP server (co-located, port 8001)        │
│    Starts automatically alongside FastAPI or Discord bot     │
│      ├─ LISIRD solar irradiance tools                        │
│      ├─ MMS SDC magnetospheric data tools                    │
│      ├─ AIM CIPS mesospheric cloud tools                     │
│      └─ HAPI heliophysics time-series tool                   │
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
| `DISCORD_TOKEN` | Discord bot token — required for `discord_bot.py` |
| `DISCORD_COMMAND_PREFIX` | Command prefix for Discord bot (default: `!`) |
| `LISIRD_BASE_URL` | LaTiS DAP2 API base URL (default: `https://lasp.colorado.edu/lisird/latis/dap2`) |
| `LISIRD_TIMEOUT` | HTTP timeout for LISIRD requests in seconds (default: `30`) |
| `MMS_SDC_BASE_URL` | MMS SDC API base URL (default: `https://lasp.colorado.edu/mms/sdc/public/files/api/v1`) |
| `MMS_SDC_TIMEOUT` | HTTP timeout for MMS SDC requests in seconds (default: `30`) |
| `AIM_BASE_URL` | AIM CIPS LaTiS DAP2 API base URL (default: `https://lasp.colorado.edu/aim/latis/dap2`) |
| `AIM_TIMEOUT` | HTTP timeout for AIM CIPS requests in seconds (default: `30`) |
| `HAPI_TIMEOUT` | HTTP timeout for HAPI server requests in seconds (default: `30`) |

### 3 — Build the LASP corpus (local)

Before indexing you need a local corpus directory containing LASP documents.
Use `build_corpus.py` (recommended) for a comprehensive multi-format corpus, or
`scrape_pdfs.py` if you only need PDFs.

```bash
cd indexer
pip install -r requirements.txt

# Recommended: full corpus (HTML text, PDFs, PDS labels, XML, GitHub docs)
python build_corpus.py          # writes to lasp_corpus/ by default

# Alternative: PDFs only
python scrape_pdfs.py           # writes to lasp_pdfs/ by default
```

`build_corpus.py` saves a `source_manifest.json` that maps every file to its
original URL so source links appear in bot answers.

### 4 — Build the FAISS index (local, GPU)

```bash
cd indexer
pip install -r requirements.txt
# For CUDA 12 GPU support:
# pip install faiss-gpu-cu12
python build_index.py /path/to/lasp/corpus
```

The script loads every `.pdf`, `.txt`, `.md`, `.xml`, and `.lbl` file in the
given directory, splits pages into chunks, generates embeddings on the local
GPU, and saves a FAISS index to `lasp_faiss_index/` (or the path specified
with `--output`).

### 5 — Run the FastAPI app locally

Before starting the app, make sure `FAISS_INDEX_DIR` in your `.env` file points
to the index directory created in step 4.  By default the indexer saves the
index to `lasp_faiss_index/` **inside the `indexer/` directory**, but the app
runs from the `app/` directory, so the paths don't line up automatically.

Open your `.env` file and set an absolute path (recommended) or a path
relative to the `app/` directory:

**Linux / macOS**
```dotenv
FAISS_INDEX_DIR=/absolute/path/to/indexer/lasp_faiss_index
# — or relative from app/ —
FAISS_INDEX_DIR=../indexer/lasp_faiss_index
```

**Windows** (use forward slashes — Python accepts them on Windows and they avoid `.env` parsing issues)
```dotenv
FAISS_INDEX_DIR=C:/Users/you/lasp-bot/indexer/lasp_faiss_index
```

> **Tip (Windows PowerShell):** you can set the variable for the current
> session without editing `.env`:
> ```powershell
> $env:FAISS_INDEX_DIR = "C:/Users/you/lasp-bot/indexer/lasp_faiss_index"
> ```
> **Tip (Windows Command Prompt):**
> ```cmd
> set FAISS_INDEX_DIR=C:/Users/you/lasp-bot/indexer/lasp_faiss_index
> ```

Then install dependencies and start the server:

```bash
cd app
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Open <http://localhost:8000/docs> for the interactive Swagger UI.

Example query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What instruments does LASP operate?"}'
```

### 6 — Build & run the Docker image (optional)

```bash
cd app
docker build -t lasp-bot:latest .
docker run -p 8000:8000 -p 8001:8001 \
  -v /path/to/lasp_faiss_index:/app/lasp_faiss_index \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e MCP_HOST=0.0.0.0 \
  lasp-bot:latest
```

The MCP server starts automatically on port **8001** alongside the FastAPI app.
Set `MCP_HOST=0.0.0.0` so that MCP clients outside the container can reach it.
To disable the MCP server, pass `-e MCP_ENABLED=false`.

### 7 — Run the Discord bot

1. [Create a Discord application and bot](https://discord.com/developers/applications), then copy the bot token.
2. Invite the bot to your server with the **Send Messages** and **Read Message History** permissions (and the `MESSAGE_CONTENT` privileged intent enabled in the Developer Portal).
3. Add `DISCORD_TOKEN` to your `.env` file:

```bash
DISCORD_TOKEN=your_discord_bot_token_here
```

4. Start the bot:

```bash
cd app
python discord_bot.py
```

The MCP server starts automatically on port **8001** alongside the Discord bot.
To disable the MCP server, set `MCP_ENABLED=false` in your `.env` file.

Once online, use `!ask` (or `!lasp`) in any channel the bot can see:

```
!ask What missions does LASP operate?
!lasp What is the MAVEN mission?
```

The bot will reply with an **Answer** and up to three **Sources** from the LASP documentation corpus.

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

## LASP MCP Server

`app/lasp_mcp.py` is a standalone **Model Context Protocol (MCP)** server that
exposes **six tools** for querying LASP data APIs. Combining all tools in a
single server means any MCP-compatible client only needs one connection to access
solar irradiance data, magnetospheric science files, mesospheric cloud
observations, and standardised heliophysics time-series.

### Running the MCP server

The MCP server starts **automatically** whenever the FastAPI app or the Discord
bot starts.  By default it listens on `http://127.0.0.1:8001/mcp` using the
streamable-HTTP transport and is ready for any MCP-compatible client (e.g.
Claude Desktop, MCP Inspector) to connect.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_ENABLED` | `true` | Set to `false` to prevent the MCP server from starting alongside the chatbot |
| `MCP_HOST` | `127.0.0.1` | Host to bind the MCP server to. Use `0.0.0.0` when running in Docker to expose it outside the container. |
| `MCP_PORT` | `8001` | Port for the MCP server |

To run the MCP server as a standalone stdio server (for direct use with a local
MCP client), disable the embedded server and run it manually:

```bash
MCP_ENABLED=false python discord_bot.py   # or: uvicorn main:app …
# then in a separate terminal:
cd app
python lasp_mcp.py
```

---

### LISIRD — solar irradiance & space weather

Access to 130+ solar datasets from LASP, NASA, NOAA, and NSO through the
[LASP Interactive Solar IRradiance Datacenter (LISIRD)](https://lasp.colorado.edu/lisird/)
via the **LaTiS DAP2 API**.

| Tool | Description |
|------|-------------|
| `list_lisird_datasets` | List all available datasets (identifiers, titles, descriptions) |
| `query_solar_irradiance` | Query measurements with optional time-range filtering and variable projection |

#### `query_solar_irradiance` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_id` | *(required)* | LaTiS dataset identifier, e.g. `sorce_tsi_24hr`. Use `list_lisird_datasets` to browse. |
| `start_date` | `None` | Start of the time range (inclusive) — ISO 8601 date or timestamp, e.g. `2003-01-01` |
| `end_date` | `None` | End of the time range (inclusive) — ISO 8601 date or timestamp, e.g. `2003-12-31` |
| `variables` | `None` | Comma-separated variable projection, e.g. `time,tsi`. Leave blank to return all variables. |
| `output_format` | `json` | Response format: `json` or `csv` |

#### Example

```python
# Fetch daily total solar irradiance from SORCE for January 2020
query_solar_irradiance(
    dataset_id="sorce_tsi_24hr",
    start_date="2020-01-01",
    end_date="2020-01-31",
    variables="time,tsi",
)
# URL: .../sorce_tsi_24hr.json?time,tsi&time>=2020-01-01&time<=2020-01-31
```

#### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LISIRD_BASE_URL` | `https://lasp.colorado.edu/lisird/latis/dap2` | LaTiS DAP2 API base URL |
| `LISIRD_TIMEOUT` | `30` | HTTP request timeout in seconds |

---

### MMS SDC — magnetospheric science files

Access to the [MMS Science Data Center](https://lasp.colorado.edu/mms/sdc/public/)
public REST API for discovering and downloading
[Magnetospheric Multiscale (MMS)](https://lasp.colorado.edu/mms/) science data.

| Tool | Description |
|------|-------------|
| `list_mms_files` | Query available science data files (returns metadata) |
| `get_mms_file_urls` | Retrieve HTTPS download URLs for matching files |

Both tools accept the same filter parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `sc_id` | `mms1`, `mms2`, `mms3`, `mms4` | Spacecraft ID (comma-separated) |
| `instrument_id` | `fgm`, `fpi`, `edp`, `hpca`, `mec`, `scm`, `edi`, `feeps`, `epd-eis`, `aspoc`, `dsp`, `ulf`, `afg`, `sdp` | Instrument (comma-separated) |
| `data_rate_mode` | `srvy`, `brst`, `fast`, `slow` | Data rate (comma-separated) |
| `data_level` | `l1a`, `l1b`, `l2`, `l2pre`, `l3`, `ql` | Processing level (comma-separated) |
| `start_date` | `YYYY-MM-DD` | Start of time range |
| `end_date` | `YYYY-MM-DD` | End of time range |
| `version` | e.g. `3.3.0` | File version (optional) |

#### Example

```python
# List survey-rate FGM level-2 files for MMS1 in March 2016
list_mms_files(
    sc_id="mms1",
    instrument_id="fgm",
    data_rate_mode="srvy",
    data_level="l2",
    start_date="2016-03-01",
    end_date="2016-03-31",
)
```

#### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MMS_SDC_BASE_URL` | `https://lasp.colorado.edu/mms/sdc/public/files/api/v1` | MMS SDC API base URL |
| `MMS_SDC_TIMEOUT` | `30` | HTTP request timeout in seconds |

---

### AIM CIPS — mesospheric noctilucent clouds

Access to the [Aeronomy of Ice in the Mesosphere (AIM)](https://lasp.colorado.edu/aim/)
mission's **Cloud Imaging and Particle Size (CIPS)** datasets via the AIM-specific
LaTiS DAP2 instance. CIPS monitors Earth's noctilucent clouds (NLCs) and uses a
day-of-year (`YYYY-DDD`) time format in its constraint expressions.

| Tool | Description |
|------|-------------|
| `query_mesospheric_data` | Query CIPS datasets with constraint-based variable and time filtering |

#### `query_mesospheric_data` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_id` | *(required)* | AIM CIPS LaTiS dataset identifier, e.g. `aim_cips_anc_na`. Browse the catalog at `https://lasp.colorado.edu/aim/latis/dap2/catalog.json`. |
| `variable_constraints` | `None` | Raw LaTiS constraint expression appended as the URL query string. Supports variable projection and time slices. Leave blank to retrieve the full dataset. |
| `output_format` | `json` | Response format: `json` or `csv` |

#### Example

```python
# Retrieve CIPS Northern-hemisphere data for a date range (day-of-year format)
query_mesospheric_data(
    dataset_id="aim_cips_anc_na",
    variable_constraints="time>=2022-305&time<2023-001",
)

# Project specific variables alongside a time constraint
query_mesospheric_data(
    dataset_id="aim_cips_anc_na",
    variable_constraints="time,albedo&time>=2022-001&time<2022-100",
)
```

#### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AIM_BASE_URL` | `https://lasp.colorado.edu/aim/latis/dap2` | AIM CIPS LaTiS DAP2 API base URL |
| `AIM_TIMEOUT` | `30` | HTTP request timeout in seconds |

---

### HAPI — standardised heliophysics time-series

The [Heliophysics Application Programmer's Interface (HAPI)](https://hapi-server.org/)
is a cross-agency standard for streaming time-series space-physics data. A single
tool call works against **any** HAPI-compliant node — LASP/LISIRD, NASA Goddard
CDAWeb, ESA AMDA, and more.

| Tool | Description |
|------|-------------|
| `hapi_time_series_stream` | Fetch time-series data from any HAPI-compliant server |

#### `hapi_time_series_stream` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `server_url` | *(required)* | Base URL of the HAPI server (without trailing slash), e.g. `https://lasp.colorado.edu/lisird/hapi` |
| `dataset` | *(required)* | Dataset (catalog entry) ID on the target server, e.g. `LISIRD3/composite_lya` |
| `start` | *(required)* | Start of the requested time range in ISO 8601 format, e.g. `2020-01-01T00:00:00Z` |
| `stop` | *(required)* | End of the requested time range (exclusive) in ISO 8601 format |
| `parameters` | `None` | Comma-separated parameter names to include. Leave blank to return all parameters. |

#### Example

```python
# Fetch composite Lyman-alpha from LASP LISIRD HAPI
hapi_time_series_stream(
    server_url="https://lasp.colorado.edu/lisird/hapi",
    dataset="LISIRD3/composite_lya",
    start="2020-01-01T00:00:00Z",
    stop="2020-01-31T23:59:59Z",
    parameters="Time,irradiance",
)

# Fetch solar-wind data from NASA Goddard CDAWeb using the same tool
hapi_time_series_stream(
    server_url="https://cdaweb.gsfc.nasa.gov/hapi",
    dataset="AC_K0_SWE",
    start="2020-06-01T00:00:00Z",
    stop="2020-06-02T00:00:00Z",
    parameters="Time,Np",
)
```

#### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HAPI_TIMEOUT` | `30` | HTTP request timeout in seconds (server URL is supplied per call) |

---

## Repository Structure

```
lasp-bot/
├── indexer/
│   ├── build_corpus.py  # Comprehensive corpus builder (HTML, PDFs, PDS, GitHub)
│   ├── build_index.py   # Local GPU script: corpus → FAISS (saved locally)
│   ├── scrape_pdfs.py   # Lightweight PDF-only scraper (subset of build_corpus)
│   ├── discovery_script.py  # One-shot mission URL discovery helper
│   └── requirements.txt
├── app/
│   ├── main.py          # FastAPI application
│   ├── rag.py           # RAG pipeline (local FAISS + Ollama)
│   ├── discord_bot.py   # Discord bot (!ask / !lasp commands)
│   ├── lasp_mcp.py      # MCP server: all LASP data API tools
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       ├── conftest.py
│       ├── test_rag.py
│       ├── test_discord_bot.py
│       └── test_lasp_mcp.py
├── .env.example
└── README.md
```
