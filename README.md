# Agentic RAG

> A production-ready, **LLM-agnostic** Retrieval-Augmented Generation system powered by [LangGraph](https://github.com/langchain-ai/langgraph) — featuring an intelligent coordinator–worker agent loop, a document ingestion pipeline, persistent session memory, MCP tool integration, and full Docker support.

---

## Table of Contents

- [What Is This?](#what-is-this)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup & Installation](#setup--installation)
  - [Option A — Local (Python)](#option-a--local-python)
  - [Option B — Docker](#option-b--docker)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
  - [1. Run the Ingestion Pipeline](#1-run-the-ingestion-pipeline)
  - [2. Run the RAG Agent](#2-run-the-rag-agent)
  - [3. Run with Docker](#3-run-with-docker)
- [Supported LLM Providers](#supported-llm-providers)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## What Is This?

**Agentic RAG** is a two-service system that combines a document ingestion pipeline with an intelligent conversational agent:

- The **ingestion pipeline** watches a folder, parses documents (PDF, DOCX, images with OCR, web pages), chunks them intelligently, embeds them using local or cloud models, and stores them in a [ChromaDB](https://www.trychroma.com/) vector database.
- The **RAG agent** is a LangGraph-powered coordinator loop that receives user queries, decides whether to retrieve context, call tools, or respond directly, and maintains persistent conversation memory across sessions using SQLite.

Unlike traditional RAG systems where retrieval is triggered on every query regardless of context, this system is **agentic** — the coordinator LLM decides *when* and *how many times* to retrieve, whether external tools are needed, and when enough context exists to synthesize a final answer.

---

## Key Features

- **Agentic coordinator loop** — the LLM drives its own retrieval and tool use, not a hard-coded pipeline
- **LLM-agnostic** — works with Ollama (local), OpenAI, Anthropic Claude, and Google Gemini out of the box
- **Local-first embeddings** — uses `sentence-transformers` (HuggingFace) by default; no external embedding API required
- **ChromaDB vector store** — persistent on-disk storage with configurable collection names
- **Persistent session memory** — conversation summaries stored in SQLite; sessions resume across restarts
- **MCP tool integration** — pluggable remote MCP (Model Context Protocol) tools via SSE
- **Rich document ingestion** — PDF, DOCX, images (OCR via Tesseract), web pages, and plain text
- **Watchdog-based live ingestion** — automatically re-ingests new files dropped into the watched folder
- **Docker support** — single image, two services (agent + ingestion) selectable via `SERVICE` env var
- **Multi-provider ready** — switch between Ollama, OpenAI, Anthropic, or Google Gemini with a single config change

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│  Watches /data folder → Parses (PDF/DOCX/Image/Web) → Chunks    │
│  → Embeds (sentence-transformers) → Stores in ChromaDB          │
└──────────────────────────────┬──────────────────────────────────┘
                               │  Vector DB (ChromaDB)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                          RAG AGENT                               │
│                                                                  │
│   User Query                                                     │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────────────────────────────┐                  │
│  │            COORDINATOR NODE                │                  │
│  │  - Injects session summary (SystemMessage) │                  │
│  │  - Decides: retrieve / call tool / respond │                  │
│  │  - Manages iteration + summarization guard │                  │
│  └──────┬─────────────────┬──────────────────┘                  │
│         │                 │                                      │
│   ┌─────▼──────┐   ┌──────▼──────┐   ┌───────────────┐         │
│   │  RETRIEVER │   │ TOOL NODE   │   │  SUMMARIZER   │         │
│   │  (ChromaDB)│   │(MCP / tools)│   │  (SQLite mem) │         │
│   └─────┬──────┘   └──────┬──────┘   └───────┬───────┘         │
│         └─────────────────┴──────────────────┘                  │
│                           │  (back to coordinator)               │
│                           ▼                                      │
│                      Final Answer                                │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  SESSION STORE   │
│  (SQLite)        │
│  • sessions      │
│  • messages      │
│  • summaries     │
└──────────────────┘
```

**Flow summary:**

1. A user query enters the **Coordinator**, which is injected with the ongoing session summary.
2. The Coordinator decides to either call the `retrieve` tool (vector search), an MCP tool, or produce a final answer directly.
3. If retrieval is chosen, the **Retriever Node** queries ChromaDB and returns scored document chunks.
4. If an external tool is needed, the **Tool Node** executes the MCP call.
5. Results loop back to the Coordinator, which synthesizes or decides to iterate further.
6. After every N turns, the **Summarizer Node** compresses conversation history and persists it to SQLite, keeping memory lean.

---

## Project Structure

```
agentic-rag/
├── ingestion_pipline/          # Document ingestion service
│   ├── data/                   # Drop files here to ingest
│   ├── main.py                 # Entry point; starts watchdog + ingestion loop
│   └── ...
├── rag-agent/                  # Conversational RAG agent service
│   ├── core/
│   │   └── run.py              # Entry point; starts the FastAPI + LangGraph agent
│   └── ...
├── Dockerfile                  # Single image; SERVICE env var selects the service
├── pyproject.toml              # Python project metadata and workspace config
├── requirements.txt            # Pinned runtime dependencies
└── .gitignore
```

---

## Requirements

### System

| Requirement | Version |
|---|---|
| Python | ≥ 3.11 |
| (Optional) Docker | ≥ 20.10 |
| (Optional) Tesseract OCR | ≥ 4.0 (for image ingestion) |
| (Optional) Ollama | Latest (for local LLM) |

### Python Dependencies (key packages)

| Package | Purpose |
|---|---|
| `langgraph >= 0.2.28` | Agent graph framework |
| `langchain-core >= 0.3.15` | LangChain core primitives |
| `langchain-ollama` / `langchain-openai` / `langchain-anthropic` / `langchain-google-genai` | LLM provider adapters |
| `chromadb >= 0.5.0` | Vector database |
| `langchain-chroma >= 0.1.0` | ChromaDB LangChain integration |
| `sentence-transformers >= 3.0.0` | Local embedding model |
| `langchain-huggingface` | HuggingFace embeddings adapter |
| `langchain-mcp-adapters >= 0.1.0` | MCP tool integration |
| `fastapi >= 0.136.3` + `uvicorn` | REST API server for the agent |
| `pdfplumber >= 0.11` | PDF parsing |
| `python-docx >= 1.1` | DOCX parsing |
| `pytesseract >= 0.3` + `pillow` | OCR for images |
| `beautifulsoup4 >= 4.12` | Web page parsing |
| `watchdog >= 4.0` | File system monitoring |
| `rank-bm25 >= 0.2.2` | BM25 hybrid retrieval |
| `pydantic-settings >= 2.0.0` | Configuration management |

All dependencies are listed in [`requirements.txt`](./requirements.txt).

---

## Setup & Installation

### Option A — Local (Python)

**1. Clone the repository**

```bash
git clone https://github.com/deepakdeepu000/agentic-rag.git
cd agentic-rag
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Install Tesseract OCR** *(required for image document ingestion)*

```bash
# Ubuntu / Debian
sudo apt-get install -y tesseract-ocr

# macOS
brew install tesseract

# Windows — download from https://github.com/UB-Mannheim/tesseract/wiki
```

**5. (Optional) Install and start Ollama** *(if using a local LLM)*

```bash
# Install from https://ollama.ai
ollama pull llama3.2       # or any model you prefer
ollama serve               # starts on http://localhost:11434
```

**6. Configure environment variables** *(see [Configuration](#configuration) below)*

---

### Option B — Docker

**1. Clone the repository**

```bash
git clone https://github.com/deepakdeepu000/agentic-rag.git
cd agentic-rag
```

**2. Build the image**

```bash
docker build -t agentic-rag .
```

The Dockerfile installs all runtime dependencies, including Tesseract OCR, and exposes port `8004`.

---

## Configuration

Create a `.env` file in the project root. The agent reads all settings via `pydantic-settings` and falls back to defaults where applicable.

```dotenv
# ── LLM Provider ─────────────────────────────────────────────────────────────
# Choose one: ollama | openai | anthropic | google
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Uncomment if using OpenAI
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o

# Uncomment if using Anthropic
# ANTHROPIC_API_KEY=sk-ant-...
# ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Uncomment if using Google
# GOOGLE_API_KEY=...
# GOOGLE_MODEL=gemini-2.0-flash

# ── Vector Store ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=documents

# ── Ingestion ─────────────────────────────────────────────────────────────────
INGESTION_WATCH_FOLDER=./ingestion_pipline/data
INGESTION_CHROMA_PERSIST_DIR=./chroma_db

# ── Agent ─────────────────────────────────────────────────────────────────────
MAX_ITERATIONS=10
SUMMARIZE_EVERY=8           # Summarize memory every N messages

# ── MCP Tools (optional) ─────────────────────────────────────────────────────
# MCP_SERVER_URL=https://your-mcp-server.com/sse
# MCP_API_KEY=your_key
```

---

## Running the Project

There are two independently runnable services. Start the **ingestion pipeline first** to populate the vector database before running the agent.

### 1. Run the Ingestion Pipeline

The ingestion pipeline monitors the `data/` folder and indexes any documents it finds into ChromaDB.

```bash
cd ingestion_pipline
python main.py
```

Drop any of the following file types into `ingestion_pipline/data/` and they will be automatically processed:

- PDF files (`.pdf`)
- Word documents (`.docx`)
- Images (`.png`, `.jpg`, `.jpeg`) — OCR via Tesseract
- Plain text files (`.txt`)
- Web pages (via URL lists or HTML files)

### 2. Run the RAG Agent

```bash
cd rag-agent
python -m core.run
```

The agent server starts on `http://localhost:8004`. You can interact with it via the REST API or a chat interface depending on your setup.

**Example request:**

```bash
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key points in the uploaded documents?", "session_id": "my-session-1"}'
```

### 3. Run with Docker

The image supports both services via the `SERVICE` environment variable.

**Run the RAG agent:**

```bash
docker run -d \
  --name rag-agent \
  -p 8004:8004 \
  -e SERVICE=rag-agent \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  agentic-rag
```

**Run the ingestion pipeline:**

```bash
docker run -d \
  --name ingestion \
  -e SERVICE=ingestion \
  -v $(pwd)/ingestion_pipline/data:/app/ingestion_pipline/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  agentic-rag
```

**Run both with Docker Compose** *(create a `docker-compose.yml`):*

```yaml
version: "3.9"
services:
  ingestion:
    build: .
    environment:
      SERVICE: ingestion
    volumes:
      - ./ingestion_pipline/data:/app/ingestion_pipline/data
      - chroma_data:/app/chroma_db
    env_file: .env

  rag-agent:
    build: .
    environment:
      SERVICE: rag-agent
    ports:
      - "8004:8004"
    volumes:
      - chroma_data:/app/chroma_db
    depends_on:
      - ingestion
    env_file: .env

volumes:
  chroma_data:
```

```bash
docker compose up
```

---

## Supported LLM Providers

| Provider | Package | Notes |
|---|---|---|
| **Ollama** (default) | `langchain-ollama` | Fully local; requires `ollama serve` running |
| **OpenAI** | `langchain-openai` | Requires `OPENAI_API_KEY` |
| **Anthropic** | `langchain-anthropic` | Requires `ANTHROPIC_API_KEY` |
| **Google Gemini** | `langchain-google-genai` | Requires `GOOGLE_API_KEY` |

Switch providers by updating `LLM_PROVIDER` (and the relevant `*_API_KEY`) in your `.env` file. No code changes needed.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph + LangChain |
| Vector database | ChromaDB (persistent on-disk) |
| Embeddings | sentence-transformers (local, HuggingFace) |
| LLM | Ollama / OpenAI / Anthropic / Google (configurable) |
| Session memory | SQLite (stdlib, zero-config) |
| Tool integration | MCP via `langchain-mcp-adapters` |
| Document parsing | pdfplumber, python-docx, pytesseract, beautifulsoup4 |
| File watching | watchdog |
| API server | FastAPI + uvicorn |
| Configuration | pydantic-settings + python-dotenv |
| Container | Docker (python:3.14-slim base) |

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests where applicable
4. Open a Pull Request with a clear description of what you changed and why

Please keep PRs focused — one feature or fix per PR makes review much faster.

---

## License

This project does not currently specify a license. All rights reserved by the author unless stated otherwise. If you intend to use this in a commercial or open-source project, please open an issue to discuss licensing.

---

*Built with LangGraph · ChromaDB · Ollama · sentence-transformers*
