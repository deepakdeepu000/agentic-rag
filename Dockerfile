# syntax=docker/dockerfile:1.7

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	CHROMA_PERSIST_DIR=/app/chroma_db \
	INGESTION_CHROMA_PERSIST_DIR=/app/chroma_db \
	INGESTION_WATCH_FOLDER=/app/ingestion_pipline/data \
	OLLAMA_BASE_URL=http://host.docker.internal:11434 \
	OLLAMA_HOST=http://host.docker.internal:11434 \
	SERVICE=rag-agent

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		build-essential \
		curl \
		tesseract-ocr \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install the shared runtime dependencies for both services.
RUN pip install --no-cache-dir \
	beautifulsoup4>=4.12 \
	chromadb>=0.5 \
	fastapi>=0.136.3 \
	langchain>=1.3.2 \
	langchain-anthropic>=0.1.16 \
	langchain-chroma>=0.1.0 \
	langchain-core>=0.3.15 \
	langchain-google-genai>=4.2.4 \
	langchain-huggingface>=0.0.3 \
	langchain-mcp-adapters>=0.1.0 \
	langchain-ollama>=0.1.0 \
	langchain-openai>=0.1.8 \
	langchain-text-splitters>=0.3.0 \
	langgraph>=0.2.28 \
	mcp>=1.0.0 \
	ollama>=0.6.2 \
	openai>=1.30 \
	pdfplumber>=0.11 \
	pillow>=10.0 \
	pydantic-settings>=2.0.0 \
	python-docx>=1.1 \
	python-dotenv>=1.0.0 \
	pytesseract>=0.3 \
	rank-bm25>=0.2.2 \
	sentence-transformers>=5.5.1 \
	tiktoken>=0.13.0 \
	typing-extensions>=4.9.0 \
	watchdog>=4.0

COPY . /app

EXPOSE 8004

CMD ["/bin/sh", "-c", "case \"$SERVICE\" in \
  rag-agent|agent) cd /app/rag-agent && python -m core.run ;; \
  ingestion|ingestion_pipline|ingestion-pipline) cd /app/ingestion_pipline && python main.py ;; \
  *) echo \"Unknown SERVICE=$SERVICE. Use rag-agent or ingestion.\" >&2; exit 1 ;; \
esac"]

