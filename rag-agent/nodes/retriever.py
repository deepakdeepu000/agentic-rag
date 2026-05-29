"""
retriever.py — Chroma vector store setup + retriever node.

Embedding providers (set EMBEDDING_PROVIDER in .env):
  huggingface  (default) — local sentence-transformers, fully offline, no Ollama needed
  ollama                 — delegates to an Ollama embedding model (e.g. nomic-embed-text)

The active provider is printed at startup. Must match whatever was used during ingestion.
"""
import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from core.config import Settings
from core.state import RAGState


logger = logging.getLogger(__name__)


# ── Embedding factory ─────────────────────────────────────────────────────────

def build_embeddings(settings: Settings) -> Embeddings:
    """
    Build the embedding function based on EMBEDDING_PROVIDER.

    huggingface (default):
        Runs sentence-transformers locally — no server, no API key.
        Set EMBEDDING_MODEL to match your ingestion pipeline's model.

    ollama:
        Calls the Ollama /api/embeddings endpoint.
        Set OLLAMA_EMBEDDING_MODEL to a model you have pulled
        (e.g. `ollama pull mxbai-embed-large`).
        Ollama must be running at OLLAMA_BASE_URL.
    """
    provider = settings.embedding_provider.strip().lower()
    logger.info("Embedding provider selected: %s", provider)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        logger.info(
            "Embeddings: ollama / %s @ %s",
            settings.ollama_embedding_model,
            settings.ollama_base_url,
        )
        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )

    # Default: huggingface
    if provider != "huggingface":
        logger.warning(
            "Unknown EMBEDDING_PROVIDER '%s', falling back to 'huggingface'.",
            provider,
        )
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info(
        "Embeddings: huggingface / %s (device=%s)",
        settings.embedding_model,
        settings.embedding_device,
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
    )


# ── Chroma helpers ────────────────────────────────────────────────────────────

def list_chroma_collections(settings: Settings) -> list[str]:
    """List all collection names in the Chroma persist directory."""
    try:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        return [col.name for col in client.list_collections()]
    except Exception as e:
        logger.warning("Could not list Chroma collections: %s", e)
        return []


def build_chroma_store(settings: Settings, embeddings: Embeddings) -> Chroma:
    """
    Connect to the existing Chroma store — read-only, no ingestion.
    The embedding function passed here must match the one used during ingestion.
    """
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def resolve_chroma_collections(settings: Settings) -> list[str]:
    """Resolve the list of target collections for isolated retrieval."""
    if not settings.chroma_collections:
        settings.chroma_collections = list_chroma_collections(settings)
    return settings.chroma_collections


def build_chroma_stores(settings: Settings, embeddings: Embeddings) -> dict[str, Chroma]:
    """Create one Chroma handle per collection to enforce isolation."""
    stores: dict[str, Chroma] = {}
    for name in resolve_chroma_collections(settings):
        stores[name] = Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
    return stores


# ── Retriever node ────────────────────────────────────────────────────────────

def make_retriever_node(vectorstores: dict[str, Chroma], settings: Settings):
    """
    Factory returning a retriever_node compatible with LangGraph's node signature.
    Executes similarity search and returns chunks + a ToolMessage for the coordinator.
    """

    async def retriever_node(state: RAGState, config: RunnableConfig) -> dict:
        query = state["query"]
        last_msg = state["messages"][-1] if state["messages"] else None

        logger.info("Retriever node invoked: query=%r", query[:200])

        # Prefer the query from the tool call args over the raw state query
        tool_call_id = "retrieve_call"
        if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                if tc["name"] == "retrieve":
                    query = tc["args"].get("query", query)
                    tool_call_id = tc.get("id", tool_call_id)
                    break

        logger.info("Retriever tool_call_id=%s", tool_call_id)

        # Isolated per-collection retrieval: top 3 each, then re-rank to top 5.
        per_collection_hits: list[dict] = []
        for collection_name, store in vectorstores.items():
            try:
                logger.info("Retrieving from collection '%s'", collection_name)
                docs_and_scores = store.similarity_search_with_score(query, k=3)
            except Exception as e:
                logger.exception("Retrieval failed for collection '%s'", collection_name)
                continue

            logger.info(
                "Collection '%s' returned %d hit(s)",
                collection_name,
                len(docs_and_scores),
            )

            for doc, score in docs_and_scores:
                per_collection_hits.append(
                    {
                        "content": doc.page_content,
                        "metadata": {**doc.metadata, "collection": collection_name},
                        "score": float(score),
                    }
                )

        # Deduplicate by (content, source) and keep the best score.
        deduped: dict[tuple[str, str], dict] = {}
        for hit in per_collection_hits:
            source = hit["metadata"].get("source", "")
            key = (hit["content"], source)
            existing = deduped.get(key)
            if existing is None or hit["score"] < existing["score"]:
                deduped[key] = hit

        # Lower score is better for Chroma distance; rank and keep top 5.
        ranked = sorted(deduped.values(), key=lambda item: item["score"])
        chunks = ranked[:5]

        if chunks:
            result_text = "\n\n".join(
                f"[score={c['score']:.3f} | source={c['metadata'].get('source', 'unknown')}]\n{c['content']}"
                for c in chunks
            )
        else:
            result_text = "No relevant documents found for this query."

        logger.info("Retriever produced %d final chunk(s)", len(chunks))

        return {
            "retrieved_chunks": chunks,
            "retrieval_attempts": state["retrieval_attempts"] + 1,
            "retrieval_done": True,
            "messages": [ToolMessage(content=result_text, tool_call_id=tool_call_id)],
        }

    return retriever_node