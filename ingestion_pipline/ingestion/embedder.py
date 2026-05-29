import logging
import math
import time
from typing import List, Sequence

import ollama
import asyncio

from core.models import Chunk
from core.chunker import _token_len
from config.config import IngestionConfig

log = logging.getLogger(__name__)

_RETRY_DELAYS = (2, 5, 10)


def _normalize_vector(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if not norm:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


def embed_texts(texts: List[str], config: IngestionConfig) -> List[List[float]]:
    if not texts:
        return []

    # Use the async implementation for concurrency; provide a sync wrapper for callers.
    try:
        return asyncio.run(embed_texts_async(texts, config))
    except RuntimeError:
        # Already running event loop (e.g., interactive env). Use existing loop to run coroutine.
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(embed_texts_async(texts, config))


async def embed_texts_async(texts: List[str], config: IngestionConfig, concurrency: int = 4) -> List[List[float]]:
    """Asynchronously embed texts in parallel batches using threads for the blocking client.

    - Splits texts into batches (based on config.embedding_batch_size).
    - Runs up to `concurrency` embed requests concurrently.
    - Retries with backoff on failure.
    """
    if not texts:
        return []

    batch_size = max(1, config.embedding_batch_size)
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    total_batches = len(batches)

    log.info("Async embedding %d text(s) in %d batch(es) using model '%s' (concurrency=%d)",
             len(texts), total_batches, config.embedding_model, concurrency)

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    results: List[List[List[float]] | None] = [None] * total_batches

    async def worker(idx: int, batch: List[str]):
        batch_token_size = sum(_token_len(t) for t in batch)
        async with sem:
            for attempt, delay in enumerate([0, * _RETRY_DELAYS], start=1):
                if delay:
                    log.warning("Embedding batch %d/%d: retry %d after %ds back-off", idx + 1, total_batches, attempt, delay)
                    await asyncio.sleep(delay)

                try:
                    def call_embed():
                        client = ollama.Client(host=config.ollama_host)
                        return client.embed(model=config.embedding_model, input=batch)

                    resp = await loop.run_in_executor(None, call_embed)
                    embeddings = getattr(resp, 'embeddings', None) or (resp.get('embeddings') if isinstance(resp, dict) else None)
                    if embeddings is None:
                        raise RuntimeError('No embeddings in response')
                    results[idx] = [_normalize_vector(vec) for vec in embeddings]
                    log.info("Batch %d/%d embedded: items=%d batch_tokens=%d", idx + 1, total_batches, len(batch), batch_token_size)
                    return
                except Exception as exc:
                    log.error("Embedding batch %d/%d attempt %d failed: %s chunksize=%d batch_tokens=%d",
                              idx + 1, total_batches, attempt, exc, len(batch), batch_token_size)
                    log.debug("Failed batch texts: %s", " | ".join(batch)[:150])
                    if attempt > len(_RETRY_DELAYS):
                        raise

    tasks = [asyncio.create_task(worker(i, b)) for i, b in enumerate(batches)]
    await asyncio.gather(*tasks)

    all_embeddings: List[List[float]] = []
    for part in results:
        if part is None:
            raise ValueError("Missing embeddings for a batch")
        all_embeddings.extend(part)

    if len(all_embeddings) != len(texts):
        raise ValueError(f"Embedding count mismatch: got {len(all_embeddings)} for {len(texts)} texts")

    return all_embeddings


def embed_chunks(chunks: List[Chunk], config: IngestionConfig) -> List[Chunk]:
    """
    Embed all chunks in batches using the configured Ollama model.
    Attaches each vector to chunk._embedding.
    """
    if not chunks:
        log.debug("embed_chunks called with empty list — nothing to do")
        return chunks

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, config)

    for chunk, embedding in zip(chunks, vectors):
        chunk._embedding = embedding

    log.info("Embedding complete: %d chunks", len(chunks))
    return chunks


def embed_query(query: str, config: IngestionConfig) -> List[float]:
    """Convenience helper for query-time retrieval."""
    vectors = embed_texts([query], config)
    return vectors[0] if vectors else []