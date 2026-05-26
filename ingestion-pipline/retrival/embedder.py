# import logging
# import time
# from typing import List

# import ollama

# from core.models import Chunk
# from config.config import IngestionConfig

# log = logging.getLogger(__name__)

# _RETRY_DELAYS = (2, 5, 10)   # seconds between embedding retries


# def embed_chunks(chunks: List[Chunk], config: IngestionConfig) -> List[Chunk]:
#     """
#     Embed all chunks in batches using the configured Ollama model.
#     Attaches each vector to chunk._embedding (consumed by chroma_store).

#     Raises on unrecoverable failure so the pipeline can back-off and retry
#     the whole file.
#     """
#     if not chunks:
#         log.debug("embed_chunks called with empty list — nothing to do")
#         return chunks

#     texts = [c.text for c in chunks]
#     batch_size = config.embedding_batch_size
#     all_embeddings: list = []

#     total_batches = (len(texts) + batch_size - 1) // batch_size
#     log.info(
#         "Embedding %d chunks in %d batch(es) using model '%s'",
#         len(chunks), total_batches, config.embedding_model,
#     )

#     client = ollama.Client(host=config.ollama_host)

#     for batch_idx, start in enumerate(range(0, len(texts), batch_size)):
#         batch = texts[start: start + batch_size]
#         batch_num = batch_idx + 1

#         for attempt, delay in enumerate([0] + list(_RETRY_DELAYS), start=1):
#             if delay:
#                 log.warning(
#                     "Embedding batch %d/%d: retry %d after %ds back-off",
#                     batch_num, total_batches, attempt, delay,
#                 )
#                 time.sleep(delay)
#             try:
#                 response = client.embed(
#                     model=config.embedding_model,
#                     input=batch,
#                 )
#                 embeddings = response["embeddings"]
#                 all_embeddings.extend(embeddings)
#                 log.debug(
#                     "Batch %d/%d embedded: %d vectors (dim=%d)",
#                     batch_num, total_batches,
#                     len(embeddings),
#                     len(embeddings[0]) if embeddings else 0,
#                 )
#                 break   # success — move to next batch

#             except Exception as exc:
#                 log.error(
#                     "Embedding batch %d/%d attempt %d failed: %s",
#                     batch_num, total_batches, attempt, exc,
#                 )
#                 if attempt > len(_RETRY_DELAYS):
#                     log.error(
#                         "Embedding exhausted retries for batch %d — re-raising",
#                         batch_num,
#                     )
#                     raise

#     # Attach vectors to chunks
#     if len(all_embeddings) != len(chunks):
#         raise ValueError(
#             f"Embedding count mismatch: got {len(all_embeddings)} "
#             f"for {len(chunks)} chunks"
#         )

#     for chunk, embedding in zip(chunks, all_embeddings):
#         chunk._embedding = embedding   # only place the vector is stored

#     log.info("Embedding complete: %d chunks", len(chunks))
#     return chunks

import logging
import math
import time
from typing import List, Sequence

import ollama

from core.models import Chunk
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

    batch_size = max(1, config.embedding_batch_size)
    total_batches = (len(texts) + batch_size - 1) // batch_size
    client = ollama.Client(host=config.ollama_host)

    all_embeddings: List[List[float]] = []

    log.info(
        "Embedding %d text(s) in %d batch(es) using model '%s'",
        len(texts), total_batches, config.embedding_model,
    )

    for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[start:start + batch_size]

        for attempt, delay in enumerate([0, * _RETRY_DELAYS], start=1):
            if delay:
                log.warning(
                    "Embedding batch %d/%d: retry %d after %ds back-off",
                    batch_idx, total_batches, attempt, delay,
                )
                time.sleep(delay)

            try:
                response = client.embed(
                    model=config.embedding_model,
                    input=batch,
                )
                embeddings = response["embeddings"]
                all_embeddings.extend(_normalize_vector(vec) for vec in embeddings)
                break
            except Exception as exc:
                log.error(
                    "Embedding batch %d/%d attempt %d failed: %s",
                    batch_idx, total_batches, attempt, exc,
                )
                if attempt > len(_RETRY_DELAYS):
                    raise

    if len(all_embeddings) != len(texts):
        raise ValueError(
            f"Embedding count mismatch: got {len(all_embeddings)} for {len(texts)} texts"
        )

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