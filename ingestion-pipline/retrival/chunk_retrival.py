import math
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence

from rank_bm25 import BM25Okapi

from core.models import Chunk, RetrievedChunk
from config.config import IngestionConfig
from retrival.embedder import embed_query
from retrival.chroma_store import dense_search


log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0

    n = min(len(a), len(b))

    dot = sum(float(a[i]) * float(b[i]) for i in range(n))

    norm_a = math.sqrt(sum(float(x) * float(x) for x in a[:n]))
    norm_b = math.sqrt(sum(float(x) * float(x) for x in b[:n]))

    if not norm_a or not norm_b:
        return 0.0

    return dot / (norm_a * norm_b)


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []

    mn = min(scores)
    mx = max(scores)

    if mx - mn < 1e-9:
        return [0.0 for _ in scores]

    return [(s - mn) / (mx - mn) for s in scores]


def build_bm25_index(chunks: List[Chunk]):
    tokenized = [tokenize(c.text) for c in chunks]
    return BM25Okapi(tokenized)


def metadata_boost(
    query: str,
    metadata: Dict[str, Any],
) -> float:
    score = 0.0

    query_lower = query.lower()

    section = str(metadata.get("section_path", "")).lower()
    filename = str(metadata.get("filename", "")).lower()

    if section and any(word in section for word in query_lower.split()):
        score += 0.10

    if filename and any(word in filename for word in query_lower.split()):
        score += 0.05

    return score


def deduplicate_results(results: List[RetrievedChunk]) -> List[RetrievedChunk]:
    seen = set()
    deduped = []

    for r in results:
        cid = r.chunk.chunk_id

        if cid in seen:
            continue

        seen.add(cid)
        deduped.append(r)

    return deduped


def hybrid_retrieve(
    query: str,
    collection,
    all_chunks: List[Chunk],
    config: IngestionConfig,
    top_k_dense: int = 30,
    top_k_final: int = 8,
) -> List[RetrievedChunk]:

    if not all_chunks:
        return []

    query = query.strip()

    # ---------------------------------------------------
    # QUERY EMBEDDING
    # ---------------------------------------------------

    query_embedding = embed_query(query, config)

    # ---------------------------------------------------
    # DENSE RETRIEVAL
    # ---------------------------------------------------

    dense_result = dense_search(
        collection=collection,
        query_embedding=query_embedding,
        top_k=top_k_dense,
    )

    dense_documents = dense_result["documents"][0]
    dense_metadatas = dense_result["metadatas"][0]
    dense_distances = dense_result["distances"][0]

    dense_scores_raw = [1.0 - float(d) for d in dense_distances]
    dense_scores = normalize_scores(dense_scores_raw)

    # ---------------------------------------------------
    # BM25
    # ---------------------------------------------------

    bm25 = build_bm25_index(all_chunks)

    query_tokens = tokenize(query)

    sparse_raw = bm25.get_scores(query_tokens)
    sparse_scores = normalize_scores(list(map(float, sparse_raw)))

    # Map chunk_id -> sparse score
    sparse_lookup = {
        chunk.chunk_id: sparse_scores[idx]
        for idx, chunk in enumerate(all_chunks)
    }

    # ---------------------------------------------------
    # MERGE
    # ---------------------------------------------------

    results = []

    for doc, metadata, dense_score in zip(
        dense_documents,
        dense_metadatas,
        dense_scores,
    ):

        chunk_id = metadata.get("chunk_id")

        sparse_score = sparse_lookup.get(chunk_id, 0.0)

        meta_boost = metadata_boost(query, metadata)

        final_score = (
            dense_score * config.retrieval.dense_weight
            + sparse_score * config.retrieval.sparse_weight
            + meta_boost
        )

        chunk = Chunk(
            chunk_id=chunk_id,
            file_hash=metadata.get("file_hash"),
            file_path=metadata.get("file_path"),
            filename=metadata.get("filename"),
            doc_type=metadata.get("doc_type"),
            chunk_index=metadata.get("chunk_index"),
            total_chunks=metadata.get("total_chunks"),
            text=doc,
            metadata=metadata,
        )

        results.append(
            RetrievedChunk(
                chunk=chunk,
                score=final_score,
                dense_score=dense_score,
                sparse_score=sparse_score,
                metadata_boost=meta_boost,
            )
        )

    # ---------------------------------------------------
    # SORT
    # ---------------------------------------------------

    results.sort(key=lambda x: x.score, reverse=True)

    # ---------------------------------------------------
    # DEDUPLICATION
    # ---------------------------------------------------

    results = deduplicate_results(results)
    
    # ---------------------------------------------------
    # FINAL TOP K
    # ---------------------------------------------------

    return results[:top_k_final]