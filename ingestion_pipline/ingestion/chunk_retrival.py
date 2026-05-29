import concurrent.futures
import re
import time
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from config.config import IngestionConfig
from core.models import Chunk, RetrievedChunk
from ingestion.chroma_store import dense_search, get_or_create_collection_by_name
from ingestion.embedder import embed_query

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def metadata_boost(query: str, metadata: Dict[str, Any]) -> float:
    score = 0.0
    q = query.lower()
    section = str(metadata.get("section_path", "")).lower()
    filename = str(metadata.get("filename", "")).lower()
    if section and any(word in section for word in q.split()):
        score += 0.10
    if filename and any(word in filename for word in q.split()):
        score += 0.05
    return score


def deduplicate_results(results: List[RetrievedChunk]) -> List[RetrievedChunk]:
    seen = set()
    out = []
    for r in results:
        cid = r.chunk.chunk_id
        if cid is not None and cid in seen:
            continue
        if cid is not None:
            seen.add(cid)
        out.append(r)
    return out


def _threaded_vector_pipeline(query: str, config: IngestionConfig, collection, top_k_dense: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    query_embedding = embed_query(query, config)
    embed_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    dense_result = dense_search(collection=collection, query_embedding=query_embedding, top_k=top_k_dense)
    db_ms = (time.perf_counter() - t1) * 1000

    return {"result": dense_result, "embed_ms": embed_ms, "db_ms": db_ms}


def _chunks_from_dense_result(dense_result: Dict[str, Any]) -> List[Chunk]:
    dense_documents = (dense_result.get("documents") or [[]])[0]
    dense_metadatas = (dense_result.get("metadatas") or [[]])[0]
    dense_ids = (dense_result.get("ids") or [[]])[0]

    chunks: List[Chunk] = []
    for idx, doc in enumerate(dense_documents):
        metadata = dense_metadatas[idx] if idx < len(dense_metadatas) and dense_metadatas[idx] else {}
        chunk_id = dense_ids[idx] if idx < len(dense_ids) else metadata.get("chunk_id")
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                file_hash=metadata.get("file_hash", ""),
                file_path=metadata.get("file_path", ""),
                filename=metadata.get("filename", ""),
                doc_type=metadata.get("doc_type", ""),
                chunk_index=metadata.get("chunk_index", 0),
                total_chunks=metadata.get("total_chunks", 0),
                text=doc or "",
                metadata=metadata,
            )
        )

    return chunks


def hybrid_retrieve(
    query: str,
    collections: List[str],
    config: IngestionConfig,
    top_k_dense: int = 30,
    top_k_final: int = 8,
) -> List[RetrievedChunk]:
    query = (query or "").strip()
    if not query:
        return []

    if not collections:
        return []

    merged: List[RetrievedChunk] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(4, len(collections)))) as executor:
        futures = {
            executor.submit(_threaded_vector_pipeline, query, config, get_or_create_collection_by_name(config, collection_name), top_k_dense): collection_name
            for collection_name in collections
        }

        for future in concurrent.futures.as_completed(futures):
            collection_name = futures[future]
            try:
                vector_payload = future.result()
            except Exception:
                continue

            dense_result = vector_payload["result"]
            dense_documents = (dense_result.get("documents") or [[]])[0]
            dense_metadatas = (dense_result.get("metadatas") or [[]])[0]
            dense_distances = (dense_result.get("distances") or [[]])[0]
            dense_ids = (dense_result.get("ids") or [[]])[0]
            dense_scores = [1.0 - float(d) for d in dense_distances]

            dense_chunks = _chunks_from_dense_result(dense_result)
            if not dense_chunks:
                continue

            bm25 = BM25Okapi([tokenize(chunk.text) for chunk in dense_chunks])
            query_tokens = tokenize(query)
            sparse_scores_raw = bm25.get_scores(query_tokens)

            candidates = []
            for idx, chunk in enumerate(dense_chunks):
                metadata = dense_metadatas[idx] if idx < len(dense_metadatas) and dense_metadatas[idx] else {}
                candidates.append(
                    RetrievedChunk(
                        chunk=Chunk(
                            chunk_id=dense_ids[idx] if idx < len(dense_ids) else chunk.chunk_id,
                            file_hash=metadata.get("file_hash", chunk.file_hash),
                            file_path=metadata.get("file_path", chunk.file_path),
                            filename=metadata.get("filename", chunk.filename),
                            doc_type=metadata.get("doc_type", chunk.doc_type),
                            chunk_index=metadata.get("chunk_index", chunk.chunk_index),
                            total_chunks=metadata.get("total_chunks", chunk.total_chunks),
                            text=dense_documents[idx] if idx < len(dense_documents) and dense_documents[idx] else chunk.text,
                            metadata=metadata or chunk.metadata,
                        ),
                        score=(dense_scores[idx] if idx < len(dense_scores) else 0.0) + sparse_scores_raw[idx] + metadata_boost(query, metadata),
                        dense_score=dense_scores[idx] if idx < len(dense_scores) else 0.0,
                        sparse_score=float(sparse_scores_raw[idx]),
                        metadata_boost=metadata_boost(query, metadata),
                    )
                )

            merged.extend(candidates)

    merged.sort(key=lambda item: item.score, reverse=True)
    return deduplicate_results(merged)[:top_k_final]




