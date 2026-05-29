

# import re
# import time
# import concurrent.futures
# from typing import List, Dict, Any, Optional

# from rank_bm25 import BM25Okapi
# from core.models import Chunk, RetrievedChunk
# from config.config import IngestionConfig
# from ingestion.embedder import embed_query
# from ingestion.chroma_store import dense_search

# _WORD_RE = re.compile(r"[A-Za-z0-9_]+")

# # Cached BM25 matrix to avoid recompiling on every query
# _LAZY_BM25_MATRIX: Optional[BM25Okapi] = None
# _CURRENT_ECOSYSTEM_KEY: Optional[str] = None


# def tokenize(text: str) -> List[str]:
#     return _WORD_RE.findall(text.lower())


# def metadata_boost(query: str, metadata: Dict[str, Any]) -> float:
#     score = 0.0
#     q = query.lower()
#     section = str(metadata.get("section_path", "")).lower()
#     filename = str(metadata.get("filename", "")).lower()
#     if section and any(word in section for word in q.split()):
#         score += 0.10
#     if filename and any(word in filename for word in q.split()):
#         score += 0.05
#     return score


# def deduplicate_results(results: List[RetrievedChunk]) -> List[RetrievedChunk]:
#     seen = set()
#     out = []
#     for r in results:
#         cid = r.chunk.chunk_id
#         if cid is not None and cid in seen:
#             continue
#         if cid is not None:
#             seen.add(cid)
#         out.append(r)
#     return out


# def _threaded_vector_pipeline(query: str, config: IngestionConfig, collection, top_k_dense: int) -> Dict[str, Any]:
#     t0 = time.perf_counter()
#     query_embedding = embed_query(query, config)
#     embed_ms = (time.perf_counter() - t0) * 1000

#     t1 = time.perf_counter()
#     dense_result = dense_search(collection=collection, query_embedding=query_embedding, top_k=top_k_dense)
#     db_ms = (time.perf_counter() - t1) * 1000

#     return {"result": dense_result, "embed_ms": embed_ms, "db_ms": db_ms}


# def hybrid_retrieve(
#     query: str,
#     collections: List[str],
#     all_chunks: Optional[List[Chunk]] = None,
#     config: IngestionConfig = None,
#     top_k_dense: int = 30,
#     top_k_final: int = 8,
# ) -> List[RetrievedChunk]:
#     global _LAZY_BM25_MATRIX, _CURRENT_ECOSYSTEM_KEY

#     query = (query or "").strip()
#     if not query:
#         return []

#     query_tokens = tokenize(query)

#     ecosystem_key = f"{len(all_chunks) if all_chunks else 0}::{all_chunks[-1].chunk_id if all_chunks else ''}"

#     # Run embedding + dense search in a thread while compiling BM25 matrix if needed
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         vector_future = executor.submit(_threaded_vector_pipeline, query, config, collection, top_k_dense)

#         if _LAZY_BM25_MATRIX is None or _CURRENT_ECOSYSTEM_KEY != ecosystem_key:
#             tokenized = [tokenize(c.text) for c in (all_chunks or [])]
#             if tokenized:
#                 _LAZY_BM25_MATRIX = BM25Okapi(tokenized)
#                 _CURRENT_ECOSYSTEM_KEY = ecosystem_key

#         sparse_scores_raw = _LAZY_BM25_MATRIX.get_scores(query_tokens) if _LAZY_BM25_MATRIX else []
#         vector_payload = vector_future.result()

#     dense_result = vector_payload["result"]

#     dense_documents = (dense_result.get("documents") or [[]])[0]
#     dense_metadatas = (dense_result.get("metadatas") or [[]])[0]
#     dense_distances = (dense_result.get("distances") or [[]])[0]
#     dense_ids = (dense_result.get("ids") or [[]])[0]
#     dense_scores_raw = [1.0 - float(d) for d in dense_distances]

#     dense_candidates = {}
#     for idx, cid in enumerate(dense_ids):
#         dense_candidates[cid] = {
#             "doc": dense_documents[idx],
#             "metadata": dense_metadatas[idx],
#             "dense_score": dense_scores_raw[idx],
#             "dense_rank": idx + 1,
#         }

#     sparse_ranked_indices = sorted(range(len(sparse_scores_raw)), key=lambda k: sparse_scores_raw[k], reverse=True)
#     sparse_lookup = {}
#     for rank, idx in enumerate(sparse_ranked_indices, start=1):
#         if not all_chunks:
#             break
#         target_chunk = all_chunks[idx]
#         sparse_lookup[target_chunk.chunk_id] = {
#             "score": float(sparse_scores_raw[idx]),
#             "sparse_rank": rank,
#             "chunk_obj": target_chunk,
#         }

#     RRF_CONSTANT = 60.0
#     merged = []
#     all_ids = set(dense_candidates.keys()).union(set(sparse_lookup.keys()))
#     for cid in all_ids:
#         d = dense_candidates.get(cid)
#         s = sparse_lookup.get(cid)

#         rrf_d = 1.0 / (RRF_CONSTANT + d["dense_rank"]) if d else 0.0
#         rrf_s = 1.0 / (RRF_CONSTANT + s["sparse_rank"]) if s else 0.0

#         metadata = d["metadata"] if d else s["chunk_obj"].metadata
#         doc_text = d["doc"] if d else s["chunk_obj"].text

#         mb = metadata_boost(query, metadata)
#         final_score = rrf_d + rrf_s + mb

#         chunk = Chunk(
#             chunk_id=cid,
#             file_hash=metadata.get("file_hash"),
#             file_path=metadata.get("file_path"),
#             filename=metadata.get("filename"),
#             doc_type=metadata.get("doc_type"),
#             chunk_index=metadata.get("chunk_index"),
#             total_chunks=metadata.get("total_chunks"),
#             text=doc_text,
#             metadata=metadata,
#         )

#         merged.append(
#             RetrievedChunk(
#                 chunk=chunk,
#                 score=final_score,
#                 dense_score=d["dense_score"] if d else 0.0,
#                 sparse_score=s["score"] if s else 0.0,
#                 metadata_boost=mb,
#             )
#         )

#     merged.sort(key=lambda x: x.score, reverse=True)
#     deduped = deduplicate_results(merged)
    # return deduped[:top_k_final]





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




