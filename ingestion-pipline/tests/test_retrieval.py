# ingestion-pipeline/test_retrieval.py
import argparse
import sys
import time
import logging
from typing import List

from config.config import IngestionConfig
from core.models import Chunk
from ingestion.chroma_store import get_or_create_collection_by_name, collection_name_for_path
from ingestion.chunk_retrival import hybrid_retrieve

# Force UTF-8 terminal encoding for clean console prints
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("RetrievalTester")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-Precision Performance & Relevance Tester for Hybrid RAG."
    )
    parser.add_argument("--query", required=True, help="The search query to evaluate.")
    parser.add_argument("--collection", help="Optional explicit Chroma collection name.")
    parser.add_argument("--file-path", help="Optional file path to resolve collection name.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final chunks to return.")
    args = parser.parse_args()

    config = IngestionConfig()

    # Resolve collection naming
    if args.collection:
        collection_name = args.collection
    elif args.file_path:
        collection_name = collection_name_for_path(
            args.file_path,
            config.watch_folder,
            config.chroma_collection
        )
    else:
        collection_name = config.chroma_collection

    log.info(f"Connecting to ChromaDB collection: '{collection_name}'")
    collection = get_or_create_collection_by_name(config, collection_name)

    # Fetch total database snapshot for BM25 mapping
    log.info("Loading document ecosystem partition from storage...")
    start_load = time.perf_counter()
    raw = collection.get(include=["documents", "metadatas"])
    load_duration = time.perf_counter() - start_load

    ids = raw.get("ids", []) or []
    documents = raw.get("documents", []) or []
    metadatas = raw.get("metadatas", []) or []
    total_records = len(ids)

    log.info(f"Loaded {total_records} chunks in {load_duration:.4f} seconds.")

    if total_records == 0:
        log.error("❌ Database collection is completely empty! Please run your ingestion pipeline first.")
        sys.exit(1)

    # Reconstruct Chunk dataclasses
    all_chunks = []
    for i, cid in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        doc = documents[i] if i < len(documents) and documents[i] else ""
        all_chunks.append(
            Chunk(
                chunk_id=cid,
                file_hash=meta.get("file_hash", ""),
                file_path=meta.get("file_path", ""),
                filename=meta.get("filename", ""),
                doc_type=meta.get("doc_type", ""),
                chunk_index=meta.get("chunk_index", 0),
                total_chunks=meta.get("total_chunks", 0),
                text=doc,
                metadata=meta,
            )
        )
    
    # log.info(f"Constructed {len(all_chunks)} Chunk objects in {const_duration:.4f} seconds.")

    # Execute Hybrid Retrieval with fine-grained performance tracking
    log.info(f"Executing hybrid search for query: '{args.query}'")
    
    start_retrieval = time.perf_counter()
    results = hybrid_retrieve(
        query=args.query,
        collection=collection,
        all_chunks=all_chunks,
        config=config,
        top_k_dense=max(args.top_k * 4, 30),
        top_k_final=args.top_k,
    )
    retret_duration = time.perf_counter() - start_retrieval

    # Render Performance Results Dashboard
    print("\n" + "="*80)
    print(" 🛠️  HYBRID RETRIEVAL PERFORMANCE DIAGNOSTICS")
    print("="*80)
    print(f" Target Collection : {collection_name}")
    print(f" Total DB Pool Size: {total_records} chunks")
    print(f" Search Query      : \"{args.query}\"")
    print("-" * 80)
    print(f" ⏱️  ChromaDB IO Fetch Time   : {load_duration * 1000:.2f} ms")
    print(f" ⏱️  Hybrid Core Processing Time: {retret_duration * 1000:.2f} ms")
    print(f" ⏱️  Total Retrieval Latency   : {(load_duration + retret_duration) * 1000:.2f} ms")
    print("="*80)

    # Output ranked chunks
    for idx, item in enumerate(results):
        meta = item.chunk.metadata or {}
        print(f"\n🥇 RANK {idx + 1}")
        print(f"  ID         : {item.chunk.chunk_id}")
        print(f"  Source File: {meta.get('filename')} (Chunk {meta.get('chunk_index')}/{meta.get('total_chunks')})")
        print(f"  Scores     : [Combined RRF: {item.score:.4f}] -> [Dense Cosine: {item.dense_score:.4f} | Sparse BM25: {item.sparse_score:.4f}]")
        print(f"  Boost Added: {item.metadata_boost:.4f}")
        print(f"  Content Preview:\n{item.chunk.text[:300]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()