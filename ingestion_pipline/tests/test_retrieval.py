# ingestion-pipeline/test_retrieval.py
import argparse
import chromadb
import sys
import time
import logging

from config.config import IngestionConfig
from ingestion.chroma_store import collection_name_for_path
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

    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    if args.collection:
        collections = [collection_name]
    else:
        collections = [c.name for c in client.list_collections()]

    if not collections:
        log.error("❌ No ChromaDB collections found. Please run your ingestion pipeline first.")
        sys.exit(1)

    total_records = 0
    for collection_name_item in collections:
        try:
            total_records += client.get_collection(name=collection_name_item).count()
        except Exception:
            continue

    # Execute Hybrid Retrieval with fine-grained performance tracking
    log.info(f"Executing hybrid search for query: '{args.query}' across {len(collections)} collection(s)")
    
    start_retrieval = time.perf_counter()
    results = hybrid_retrieve(
        query=args.query,
        collections=collections,
        config=config,
        top_k_dense=max(args.top_k * 4, 30),
        top_k_final=args.top_k,
    )
    retrieval_duration = time.perf_counter() - start_retrieval

    # Render Performance Results Dashboard
    print("\n" + "="*80)
    print(" 🛠️  HYBRID RETRIEVAL PERFORMANCE DIAGNOSTICS")
    print("="*80)
    print(f" Target Collections: {', '.join(collections)}")
    print(f" Total DB Pool Size: {total_records} chunks")
    print(f" Search Query      : \"{args.query}\"")
    print("-" * 80)
    print(f" ⏱️  Hybrid Core Processing Time: {retrieval_duration * 1000:.2f} ms")
    print(f" ⏱️  Total Retrieval Latency   : {retrieval_duration * 1000:.2f} ms")
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