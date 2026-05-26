import argparse
import sys

from config.config import IngestionConfig
from ingestion.chroma_store import (
    collection_name_for_path,
    get_or_create_collection_by_name,
)

from ingestion.chunk_retrival import hybrid_retrieve
# from ingestion.reranker import rerank_chunks
from core.models import Chunk


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Test retrieval relevance against a Chroma collection."
    )
    parser.add_argument("--query", required=True, help="Natural-language query to search for.")
    parser.add_argument(
        "--file-path",
        help="Optional file path used to pick the collection from the top-level data subfolder.",
    )
    parser.add_argument(
        "--collection",
        help="Optional explicit collection name. Overrides --file-path.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    args = parser.parse_args()

    config = IngestionConfig()

    if args.collection:
        collection_name = args.collection
    elif args.file_path:
        collection_name = collection_name_for_path(
            args.file_path,
            config.watch_folder,
            config.chroma_collection,
        )
    else:
        collection_name = config.chroma_collection

    collection = get_or_create_collection_by_name(config, collection_name)

    # Hybrid retrieval needs all chunks for BM25; reconstruct them from the collection.
    raw = collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids", []) or []
    documents = raw.get("documents", []) or []
    metadatas = raw.get("metadatas", []) or []

    all_chunks = []
    for i, cid in enumerate(ids):
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        doc = documents[i] if i < len(documents) and documents[i] else ""
        all_chunks.append(
            Chunk(
                chunk_id=cid,
                file_hash=metadata.get("file_hash", ""),
                file_path=metadata.get("file_path", ""),
                filename=metadata.get("filename", ""),
                doc_type=metadata.get("doc_type", ""),
                chunk_index=metadata.get("chunk_index", 0),
                total_chunks=metadata.get("total_chunks", 0),
                text=doc,
                metadata=metadata,
            )
        )

    results = hybrid_retrieve(
        query=args.query,
        collection=collection,
        all_chunks=all_chunks,
        config=config,
        top_k_dense=max(args.top_k * 4, args.top_k),
        top_k_final=args.top_k,
    )

    print(f"Collection: {collection_name}")
    print(f"Query: {args.query}")
    print("-")
    for idx, item in enumerate(results):
        metadata = item.chunk.metadata or {}
        preview = item.chunk.text
        print(f"Rank {idx + 1}")
        print(f"chunk_id: {item.chunk.chunk_id}")
        print(f"score: {item.score:.4f}")
        print(f"dense_score: {item.dense_score:.4f}")
        print(f"sparse_score: {item.sparse_score:.4f}")
        print(f"metadata_boost: {item.metadata_boost:.4f}")
        print(f"file_path: {metadata.get('file_path')}")
        print(f"filename: {metadata.get('filename')}")
        print(f"document length: {len(preview)}")
        print(f"document preview: {preview}")
        print("-")


if __name__ == "__main__":
    main()