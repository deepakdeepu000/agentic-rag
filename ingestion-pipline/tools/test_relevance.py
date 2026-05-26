import argparse
import sys
from pathlib import Path

from config.config import IngestionConfig
from retrival.chroma_store import (
    collection_name_for_path,
    query_relevant_chunks,
)


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

    result = query_relevant_chunks(config, collection_name, args.query, args.top_k)

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    print(f"Collection: {collection_name}")
    print(f"Query: {args.query}")
    print("-")
    for idx, chunk_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        distance = distances[idx] if idx < len(distances) else None
        document = documents[idx] if idx < len(documents) else ""
        preview = document
        print(f"Rank {idx + 1}")
        print(f"chunk_id: {chunk_id}")
        print(f"distance: {distance}")
        print(f"file_path: {metadata.get('file_path')}")
        print(f"filename: {metadata.get('filename')}")
        print(f"document length: {len(document)}")
        print(f"document preview: {preview}")
        print("-")


if __name__ == "__main__":
    main()