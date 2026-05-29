import logging
import os
import re
import shutil
from typing import List
from pathlib import Path

import chromadb
import ollama

from core.models import Chunk
from config.config import IngestionConfig

log = logging.getLogger(__name__)


def collection_name_for_path(file_path: str, watch_folder: str, default: str = "documents") -> str:
    """Map a file path to a top-level collection name.

    Examples:
    - watch/data/profile/a.txt -> profile
    - watch/data/technologies/x.md -> technologies
    - watch/data/root-file.txt -> documents (default)
    """
    try:
        relative_path = Path(file_path).resolve().relative_to(Path(watch_folder).resolve())
    except Exception:
        relative_path = Path(file_path)

    parts = relative_path.parts if isinstance(relative_path, Path) else Path(relative_path).parts
    top_level = parts[0] if parts and len(parts) > 1 else default

    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", top_level).strip("._-")
    return normalized.lower() or default


def get_or_create_collection(config: IngestionConfig):
    log.info(
        "Connecting to ChromaDB: persist_dir=%s collection=%s",
        config.chroma_persist_dir, config.chroma_collection,
    )
    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name=config.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(
        "Collection '%s' ready — %d existing chunk(s)",
        config.chroma_collection, collection.count(),
    )
    return collection


def get_or_create_collection_by_name(config: IngestionConfig, collection_name: str):
    log.info(
        "Resolving ChromaDB collection: collection=%s",
        collection_name,
    )
    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(
        "Collection '%s' ready — %d existing chunk(s)",
        collection_name, collection.count(),
    )
    return collection


def get_or_create_collection_for_path(config: IngestionConfig, file_path: str):
    collection_name = collection_name_for_path(file_path, config.watch_folder, config.chroma_collection)
    log.info(
        "Resolving ChromaDB collection for file=%s -> collection=%s",
        file_path, collection_name,
    )
    return get_or_create_collection_by_name(config, collection_name)


def upsert_chunks(chunks: List[Chunk], collection) -> int:
    """
    Upsert into ChromaDB.
    Deterministic chunk IDs make this idempotent — re-ingesting the same
    file overwrites existing chunks, never duplicates them.

    Returns the number of chunks upserted.
    """
    if not chunks:
        log.debug("upsert_chunks: empty list — nothing to do")
        return 0

    ids        = [c.chunk_id for c in chunks]
    documents  = [c.text for c in chunks]
    embeddings = [c._embedding for c in chunks]
    # Filter out None values and the raw embedding key from metadata
    metadatas  = [
        {k: v for k, v in c.metadata.items() if k != "embedding" and v is not None}
        for c in chunks
    ]

    log.debug("Upserting %d chunks into collection '%s'", len(chunks), collection.name)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    log.info("Upserted %d chunk(s) into ChromaDB collection '%s'", len(chunks), collection.name)
    return len(chunks)



def dense_search(
    collection,
    query_embedding,
    top_k: int = 30,
):
    """
    Dense vector retrieval ONLY.

    No reranking.
    No hybrid logic.
    No metadata scoring.
    """

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )


def delete_chunks_by_file_hash(
    collection,
    file_hash: str,
):
    """
    Remove all chunks belonging to a file.
    """

    collection.delete(
        where={
            "file_hash": file_hash
        }
    )

    log.info(
        "Deleted chunks for file_hash=%s",
        file_hash,
    )



def clear_chroma_db(config: IngestionConfig) -> None:
    """Delete the ChromaDB persist directory (sqlite file + collections).

    This forcibly removes `config.chroma_persist_dir`. Use when you want
    a full reset of the persisted Chroma store. Caller should ensure no
    processes are using the DB.
    """
    path = config.chroma_persist_dir
    if not os.path.exists(path):
        log.info("Chroma persist dir '%s' does not exist — nothing to do", path)
        return

    try:
        shutil.rmtree(path)
        log.info("Removed Chroma persist dir '%s' (full reset)", path)
    except Exception as exc:
        log.error("Failed to remove Chroma persist dir '%s': %s", path, exc)
        raise