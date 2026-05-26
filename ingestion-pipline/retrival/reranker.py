import logging
from typing import List

from sentence_transformers import CrossEncoder

from core.retriever import RetrievedChunk

log = logging.getLogger(__name__)

# ---------------------------------------------------------
# GLOBAL MODEL CACHE
# ---------------------------------------------------------

_RERANKER_MODEL = None


def get_reranker(
    model_name: str = "BAAI/bge-reranker-large",
):
    """
    Lazy-load reranker once.

    CrossEncoder models are expensive to load.
    """

    global _RERANKER_MODEL

    if _RERANKER_MODEL is None:

        log.info(
            "Loading reranker model: %s",
            model_name,
        )

        _RERANKER_MODEL = CrossEncoder(
            model_name,
            trust_remote_code=True,
        )

    return _RERANKER_MODEL


def rerank_chunks(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    top_k: int = 5,
    reranker_model: str = "BAAI/bge-reranker-large",
) -> List[RetrievedChunk]:
    """
    Cross-encoder reranking.

    Input:
        query
        retrieved chunks from hybrid retrieval

    Output:
        best answer chunks ranked by relevance
    """

    if not retrieved_chunks:
        return []

    model = get_reranker(reranker_model)


    pairs = [
        [query, r.chunk.text]
        for r in retrieved_chunks
    ]

    scores = model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )


    reranked = []

    for result, rerank_score in zip(
        retrieved_chunks,
        scores,
    ):

        # Combine hybrid score + rerank score
        # rerank dominates because it is more accurate
        final_score = (
            result.score * 0.25
            + float(rerank_score) * 0.75
        )

        result.score = final_score

        # Attach extra metadata for debugging
        result.chunk.metadata["rerank_score"] = float(rerank_score)

        reranked.append(result)

    reranked.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return reranked[:top_k]