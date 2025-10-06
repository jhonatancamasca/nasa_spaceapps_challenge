from fastapi import APIRouter, Query
from typing import Literal
from src.core.retrieval import retrieve, retrieve_documents
from src.config import get_settings

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/")
def search(
    query: str,
    k: int = Query(None, description="Number of top results to retrieve"),
    threshold: float = Query(None, description="Minimum relevance threshold"),
    source: Literal["chunk", "asset_chunk"] = Query(
        "chunk",
        description="Data source: 'chunk' for text, 'asset_chunk' for image embeddings",
    ),
):
    """
    Search for relevant text or image chunks based on the input query.

    Args:
        query (str): User input text query.
        k (int, optional): Number of results to return.
        threshold (float, optional): Minimum relevance score.
        source (Literal["chunk", "asset_chunk"]): Which table to search in.

    Returns:
        dict: Query info, count, and list of retrieved chunks.
    """
    settings = get_settings()
    k = k or settings.default_k
    threshold = threshold or settings.default_threshold

    chunks = retrieve(query, k=k, threshold=threshold, source=source)

    return {
        "query": query,
        "source": source,
        "count": len(chunks),
        "results": [c.dict() for c in chunks],
    }


@router.get("/search_documents")
def search_documents(
    query: str,
    k: int = Query(None, description="Number of top results to retrieve"),
    threshold: float = Query(None, description="Minimum relevance threshold"),
    source: Literal["chunk", "asset_chunk"] = Query(
        "chunk",
        description="Data source: 'chunk' for text, 'asset_chunk' for image embeddings",
    ),
):
    """
    Search for relevant text or image chunks based on the input query.

    Args:
        query (str): User input text query.
        k (int, optional): Number of results to return.
        threshold (float, optional): Minimum relevance score.
        source (Literal["chunk", "asset_chunk"]): Which table to search in.

    Returns:
        dict: Query info, count, and list of retrieved chunks.
    """
    settings = get_settings()
    k = k or settings.default_k
    threshold = threshold or settings.default_threshold

    chunks = retrieve_documents(query, k=k, threshold=threshold, source=source)

    return {
        "query": query,
        "source": source,
        "count": len(chunks),
        "results": [c.dict() for c in chunks],
    }
