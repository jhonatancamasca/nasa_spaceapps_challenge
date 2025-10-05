from fastapi import APIRouter, Query, Depends
from src.models.schemas import QueryInput
from src.core.retrieval import retrieve
from src.config import get_settings
router = APIRouter(prefix="/v1/search", tags=["search"])

@router.get("/")
def search(query: str, k: int = Query(None), threshold: float = Query(None)):
    settings = get_settings()
    k = k or settings.default_k
    threshold = threshold or settings.default_threshold
    chunks = retrieve(query, k=k, threshold=threshold)
    return {"query": query, "count": len(chunks), "results": [c.dict() for c in chunks]}