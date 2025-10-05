from typing import List
from src.models.schemas import Chunk
import random
from src.config import get_settings

settings = get_settings()

# NOTE: Replace with real DB/pgvector logic. This is a stub.

def retrieve(query: str, k: int = None, threshold: float | None = None) -> List[Chunk]:
    k = k or settings.default_k
    threshold = threshold or settings.default_threshold

    results = []
    for i in range(12):
        score = round(random.uniform(0.5, 0.99), 3)
        chunk = Chunk(
            document_uuid=str(i),
            title=f"Document {i}",
            chunk_content="Simulated chunk matching the query",
            chunk_s3_path=None,
            chunk_type="text_chunk",
            score=score,
        )
        if score >= threshold:
            results.append(chunk)
    results_sorted = sorted(results, key=lambda c: c.score, reverse=True)[:k]
    return results_sorted