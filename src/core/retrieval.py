from typing import List, Literal
import numpy as np
from sqlalchemy import text, bindparam, create_engine
from pgvector.sqlalchemy import Vector
from src.config import get_settings
from src.core.embeddings import embed_query
from src.models.schemas import Chunk


settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)


def retrieve(
    query: str,
    k: int | None = None,
    threshold: float | None = None,
    source: Literal["chunk", "asset_chunk"] = "chunk",
) -> List[Chunk]:
    """
    Retrieve top-k relevant chunks (text or image) from PostgreSQL using pgvector cosine distance.

    Args:
        query (str): The user query to embed and search for.
        k (int, optional): Number of top results to retrieve.
        threshold (float, optional): Minimum relevance score.
        source (Literal["chunk", "asset_chunk"]): Which table to query ('chunk' or 'asset_chunk').

    Returns:
        List[Chunk]: List of relevant Chunk objects with scores.
    """
    k = k or settings.default_k
    threshold = threshold or settings.default_threshold

    # Get embedding for query text
    query_embedding = embed_query(query)
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.astype(np.float32).tolist()

    # Choose table depending on source
    if source == "asset_chunk":
        sql = """
            SELECT 
                ac.asset_chunk_uuid AS chunk_uuid,
                d.title AS document_title,
                a.asset_s3_path AS chunk_s3_path,
                'image' AS chunk_type,
                (1 - (ac.embedding <=> :q)) AS relevance
            FROM genai.asset_chunk AS ac
            JOIN genai.asset AS a ON ac.asset_uuid = a.asset_uuid
            JOIN genai.document AS d ON a.document_uuid = d.document_uuid
            ORDER BY ac.embedding <=> :q
            LIMIT :limit
        """
    else:
        sql = """
            SELECT 
                c.chunk_uuid,
                d.title AS document_title,
                c.chunk_s3_path,
                'text' AS chunk_type,
                (1 - (c.embedding <=> :q)) AS relevance
            FROM genai.chunk AS c
            JOIN genai.document AS d ON c.document_uuid = d.document_uuid
            ORDER BY c.embedding <=> :q
            LIMIT :limit
        """

    stmt = text(sql).bindparams(bindparam("q", type_=Vector(settings.embedded_dim)))

    with engine.begin() as conn:
        rows = conn.execute(stmt, {"q": query_embedding, "limit": k}).mappings().all()

    # Filter and map results
    filtered = [
        Chunk(
            document_uuid=str(r["chunk_uuid"]),
            title=r["document_title"],
            chunk_s3_path=r["chunk_s3_path"],
            chunk_type=r["chunk_type"],
            score=float(r["relevance"]),
        )
        for r in rows
        if r["relevance"] >= threshold
    ]

    filtered.sort(key=lambda c: c.score, reverse=True)
    return filtered
