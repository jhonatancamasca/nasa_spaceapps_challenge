from typing import List, Literal
import numpy as np
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
from src.config import get_settings
from src.core.embeddings import embed_query
from src.models.schemas import Chunk
import re
settings = get_settings()

# --- S3 Client ---
s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    region_name=settings.aws_region,
)


def download_from_s3(s3_path: str) -> str:
    print("the path is:", s3_path)
    if not s3_path:
        return ""

    # Case 1: s3://bucket/key
    if s3_path.startswith("s3://"):
        bucket, key = s3_path.replace("s3://", "").split("/", 1)

    # Case 2: https://bucket.s3.region.amazonaws.com/key
    elif s3_path.startswith("https://"):
        match = re.match(
            r"https://(.+?)\.s3(?:[.-](.+?))?\.amazonaws\.com/(.+)", s3_path
        )
        if not match:
            return ""
        bucket = match.group(1)
        key = match.group(3)
    else:
        return ""

    # Download the object
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        import base64

        return base64.b64encode(data).decode("utf-8")


# --- DB Connection ---
def get_db_connection():
    return psycopg2.connect(
        host="bioseekers-db.cjg2c8w8choo.us-east-2.rds.amazonaws.com",
        port="5432",
        database="bioseekers_nasa_2025",
        user="postgres",
        password="bioseekers123",
        cursor_factory=RealDictCursor,
    )


# --- Convert embedding list to pgvector string ---
def vector_to_pgvector(arr: list[float]) -> str:
    return f"ARRAY[{','.join(map(str, arr))}]::vector"


# --- Retrieve top-k relevant chunks ---
def retrieve(
    query: str,
    k: int | None = None,
    threshold: float | None = None,
    source: Literal["chunk", "asset_chunk", "both"] = "chunk",
) -> List[Chunk]:
    k = k or settings.default_k
    threshold = threshold or settings.default_threshold

    query_embedding = embed_query(query)
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.astype(np.float32).tolist()
    vec_literal = vector_to_pgvector(query_embedding)

    if source == "asset_chunk":
        sql = f"""
            SELECT 
                ac.asset_chunk_uuid AS chunk_uuid,
                d.title AS document_title,
                a.asset_s3_path AS chunk_s3_path,
                'image' AS chunk_type,
                (1 - (ac.embedding <=> {vec_literal})) AS relevance
            FROM genai.asset_chunk AS ac
            JOIN genai.asset AS a ON ac.asset_uuid = a.asset_uuid
            JOIN genai.document AS d ON a.document_uuid = d.document_uuid
            ORDER BY ac.embedding <=> {vec_literal}
            LIMIT %s
        """
    elif source == "chunk":
        sql = f"""
            SELECT 
                c.chunk_uuid,
                d.title AS document_title,
                c.chunk_s3_path,
                'text' AS chunk_type,
                (1 - (c.embedding <=> {vec_literal})) AS relevance
            FROM genai.chunk AS c
            JOIN genai.document AS d ON c.document_uuid = d.document_uuid
            ORDER BY c.embedding <=> {vec_literal}
            LIMIT %s
        """
    else:  # both
        sql = f"""
            SELECT * FROM (
                SELECT 
                    c.chunk_uuid,
                    d.title AS document_title,
                    c.chunk_s3_path,
                    'text' AS chunk_type,
                    (1 - (c.embedding <=> {vec_literal})) AS relevance
                FROM genai.chunk AS c
                JOIN genai.document AS d ON c.document_uuid = d.document_uuid
                UNION ALL
                SELECT 
                    ac.asset_chunk_uuid AS chunk_uuid,
                    d.title AS document_title,
                    a.asset_s3_path AS chunk_s3_path,
                    'image' AS chunk_type,
                    (1 - (ac.embedding <=> {vec_literal})) AS relevance
                FROM genai.asset_chunk AS ac
                JOIN genai.asset AS a ON ac.asset_uuid = a.asset_uuid
                JOIN genai.document AS d ON a.document_uuid = d.document_uuid
            ) AS combined
            ORDER BY relevance DESC
            LIMIT %s
        """

    results = []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (k,))
            rows = cur.fetchall()
            for r in rows:
                if r.get("relevance") is not None and r["relevance"] >= threshold:
                    content = download_from_s3(r["chunk_s3_path"])
                    results.append(
                        Chunk(
                            chunk_uuid=str(r["chunk_uuid"]),
                            title=r["document_title"],
                            chunk_s3_path=r["chunk_s3_path"],
                            chunk_type=r["chunk_type"],
                            score=float(r["relevance"]),
                            chunk_content=content,
                        )
                    )

    results.sort(key=lambda c: c.score, reverse=True)
    return results


# --- Retrieve chunks by document UUID ---
def retrieve_by_document_uuid(
    document_uuid: str | list[str],
    source: Literal["chunk", "asset_chunk", "both"] = "chunk",
) -> List[Chunk]:
    if isinstance(document_uuid, str):
        document_uuids = [document_uuid]
    else:
        document_uuids = document_uuid

    if source == "asset_chunk":
        sql = """
            SELECT 
                ac.asset_chunk_uuid AS chunk_uuid,
                d.document_uuid,
                d.title AS document_title,
                a.asset_s3_path AS chunk_s3_path,
                'image' AS chunk_type
            FROM genai.asset_chunk AS ac
            JOIN genai.asset AS a ON ac.asset_uuid = a.asset_uuid
            JOIN genai.document AS d ON a.document_uuid = d.document_uuid
            WHERE d.document_uuid = ANY(%s)
            ORDER BY d.document_uuid, ac.created_at ASC
        """
        params = (document_uuids,)
    elif source == "chunk":
        sql = """
            SELECT 
                c.chunk_uuid,
                d.document_uuid,
                d.title AS document_title,
                c.chunk_s3_path,
                'text' AS chunk_type
            FROM genai.chunk AS c
            JOIN genai.document AS d ON c.document_uuid = d.document_uuid
            WHERE d.document_uuid = ANY(%s)
            ORDER BY d.document_uuid, c.created_at ASC
        """
        params = (document_uuids,)
    else:  # both
        sql = """
            SELECT * FROM (
                SELECT 
                    c.chunk_uuid,
                    d.document_uuid,
                    d.title AS document_title,
                    c.chunk_s3_path,
                    'text' AS chunk_type,
                    c.created_at
                FROM genai.chunk AS c
                JOIN genai.document AS d ON c.document_uuid = d.document_uuid
                WHERE d.document_uuid = ANY(%s)
                UNION ALL
                SELECT 
                    ac.asset_chunk_uuid AS chunk_uuid,
                    d.document_uuid,
                    d.title AS document_title,
                    a.asset_s3_path AS chunk_s3_path,
                    'image' AS chunk_type,
                    ac.created_at
                FROM genai.asset_chunk AS ac
                JOIN genai.asset AS a ON ac.asset_uuid = a.asset_uuid
                JOIN genai.document AS d ON a.document_uuid = d.document_uuid
                WHERE d.document_uuid = ANY(%s)
            ) AS combined
            ORDER BY document_uuid, created_at ASC
        """
        params = (document_uuids, document_uuids)

    results = []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            for r in rows:
                content = download_from_s3(r["chunk_s3_path"])
                results.append(
                    Chunk(
                        document_uuid=str(r["document_uuid"]),
                        title=r["document_title"],
                        chunk_s3_path=r["chunk_s3_path"],
                        chunk_type=r["chunk_type"],
                        content=content,
                    )
                )
    return results


# --- Retrieve documents by query ---
def retrieve_documents(
    query: str,
    k: int | None = None,
    threshold: float | None = None,
    source: Literal["chunk", "asset_chunk", "both"] = "chunk",
) -> List[Chunk]:
    k = k or settings.default_k
    threshold = threshold or settings.default_threshold

    query_embedding = embed_query(query)
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.astype(np.float32).tolist()
    vec_literal = vector_to_pgvector(query_embedding)

    if source == "asset_chunk":
        sql = f"""
            SELECT 
                ac.asset_chunk_uuid AS chunk_uuid,
                d.title AS document_title,
                a.asset_s3_path AS chunk_s3_path,
                'image' AS chunk_type,
                (1 - (ac.embedding <=> {vec_literal})) AS relevance
            FROM genai.asset_chunk AS ac
            JOIN genai.asset AS a ON ac.asset_uuid = a.asset_uuid
            JOIN genai.document AS d ON a.document_uuid = d.document_uuid
            ORDER BY ac.embedding <=> {vec_literal}
            LIMIT %s
        """
    elif source == "chunk":
        sql = f"""
            SELECT 
                c.chunk_uuid,
                d.title AS document_title,
                c.chunk_s3_path,
                'text' AS chunk_type,
                (1 - (c.embedding <=> {vec_literal})) AS relevance
            FROM genai.chunk AS c
            JOIN genai.document AS d ON c.document_uuid = d.document_uuid
            ORDER BY c.embedding <=> {vec_literal}
            LIMIT %s
        """
    else:  # both
        sql = f"""
            SELECT * FROM (
                SELECT 
                    c.chunk_uuid,
                    d.title AS document_title,
                    c.chunk_s3_path,
                    'text' AS chunk_type,
                    (1 - (c.embedding <=> {vec_literal})) AS relevance
                FROM genai.chunk AS c
                JOIN genai.document AS d ON c.document_uuid = d.document_uuid
                UNION ALL
                SELECT 
                    ac.asset_chunk_uuid AS chunk_uuid,
                    d.title AS document_title,
                    a.asset_s3_path AS chunk_s3_path,
                    'image' AS chunk_type,
                    (1 - (ac.embedding <=> {vec_literal})) AS relevance
                FROM genai.asset_chunk AS ac
                JOIN genai.asset AS a ON ac.asset_uuid = a.asset_uuid
                JOIN genai.document AS d ON a.document_uuid = d.document_uuid
            ) AS combined
            ORDER BY relevance DESC
            LIMIT %s
        """

    results = []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (k,))
            rows = cur.fetchall()
            for r in rows:
                relevance = r.get("relevance")
                if relevance is not None and relevance >= threshold:
                    content = download_from_s3(r["chunk_s3_path"])
                    results.append(
                        Chunk(
                            chunk_uuid=str(r["chunk_uuid"]),
                            title=r["document_title"],
                            chunk_s3_path=r["chunk_s3_path"],
                            chunk_type=r["chunk_type"],
                            score=float(relevance),
                            content=content,
                        )
                    )
    results.sort(key=lambda c: c.score, reverse=True)
    return results
