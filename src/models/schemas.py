from pydantic import BaseModel
from typing import Optional, List

class DocumentInput(BaseModel):
    s3_path: str
    title: str
    document_uuid: Optional[str] = None

class QueryInput(BaseModel):
    query: str

class Chunk(BaseModel):
    document_uuid: str
    title: str
    chunk_content: str | None = None
    chunk_s3_path: str | None = None
    chunk_type: str = "text_chunk"
    score: float