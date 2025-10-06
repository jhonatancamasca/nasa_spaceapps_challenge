import enum
from pydantic import BaseModel
from typing import Optional, List


class DocumentInput(BaseModel):
    s3_path: str
    title: str
    document_uuid: Optional[str] = None


class QueryInput(BaseModel):
    query: str


class Chunk(BaseModel):
    chunk_uuid: str  # ID of the chunk itself
    title: str  # Document title
    chunk_content: str | None = None  # Content downloaded from S3
    chunk_s3_path: str | None = None  # S3 path of the chunk
    chunk_type: str = "text_chunk"  # 'text' or 'image'
    score: float


class SummaryTypeEnum(str, enum.Enum):
    """Defines what type of summarization or report was generated."""

    query = "query"  # Summarization based on a query
    document = "document"  # Summarization based on a document
    scientific = "scientific"  # Scientific structured report
    general = "general"  # Generic summarization


class SummaryStatusEnum(str, enum.Enum):
    """Defines the current status of the summary generation process."""

    pending = "pending"
    processing = "processing"
    finished = "finished"
    failed = "failed"
