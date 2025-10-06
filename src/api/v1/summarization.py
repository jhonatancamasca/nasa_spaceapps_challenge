from typing import Literal
from fastapi import APIRouter, BackgroundTasks, Query
from src.models.schemas import DocumentInput
from src.services.background_tasks import generate_scientific_report_background

router = APIRouter(prefix="/v1/summarization", tags=["summarization"])


@router.get("/analyze")
def analyze_scientific_query(
    query: str,
    background_tasks: BackgroundTasks,
    k: int = Query(None),
    threshold: float = Query(None),
    source: Literal["chunk", "asset_chunk"] = Query("chunk"),
):
    """
    Launch background report generation.
    """
    background_tasks.add_task(
        generate_scientific_report_background,
        query=query,
        k=k,
        threshold=threshold,
        source=source,
    )

    return {
        "status": "processing",
        "message": f"Scientific report generation started for query '{query}'",
    }


@router.get("/status/{document_uuid}")
def status(document_uuid: str):
    # Replace with DB lookup
    return {"document_uuid": document_uuid, "status": "completed"}
