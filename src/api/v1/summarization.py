from fastapi import APIRouter, BackgroundTasks
from src.models.schemas import DocumentInput
from src.services.background_tasks import add_document_summarization_task

router = APIRouter(prefix="/v1/summarization", tags=["summarization"])

@router.post("/start")
def start(doc: DocumentInput, background_tasks: BackgroundTasks):
    # For production, validate doc exists in storage
    # Simulate chunk extraction (in real life you'd extract text and call vectorizer)
    dummy_chunks = ["chunk 1 text", "chunk 2 text"]
    add_document_summarization_task(background_tasks, doc.document_uuid or "generated-uuid", dummy_chunks)
    return {"document_uuid": doc.document_uuid or "generated-uuid", "status": "processing"}

@router.get("/status/{document_uuid}")
def status(document_uuid: str):
    # Replace with DB lookup
    return {"document_uuid": document_uuid, "status": "completed"}