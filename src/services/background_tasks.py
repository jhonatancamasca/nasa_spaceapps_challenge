from fastapi import BackgroundTasks
from src.core.summarizer import generate_summary_for_document
from src.config import get_settings
from src.models.schemas import DocumentInput

# a small wrapper so you can import and add tasks easily

def add_document_summarization_task(background_tasks: BackgroundTasks, document_uuid: str, chunks: list[str]):
    background_tasks.add_task(_background_summarize, document_uuid, chunks)


def _background_summarize(document_uuid: str, chunks: list[str]):
    # run sequentially; this will be executed in worker thread by FastAPI
    result = generate_summary_for_document(document_uuid, chunks)
    # store result to local file or update in memory store
    print(f"Finished summary for {document_uuid}")
