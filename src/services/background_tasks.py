import os
import traceback
from datetime import datetime
from typing import Literal, List, Optional
import uuid

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

from src.core.llm import get_llm_client
from src.core.retrieval import retrieve
from src.utils.s3_utils import download_images_from_s3, upload_file_to_s3
from src.utils.pdf_generator import generate_pdf_with_images
from src.db.session import get_db
from src.models.schemas import SummaryTypeEnum, SummaryStatusEnum
from datetime import datetime
import uuid


from sqlalchemy import Column, DateTime, String, Enum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


# --- 🔹 Summary model ---
class Summary(Base):
    __tablename__ = "summaries"

    summary_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(Enum(SummaryTypeEnum), nullable=False, default=SummaryTypeEnum.query)
    status = Column(
        Enum(SummaryStatusEnum), nullable=False, default=SummaryStatusEnum.pending
    )

    query = Column(String, nullable=True)
    summary_report_s3_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationship to SummaryDocument
    documents = relationship(
        "SummaryDocument", back_populates="summary", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Summary(uuid={self.summary_uuid}, type={self.type}, status={self.status})>"


# --- 🔹 SummaryDocument model ---
class SummaryDocument(Base):
    __tablename__ = "summary_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    summary_uuid = Column(
        UUID(as_uuid=True), ForeignKey("summaries.summary_uuid", ondelete="CASCADE")
    )
    document_uuid = Column(UUID(as_uuid=True), nullable=False)

    # Relationship back to Summary
    summary = relationship("Summary", back_populates="documents")

    def __repr__(self):
        return f"<SummaryDocument(summary_uuid={self.summary_uuid}, document_uuid={self.document_uuid})>"


class ScientificReport(BaseModel):
    title: str = Field(..., description="Title of the scientific report")
    executive_summary: str
    key_findings: List[str]
    knowledge_gaps: List[str]
    consensus_or_disagreement: Optional[str] = Field(
        None, description="Areas of consensus or disagreement"
    )
    actionable_insights: List[str]
    recommended_next_steps: List[str]
    relevant_sections: Optional[List[str]] = Field(
        None, description="Relevant sections or topics mentioned"
    )


def generate_scientific_report_background(
    query: str,
    k: int | None = None,
    threshold: float | None = None,
    source: Literal["chunk", "asset_chunk"] = "chunk",
    document_uuid: Optional[str] = None,
):
    """
    Background task:
    1. Retrieve relevant content
    2. Generate structured report via LLM (validated with Pydantic)
    3. Create a PDF with visuals
    4. Upload to S3
    5. Store metadata in DB
    """

    summary = None
    local_images = []
    local_pdf_path = None

    try:
        # --- 1. Retrieve chunks ---
        chunks = retrieve(query, k=k, threshold=threshold, source=source)
        if not chunks:
            raise ValueError("No relevant chunks found for query.")

        text_chunks = [
            c.page_content for c in chunks if getattr(c, "page_content", None)
        ]
        combined_text = "\n\n".join(text_chunks[:10])

        # --- 2. Define parser and prompt ---
        parser = PydanticOutputParser(pydantic_object=ScientificReport)

        prompt_template = """
            You are a scientific research analyst writing a comprehensive report.

            Emerging approaches in informatics and AI offer an opportunity to rethink how this information can be organized and summarized to describe research progress, identify gaps, and provide actionable information.

            Write a structured scientific report with the following sections:
            - title
            - executive_summary
            - key_findings
            - knowledge_gaps
            - consensus_or_disagreement
            - actionable_insights
            - recommended_next_steps
            - relevant_sections

            Context:
            {context}

            {format_instructions}
        """
        prompt = PromptTemplate(
            input_variables=["context"],
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        #llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        llm = get_llm_client()
        chain = LLMChain(prompt=prompt, llm=llm)

        # --- 3. Generate report ---
        raw_output = chain.run({"context": combined_text})
        report_model = parser.parse(raw_output)

        # --- 4. Download related images ---
        if source == "asset_chunk":
            image_paths = [
                c.asset_s3_path for c in chunks if getattr(c, "asset_s3_path", None)
            ]
            local_images = download_images_from_s3(image_paths)

        # --- 5. Generate PDF ---
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sanitized_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in query)
        local_pdf_path = f"/tmp/{sanitized_query}_scientific_report_{timestamp}.pdf"

        generate_pdf_with_images(
            report_model.dict(), local_images, output_path=local_pdf_path
        )

        # --- 6. Upload to S3 ---
        s3_pdf_url = upload_file_to_s3(local_pdf_path, s3_prefix="scientific_reports")

        # --- 7. Persist in DB ---
        with get_db() as db:
            summary = Summary(
                type=SummaryTypeEnum.query,
                summary_report_s3_path=s3_pdf_url,
                status=SummaryStatusEnum.finished,
            )
            db.add(summary)
            db.commit()
            db.refresh(summary)

            if document_uuid:
                summary_doc = SummaryDocument(
                    summary_uuid=summary.summary_uuid,
                    document_uuid=document_uuid,
                )
                db.add(summary_doc)
                db.commit()

        # ✅ Optional logging (background tasks don’t return values)
        print(f"[INFO] Scientific report generated for '{query}' → {s3_pdf_url}")

    except Exception as e:
        error_msg = f"[ERROR] Failed to generate report for '{query}': {e}"
        print(error_msg)
        traceback.print_exc()

        # Mark summary as failed if DB record was created
        if summary:
            with get_db() as db:
                summary.status = SummaryStatusEnum.failed
                db.commit()

    finally:
        # --- 8. Cleanup ---
        if local_pdf_path and os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)
        for img_path in local_images:
            if os.path.exists(img_path):
                os.remove(img_path)
