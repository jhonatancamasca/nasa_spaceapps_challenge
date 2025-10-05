from src.core.prompt_engine import build_prompt
from src.core.llm import get_llm_client

llm = get_llm_client()


def summarize_text(context: str, n_sentences: int = 5) -> str:
    prompt = build_prompt("summarize", context=context, n_sentences=n_sentences)
    res = llm.generate(prompt)
    return res.get("text") or ""


def generate_summary_for_document(document_uuid: str, text_chunks: list[str]):
    # naive: concatenate top chunks then summarize
    joined = "\n\n".join(text_chunks[:10])
    summary = summarize_text(joined, n_sentences=6)
    # store summary to local storage or S3 depending on config
    return {"summary": summary}