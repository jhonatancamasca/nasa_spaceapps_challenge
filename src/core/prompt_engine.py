from typing import Dict

# central place for prompt templates and prompt building
TEMPLATES = {
    "summarize": "Summarize the following text in {n_sentences} sentences:\n\n{context}",
    "qa": "You are an assistant. Answer the question using the provided context.\nContext:\n{context}\nQuestion: {question}",
}


def build_prompt(template_name: str, **kwargs) -> str:
    template = TEMPLATES.get(template_name)
    if not template:
        raise ValueError("Unknown template")
    return template.format(**kwargs)