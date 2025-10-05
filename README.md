# AI Knowledge API — Project Structure & Core Code

This document provides a complete repo layout, `.env` example, `Dockerfile`, and ready-to-drop-in FastAPI code for the features you requested: **search (retrieval with threshold)**, **summarization**, **QA sessions**, **prompt/LLM loader**, and background handling using FastAPI `BackgroundTasks`. No Celery/MinIO — everything runs inside the same repo.

---

## Repo tree (suggested)

```
ai-knowledge-api/
├─ .env.example
├─ Dockerfile
├─ requirements.txt
├─ README.md
├─ alembic.ini            # optional, if you use migrations
├─ src/
│  ├─ main.py
│  ├─ app.py              # FastAPI application factory
│  ├─ config.py           # loads env
│  ├─ api/
│  │  ├─ __init__.py
│  │  ├─ routes.py        # includes routers registration
│  │  ├─ v1/
│  │  │  ├─ __init__.py
│  │  │  ├─ search.py
│  │  │  ├─ summarization.py
│  │  │  ├─ sessions.py
│  │  │  └─ qa.py
│  ├─ core/
│  │  ├─ __init__.py
│  │  ├─ llm.py           # LLM loader + wrapper
│  │  ├─ prompt_engine.py # prompt templates, prompt-building helpers
│  │  ├─ retrieval.py     # retrieval stubs (pgvector / vector DB)
│  │  ├─ summarizer.py    # summarization orchestration
│  │  └─ storage.py       # s3/minimal wrapper or local file stub (you said no minio)
│  ├─ services/
│  │  ├─ __init__.py
│  │  └─ background_tasks.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ schemas.py       # pydantic models
│  └─ tests/              # unit tests
└─ scripts/
   └─ start.sh
```

---

## .env.example

```
# FastAPI / environment
APP_NAME=AI_KNOWLEDGE_API
ENV=development
HOST=0.0.0.0
PORT=8000
WORKERS=1

# LLM provider (example values, customize)
LLM_PROVIDER=openai  # or local, llama, etc.
OPENAI_API_KEY=sk-REPLACE_ME
OPENAI_MODEL=gpt-4o-mini

# DB / pgvector (optional)
DATABASE_URL=postgresql://user:pass@localhost:5432/ai_db

# Storage (since no MinIO, we use S3-compatible or local)
STORAGE_TYPE=local  # local | s3
LOCAL_STORAGE_PATH=./data

# Other
DEFAULT_K=5
DEFAULT_THRESHOLD=0.75
```

---

## Dockerfile

```
FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## requirements.txt (minimal)

```
fastapi>=0.95
uvicorn[standard]>=0.21
pydantic>=1.10
python-dotenv>=1.0
requests>=2.28
httpx>=0.24
typing_extensions
psycopg2-binary>=2.9  # optional
sqlalchemy>=1.4        # optional
openai                 # optional if you use OpenAI
pytest
```

---

## Design pattern & philosophy

* **Application factory** (`app.py` + `main.py`) to allow testing and multiple workers.
* **Router-service-repository**: API routers call into `core` services (stateless), which call lower-level adapters (retrieval, storage). This separation makes it easy to swap LLM providers or vector DBs.
* **Dependency injection** via FastAPI `Depends` for config and clients.
* **Background tasks**: use FastAPI's `BackgroundTasks` for simple jobs (summaries). For heavier workloads you can later replace with a task queue — but you asked to keep it inside the repo.

---

## Key files (full code snippets)

### `src/config.py`

```python
from functools import lru_cache
from pydantic_settings  import BaseSettings

class Settings(BaseSettings):
    app_name: str = "AI Knowledge API"
    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    database_url: str | None = None

    default_k: int = 5
    default_threshold: float = 0.75

    storage_type: str = "local"
    local_storage_path: str = "./data"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

---

### `src/models/schemas.py`

```python
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
```

---

### `src/core/llm.py` (LLM loader & wrapper)

```python
from typing import Any, Dict
from src.config import get_settings

settings = get_settings()

class LLMClient:
    def __init__(self, provider: str = None):
        self.provider = provider or settings.llm_provider
        # lazy loaded client
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.provider == "openai":
            import openai
            openai.api_key = settings.openai_api_key
            self._client = openai
        else:
            # placeholder for other providers
            self._client = None

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        self._ensure_client()
        if self.provider == "openai":
            resp = self._client.ChatCompletion.create(
                model=settings.openai_model,
                messages=[{"role":"user","content":prompt}],
                **kwargs,
            )
            # adapt to simple dict
            return {"text": resp.choices[0].message.content}
        # fallback
        return {"text": ""}

# helper single instance
_llm_client = None

def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
```

---

### `src/core/prompt_engine.py`

```python
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
```

---

### `src/core/retrieval.py` (stub for pgvector retrieval)

```python
from typing import List
from src.models.schemas import Chunk
import random
from src.config import get_settings

settings = get_settings()

# NOTE: Replace with real DB/pgvector logic. This is a stub.

def retrieve(query: str, k: int = None, threshold: float | None = None) -> List[Chunk]:
    k = k or settings.default_k
    threshold = threshold or settings.default_threshold

    results = []
    for i in range(12):
        score = round(random.uniform(0.5, 0.99), 3)
        chunk = Chunk(
            document_uuid=str(i),
            title=f"Document {i}",
            chunk_content="Simulated chunk matching the query",
            chunk_s3_path=None,
            chunk_type="text_chunk",
            score=score,
        )
        if score >= threshold:
            results.append(chunk)
    results_sorted = sorted(results, key=lambda c: c.score, reverse=True)[:k]
    return results_sorted
```

---

### `src/core/summarizer.py` (summarization orchestration)

```python
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
```

---

### `src/services/background_tasks.py`

```python
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
```

---

### `src/api/v1/search.py`

```python
from fastapi import APIRouter, Query, Depends
from src.models.schemas import QueryInput
from src.core.retrieval import retrieve
from src.core.config import get_settings

router = APIRouter(prefix="/v1/search", tags=["search"])

@router.get("/")
def search(query: str, k: int = Query(None), threshold: float = Query(None)):
    settings = get_settings()
    k = k or settings.default_k
    threshold = threshold or settings.default_threshold
    chunks = retrieve(query, k=k, threshold=threshold)
    return {"query": query, "count": len(chunks), "results": [c.dict() for c in chunks]}
```

---

### `src/api/v1/summarization.py`

```python
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
```

---

### `src/api/v1/sessions.py` (QA sessions)

```python
from fastapi import APIRouter
from src.models.schemas import QueryInput
from uuid import uuid4

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])

# naive in-memory sessions store
SESSIONS = {}

@router.post("/")
def create_session():
    sid = str(uuid4())
    SESSIONS[sid] = {"session_uuid": sid, "conversations": []}
    return {"session_uuid": sid}

@router.post("/{session_uuid}/query")
def ask(session_uuid: str, query: QueryInput):
    # perform retrieval + prompt + llm
    # stubbed answer
    answer = "This is a stubbed answer."
    SESSIONS[session_uuid]["conversations"].append({"role": "user", "query": query.query, "answer": answer})
    return {"answer": answer}

@router.get("/{session_uuid}")
def get_session(session_uuid: str):
    return SESSIONS.get(session_uuid, {"error": "not found"})
```

---

### `src/api/routes.py` (register routers)

```python
from fastapi import APIRouter
from src.api.v1 import search, summarization, sessions, qa

api_router = APIRouter()
api_router.include_router(search.router)
api_router.include_router(summarization.router)
api_router.include_router(sessions.router)
# optionally include qa router if separate
```

---

### `src/app.py` (application factory)

```python
from fastapi import FastAPI
from src.config import get_settings
from src.api.routes import api_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(api_router, prefix="/api")

    # simple healthcheck
    @app.get("/health")
    def health():
        return {"status": "ok", "app": settings.app_name}

    return app
```

---

### `src/main.py`

```python
import uvicorn
from src.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## README — Basic usage

1. Copy `.env.example` to `.env` and fill in keys.
2. Build docker image: `docker build -t ai-knowledge-api .`
3. Run locally: `uvicorn src.main:app --reload`
4. Endpoints:

   * `GET /api/v1/search/?query=...&k=5&threshold=0.75`
   * `POST /api/v1/summarization/start` with `DocumentInput`
   * `GET /api/v1/summarization/status/{document_uuid}`
   * `POST /api/v1/sessions/` to create session
   * `POST /api/v1/sessions/{session_uuid}/query` to ask

---

## Notes & next steps

* Replace the retrieval stub with real pgvector queries when you integrate Postgres + pgvector.
* Swap the LLM client implementation in `src/core/llm.py` to any provider you want (local Llama, OpenAI, Anthropic) — keep the `LLMClient` interface stable so the rest of the code doesn't change.
* Add persistent storage (Postgres / file) for sessions and summaries.
* Add robust error handling, logging, and monitoring for production use.

---

If you'd like, I can now:

* generate each file here in the repo as separate code blocks, or
* create a zip containing the scaffold, or
* convert the stub retrieval to Postgres + pgvector queries (I can provide the SQL and Python examples).

Tell me which of those you want to do next and I'll produce it immediately.
