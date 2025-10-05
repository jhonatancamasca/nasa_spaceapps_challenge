Absolutely! Here’s a **fully polished, copy-paste ready README** that includes your previous content **plus the Docker Compose instructions**. It’s structured, clear, and developer-friendly.

---

# AI Knowledge API

**AI Knowledge API** is a FastAPI backend for **document search, summarization, and QA sessions**. It supports:

* **Vector search** with threshold (`pgvector` compatible)
* **LLM-powered summarization**
* **Interactive QA sessions**
* **Background task processing** using FastAPI `BackgroundTasks`
* **Modular LLM integration** (OpenAI, local LLaMA, Anthropic, etc.)

---

## Features

* **Document Search:** Retrieve top-k relevant chunks.
* **Summarization:** Summarize document chunks asynchronously.
* **QA Sessions:** Interactive sessions over stored documents.
* **Flexible LLM:** Easily swap providers without changing the API code.
* **Background Tasks:** No external task queue required.
* **Modular & Extensible:** Swap storage, DB, or LLM backend easily.

---

## Project Structure

```
ai-knowledge-api/
├─ .env.example
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ README.md
├─ alembic.ini            # optional
├─ src/
│  ├─ main.py
│  ├─ app.py              # FastAPI factory
│  ├─ config.py
│  ├─ api/
│  │  ├─ __init__.py
│  │  ├─ routes.py
│  │  ├─ v1/
│  │  │  ├─ __init__.py
│  │  │  ├─ search.py
│  │  │  ├─ summarization.py
│  │  │  ├─ sessions.py
│  │  │  └─ qa.py
│  ├─ core/
│  │  ├─ __init__.py
│  │  ├─ llm.py
│  │  ├─ prompt_engine.py
│  │  ├─ retrieval.py
│  │  ├─ summarizer.py
│  │  └─ storage.py
│  ├─ services/
│  │  ├─ __init__.py
│  │  └─ background_tasks.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ schemas.py
│  └─ tests/
└─ scripts/
   └─ start.sh
```

---



## Setup

1. **Clone the repo:**

```bash
git clone https://github.com/your-org/ai-knowledge-api.git
cd ai-knowledge-api
```

2. **Start the stack:**

```bash
docker compose up --build
```

3. **Verify API is running:**

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "app": "AI Knowledge API"
}
```

4. **Stop the stack:**

```bash
docker compose down
```

---

## API Endpoints

### Health

```
GET /health
```

### Search Documents

```
GET /api/v1/search/?query=...&k=5&threshold=0.75
```

### Summarization

```
POST /api/v1/summarization/start
GET /api/v1/summarization/status/{document_uuid}
```

### QA Sessions

```
POST /api/v1/sessions/
POST /api/v1/sessions/{session_uuid}/query
GET /api/v1/sessions/{session_uuid}
```


