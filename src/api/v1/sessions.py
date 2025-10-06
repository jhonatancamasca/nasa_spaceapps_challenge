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
    answer = "This is a stubbed answer."
    SESSIONS[session_uuid]["conversations"].append({"role": "user", "query": query.query, "answer": answer})
    return {"answer": answer}

@router.get("/{session_uuid}")
def get_session(session_uuid: str):
    return SESSIONS.get(session_uuid, {"error": "not found"})