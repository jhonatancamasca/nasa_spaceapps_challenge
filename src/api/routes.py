from fastapi import APIRouter
from src.api.v1 import search, summarization, sessions, qa

api_router = APIRouter()
api_router.include_router(search.router)
api_router.include_router(summarization.router)
api_router.include_router(sessions.router)