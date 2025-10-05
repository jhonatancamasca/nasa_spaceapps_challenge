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