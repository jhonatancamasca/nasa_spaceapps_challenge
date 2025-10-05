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