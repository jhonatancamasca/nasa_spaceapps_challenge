from typing import Any, Dict, Optional
from src.config import get_settings

settings = get_settings()


class LLMClient:
    def __init__(self, provider: str = None):
        self.provider = provider or settings.llm_provider
        self._client = None
        self.model = self._get_model_for_provider()

    def _get_model_for_provider(self) -> str:
        """Get the appropriate model for the current provider."""
        model_mapping = {
            "openai": getattr(settings, "openai_model", "gpt-3.5-turbo"),
            "groq": getattr(settings, "groq_model", "mixtral-8x7b-32768"),
            "ollama": getattr(settings, "ollama_model", "llama2"),
        }
        return model_mapping.get(self.provider, "")

    def _ensure_client(self):
        if self._client is not None:
            return

        if self.provider == "openai":
            import openai

            openai.api_key = settings.openai_api_key
            self._client = openai

        elif self.provider == "groq":
            import groq

            self._client = groq.Client(api_key=settings.groq_api_key)

        elif self.provider == "ollama":
            import ollama

            self._client = ollama.Client(
                host=getattr(settings, "ollama_host", "http://localhost:11434")
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        self._ensure_client()

        if self.provider == "openai":
            resp = self._client.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return {"text": resp.choices[0].message.content}

        elif self.provider == "groq":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return {"text": resp.choices[0].message.content}

        elif self.provider == "ollama":
            resp = self._client.generate(
                model=self.model,
                prompt=prompt,
                **kwargs,
            )
            return {"text": resp["response"]}

        else:
            return {"text": ""}


# Configuration helper to validate required settings
def validate_llm_config(provider: str) -> None:
    """Validate that required settings are present for the given provider."""
    required_settings = {
        "openai": ["openai_api_key", "openai_model"],
        "groq": ["groq_api_key", "groq_model"],
        "ollama": ["ollama_model"],  # API key not required for local Ollama
    }

    required = required_settings.get(provider, [])
    for setting in required:
        if not hasattr(settings, setting) or not getattr(settings, setting):
            raise ValueError(f"Missing required setting for {provider}: {setting}")


# helper single instance
_llm_client = None


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    global _llm_client
    provider = provider or settings.llm_provider

    validate_llm_config(provider)

    if _llm_client is None or _llm_client.provider != provider:
        _llm_client = LLMClient(provider)
    return _llm_client
