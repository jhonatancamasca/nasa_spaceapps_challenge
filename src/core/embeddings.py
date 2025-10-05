import os
import numpy as np
from huggingface_hub import InferenceClient
from functools import lru_cache
from src.config import get_settings

settings = get_settings()


@lru_cache()
def get_hf_client() -> InferenceClient:
    """Initialize and cache the Hugging Face Inference client."""
    return InferenceClient(provider="hf-inference", api_key=settings.HF_TOKEN)


def embed_query(
    text: str | list[str],
    model: str = "mixedbread-ai/mxbai-embed-large-v1",
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate normalized embeddings using the Hugging Face Inference API.

    Args:
        text: A single string or a list of strings.
        model: Embedding model to use.
        normalize: Whether to apply L2 normalization (recommended for cosine similarity).

    Returns:
        np.ndarray: 2D array (n, d) or 1D array (d,) if a single text input.
    """
    client = get_hf_client()

    # Request embeddings
    result = client.feature_extraction(text, model=model)

    # Convert to NumPy array
    emb = np.array(result, dtype=np.float32)

    return emb
