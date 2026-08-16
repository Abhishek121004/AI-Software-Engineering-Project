from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.config import settings

try:  # pragma: no cover - optional runtime dependency is available in the app environment
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except Exception:  # pragma: no cover - tests can still run with injected doubles
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None


@dataclass(slots=True)
class GeminiModels:
    chat_model: object
    embeddings_model: object


def has_gemini_runtime() -> bool:
    return settings.has_gemini_key and ChatGoogleGenerativeAI is not None and GoogleGenerativeAIEmbeddings is not None


def create_gemini_chat_model(model_name: Optional[str] = None, temperature: float = 0.0):
    if not settings.has_gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("langchain_google_genai is not available.")
    return ChatGoogleGenerativeAI(
        model=model_name or settings.gemini_chat_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
    )


def create_gemini_embeddings_model(model_name: Optional[str] = None):
    if not settings.has_gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    if GoogleGenerativeAIEmbeddings is None:
        raise RuntimeError("langchain_google_genai is not available.")
    return GoogleGenerativeAIEmbeddings(
        model=model_name or settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
    )

