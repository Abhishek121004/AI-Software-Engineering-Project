from __future__ import annotations

import hashlib
import json
import math
import re
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from app.core.models import RetrievedChunk

try:  # pragma: no cover - optional acceleration
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - dependency not installed in workspace
    SentenceTransformer = None

GoogleGenerativeAIEmbeddings = None


@dataclass(slots=True)
class IndexedChunk:
    content: str
    metadata: Dict[str, Any]
    vector: List[float]


class _EmbeddingBackend:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class _SentenceTransformerBackend(_EmbeddingBackend):
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(value) for value in vector] for vector in self.model.encode(texts, normalize_embeddings=True)]

    def embed_query(self, text: str) -> List[float]:
        return [float(value) for value in self.model.encode([text], normalize_embeddings=True)[0]]


class _HashedBackend(_EmbeddingBackend):
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    @staticmethod
    def _tokens(text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text.lower())
        expanded: List[str] = []
        for token in tokens:
            expanded.append(token)
            if "_" in token:
                expanded.extend(part for part in token.split("_") if part)
        return expanded

    def _vector(self, text: str) -> List[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0
            if len(token) > 6:
                vector[(index + 7) % self.dimension] += 0.25
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


class CodeVectorStore:
    """Persistent repository-scoped vector index with metadata filtering."""

    def __init__(
        self,
        persist_dir: str | Path,
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
    ) -> None:
        self.persist_dir = Path(persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "index.json"
        self.embeddings_model_name = embeddings_model_name

        prefer_sentence_transformers = os.getenv("USE_SENTENCE_TRANSFORMERS", "").lower() in {"1", "true", "yes"}

        if callable(GoogleGenerativeAIEmbeddings):
            self.embeddings = GoogleGenerativeAIEmbeddings(  # type: ignore[operator]
                model=self.embeddings_model_name,
                google_api_key=None,
            )
        elif prefer_sentence_transformers and SentenceTransformer is not None:
            self.embeddings: _EmbeddingBackend = _SentenceTransformerBackend(self.embeddings_model_name)
        else:
            self.embeddings = _HashedBackend(dimension=embedding_dimension)

        self._entries: List[IndexedChunk] = []
        self._load()

    @property
    def count(self) -> int:
        return len(self._entries)

    def _load(self) -> None:
        if not self.index_path.exists():
            self._entries = []
            return

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self._entries = [IndexedChunk(**entry) for entry in payload.get("entries", [])]

    def _save(self) -> None:
        payload = {"entries": [asdict(entry) for entry in self._entries]}
        self.index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    @staticmethod
    def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return normalized

    @staticmethod
    def _metadata_matches(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    @staticmethod
    def _identifier_terms(text: str) -> List[str]:
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)

    def add_chunks(self, chunks: List[Dict[str, Any]], repository_id: str) -> None:
        texts: List[str] = []
        records: List[Dict[str, Any]] = []

        for chunk in chunks:
            metadata = self._normalize_metadata(dict(chunk["metadata"]))
            metadata["repository_id"] = repository_id
            texts.append(chunk["content"])
            records.append({"content": chunk["content"], "metadata": metadata})

        if not texts:
            return

        vectors = self.embeddings.embed_documents(texts)
        for record, vector in zip(records, vectors):
            self._entries.append(
                IndexedChunk(
                    content=record["content"],
                    metadata=record["metadata"],
                    vector=[float(value) for value in vector],
                )
            )

        self._save()

    def clear_repository(self, repository_id: str) -> None:
        self._entries = [entry for entry in self._entries if entry.metadata.get("repository_id") != repository_id]
        self._save()

    def list_repositories(self) -> List[str]:
        repositories = sorted({str(entry.metadata.get("repository_id", "")) for entry in self._entries if entry.metadata.get("repository_id")})
        return repositories

    def _score_entry(self, query_vector: np.ndarray, query: str, entry: IndexedChunk) -> float:
        candidate = np.asarray(entry.vector, dtype=np.float32)
        denom = float(np.linalg.norm(candidate) * np.linalg.norm(query_vector))
        semantic_score = float(np.dot(query_vector, candidate) / denom) if denom else 0.0

        boost = 0.0
        query_terms = {term.lower() for term in self._identifier_terms(query)}
        candidate_terms = {
            str(entry.metadata.get("symbol", "")).lower(),
            str(entry.metadata.get("file_path", "")).lower(),
            entry.content.lower(),
        }
        for term in query_terms:
            if not term:
                continue
            if any(term in candidate_text for candidate_text in candidate_terms):
                boost += 0.12

        if query_terms and str(entry.metadata.get("symbol", "")).lower() in query_terms:
            boost += 0.2

        return semantic_score + boost

    def search(
        self,
        query: str,
        repository_id: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        filters = {"repository_id": repository_id}
        if filter_dict:
            filters.update(filter_dict)

        query_vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        scored: List[RetrievedChunk] = []

        for entry in self._entries:
            if not self._metadata_matches(entry.metadata, filters):
                continue
            score = self._score_entry(query_vector, query, entry)
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            scored.append(RetrievedChunk(content=entry.content, metadata=dict(entry.metadata), score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    def exact_match(self, term: str, repository_id: str, k: int = 10) -> List[RetrievedChunk]:
        term_lower = term.lower()
        matches: List[RetrievedChunk] = []
        for entry in self._entries:
            if entry.metadata.get("repository_id") != repository_id:
                continue
            haystack = " ".join(
                [
                    str(entry.metadata.get("symbol", "")),
                    str(entry.metadata.get("file_path", "")),
                    entry.content,
                ]
            ).lower()
            if term_lower in haystack:
                matches.append(
                    RetrievedChunk(content=entry.content, metadata=dict(entry.metadata), score=1.0 + haystack.count(term_lower) * 0.05)
                )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:k]
