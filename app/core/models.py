from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class RepositoryFile:
    file_path: str
    content: str
    language: str


@dataclass(slots=True)
class CodeChunk:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    content: str
    metadata: Dict[str, Any]
    score: float


@dataclass(slots=True)
class SourceReference:
    file_path: str
    symbol: str = "general"
    chunk_type: str = "general"
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    score: Optional[float] = None


@dataclass(slots=True)
class RepositoryCorpus:
    repository_id: str
    root_dir: Path
    files: List[RepositoryFile]
    chunks: List[CodeChunk]


@dataclass(slots=True)
class RAGContext:
    question: str
    repository_id: str
    context_text: str
    sources: List[SourceReference]
    retrieved_chunks: List[RetrievedChunk]
    memory_snippet: str = ""


@dataclass(slots=True)
class AnswerBundle:
    question: str
    answer: str
    sources: List[SourceReference]
    retrieved_chunks: List[RetrievedChunk]
    context_text: str
    latency_ms: float

