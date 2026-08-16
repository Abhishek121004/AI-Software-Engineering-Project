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


@dataclass(slots=True)
class ArchitectureReport:
    repository_id: str
    summary: str
    top_level_directories: List[Dict[str, Any]]
    entrypoints: List[str]
    languages: Dict[str, int]
    dependency_edges: List[Dict[str, str]]
    notable_symbols: List[Dict[str, Any]]


@dataclass(slots=True)
class DependencyReport:
    repository_id: str
    summary: str
    internal_edges: List[Dict[str, str]]
    external_imports: List[str]
    fan_in: Dict[str, int]
    fan_out: Dict[str, int]


@dataclass(slots=True)
class CodeReviewIssue:
    severity: str
    file_path: str
    symbol: str
    line: Optional[int]
    message: str
    recommendation: str


@dataclass(slots=True)
class CodeReviewReport:
    repository_id: str
    summary: str
    issues: List[CodeReviewIssue]


@dataclass(slots=True)
class DocumentationArtifact:
    title: str
    markdown: str
    sources: List[SourceReference]


@dataclass(slots=True)
class TestArtifact:
    title: str
    markdown: str
    code: str
    source_file: str
    symbol: str
