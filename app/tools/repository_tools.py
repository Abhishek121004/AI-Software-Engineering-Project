from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.models import RepositoryCorpus, RetrievedChunk
from app.retrieval.vector_store import CodeVectorStore


@dataclass
class RepositoryTools:
    corpus: RepositoryCorpus
    vector_store: CodeVectorStore

    def __post_init__(self) -> None:
        self.root_dir = self.corpus.root_dir.resolve()
        self._files_by_path = {file.file_path: file for file in self.corpus.files}

    def _safe_relative_path(self, file_path: str) -> Path:
        candidate = Path(file_path)
        if candidate.is_absolute():
            raise ValueError("Absolute paths are not allowed.")

        resolved = (self.root_dir / candidate).resolve()
        if self.root_dir not in resolved.parents and resolved != self.root_dir:
            raise ValueError("Path escapes the repository root.")
        return resolved

    def search_code(self, query: str, top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        return self.vector_store.search(query=query, repository_id=self.corpus.repository_id, k=top_k, filter_dict=filter_dict)

    def read_file(self, file_path: str) -> Dict[str, Any]:
        resolved = self._safe_relative_path(file_path)
        relative_path = resolved.relative_to(self.root_dir).as_posix()
        file_entry = self._files_by_path.get(relative_path)
        if file_entry is None:
            if not resolved.exists() or not resolved.is_file():
                raise FileNotFoundError(f"File not found in repository: {file_path}")
            content = resolved.read_text(encoding="utf-8")
            language = resolved.suffix.lstrip(".").lower() or "text"
        else:
            content = file_entry.content
            language = file_entry.language
        return {"file_path": relative_path, "language": language, "content": content}

    def find_function(self, symbol: str) -> List[RetrievedChunk]:
        symbol_lower = symbol.lower()
        matches: List[RetrievedChunk] = []
        for chunk in self.corpus.chunks:
            if str(chunk.metadata.get("symbol", "")).lower() == symbol_lower:
                matches.append(
                    RetrievedChunk(content=chunk.content, metadata=dict(chunk.metadata), score=1.0)
                )
        if matches:
            return matches
        return self.search_code(symbol, top_k=5, filter_dict={"chunk_type": "function"})

    def find_references(self, symbol: str) -> List[Dict[str, Any]]:
        symbol_lower = symbol.lower()
        results: List[Dict[str, Any]] = []
        for file in self.corpus.files:
            if symbol_lower not in file.content.lower():
                continue
            results.append({"file_path": file.file_path, "language": file.language, "symbol": symbol})
        return results

    def get_repository_tree(self) -> Dict[str, Any]:
        paths = sorted(file.file_path for file in self.corpus.files)
        return {"repository_id": self.corpus.repository_id, "files": paths, "count": len(paths)}
