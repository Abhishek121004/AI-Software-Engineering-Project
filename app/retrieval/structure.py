from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from app.core.models import CodeChunk, RepositoryCorpus, RetrievedChunk


def _split_identifier(value: str) -> List[str]:
    parts: List[str] = []
    for token in re.split(r"[^A-Za-z0-9_./-]+", value):
        if not token:
            continue
        parts.append(token.lower())
        snake_parts = [part for part in token.split("_") if part]
        parts.extend(part.lower() for part in snake_parts if part.lower() not in parts)
        camel_parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token).split()
        parts.extend(part.lower() for part in camel_parts if part.lower() not in parts)
    return list(dict.fromkeys(parts))


def _chunk_text_tokens(chunk: CodeChunk) -> List[str]:
    meta = chunk.metadata
    tokens = []
    tokens.extend(_split_identifier(str(meta.get("file_path", ""))))
    tokens.extend(_split_identifier(str(meta.get("symbol", ""))))
    tokens.extend(_split_identifier(str(meta.get("chunk_type", ""))))
    tokens.extend(_split_identifier(chunk.content[:400]))
    return list(dict.fromkeys(token for token in tokens if token))


def _parse_imports_for_file(file_path: str, content: str, language: str) -> List[str]:
    imports: List[str] = []
    if language == "python":
        for match in re.finditer(r"^\s*from\s+([A-Za-z0-9_./]+)\s+import\s+", content, flags=re.MULTILINE):
            imports.append(match.group(1))
        for match in re.finditer(r"^\s*import\s+([A-Za-z0-9_./, ]+)", content, flags=re.MULTILINE):
            imports.extend(part.strip() for part in match.group(1).split(","))
    elif language in {"javascript", "typescript", "jsx", "tsx"}:
        patterns = [
            r"""^\s*import\s+.*?\s+from\s+["']([^"']+)["']""",
            r"""^\s*import\s+["']([^"']+)["']""",
            r"""require\(\s*["']([^"']+)["']\s*\)""",
            r"""^\s*export\s+.*?\s+from\s+["']([^"']+)["']""",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, content, flags=re.MULTILINE):
                imports.append(match.group(1))
    elif language == "java":
        for match in re.finditer(r"^\s*import\s+([A-Za-z0-9_.*]+)", content, flags=re.MULTILINE):
            imports.append(match.group(1))
    elif language == "cpp":
        for match in re.finditer(r"""^\s*#include\s+[<"]([^>"]+)[>"]""", content, flags=re.MULTILINE):
            imports.append(match.group(1))
    return list(dict.fromkeys(imports))


@dataclass(slots=True)
class StructureHit:
    chunk: CodeChunk
    score: float
    reasons: List[str]


class RepositoryStructureIndex:
    """Inspectable code-structure index for symbol, path, and dependency signals."""

    def __init__(self, corpus: RepositoryCorpus) -> None:
        self.corpus = corpus
        self._symbol_to_chunks: Dict[str, List[CodeChunk]] = defaultdict(list)
        self._file_to_chunks: Dict[str, List[CodeChunk]] = defaultdict(list)
        self._language_to_chunks: Dict[str, List[CodeChunk]] = defaultdict(list)
        self._imports_by_file: Dict[str, List[str]] = {}
        self._imported_by: Dict[str, List[str]] = defaultdict(list)
        self._token_counts: Counter[str] = Counter()

        for chunk in corpus.chunks:
            file_path = str(chunk.metadata.get("file_path", ""))
            symbol = str(chunk.metadata.get("symbol", "general")).lower()
            language = str(chunk.metadata.get("language", ""))
            self._symbol_to_chunks[symbol].append(chunk)
            self._file_to_chunks[file_path.lower()].append(chunk)
            self._language_to_chunks[language.lower()].append(chunk)
            for token in _chunk_text_tokens(chunk):
                self._token_counts[token] += 1

        for file in corpus.files:
            imports = _parse_imports_for_file(file.file_path, file.content, file.language)
            self._imports_by_file[file.file_path] = imports
            for imported in imports:
                self._imported_by[imported.lower()].append(file.file_path)

    @staticmethod
    def query_tokens(query: str) -> List[str]:
        return _split_identifier(query)

    def _score_chunk(self, query: str, chunk: CodeChunk) -> StructureHit:
        tokens = self.query_tokens(query)
        file_path = str(chunk.metadata.get("file_path", "")).lower()
        symbol = str(chunk.metadata.get("symbol", "general")).lower()
        chunk_type = str(chunk.metadata.get("chunk_type", "general")).lower()
        language = str(chunk.metadata.get("language", "")).lower()
        score = 0.0
        reasons: List[str] = []

        for token in tokens:
            if token in symbol:
                score += 1.5
                reasons.append(f"symbol match:{token}")
            if token in file_path:
                score += 1.2
                reasons.append(f"path match:{token}")
            if token in language:
                score += 0.5
                reasons.append(f"language match:{token}")

        if query.lower() in file_path:
            score += 1.0
            reasons.append("full file path match")

        if query.lower() in symbol:
            score += 1.5
            reasons.append("full symbol match")

        if any(term in query.lower() for term in ("function", "method", "call", "logic")) and chunk_type == "function":
            score += 0.7
            reasons.append("function-oriented query")
        if any(term in query.lower() for term in ("class", "object", "model")) and chunk_type == "class":
            score += 0.7
            reasons.append("class-oriented query")
        if any(term in query.lower() for term in ("module", "dependency", "import", "reference")):
            imports = self._imports_by_file.get(chunk.metadata.get("file_path", ""), [])
            if imports:
                score += 0.4
                reasons.append("dependency context")
        if chunk_type != "general" and chunk_type in query.lower():
            score += 0.25
            reasons.append("chunk-type hint")
        if chunk.content and any(token in chunk.content.lower() for token in tokens):
            score += 0.2
            reasons.append("content token overlap")

        return StructureHit(chunk=chunk, score=score, reasons=reasons)

    def search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict[str, object]] = None) -> List[RetrievedChunk]:
        hits: List[RetrievedChunk] = []
        for chunk in self.corpus.chunks:
            if filter_dict:
                metadata = chunk.metadata
                if any(metadata.get(key) != value for key, value in filter_dict.items()):
                    continue
            hit = self._score_chunk(query, chunk)
            if hit.score <= 0:
                continue
            metadata = dict(hit.chunk.metadata)
            metadata["structure_reasons"] = hit.reasons
            hits.append(RetrievedChunk(content=hit.chunk.content, metadata=metadata, score=hit.score))

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def symbol_hits(self, symbol: str) -> List[RetrievedChunk]:
        chunks = self._symbol_to_chunks.get(symbol.lower(), [])
        results = [
            RetrievedChunk(content=chunk.content, metadata=dict(chunk.metadata), score=2.0)
            for chunk in chunks
        ]
        return results

    def file_hits(self, file_path: str) -> List[RetrievedChunk]:
        chunks = self._file_to_chunks.get(file_path.lower(), [])
        results = [
            RetrievedChunk(content=chunk.content, metadata=dict(chunk.metadata), score=1.6)
            for chunk in chunks
        ]
        return results

    def related_files_for_symbol(self, symbol: str) -> List[str]:
        related = self._imported_by.get(symbol.lower(), [])
        return sorted(dict.fromkeys(related))

    def import_map(self) -> Dict[str, List[str]]:
        return {file_path: list(imports) for file_path, imports in self._imports_by_file.items()}
