from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(slots=True)
class SymbolSpan:
    name: str
    chunk_type: str
    start_line: int
    end_line: int


class CodeAwareChunker:
    """Chunks repository files on logical code boundaries."""

    def __init__(self, chunk_size: int = 80, chunk_overlap: int = 12) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))

    def _split_range(self, start_line: int, end_line: int) -> Iterable[tuple[int, int]]:
        current = start_line
        while current <= end_line:
            chunk_end = min(end_line, current + self.chunk_size - 1)
            yield current, chunk_end
            if chunk_end >= end_line:
                break
            current = max(current + 1, chunk_end - self.chunk_overlap + 1)

    def _python_symbols(self, content: str) -> List[SymbolSpan]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        symbols: List[SymbolSpan] = []

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.class_stack: List[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                end_line = getattr(node, "end_lineno", node.lineno)
                symbols.append(SymbolSpan(node.name, "class", node.lineno, end_line))
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                end_line = getattr(node, "end_lineno", node.lineno)
                symbols.append(SymbolSpan(node.name, "function", node.lineno, end_line))
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                end_line = getattr(node, "end_lineno", node.lineno)
                symbols.append(SymbolSpan(node.name, "function", node.lineno, end_line))
                self.generic_visit(node)

        Visitor().visit(tree)
        return sorted(symbols, key=lambda item: (item.start_line, item.end_line, item.name))

    def _regex_symbols(self, content: str, language: str) -> List[SymbolSpan]:
        lines = content.splitlines()
        if not lines:
            return []

        patterns = [
            (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("), "function"),
            (re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b"), "class"),
            (re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*=>"), "function"),
            (re.compile(r"^\s*#\s+(.+)$"), "section"),
            (re.compile(r"^\s*##+\s+(.+)$"), "section"),
        ]

        symbol_starts: List[tuple[int, str, str]] = []
        for index, line in enumerate(lines, start=1):
            for pattern, chunk_type in patterns:
                match = pattern.search(line)
                if match:
                    name = match.group(1).strip()
                    if chunk_type == "section":
                        name = re.sub(r"\s+", " ", name)
                    symbol_starts.append((index, name, chunk_type))
                    break

        if not symbol_starts:
            return []

        spans: List[SymbolSpan] = []
        for i, (start_line, name, chunk_type) in enumerate(symbol_starts):
            if i + 1 < len(symbol_starts):
                end_line = symbol_starts[i + 1][0] - 1
            else:
                end_line = len(lines)
            if end_line < start_line:
                end_line = start_line
            spans.append(SymbolSpan(name=name, chunk_type=chunk_type, start_line=start_line, end_line=end_line))
        return spans

    def _gaps(self, symbols: List[SymbolSpan], total_lines: int) -> List[SymbolSpan]:
        if not symbols:
            return [SymbolSpan("general", "general", 1, total_lines)]

        gaps: List[SymbolSpan] = []
        cursor = 1
        for symbol in symbols:
            if cursor < symbol.start_line:
                gaps.append(SymbolSpan("general", "general", cursor, symbol.start_line - 1))
            cursor = max(cursor, symbol.end_line + 1)
        if cursor <= total_lines:
            gaps.append(SymbolSpan("general", "general", cursor, total_lines))
        return gaps

    def _sections(self, content: str, language: str) -> List[SymbolSpan]:
        lines = content.splitlines()
        if not lines:
            return []

        if language == "python":
            symbols = self._python_symbols(content)
        else:
            symbols = self._regex_symbols(content, language)

        sections = [*symbols, *self._gaps(symbols, len(lines))]
        sections.sort(key=lambda item: (item.start_line, item.end_line, item.name))
        return sections

    def chunk_file(self, file_data: Dict[str, str], repository_id: str = "repository") -> List[Dict[str, Any]]:
        file_path = file_data["file_path"]
        content = file_data["content"]
        language = file_data["language"]
        lines = content.splitlines()
        if not lines:
            return []

        sections = self._sections(content, language)
        if not sections:
            sections = [SymbolSpan("general", "general", 1, len(lines))]

        chunks: List[Dict[str, Any]] = []
        chunk_index = 0
        for section in sections:
            for start_line, end_line in self._split_range(section.start_line, section.end_line):
                start_index = start_line - 1
                end_index = end_line
                chunk_lines = lines[start_index:end_index]
                chunk_text = "\n".join(chunk_lines).strip("\n")
                if not chunk_text.strip():
                    continue

                metadata = {
                    "repository_id": repository_id,
                    "file_path": file_path,
                    "language": language,
                    "symbol": section.name,
                    "chunk_type": section.chunk_type,
                    "start_line": start_line,
                    "end_line": end_line,
                    "chunk_index": chunk_index,
                }
                chunks.append({"content": chunk_text, "metadata": metadata})
                chunk_index += 1

        return chunks

    def chunk_all(self, files_list: List[Dict[str, str]], repository_id: str = "repository") -> List[Dict[str, Any]]:
        all_chunks: List[Dict[str, Any]] = []
        for file_data in files_list:
            all_chunks.extend(self.chunk_file(file_data, repository_id=repository_id))
        return all_chunks
