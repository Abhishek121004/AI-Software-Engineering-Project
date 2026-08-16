from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.core.models import (
    ArchitectureReport,
    CodeReviewIssue,
    CodeReviewReport,
    DependencyReport,
    DocumentationArtifact,
    RetrievedChunk,
    SourceReference,
    TestArtifact,
)
from app.core.repository import RepositoryIndexer
from app.memory.conversation import ConversationMemory
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rag import RepositoryRAG
from app.tools.repository_tools import RepositoryTools


_ENTRYPOINT_NAMES = {
    "main.py",
    "app.py",
    "server.py",
    "index.js",
    "index.ts",
    "server.js",
    "server.ts",
    "__main__.py",
}


@dataclass(slots=True)
class ImportRecord:
    source_file: str
    imported: str
    kind: str
    internal: bool


class RepositoryIntelligence:
    """Repository-focused analysis and generation utilities built on top of RAG."""

    def __init__(
        self,
        tools: RepositoryTools,
        rag: Optional[RepositoryRAG] = None,
        retriever: Optional[HybridRetriever] = None,
    ) -> None:
        self.tools = tools
        self.rag = rag
        self.corpus = tools.corpus
        self.hybrid_retriever = retriever or HybridRetriever(self.corpus, tools.vector_store)

    def _files_by_language(self, language: Optional[str] = None):
        for file in self.corpus.files:
            if language is None or file.language == language:
                yield file

    def _chunk_sources_for_file(self, file_path: str) -> List[SourceReference]:
        sources: List[SourceReference] = []
        for chunk in self.corpus.chunks:
            if str(chunk.metadata.get("file_path")) != file_path:
                continue
            sources.append(
                SourceReference(
                    file_path=file_path,
                    symbol=str(chunk.metadata.get("symbol", "general")),
                    chunk_type=str(chunk.metadata.get("chunk_type", "general")),
                    start_line=int(chunk.metadata["start_line"]) if chunk.metadata.get("start_line") is not None else None,
                    end_line=int(chunk.metadata["end_line"]) if chunk.metadata.get("end_line") is not None else None,
                )
            )
        return sources

    def _file_symbol_names(self, file_path: str) -> List[str]:
        return sorted(
            {
                str(chunk.metadata.get("symbol", "general"))
                for chunk in self.corpus.chunks
                if str(chunk.metadata.get("file_path")) == file_path and str(chunk.metadata.get("symbol", "general")) != "general"
            }
        )

    def _top_level_directory_stats(self) -> List[Dict[str, object]]:
        counts: Counter[str] = Counter()
        for file in self.corpus.files:
            parts = Path(file.file_path).parts
            top = parts[0] if len(parts) > 1 else "."
            counts[top] += 1
        return [{"directory": name, "file_count": count} for name, count in counts.most_common()]

    def _entrypoints(self) -> List[str]:
        entrypoints: List[str] = []
        for file in self.corpus.files:
            path = Path(file.file_path)
            if path.name in _ENTRYPOINT_NAMES:
                entrypoints.append(file.file_path)
                continue
            if file.language == "python" and "if __name__ == \"__main__\"" in file.content:
                entrypoints.append(file.file_path)
        return entrypoints

    def _language_counts(self) -> Dict[str, int]:
        return dict(Counter(file.language for file in self.corpus.files))

    def _python_imports(self, file) -> List[ImportRecord]:
        try:
            tree = ast.parse(file.content)
        except SyntaxError:
            return []

        records: List[ImportRecord] = []
        known_targets = self._known_internal_targets()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    records.append(
                        ImportRecord(
                            source_file=file.file_path,
                            imported=imported,
                            kind="import",
                            internal=self._is_internal_target(imported, known_targets),
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = "." * node.level + module
                for alias in node.names:
                    imported = f"{module}:{alias.name}" if module else alias.name
                    records.append(
                        ImportRecord(
                            source_file=file.file_path,
                            imported=imported,
                            kind="from",
                            internal=self._is_internal_target(module or alias.name, known_targets),
                        )
                    )
        return records

    def _js_imports(self, file) -> List[ImportRecord]:
        records: List[ImportRecord] = []
        known_targets = self._known_internal_targets()
        patterns = [
            re.compile(r"""^\s*import\s+.*?\s+from\s+["']([^"']+)["']""", re.MULTILINE),
            re.compile(r"""^\s*import\s+["']([^"']+)["']""", re.MULTILINE),
            re.compile(r"""require\(\s*["']([^"']+)["']\s*\)""", re.MULTILINE),
            re.compile(r"""^\s*export\s+.*?\s+from\s+["']([^"']+)["']""", re.MULTILINE),
        ]
        for pattern in patterns:
            for match in pattern.finditer(file.content):
                imported = match.group(1)
                records.append(
                    ImportRecord(
                        source_file=file.file_path,
                        imported=imported,
                        kind="js",
                        internal=imported.startswith(".") or self._is_internal_target(imported, known_targets),
                    )
                )
        return records

    def _java_imports(self, file) -> List[ImportRecord]:
        records: List[ImportRecord] = []
        known_targets = self._known_internal_targets()
        for line in file.content.splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                imported = stripped.removeprefix("import ").rstrip(";")
                records.append(
                    ImportRecord(
                        source_file=file.file_path,
                        imported=imported,
                        kind="java",
                        internal=self._is_internal_target(imported, known_targets),
                    )
                )
        return records

    def _cpp_imports(self, file) -> List[ImportRecord]:
        records: List[ImportRecord] = []
        for match in re.finditer(r"""^\s*#include\s+[<"]([^>"]+)[>"]""", file.content, flags=re.MULTILINE):
            imported = match.group(1)
            records.append(
                ImportRecord(
                    source_file=file.file_path,
                    imported=imported,
                    kind="cpp",
                    internal=imported.startswith('"') or imported.startswith("."),
                )
            )
        return records

    def _known_internal_targets(self) -> List[str]:
        targets: List[str] = []
        for file in self.corpus.files:
            path = Path(file.file_path)
            targets.append(path.stem)
            targets.append(path.with_suffix("").as_posix())
            targets.extend(path.with_suffix("").parts)
        return list(dict.fromkeys(targets))

    @staticmethod
    def _is_internal_target(candidate: str, known_targets: Sequence[str]) -> bool:
        candidate = candidate.replace("/", ".").replace(":", ".").replace("-", "_")
        if candidate.startswith("."):
            return True
        tail = candidate.split(".")[-1]
        return candidate in known_targets or tail in known_targets or candidate.replace(".", "/") in known_targets

    def dependency_analysis(self) -> DependencyReport:
        records: List[ImportRecord] = []
        for file in self.corpus.files:
            if file.language == "python":
                records.extend(self._python_imports(file))
            elif file.language in {"javascript", "typescript", "jsx", "tsx"}:
                records.extend(self._js_imports(file))
            elif file.language == "java":
                records.extend(self._java_imports(file))
            elif file.language == "cpp":
                records.extend(self._cpp_imports(file))

        internal_edges: List[Dict[str, str]] = []
        external_imports: List[str] = []
        fan_in: Counter[str] = Counter()
        fan_out: Counter[str] = Counter()

        for record in records:
            if record.internal:
                internal_edges.append(
                    {
                        "source_file": record.source_file,
                        "target": record.imported,
                        "kind": record.kind,
                    }
                )
                fan_out[record.source_file] += 1
                fan_in[record.imported] += 1
            else:
                external_imports.append(record.imported)

        unique_external = sorted(dict.fromkeys(external_imports))
        summary = (
            f"Found {len(internal_edges)} internal dependency edge(s) and {len(unique_external)} external import(s)."
        )
        return DependencyReport(
            repository_id=self.corpus.repository_id,
            summary=summary,
            internal_edges=internal_edges,
            external_imports=unique_external,
            fan_in=dict(fan_in),
            fan_out=dict(fan_out),
        )

    def architecture_analysis(self) -> ArchitectureReport:
        dependencies = self.dependency_analysis()
        notable_symbols: List[Dict[str, object]] = []
        for file in self.corpus.files:
            symbols = self._file_symbol_names(file.file_path)
            if symbols:
                notable_symbols.append({"file_path": file.file_path, "symbols": symbols[:5]})

        summary = (
            f"{len(self.corpus.files)} file(s), {len(self.corpus.chunks)} chunk(s), "
            f"{len(self._language_counts())} language group(s), and {len(dependencies.internal_edges)} internal dependency link(s)."
        )
        return ArchitectureReport(
            repository_id=self.corpus.repository_id,
            summary=summary,
            top_level_directories=self._top_level_directory_stats(),
            entrypoints=self._entrypoints(),
            languages=self._language_counts(),
            dependency_edges=dependencies.internal_edges,
            notable_symbols=notable_symbols,
        )

    def code_search(self, query: str, top_k: int = 5):
        return self.hybrid_retriever.retrieve(
            query=query,
            repository_id=self.corpus.repository_id,
            top_k=top_k,
        ).chunks

    def code_review(self, target_file: Optional[str] = None) -> CodeReviewReport:
        files = [file for file in self.corpus.files if target_file is None or file.file_path == target_file]
        issues: List[CodeReviewIssue] = []

        for file in files:
            lines = file.content.splitlines()
            for index, line in enumerate(lines, start=1):
                if "TODO" in line or "FIXME" in line:
                    issues.append(
                        CodeReviewIssue(
                            severity="medium",
                            file_path=file.file_path,
                            symbol="general",
                            line=index,
                            message="TODO/FIXME marker remains in the code.",
                            recommendation="Either complete the implementation or capture the follow-up in an issue.",
                        )
                    )
                if "print(" in line or "console.log(" in line:
                    issues.append(
                        CodeReviewIssue(
                            severity="low",
                            file_path=file.file_path,
                            symbol="general",
                            line=index,
                            message="Debug-style output is present.",
                            recommendation="Remove or replace with structured logging before shipping.",
                        )
                    )
                if re.search(r"\bexcept\s+Exception\b", line) or re.search(r"\bbare\s+except\b", line):
                    issues.append(
                        CodeReviewIssue(
                            severity="high",
                            file_path=file.file_path,
                            symbol="general",
                            line=index,
                            message="Broad exception handling can hide failures.",
                            recommendation="Catch specific exceptions and preserve the failure context.",
                        )
                    )
                if re.search(r"(password|api[_-]?key|secret|token)\s*=", line, flags=re.IGNORECASE):
                    issues.append(
                        CodeReviewIssue(
                            severity="high",
                            file_path=file.file_path,
                            symbol="general",
                            line=index,
                            message="Potential hard-coded credential or secret pattern found.",
                            recommendation="Move sensitive values into environment variables or secret storage.",
                        )
                    )

            if file.language == "python":
                try:
                    tree = ast.parse(file.content)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        end_line = getattr(node, "end_lineno", node.lineno)
                        span = end_line - node.lineno + 1
                        if span > 40:
                            issues.append(
                                CodeReviewIssue(
                                    severity="medium",
                                    file_path=file.file_path,
                                    symbol=node.name,
                                    line=node.lineno,
                                    message=f"Function is {span} lines long.",
                                    recommendation="Consider extracting smaller helpers to improve readability and testability.",
                                )
                            )
                        if len(node.args.args) > 4:
                            issues.append(
                                CodeReviewIssue(
                                    severity="medium",
                                    file_path=file.file_path,
                                    symbol=node.name,
                                    line=node.lineno,
                                    message="Function has a large parameter list.",
                                    recommendation="Group related parameters into a config or data object.",
                                )
                            )
                        if not ast.get_docstring(node) and not node.name.startswith("_"):
                            issues.append(
                                CodeReviewIssue(
                                    severity="low",
                                    file_path=file.file_path,
                                    symbol=node.name,
                                    line=node.lineno,
                                    message="Public function is missing a docstring.",
                                    recommendation="Add a short docstring that explains inputs, outputs, and side effects.",
                                )
                            )

        summary = f"Found {len(issues)} review finding(s) across {len(files)} file(s)."
        return CodeReviewReport(repository_id=self.corpus.repository_id, summary=summary, issues=issues)

    def generate_documentation(self, target_file: Optional[str] = None, symbol: Optional[str] = None) -> DocumentationArtifact:
        if target_file:
            file = next((item for item in self.corpus.files if item.file_path == target_file), None)
            if file is None:
                raise FileNotFoundError(f"File not found in corpus: {target_file}")
            sources = self._chunk_sources_for_file(file.file_path)
            symbols = self._file_symbol_names(file.file_path)
            dependency_report = self.dependency_analysis()
            markdown = [
                f"# Documentation for `{file.file_path}`",
                "",
                f"- Language: `{file.language}`",
                f"- Symbols: {', '.join(symbols) if symbols else 'None found'}",
                f"- Direct dependencies observed: {sum(1 for edge in dependency_report.internal_edges if edge['source_file'] == file.file_path)}",
                "",
                "## Summary",
                f"This file is part of repository `{self.corpus.repository_id}` and contains {len(file.content.splitlines())} line(s).",
                "",
                "## Notes",
                "Use this file as a source of truth for the repository behavior described by its symbols and chunks.",
            ]
            return DocumentationArtifact(title=f"Documentation: {file.file_path}", markdown="\n".join(markdown), sources=sources)

        if symbol:
            matches = [chunk for chunk in self.corpus.chunks if str(chunk.metadata.get("symbol")) == symbol]
            if not matches:
                raise ValueError(f"No symbol found with name: {symbol}")
            source_file = str(matches[0].metadata.get("file_path"))
            sources = self._chunk_sources_for_file(source_file)
            markdown = [
                f"# Documentation for `{symbol}`",
                "",
                f"- Source file: `{source_file}`",
                f"- Chunk type: `{matches[0].metadata.get('chunk_type', 'general')}`",
                "",
                "## Behavior",
                "The repository evidence suggests this symbol should be documented using the retrieved code context.",
            ]
            return DocumentationArtifact(title=f"Documentation: {symbol}", markdown="\n".join(markdown), sources=sources)

        architecture = self.architecture_analysis()
        markdown = [
            "# Repository Documentation",
            "",
            f"Repository id: `{self.corpus.repository_id}`",
            "",
            "## Architecture",
            architecture.summary,
            "",
            "## Top-level directories",
        ]
        for item in architecture.top_level_directories:
            markdown.append(f"- `{item['directory']}`: {item['file_count']} file(s)")
        markdown.extend(["", "## Entry points"])
        markdown.extend(f"- `{entry}`" for entry in architecture.entrypoints or ["None detected"])
        return DocumentationArtifact(title="Repository Documentation", markdown="\n".join(markdown), sources=[])

    def generate_unit_tests(self, target_file: Optional[str] = None, symbol: Optional[str] = None) -> TestArtifact:
        if target_file is None and symbol is None:
            raise ValueError("Either target_file or symbol must be provided.")

        file = None
        if target_file:
            file = next((item for item in self.corpus.files if item.file_path == target_file), None)
            if file is None:
                raise FileNotFoundError(f"File not found in corpus: {target_file}")
        elif symbol:
            matches = [chunk for chunk in self.corpus.chunks if str(chunk.metadata.get("symbol")) == symbol]
            if not matches:
                raise ValueError(f"No symbol found with name: {symbol}")
            file = next((item for item in self.corpus.files if item.file_path == str(matches[0].metadata.get("file_path"))), None)

        assert file is not None

        if file.language != "python":
            markdown = [
                f"# Unit Tests for `{file.file_path}`",
                "",
                "This generator currently emits Python/pytest test skeletons. For non-Python code, use the code review and documentation outputs as a starting point.",
            ]
            return TestArtifact(
                title=f"Unit tests for {file.file_path}",
                markdown="\n".join(markdown),
                code="",
                source_file=file.file_path,
                symbol=symbol or "general",
            )

        try:
            tree = ast.parse(file.content)
        except SyntaxError:
            raise ValueError(f"Unable to parse Python source for unit-test generation: {file.file_path}")

        target_nodes: List[ast.AST] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if symbol is None or node.name == symbol:
                    target_nodes.append(node)
            elif isinstance(node, ast.ClassDef) and (symbol is None or node.name == symbol):
                target_nodes.append(node)

        if not target_nodes and symbol:
            raise ValueError(f"No public function or class found for symbol: {symbol}")

        code_lines = ["import pytest", "", ""]
        module_name = Path(file.file_path).with_suffix("").as_posix().replace("/", ".")
        code_lines.append(f"from {module_name} import *  # adjust imports as needed")
        code_lines.append("")

        for node in target_nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [arg.arg for arg in node.args.args if arg.arg not in {"self", "cls"}]
                call_args = ", ".join("None" for _ in params)
                code_lines.extend(
                    [
                        f"def test_{node.name}_returns_expected_result():",
                        f"    result = {node.name}({call_args})" if params else f"    result = {node.name}()",
                        "    assert result is not None",
                        "",
                    ]
                )
            elif isinstance(node, ast.ClassDef):
                code_lines.extend(
                    [
                        f"def test_{node.name.lower()}_can_be_instantiated():",
                        f"    instance = {node.name}()",
                        "    assert instance is not None",
                        "",
                    ]
                )

        markdown = [
            f"# Unit Test Plan for `{file.file_path}`",
            "",
            "## Generated skeleton",
            "The code block below is a starting point. Fill in realistic assertions for repository-specific behavior.",
            "",
            "```python",
            *code_lines,
            "```",
        ]
        return TestArtifact(
            title=f"Unit tests for {file.file_path}",
            markdown="\n".join(markdown),
            code="\n".join(code_lines),
            source_file=file.file_path,
            symbol=symbol or "general",
        )
