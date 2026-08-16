from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.core.models import RetrievedChunk, RepositoryCorpus
from app.evaluation.metrics import answer_relevance, mean_reciprocal_rank, precision_at_k, recall_at_k, token_overlap_score
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import CodeVectorStore


@dataclass(slots=True)
class BenchmarkCase:
    question: str
    expected_files: List[str]
    expected_symbols: List[str] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkRow:
    mode: str
    question: str
    retrieved_files: List[str]
    retrieved_symbols: List[str]
    recall_at_k: float
    precision_at_k: float
    mrr: float
    answer_relevance: float
    context_relevance: float
    latency_ms: float


@dataclass
class BenchmarkReport:
    rows: List[BenchmarkRow] = field(default_factory=list)

    def summary(self) -> Dict[str, Dict[str, float]]:
        aggregates: Dict[str, List[BenchmarkRow]] = {}
        for row in self.rows:
            aggregates.setdefault(row.mode, []).append(row)

        summary: Dict[str, Dict[str, float]] = {}
        for mode, rows in aggregates.items():
            count = float(len(rows))
            summary[mode] = {
                "recall_at_k": sum(row.recall_at_k for row in rows) / count,
                "precision_at_k": sum(row.precision_at_k for row in rows) / count,
                "mrr": sum(row.mrr for row in rows) / count,
                "answer_relevance": sum(row.answer_relevance for row in rows) / count,
                "context_relevance": sum(row.context_relevance for row in rows) / count,
                "latency_ms": sum(row.latency_ms for row in rows) / count,
            }
        return summary

    def to_table(self) -> List[Dict[str, object]]:
        return [
            {
                "mode": row.mode,
                "question": row.question,
                "retrieved_files": row.retrieved_files,
                "retrieved_symbols": row.retrieved_symbols,
                "recall_at_k": row.recall_at_k,
                "precision_at_k": row.precision_at_k,
                "mrr": row.mrr,
                "answer_relevance": row.answer_relevance,
                "context_relevance": row.context_relevance,
                "latency_ms": row.latency_ms,
            }
            for row in self.rows
        ]

    def to_json(self) -> str:
        return json.dumps({"summary": self.summary(), "rows": self.to_table()}, indent=2)


class RepositoryBenchmarkRunner:
    """Runs repeatable retrieval benchmarks over repository questions."""

    def __init__(
        self,
        corpus: RepositoryCorpus,
        vector_store: CodeVectorStore,
        hybrid_retriever: Optional[HybridRetriever] = None,
    ) -> None:
        self.corpus = corpus
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(corpus, vector_store)

    @staticmethod
    def _chunk_files(chunks: Sequence[RetrievedChunk]) -> List[str]:
        return [str(chunk.metadata.get("file_path", "")) for chunk in chunks]

    @staticmethod
    def _chunk_symbols(chunks: Sequence[RetrievedChunk]) -> List[str]:
        return [str(chunk.metadata.get("symbol", "")) for chunk in chunks]

    def _vector_retrieve(self, question: str, top_k: int) -> List[RetrievedChunk]:
        return self.vector_store.search(question, repository_id=self.corpus.repository_id, k=top_k)

    def _hybrid_retrieve(self, question: str, top_k: int) -> List[RetrievedChunk]:
        return self.hybrid_retriever.retrieve(question, repository_id=self.corpus.repository_id, top_k=top_k).chunks

    def run(self, cases: Sequence[BenchmarkCase], top_k: int = 5, modes: Sequence[str] = ("vector", "hybrid")) -> BenchmarkReport:
        rows: List[BenchmarkRow] = []
        for case in cases:
            for mode in modes:
                start = time.perf_counter()
                if mode == "vector":
                    retrieved = self._vector_retrieve(case.question, top_k=top_k)
                elif mode == "hybrid":
                    retrieved = self._hybrid_retrieve(case.question, top_k=top_k)
                else:
                    raise ValueError(f"Unsupported benchmark mode: {mode}")
                latency_ms = (time.perf_counter() - start) * 1000.0
                retrieved_files = self._chunk_files(retrieved)
                retrieved_symbols = self._chunk_symbols(retrieved)
                rows.append(
                    BenchmarkRow(
                        mode=mode,
                        question=case.question,
                        retrieved_files=retrieved_files,
                        retrieved_symbols=retrieved_symbols,
                        recall_at_k=recall_at_k(retrieved_files, case.expected_files),
                        precision_at_k=precision_at_k(retrieved_files, case.expected_files),
                        mrr=mean_reciprocal_rank(retrieved_files, case.expected_files),
                        answer_relevance=answer_relevance(" ".join(retrieved_files), case.question),
                        context_relevance=token_overlap_score(" ".join(retrieved_symbols), case.question),
                        latency_ms=latency_ms,
                    )
                )
        return BenchmarkReport(rows=rows)

    @staticmethod
    def load_cases(path: str | Path) -> List[BenchmarkCase]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return RepositoryBenchmarkRunner.load_cases_from_payload(payload)

    @staticmethod
    def load_cases_from_payload(payload: Sequence[Dict[str, Any]]) -> List[BenchmarkCase]:
        cases: List[BenchmarkCase] = []
        for item in payload:
            cases.append(
                BenchmarkCase(
                    question=item["question"],
                    expected_files=list(item.get("expected_files", [])),
                    expected_symbols=list(item.get("expected_symbols", [])),
                )
            )
        return cases

    def auto_cases(self, limit: int = 5) -> List[BenchmarkCase]:
        cases: List[BenchmarkCase] = []
        seen_files: set[str] = set()
        for chunk in self.corpus.chunks:
            symbol = str(chunk.metadata.get("symbol", "general"))
            file_path = str(chunk.metadata.get("file_path", ""))
            if not file_path or file_path in seen_files:
                continue
            if symbol and symbol != "general":
                cases.append(
                    BenchmarkCase(
                        question=f"Where is `{symbol}` defined?",
                        expected_files=[file_path],
                        expected_symbols=[symbol],
                    )
                )
                seen_files.add(file_path)
            if len(cases) >= limit:
                break

        if not cases:
            for file in self.corpus.files[:limit]:
                cases.append(
                    BenchmarkCase(
                        question=f"What does {file.file_path} do?",
                        expected_files=[file.file_path],
                        expected_symbols=[],
                    )
                )
        return cases[:limit]
