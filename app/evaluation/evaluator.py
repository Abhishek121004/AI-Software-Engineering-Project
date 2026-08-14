from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from app.evaluation.metrics import answer_relevance, mean_reciprocal_rank, precision_at_k, recall_at_k, token_overlap_score


@dataclass(slots=True)
class EvaluationCase:
    question: str
    expected_files: List[str]


@dataclass(slots=True)
class EvaluationResult:
    question: str
    retrieved_files: List[str]
    recall_at_k: float
    precision_at_k: float
    mrr: float
    answer_relevance: float
    context_relevance: float
    latency_ms: float


@dataclass
class EvaluationReport:
    results: List[EvaluationResult] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        if not self.results:
            return {
                "recall_at_k": 0.0,
                "precision_at_k": 0.0,
                "mrr": 0.0,
                "answer_relevance": 0.0,
                "context_relevance": 0.0,
                "latency_ms": 0.0,
            }

        count = float(len(self.results))
        return {
            "recall_at_k": sum(result.recall_at_k for result in self.results) / count,
            "precision_at_k": sum(result.precision_at_k for result in self.results) / count,
            "mrr": sum(result.mrr for result in self.results) / count,
            "answer_relevance": sum(result.answer_relevance for result in self.results) / count,
            "context_relevance": sum(result.context_relevance for result in self.results) / count,
            "latency_ms": sum(result.latency_ms for result in self.results) / count,
        }


class RepositoryEvaluator:
    def __init__(self, k: int = 5) -> None:
        self.k = k

    def evaluate_case(self, question: str, expected_files: Sequence[str], retrieved_files: Sequence[str], answer: str, context: str, latency_ms: float) -> EvaluationResult:
        return EvaluationResult(
            question=question,
            retrieved_files=list(retrieved_files),
            recall_at_k=recall_at_k(retrieved_files, expected_files),
            precision_at_k=precision_at_k(retrieved_files, expected_files),
            mrr=mean_reciprocal_rank(retrieved_files, expected_files),
            answer_relevance=answer_relevance(answer, question),
            context_relevance=token_overlap_score(answer, context),
            latency_ms=latency_ms,
        )

