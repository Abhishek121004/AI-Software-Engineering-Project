from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    if not expected:
        return 0.0
    hits = sum(1 for item in expected if item in retrieved)
    return hits / float(len(expected))


def precision_at_k(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for item in retrieved if item in expected)
    return hits / float(len(retrieved))


def mean_reciprocal_rank(ranked: Sequence[str], expected: Sequence[str]) -> float:
    expected_set = set(expected)
    for index, item in enumerate(ranked, start=1):
        if item in expected_set:
            return 1.0 / float(index)
    return 0.0


def token_overlap_score(answer: str, context: str) -> float:
    answer_tokens = {token.lower() for token in answer.split() if token.strip()}
    context_tokens = {token.lower() for token in context.split() if token.strip()}
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / float(len(answer_tokens))


def answer_relevance(answer: str, question: str) -> float:
    answer_tokens = {token.lower() for token in answer.split() if token.strip()}
    question_tokens = {token.lower() for token in question.split() if token.strip()}
    if not question_tokens:
        return 0.0
    return len(answer_tokens & question_tokens) / float(len(question_tokens))

