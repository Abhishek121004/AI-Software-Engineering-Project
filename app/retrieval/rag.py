from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol

from app.core.models import AnswerBundle, RetrievedChunk, SourceReference
from app.retrieval.context import ContextBuilder
from app.retrieval.vector_store import CodeVectorStore


class TextGenerator(Protocol):
    def __call__(self, prompt: str) -> str: ...


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 5
    similarity_threshold: Optional[float] = None


class RepositoryRAG:
    """Hybrid repository RAG pipeline with visible context construction."""

    def __init__(
        self,
        vector_store: CodeVectorStore,
        generator: Optional[TextGenerator] = None,
        context_builder: Optional[ContextBuilder] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.vector_store = vector_store
        self.generator = generator or self._default_generator
        self.context_builder = context_builder or ContextBuilder()
        self.config = config or RetrievalConfig()

    @staticmethod
    def _default_generator(prompt: str) -> str:
        return prompt

    @staticmethod
    def _extract_terms(question: str) -> List[str]:
        terms: List[str] = []
        for token in question.replace("`", " ").split():
            cleaned = "".join(ch for ch in token if ch.isalnum() or ch in {"_", ".", "-"})
            if cleaned:
                terms.append(cleaned)
        return terms

    def retrieve(
        self,
        question: str,
        repository_id: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, object]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        k = top_k or self.config.top_k
        threshold = self.config.similarity_threshold if similarity_threshold is None else similarity_threshold

        semantic_results = self.vector_store.search(
            query=question,
            repository_id=repository_id,
            k=max(k * 2, k),
            filter_dict=filter_dict,
            similarity_threshold=threshold,
        )

        exact_results: List[RetrievedChunk] = []
        for term in self._extract_terms(question):
            if len(term) < 3:
                continue
            exact_results.extend(self.vector_store.exact_match(term, repository_id=repository_id, k=k))

        merged: "OrderedDict[tuple[str, int, int], RetrievedChunk]" = OrderedDict()
        for candidate in [*exact_results, *semantic_results]:
            meta = candidate.metadata
            key = (
                str(meta.get("file_path", "")),
                int(meta.get("start_line", 0) or 0),
                int(meta.get("end_line", 0) or 0),
            )
            if key not in merged or candidate.score > merged[key].score:
                merged[key] = candidate

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[:k]

    @staticmethod
    def _build_answer_prompt(context_text: str) -> str:
        return (
            "You are a repository QA assistant.\n"
            "Use only the repository evidence below.\n"
            "If the evidence is insufficient, say so explicitly.\n"
            "Cite the relevant file paths and line ranges in your answer.\n\n"
            f"{context_text}\n\n"
            "Answer:"
        )

    @staticmethod
    def _fallback_answer(question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "I do not have enough repository evidence to answer this question."

        top = retrieved_chunks[0]
        file_path = top.metadata.get("file_path", "unknown file")
        symbol = top.metadata.get("symbol", "general")
        start_line = top.metadata.get("start_line")
        end_line = top.metadata.get("end_line")

        answer = [f"The strongest evidence points to `{file_path}`."]
        if symbol and symbol != "general":
            answer.append(f"The relevant symbol appears to be `{symbol}`.")
        if start_line is not None and end_line is not None:
            answer.append(f"It is covered by lines {start_line}-{end_line}.")
        answer.append("Repository excerpts were used to ground this response.")
        return " ".join(answer)

    def answer(
        self,
        question: str,
        repository_id: str,
        memory_snippet: str = "",
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, object]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> AnswerBundle:
        start = time.perf_counter()
        retrieved_chunks = self.retrieve(
            question=question,
            repository_id=repository_id,
            top_k=top_k,
            filter_dict=filter_dict,
            similarity_threshold=similarity_threshold,
        )
        context = self.context_builder.build(
            question=question,
            repository_id=repository_id,
            retrieved_chunks=retrieved_chunks,
            memory_snippet=memory_snippet,
        )
        prompt = self._build_answer_prompt(context.context_text)
        generated = self.generator(prompt)
        answer_text = generated if generated.strip() != prompt.strip() else self._fallback_answer(question, retrieved_chunks)

        latency_ms = (time.perf_counter() - start) * 1000.0
        return AnswerBundle(
            question=question,
            answer=answer_text,
            sources=context.sources,
            retrieved_chunks=retrieved_chunks,
            context_text=context.context_text,
            latency_ms=latency_ms,
        )

    @staticmethod
    def format_sources(sources: List[SourceReference]) -> str:
        if not sources:
            return "Sources:\n- None"

        lines = ["Sources:"]
        for source in sources:
            location = f"{source.file_path}"
            if source.start_line is not None and source.end_line is not None:
                location = f"{location} ({source.start_line}-{source.end_line})"
            lines.append(f"- {location} :: {source.symbol}")
        return "\n".join(lines)

