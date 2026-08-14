from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.models import RAGContext, RetrievedChunk, SourceReference


@dataclass(slots=True)
class ContextBuilder:
    """Builds compact prompt context from retrieved repository evidence."""

    def build(
        self,
        question: str,
        repository_id: str,
        retrieved_chunks: List[RetrievedChunk],
        memory_snippet: str = "",
    ) -> RAGContext:
        sources: List[SourceReference] = []
        lines: List[str] = []

        if memory_snippet.strip():
            lines.append("Conversation context:")
            lines.append(memory_snippet.strip())
            lines.append("")

        lines.append(f"Repository: {repository_id}")
        lines.append(f"Question: {question}")
        lines.append("")
        lines.append("Relevant evidence:")

        for index, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.metadata
            source = SourceReference(
                file_path=str(meta.get("file_path", "")),
                symbol=str(meta.get("symbol", "general")),
                chunk_type=str(meta.get("chunk_type", "general")),
                start_line=int(meta["start_line"]) if meta.get("start_line") is not None else None,
                end_line=int(meta["end_line"]) if meta.get("end_line") is not None else None,
                score=chunk.score,
            )
            sources.append(source)
            lines.append(
                f"[{index}] {source.file_path} | {source.symbol} | lines {source.start_line}-{source.end_line} | score {chunk.score:.3f}"
            )
            lines.append(chunk.content.strip())
            lines.append("")

        context_text = "\n".join(lines).strip()
        return RAGContext(
            question=question,
            repository_id=repository_id,
            context_text=context_text,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            memory_snippet=memory_snippet,
        )

