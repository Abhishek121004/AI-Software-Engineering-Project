from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.models import RepositoryCorpus, RetrievedChunk
from app.retrieval.structure import RepositoryStructureIndex
from app.retrieval.vector_store import CodeVectorStore


@dataclass(slots=True)
class RetrievalTraceItem:
    source: str
    score: float
    reason: str


@dataclass(slots=True)
class HybridRetrievalResult:
    chunks: List[RetrievedChunk]
    trace: List[RetrievalTraceItem]


class HybridRetriever:
    """Combines semantic vector search with structure-aware code retrieval."""

    def __init__(
        self,
        corpus: RepositoryCorpus,
        vector_store: CodeVectorStore,
        structure_index: Optional[RepositoryStructureIndex] = None,
    ) -> None:
        self.corpus = corpus
        self.vector_store = vector_store
        self.structure_index = structure_index or RepositoryStructureIndex(corpus)

    @staticmethod
    def _chunk_key(chunk: RetrievedChunk) -> tuple[str, int, int]:
        metadata = chunk.metadata
        return (
            str(metadata.get("file_path", "")),
            int(metadata.get("start_line", 0) or 0),
            int(metadata.get("end_line", 0) or 0),
        )

    @staticmethod
    def _merge_scores(current: float, candidate: float, weight: float) -> float:
        return current + (candidate * weight)

    def retrieve(
        self,
        query: str,
        repository_id: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, object]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> HybridRetrievalResult:
        semantic_hits = self.vector_store.search(
            query=query,
            repository_id=repository_id,
            k=max(top_k * 3, top_k),
            filter_dict=filter_dict,
            similarity_threshold=similarity_threshold,
        )
        structure_hits = self.structure_index.search(query=query, top_k=max(top_k * 3, top_k), filter_dict=filter_dict)
        symbol_hits: List[RetrievedChunk] = []
        for token in self.structure_index.query_tokens(query):
            symbol_hits.extend(self.structure_index.symbol_hits(token))
        file_hits: List[RetrievedChunk] = []
        for token in self.structure_index.query_tokens(query):
            if "." in token or "/" in token:
                file_hits.extend(self.structure_index.file_hits(token))

        merged: "OrderedDict[tuple[str, int, int], RetrievedChunk]" = OrderedDict()
        traces: Dict[tuple[str, int, int], List[RetrievalTraceItem]] = {}

        def add_hits(hits: List[RetrievedChunk], weight: float, source: str) -> None:
            for hit in hits:
                key = self._chunk_key(hit)
                weighted_score = hit.score * weight
                if key not in merged:
                    merged[key] = RetrievedChunk(content=hit.content, metadata=dict(hit.metadata), score=weighted_score)
                else:
                    merged[key] = RetrievedChunk(
                        content=merged[key].content,
                        metadata=dict(merged[key].metadata),
                        score=self._merge_scores(merged[key].score, hit.score, weight),
                    )
                traces.setdefault(key, []).append(
                    RetrievalTraceItem(
                        source=source,
                        score=weighted_score,
                        reason=", ".join(hit.metadata.get("structure_reasons", [])) if hit.metadata.get("structure_reasons") else source,
                    )
                )

        add_hits(semantic_hits, 1.0, "semantic")
        add_hits(structure_hits, 1.3, "structure")
        add_hits(symbol_hits, 1.6, "symbol")
        add_hits(file_hits, 1.2, "file")

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]
        for chunk in ranked:
            key = self._chunk_key(chunk)
            if key in traces:
                chunk.metadata = dict(chunk.metadata)
                chunk.metadata["retrieval_trace"] = [
                    {"source": item.source, "score": item.score, "reason": item.reason}
                    for item in traces[key]
                ]
        trace_items = [item for chunk in ranked for item in traces.get(self._chunk_key(chunk), [])]
        return HybridRetrievalResult(chunks=ranked, trace=trace_items)

