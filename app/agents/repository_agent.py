from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from app.core.models import AnswerBundle
from app.memory.conversation import ConversationMemory
from app.retrieval.rag import RepositoryRAG
from app.tools.repository_tools import RepositoryTools


@dataclass(slots=True)
class AgentTrace:
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, action: str, detail: Any) -> None:
        self.steps.append({"action": action, "detail": detail})


@dataclass
class RepositoryAgent:
    """Heuristic assistant that chooses repository tools before answering."""

    rag: RepositoryRAG
    tools: RepositoryTools
    memory: ConversationMemory | None = None

    def _route(self, question: str) -> str:
        q = question.lower()
        if any(word in q for word in ("tree", "structure", "layout", "directory")):
            return "tree"
        if any(word in q for word in ("where is", "used", "reference", "references", "find", "locate")):
            return "references"
        if any(word in q for word in ("read file", "open file", "show file", "display file")):
            return "read"
        if any(word in q for word in ("function", "method", "class", "symbol")):
            return "function"
        return "rag"

    @staticmethod
    def _extract_identifier(question: str) -> str:
        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
        stopwords = {"where", "what", "which", "how", "why", "used", "use", "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on", "for"}
        filtered = [token for token in candidates if token.lower() not in stopwords]
        if not filtered:
            return candidates[-1] if candidates else question.strip()
        return max(filtered, key=len)

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        memory = self.memory
        routed_question = memory.resolve_follow_up(question) if memory else question
        trace = AgentTrace()
        trace.add("route", self._route(question))

        route = trace.steps[-1]["detail"]
        if route == "tree":
            result = self.tools.get_repository_tree()
            trace.add("tool", "get_repository_tree")
            answer = f"Repository {result['repository_id']} contains {result['count']} indexed files."
            return {"answer": answer, "trace": trace.steps, "result": result}

        if route == "read":
            file_path = self._extract_identifier(question)
            result = self.tools.read_file(file_path)
            trace.add("tool", {"name": "read_file", "file_path": file_path})
            return {"answer": result["content"], "trace": trace.steps, "result": result}

        if route == "function":
            symbol = self._extract_identifier(question).strip("`")
            result = self.tools.find_function(symbol)
            trace.add("tool", {"name": "find_function", "symbol": symbol})
            if result:
                best = result[0]
                answer = f"{symbol} is documented in {best.metadata.get('file_path')}."
            else:
                answer = "I could not find a function match in the indexed repository."
            return {"answer": answer, "trace": trace.steps, "result": result}

        if route == "references":
            symbol = self._extract_identifier(question).strip("`")
            result = self.tools.find_references(symbol)
            trace.add("tool", {"name": "find_references", "symbol": symbol})
            answer = f"Found {len(result)} reference(s) for {symbol}."
            return {"answer": answer, "trace": trace.steps, "result": result}

        bundle: AnswerBundle = self.rag.answer(routed_question, self.tools.corpus.repository_id, top_k=top_k)
        trace.add("tool", {"name": "rag.answer", "top_k": top_k})
        if memory:
            memory.add_user_message(question)
            memory.add_assistant_message(bundle.answer)
        return {
            "answer": bundle.answer,
            "sources": bundle.sources,
            "retrieved_chunks": bundle.retrieved_chunks,
            "context_text": bundle.context_text,
            "latency_ms": bundle.latency_ms,
            "trace": trace.steps,
        }
