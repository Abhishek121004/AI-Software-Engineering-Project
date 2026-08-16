from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from app.analysis.repository_intelligence import RepositoryIntelligence
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
    intelligence: RepositoryIntelligence | None = None

    def _route(self, question: str) -> str:
        q = question.lower()
        if any(word in q for word in ("architecture", "architecture analysis", "system design", "structure overview")):
            return "architecture"
        if any(word in q for word in ("dependency", "dependencies", "module analysis", "imports", "import graph")):
            return "dependency"
        if any(word in q for word in ("code review", "review this", "review the code", "issues", "bugs", "smells")):
            return "review"
        if any(word in q for word in ("documentation", "generate docs", "docs", "readme")):
            return "documentation"
        if any(word in q for word in ("unit test", "unit tests", "test generation", "pytest", "tests for")):
            return "tests"
        if any(word in q for word in ("search code", "find code", "search", "lookup", "locate symbol")):
            return "search"
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

    @staticmethod
    def _extract_file_path(question: str) -> Optional[str]:
        match = re.search(r"([A-Za-z0-9_\-./\\]+?\.[A-Za-z0-9]+)", question)
        if match:
            return match.group(1).replace("\\", "/")
        return None

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        memory = self.memory
        routed_question = memory.resolve_follow_up(question) if memory else question
        trace = AgentTrace()
        trace.add("route", self._route(question))

        route = trace.steps[-1]["detail"]
        if route == "architecture" and self.intelligence is not None:
            report = self.intelligence.architecture_analysis()
            trace.add("tool", "architecture_analysis")
            answer = report.summary
            return {"answer": answer, "trace": trace.steps, "result": report}

        if route == "dependency" and self.intelligence is not None:
            report = self.intelligence.dependency_analysis()
            trace.add("tool", "dependency_analysis")
            answer = report.summary
            return {"answer": answer, "trace": trace.steps, "result": report}

        if route == "review" and self.intelligence is not None:
            target_file = self._extract_file_path(question)
            report = self.intelligence.code_review(target_file=target_file)
            trace.add("tool", {"name": "code_review", "target_file": target_file})
            answer = report.summary
            return {"answer": answer, "trace": trace.steps, "result": report}

        if route == "documentation" and self.intelligence is not None:
            target_file = self._extract_file_path(question)
            symbol = self._extract_identifier(question)
            artifact = self.intelligence.generate_documentation(target_file=target_file, symbol=None if target_file else symbol)
            trace.add("tool", {"name": "generate_documentation", "target_file": target_file, "symbol": symbol})
            return {"answer": artifact.markdown, "trace": trace.steps, "result": artifact}

        if route == "tests" and self.intelligence is not None:
            target_file = self._extract_file_path(question)
            symbol = self._extract_identifier(question)
            artifact = self.intelligence.generate_unit_tests(target_file=target_file, symbol=None if target_file else symbol)
            trace.add("tool", {"name": "generate_unit_tests", "target_file": target_file, "symbol": symbol})
            return {"answer": artifact.markdown, "trace": trace.steps, "result": artifact}

        if route == "search" and self.intelligence is not None:
            results = self.intelligence.code_search(routed_question, top_k=top_k)
            trace.add("tool", {"name": "code_search", "top_k": top_k})
            answer_lines = [f"{len(results)} result(s) found."]
            for item in results:
                answer_lines.append(
                    f"- {item.metadata.get('file_path')} :: {item.metadata.get('symbol')} ({item.score:.3f})"
                )
            return {"answer": "\n".join(answer_lines), "trace": trace.steps, "result": results}

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
