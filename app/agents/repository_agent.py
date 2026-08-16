from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.analysis.repository_intelligence import RepositoryIntelligence
from app.core.gemini import create_gemini_chat_model, has_gemini_runtime
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
    """Constrained tool-calling repository assistant."""

    rag: RepositoryRAG
    tools: RepositoryTools
    memory: ConversationMemory | None = None
    intelligence: RepositoryIntelligence | None = None
    llm: Any | None = None
    max_iterations: int = 4

    def __post_init__(self) -> None:
        if self.intelligence is None:
            self.intelligence = RepositoryIntelligence(tools=self.tools, rag=self.rag)

    def _build_tools(self):
        intelligence = self.intelligence
        rag = self.rag
        repository_id = self.tools.corpus.repository_id

        @tool
        def repository_qa(question: str, top_k: int = 5) -> str:
            """Answer a repository question using grounded RAG evidence."""
            bundle: AnswerBundle = rag.answer(question, repository_id=repository_id, top_k=top_k)
            payload = {
                "answer": bundle.answer,
                "sources": [asdict(source) for source in bundle.sources],
                "retrieved_chunks": [
                    {
                        "content": chunk.content,
                        "metadata": dict(chunk.metadata),
                        "score": chunk.score,
                    }
                    for chunk in bundle.retrieved_chunks
                ],
                "context_text": bundle.context_text,
                "latency_ms": bundle.latency_ms,
            }
            return json.dumps(payload)

        @tool
        def search_code(query: str, top_k: int = 5) -> str:
            """Find relevant repository chunks for a code or architecture search query."""
            results = intelligence.code_search(query, top_k=top_k) if intelligence else []
            if not results:
                return "No matching code chunks found."
            lines = [f"Found {len(results)} result(s)."]
            for item in results:
                lines.append(
                    f"- {item.metadata.get('file_path')} :: {item.metadata.get('symbol')} "
                    f"({item.metadata.get('start_line')}-{item.metadata.get('end_line')}) score={item.score:.3f}"
                )
            return "\n".join(lines)

        @tool
        def repository_architecture_analysis() -> str:
            """Summarize repository architecture, entry points, and top-level modules."""
            report = intelligence.architecture_analysis() if intelligence else None
            if report is None:
                return "Repository intelligence is unavailable."
            return report.summary

        @tool
        def dependency_module_analysis() -> str:
            """Analyze internal dependencies, module boundaries, and external imports."""
            report = intelligence.dependency_analysis() if intelligence else None
            if report is None:
                return "Repository intelligence is unavailable."
            return report.summary

        @tool
        def code_review(target_file: str = "") -> str:
            """Review a file or the repository for issues, smells, and risks."""
            report = intelligence.code_review(target_file=target_file or None) if intelligence else None
            if report is None:
                return "Repository intelligence is unavailable."
            if not report.issues:
                return report.summary + "\nNo issues found."
            lines = [report.summary]
            for issue in report.issues[:12]:
                lines.append(
                    f"- [{issue.severity}] {issue.file_path}:{issue.line or '?'} {issue.message} "
                    f"Recommendation: {issue.recommendation}"
                )
            return "\n".join(lines)

        @tool
        def documentation_generation(target_file: str = "", symbol: str = "") -> str:
            """Generate documentation for a file, symbol, or the whole repository."""
            artifact = intelligence.generate_documentation(
                target_file=target_file or None,
                symbol=symbol or None,
            ) if intelligence else None
            if artifact is None:
                return "Repository intelligence is unavailable."
            return artifact.markdown

        @tool
        def unit_test_generation(target_file: str = "", symbol: str = "") -> str:
            """Generate unit-test skeletons for a repository file or symbol."""
            artifact = intelligence.generate_unit_tests(
                target_file=target_file or None,
                symbol=symbol or None,
            ) if intelligence else None
            if artifact is None:
                return "Repository intelligence is unavailable."
            return artifact.markdown

        @tool
        def read_file(file_path: str) -> str:
            """Read a file from the indexed repository workspace."""
            result = self.tools.read_file(file_path)
            return result["content"]

        @tool
        def find_function(symbol: str) -> str:
            """Find the function or class definition for a symbol."""
            results = self.tools.find_function(symbol)
            if not results:
                return "No matching function or class found."
            lines = [f"Found {len(results)} match(es)."]
            for item in results[:8]:
                lines.append(
                    f"- {item.metadata.get('file_path')} :: {item.metadata.get('symbol')} "
                    f"({item.metadata.get('start_line')}-{item.metadata.get('end_line')})"
                )
            return "\n".join(lines)

        @tool
        def find_references(symbol: str) -> str:
            """Find repository references to a symbol."""
            results = self.tools.find_references(symbol)
            if not results:
                return "No references found."
            lines = [f"Found {len(results)} reference(s)."]
            for item in results[:12]:
                lines.append(f"- {item['file_path']} ({item.get('language', 'unknown')})")
            return "\n".join(lines)

        @tool
        def repository_tree() -> str:
            """Inspect the repository tree and file count."""
            result = self.tools.get_repository_tree()
            return json.dumps(result, indent=2)

        return [
            repository_qa,
            search_code,
            repository_architecture_analysis,
            dependency_module_analysis,
            code_review,
            documentation_generation,
            unit_test_generation,
            read_file,
            find_function,
            find_references,
            repository_tree,
        ]

    def _create_llm(self):
        if self.llm is not None:
            return self.llm
        if has_gemini_runtime():
            return create_gemini_chat_model()
        raise RuntimeError("GEMINI_API_KEY is required for the tool-calling agent.")

    @staticmethod
    def _tool_map(tools) -> Dict[str, Any]:
        mapping = {}
        for tool_item in tools:
            mapping[tool_item.name] = tool_item
        return mapping

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a constrained repository assistant.\n"
            "Only use the provided tools.\n"
            "Prefer repository evidence over general knowledge.\n"
            "For codebase questions, cite the files and line ranges when the tools provide them.\n"
            "For architecture, dependency, review, documentation, and test requests, call the matching tool.\n"
            "Do not invent repository facts.\n"
        )

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        tools = self._build_tools()
        tool_map = self._tool_map(tools)
        model = self._create_llm()
        if hasattr(model, "bind_tools"):
            model = model.bind_tools(tools)

        trace = AgentTrace()
        messages: List[Any] = [SystemMessage(content=self._system_prompt())]
        if self.memory:
            memory_context = self.memory.recent_context()
            if memory_context:
                messages.append(SystemMessage(content=f"Conversation memory:\n{memory_context}"))
        messages.append(HumanMessage(content=question))
        trace.add("user", question)
        structured_payload: Dict[str, Any] | None = None

        for _ in range(self.max_iterations):
            response = model.invoke(messages)
            if not isinstance(response, AIMessage):
                content = getattr(response, "content", str(response))
                response = AIMessage(content=content)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                final_answer = response.content or ""
                if structured_payload is not None:
                    final_answer = str(structured_payload.get("answer", final_answer))
                if self.memory:
                    self.memory.add_user_message(question)
                    self.memory.add_assistant_message(final_answer)
                if structured_payload is not None:
                    return {
                        "answer": final_answer,
                        "sources": structured_payload.get("sources", []),
                        "retrieved_chunks": structured_payload.get("retrieved_chunks", []),
                        "context_text": structured_payload.get("context_text", ""),
                        "latency_ms": structured_payload.get("latency_ms", 0.0),
                        "trace": trace.steps,
                    }
                return {"answer": final_answer, "trace": trace.steps}

            trace.add("model_tool_calls", [call.get("name") for call in tool_calls])
            for call in tool_calls:
                tool_name = call.get("name")
                tool_args = call.get("args") or {}
                tool_call_id = call.get("id")
                tool_item = tool_map.get(tool_name)
                if tool_item is None:
                    tool_result = f"Tool not found: {tool_name}"
                else:
                    try:
                        tool_result = tool_item.invoke(tool_args)
                        tool_result_text = getattr(tool_result, "content", tool_result)
                        if tool_name == "repository_qa":
                            try:
                                structured_payload = json.loads(str(tool_result_text))
                            except json.JSONDecodeError:
                                structured_payload = None
                        tool_result = tool_result_text
                    except Exception as exc:  # pragma: no cover - tool guard
                        tool_result = f"Tool error for {tool_name}: {exc}"
                trace.add("tool", {"name": tool_name, "args": tool_args})
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id or tool_name))

        fallback = "I could not complete the request with the available tools."
        if self.memory:
            self.memory.add_user_message(question)
            self.memory.add_assistant_message(fallback)
        if structured_payload is not None:
            answer_text = str(structured_payload.get("answer", fallback))
            return {
                "answer": answer_text,
                "sources": structured_payload.get("sources", []),
                "retrieved_chunks": structured_payload.get("retrieved_chunks", []),
                "context_text": structured_payload.get("context_text", ""),
                "latency_ms": structured_payload.get("latency_ms", 0.0),
                "trace": trace.steps,
            }
        return {"answer": fallback, "trace": trace.steps}
