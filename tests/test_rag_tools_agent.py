from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.agents.repository_agent import RepositoryAgent
from app.core.repository import RepositoryIndexOptions, RepositoryIndexer
from app.memory.conversation import ConversationMemory
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rag import RepositoryRAG
from app.retrieval.vector_store import CodeVectorStore
from app.tools.repository_tools import RepositoryTools


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_sample_repo(tmp_path: Path):
    write_file(
        tmp_path / "auth.py",
        """def create_token(user_id):
    return f"token:{user_id}"


def validate_token(token):
    return token.startswith("token:")
""",
    )
    write_file(
        tmp_path / "service.py",
        """from auth import create_token


def login(user_id):
    token = create_token(user_id)
    return token
""",
    )

    indexer = RepositoryIndexer()
    corpus = indexer.build_corpus(RepositoryIndexOptions(root_dir=tmp_path, repository_id="sample"))
    store = CodeVectorStore(persist_dir=tmp_path / "chromadb")
    store.add_chunks([{"content": chunk.content, "metadata": chunk.metadata} for chunk in corpus.chunks], corpus.repository_id)
    tools = RepositoryTools(corpus=corpus, vector_store=store)
    retriever = HybridRetriever(corpus=corpus, vector_store=store)
    rag = RepositoryRAG(vector_store=store, retriever=retriever)
    return corpus, store, tools, rag


class FakeToolCallingLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tools = None
        self.calls = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_rag_pipeline_returns_sources(tmp_path):
    corpus, store, _, rag = build_sample_repo(tmp_path)

    bundle = rag.answer("Where is create_token defined?", repository_id=corpus.repository_id)

    assert "create_token" in bundle.answer or "token" in bundle.answer.lower()
    assert len(bundle.sources) > 0
    assert any(source.file_path == "auth.py" for source in bundle.sources)


def test_repository_tools_and_agent(tmp_path):
    corpus, store, tools, rag = build_sample_repo(tmp_path)
    memory = ConversationMemory()
    fake_llm = FakeToolCallingLLM(
        [
            AIMessage(
                tool_calls=[
                    {"name": "repository_qa", "args": {"question": "Where is create_token used?", "top_k": 5}, "id": "call-1"}
                ],
                content="",
            ),
            AIMessage(content="create_token is used in service.py."),
        ]
    )
    agent = RepositoryAgent(rag=rag, tools=tools, memory=memory, llm=fake_llm)

    tree = tools.get_repository_tree()
    assert tree["count"] == 2

    file_data = tools.read_file("auth.py")
    assert "create_token" in file_data["content"]

    function_matches = tools.find_function("create_token")
    assert function_matches
    assert function_matches[0].metadata["file_path"] == "auth.py"

    references = tools.find_references("create_token")
    assert any(item["file_path"] == "service.py" for item in references)

    with pytest.raises(ValueError):
        tools.read_file("../secrets.txt")

    response = agent.answer("Where is create_token used?")
    assert "answer" in response
    assert response["trace"]
    assert "service.py" in response["answer"] or any("service.py" in str(source) for source in response.get("sources", []))
    assert response.get("sources")
