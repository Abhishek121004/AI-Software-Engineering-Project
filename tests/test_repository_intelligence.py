from pathlib import Path

from app.analysis.repository_intelligence import RepositoryIntelligence
from app.core.repository import RepositoryIndexOptions, RepositoryIndexer
from app.retrieval.vector_store import CodeVectorStore
from app.tools.repository_tools import RepositoryTools


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_intelligence_repo(tmp_path: Path):
    write_file(
        tmp_path / "main.py",
        """from service import login


if __name__ == "__main__":
    login("demo")
""",
    )
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
    # TODO: add password verification
    token = create_token(user_id)
    return token
""",
    )

    indexer = RepositoryIndexer()
    corpus = indexer.build_corpus(RepositoryIndexOptions(root_dir=tmp_path, repository_id="sample"))
    store = CodeVectorStore(persist_dir=tmp_path / "chromadb")
    store.add_chunks([{"content": chunk.content, "metadata": chunk.metadata} for chunk in corpus.chunks], corpus.repository_id)
    tools = RepositoryTools(corpus=corpus, vector_store=store)
    intelligence = RepositoryIntelligence(tools=tools)
    return corpus, intelligence


def test_architecture_and_dependency_analysis(tmp_path):
    corpus, intelligence = build_intelligence_repo(tmp_path)

    architecture = intelligence.architecture_analysis()
    assert architecture.repository_id == corpus.repository_id
    assert any(item["directory"] == "." or item["directory"] == "main.py" for item in architecture.top_level_directories)
    assert "main.py" in architecture.entrypoints
    assert architecture.languages["python"] == 3

    dependencies = intelligence.dependency_analysis()
    assert any(edge["source_file"] == "service.py" for edge in dependencies.internal_edges)
    assert any("auth" in edge["target"] for edge in dependencies.internal_edges)


def test_review_docs_and_tests_generation(tmp_path):
    _, intelligence = build_intelligence_repo(tmp_path)

    review = intelligence.code_review(target_file="service.py")
    assert review.issues
    assert any("TODO" in issue.message for issue in review.issues)

    docs = intelligence.generate_documentation(target_file="auth.py")
    assert "auth.py" in docs.markdown
    assert docs.sources

    tests = intelligence.generate_unit_tests(target_file="auth.py")
    assert "test_create_token" in tests.code
    assert "pytest" in tests.code

