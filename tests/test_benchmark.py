from pathlib import Path

from app.core.repository import RepositoryIndexOptions, RepositoryIndexer
from app.evaluation.benchmark import RepositoryBenchmarkRunner
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import CodeVectorStore


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_repository_benchmark_runner(tmp_path):
    write_file(
        tmp_path / "auth.py",
        """def create_token(user_id):
    return f"token:{user_id}"
""",
    )
    write_file(
        tmp_path / "service.py",
        """from auth import create_token


def login(user_id):
    return create_token(user_id)
""",
    )

    indexer = RepositoryIndexer()
    corpus = indexer.build_corpus(RepositoryIndexOptions(root_dir=tmp_path, repository_id="sample"))
    store = CodeVectorStore(persist_dir=tmp_path / "chromadb")
    store.add_chunks([{"content": chunk.content, "metadata": chunk.metadata} for chunk in corpus.chunks], corpus.repository_id)
    retriever = HybridRetriever(corpus=corpus, vector_store=store)
    runner = RepositoryBenchmarkRunner(corpus=corpus, vector_store=store, hybrid_retriever=retriever)

    cases = runner.auto_cases(limit=2)
    report = runner.run(cases, top_k=2)

    summary = report.summary()
    assert "vector" in summary
    assert "hybrid" in summary
    assert len(report.rows) == 4
    assert all(row.mode in {"vector", "hybrid"} for row in report.rows)

