from app.core.repository import RepositoryIndexer


def test_repo_slug_from_github_url():
    indexer = RepositoryIndexer()

    assert indexer._repo_slug_from_url("https://github.com/openai/codex.git") == "codex"
    assert indexer._repo_slug_from_url("https://github.com/openai/codex") == "codex"

