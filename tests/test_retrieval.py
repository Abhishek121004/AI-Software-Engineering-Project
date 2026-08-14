import pytest
from unittest.mock import MagicMock
from app.retrieval.vector_store import CodeVectorStore

@pytest.fixture(autouse=True)
def mock_google_embeddings(monkeypatch):
    """Mocks GoogleGenerativeAIEmbeddings to run tests completely offline."""
    mock_inst = MagicMock()
    mock_inst.embed_documents.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    mock_inst.embed_query.side_effect = lambda text: [0.1] * 768
    
    monkeypatch.setattr(
        "app.retrieval.vector_store.GoogleGenerativeAIEmbeddings",
        lambda *args, **kwargs: mock_inst
    )
    return mock_inst

def test_vector_store_operations(tmp_path):
    """Verify document insertion, repo-scoped filtering, and index cleaning."""
    persist_dir = tmp_path / "chromadb"
    
    # Initialize store
    # Use a small mock embedding model or standard fast CPU model
    # "sentence-transformers/all-MiniLM-L6-v2" is cached/downloaded automatically
    store = CodeVectorStore(persist_dir=persist_dir)

    # Prepare chunks for Repository A
    repo_a_chunks = [
        {
            "content": "def calculate_price(quantity, price):\n    return quantity * price",
            "metadata": {"file_path": "pricing.py", "language": "python", "chunk_type": "function", "symbol": "calculate_price"}
        },
        {
            "content": "class AuthController:\n    def login(self):\n        pass",
            "metadata": {"file_path": "auth.py", "language": "python", "chunk_type": "class", "symbol": "AuthController"}
        }
    ]

    # Prepare chunks for Repository B
    repo_b_chunks = [
        {
            "content": "function processOrder(orderId) {\n    console.log(orderId);\n}",
            "metadata": {"file_path": "order.js", "language": "javascript", "chunk_type": "function", "symbol": "processOrder"}
        }
    ]

    # Add documents
    store.add_chunks(repo_a_chunks, repository_id="repo_a")
    store.add_chunks(repo_b_chunks, repository_id="repo_b")

    # Verify search filters by repository_id
    results_a = store.search(query="price calculation method", repository_id="repo_a", k=2)
    assert len(results_a) > 0
    assert all(r.metadata["repository_id"] == "repo_a" for r in results_a)
    assert "pricing.py" in [r.metadata["file_path"] for r in results_a]

    # Verify order search returns repo_b documents
    results_b = store.search(query="order process system", repository_id="repo_b", k=2)
    assert len(results_b) == 1
    assert results_b[0].metadata["repository_id"] == "repo_b"
    assert results_b[0].metadata["file_path"] == "order.js"

    # Verify sub-filtering on metadata works (e.g. searching only for class definitions in repo_a)
    class_results = store.search(
        query="login authentication handler",
        repository_id="repo_a",
        k=2,
        filter_dict={"chunk_type": "class"}
    )
    assert len(class_results) == 1
    assert class_results[0].metadata["symbol"] == "AuthController"

    # Clear repository A and verify it's removed but repository B remains
    store.clear_repository("repo_a")
    
    empty_results_a = store.search(query="price calculation method", repository_id="repo_a", k=2)
    assert len(empty_results_a) == 0

    remaining_results_b = store.search(query="order process system", repository_id="repo_b", k=2)
    assert len(remaining_results_b) == 1
