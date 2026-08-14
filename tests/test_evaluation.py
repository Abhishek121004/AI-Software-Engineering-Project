from app.evaluation.metrics import answer_relevance, mean_reciprocal_rank, precision_at_k, recall_at_k, token_overlap_score


def test_retrieval_metrics():
    retrieved = ["auth.py", "service.py", "README.md"]
    expected = ["auth.py", "config.py"]

    assert recall_at_k(retrieved, expected) == 0.5
    assert precision_at_k(retrieved, expected) == 1 / 3
    assert mean_reciprocal_rank(retrieved, expected) == 1.0


def test_generation_metrics():
    answer = "The token is created in auth.py"
    question = "Where is the token created?"
    context = "auth.py creates the token"

    assert answer_relevance(answer, question) > 0
    assert token_overlap_score(answer, context) > 0

