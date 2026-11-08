"""Regression tests against the saved outputs of the Week 8 Day 2 notebook.

Run from backend/:  .venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

# question, expected top score, expected source file (from the notebook's saved outputs)
PARITY_CASES = [
    ("What is precision?", 0.662, "12_evaluation_metrics.txt"),
    ("What does top-k mean in retrieval?", 0.716, "06_rag_concepts.txt"),
    ("How do I handle missing values in pandas?", 0.695, "08_numpy_pandas.txt"),
    ("What is the difference between classification and regression?", 0.729, "09_ml_scikit_learn.txt"),
    ("What is LangChain?", 0.319, "09_ml_scikit_learn.txt"),
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_index_matches_notebook(client):
    h = client.get("/api/health").json()
    assert h["chunks"] == 1342, h
    assert h["documents"] == 13, h
    assert h["embedding_dim"] == 384, h


def test_corpus_is_the_same_13_files(client):
    docs = client.get("/api/documents").json()
    assert len(docs) == 13
    assert sum(d["characters"] for d in docs) == 276_888
    assert sum(d["chunks"] for d in docs) == 1342


@pytest.mark.parametrize("question,expected,source", PARITY_CASES)
def test_semantic_scores_match_notebook(client, question, expected, source):
    r = client.post("/api/search", json={"question": question, "k": 3, "mode": "semantic"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hits"][0]["source"] == source, (question, body["hits"][0]["source"])
    assert abs(body["top_score"] - expected) < 0.02, (question, body["top_score"], expected)


def test_low_score_is_refused(client):
    r = client.post("/api/ask", json={"question": "What is LangChain?", "k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["refused"] is True
    assert "I do not know" in body["answer"]
    assert body["model"] == "guardrail"


def test_real_groq_answer_is_grounded(client):
    """Integration test: uses the real LLM key from .env."""
    r = client.post("/api/ask", json={"question": "What is precision?", "k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["refused"] is False
    assert body["model"].startswith("groq:"), body["model"]
    assert len(body["answer"].strip()) > 20
    assert "precision" in body["answer"].lower()


def test_hybrid_mode_runs_and_reports_scores(client):
    r = client.post("/api/search", json={
        "question": "What accuracy did fine-tuning reach on SMS spam, and what is RAG?",
        "k": 5, "mode": "hybrid",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["hits"]) == 5
    assert 0.0 <= body["top_score"] <= 1.0
