"""Semantic retrieval (FAISS) and hybrid retrieval (FAISS + TF-IDF)."""
from __future__ import annotations

import joblib
import numpy as np

from app.config import settings
from rag.embeddings import embed_one
from rag.index_store import VectorIndex


class Retriever:
    def __init__(self, index: VectorIndex, records: list[dict], tfidf_bundle: dict):
        self.index = index
        self.records = records
        self.vectorizer = tfidf_bundle["vectorizer"]
        self.tfidf_matrix = tfidf_bundle["matrix"]

    def _hit(self, rank: int, score: float, idx: int) -> dict:
        return {
            "rank": rank,
            "score": float(score),
            "source": self.records[idx]["source"],
            "chunk": self.records[idx]["text"],
        }

    def semantic(self, question: str, k: int) -> tuple[list[dict], float]:
        q = embed_one(question)
        scores, ids = self.index.search(q, k)
        hits = [
            self._hit(rank + 1, float(scores[0][rank]), int(ids[0][rank]))
            for rank in range(len(ids[0]))
            if int(ids[0][rank]) >= 0
        ]
        return hits, hits[0]["score"] if hits else 0.0

    def hybrid(self, question: str, k: int, candidates: int | None = None) -> tuple[list[dict], float]:
        candidates = candidates or settings.hybrid_candidates
        q = embed_one(question)
        scores, ids = self.index.search(q, candidates)
        sem = {int(ids[0][i]): float(scores[0][i]) for i in range(len(ids[0]))}

        key_scores = (self.vectorizer.transform([question]) @ self.tfidf_matrix.T).toarray()[0]
        key_top = key_scores.argsort()[::-1][:candidates]
        key = {int(i): float(key_scores[i]) for i in key_top}
        max_key = max(key.values()) if key else 1.0

        combined: dict[int, float] = {}
        for i, s in sem.items():
            combined[i] = combined.get(i, 0) + s
        for i, s in key.items():
            combined[i] = combined.get(i, 0) + (s / max_key if max_key > 0 else 0)
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

        ids_ranked = [i for i, _ in ranked]
        sem_scores = self.index.semantic_scores(q, ids_ranked)
        hits = [
            self._hit(rank, score, idx)
            for rank, (idx, score) in enumerate(ranked, start=1)
        ]
        top_semantic = float(max(sem_scores)) if len(sem_scores) else 0.0
        return hits, top_semantic

    def retrieve(self, question: str, k: int, mode: str) -> tuple[list[dict], float]:
        if mode == "hybrid":
            return self.hybrid(question, k)
        return self.semantic(question, k)


def load_tfidf(path):
    return joblib.load(path)
