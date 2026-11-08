"""Vector index with FAISS when available and an exact NumPy fallback.

Both backends store L2-normalized vectors and use inner product, which is
cosine similarity. Scores must therefore match the notebook's IndexFlatIP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss

    HAS_FAISS = True
except ImportError:  # faiss-cpu has no wheel on some Python versions yet
    faiss = None
    HAS_FAISS = False


class VectorIndex:
    def __init__(self, dim: int = 384, vectors: np.ndarray | None = None):
        self.dim = dim
        self.vectors = vectors
        self._faiss = None
        if HAS_FAISS and vectors is not None:
            self._faiss = faiss.IndexFlatIP(dim)
            self._faiss.add(np.ascontiguousarray(vectors, dtype="float32"))

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    @property
    def ntotal(self) -> int:
        return 0 if self.vectors is None else len(self.vectors)

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(np.asarray(v, dtype="float32"))
        if HAS_FAISS:
            faiss.normalize_L2(v)
        else:
            v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        return v

    @classmethod
    def build(cls, vectors: np.ndarray, dim: int) -> "VectorIndex":
        return cls(dim=dim, vectors=cls.normalize(vectors))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self.normalize(query)
        k = max(1, min(k, self.ntotal))
        if self._faiss is not None:
            return self._faiss.search(q, k)
        scores = q @ self.vectors.T
        order = np.argsort(-scores[0])[:k]
        return scores[0][order].reshape(1, -1), order.reshape(1, -1)

    def semantic_scores(self, query: np.ndarray, ids: list[int]) -> np.ndarray:
        """Exact cosine scores for specific ids, even for hybrid-ranked hits."""
        q = self.normalize(query)
        return (q[0] @ self.vectors[ids].T).astype("float32")

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "backend.txt").write_text(self.backend, encoding="utf-8")
        if self._faiss is not None:
            faiss.write_index(self._faiss, str(folder / "rag.index"))

    @classmethod
    def load(cls, folder: Path, dim: int = 384) -> "VectorIndex":
        vectors = np.load(folder / "vectors.npy")
        backend = (folder / "backend.txt").read_text(encoding="utf-8").strip()
        if backend == "faiss" and not HAS_FAISS:
            # identical results, just no native faiss index
            return cls(dim=dim, vectors=vectors)
        return cls(dim=dim, vectors=vectors)
