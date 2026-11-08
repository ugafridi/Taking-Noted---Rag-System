"""Orchestrates retrieval -> grounding guardrail -> prompt -> generation."""
from __future__ import annotations

import json
import time

import joblib

from app.config import settings
from rag import embeddings
from rag.generate import GenerationError, Generator
from rag.index_store import VectorIndex
from rag.ingest import build_index, load_documents
from rag.prompt import build_prompt
from rag.retrieve import Retriever

REFUSAL = (
    "I do not know. The course documents do not appear to contain the answer "
    "(best match scored {top:.3f}, below the {threshold:.2f} threshold)."
)


class RAGSystem:
    def __init__(self, index, records, manifest, retriever, generator):
        self.index = index
        self.records = records
        self.manifest = manifest
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def load(cls) -> "RAGSystem":
        folder = settings.index_dir
        required = ["vectors.npy", "backend.txt", "chunks.json", "tfidf.joblib", "manifest.json"]
        missing = [f for f in required if not (folder / f).exists()]
        if missing:
            raise RuntimeError(
                f"Index not built (missing {', '.join(missing)}). Run: python -m rag.ingest"
            )
        index = VectorIndex.load(folder, settings.embedding_dim)
        records = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        tfidf = joblib.load(folder / "tfidf.joblib")
        retriever = Retriever(index, records, tfidf)
        embeddings.load_model()  # warm up so the first question is fast
        return cls(index, records, manifest, retriever, Generator(settings))

    def documents(self) -> list[dict]:
        docs = load_documents()
        per_doc = self.manifest.get("per_document", {})
        return [
            {
                "name": name,
                "characters": len(text),
                "chunks": per_doc.get(name, 0),
            }
            for name, text in docs.items()
        ]

    def _threshold(self, override: float | None) -> float:
        return settings.top_score_threshold if override is None else override

    def search(self, question: str, k: int, mode: str) -> dict:
        hits, top = self.retriever.retrieve(question, k, mode)
        return {"question": question, "hits": hits, "top_score": top, "mode": mode}

    def ask(self, question: str, k: int, mode: str, threshold: float | None = None) -> dict:
        start = time.perf_counter()
        thr = self._threshold(threshold)
        hits, top = self.retriever.retrieve(question, k, mode)
        refused = top < thr

        if refused:
            answer = REFUSAL.format(top=top, threshold=thr)
            model = "guardrail"
        else:
            prompt = build_prompt(question, hits)
            try:
                answer, model = self.generator.generate(prompt)
            except GenerationError as exc:
                answer = f"Generation failed: {exc}"
                model = "error"

        return {
            "answer": answer,
            "hits": hits,
            "top_score": top,
            "refused": refused,
            "model": model,
            "mode": mode,
            "k": k,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    def ask_stream(self, question: str, k: int, mode: str, threshold: float | None = None):
        """Yields SSE-ready dict events: meta, token, done (or error)."""
        start = time.perf_counter()
        thr = self._threshold(threshold)
        hits, top = self.retriever.retrieve(question, k, mode)
        refused = top < thr
        collected: list[str] = []

        if refused:
            text = REFUSAL.format(top=top, threshold=thr)
            yield {"type": "meta", "hits": hits, "top_score": top,
                   "refused": True, "model": "guardrail", "mode": mode, "k": k}
            yield {"type": "token", "text": text}
            collected.append(text)
            model = "guardrail"
        else:
            prompt = build_prompt(question, hits)
            model = "unknown"
            try:
                for kind, payload in self.generator.stream(prompt):
                    if kind == "start":
                        model = payload
                        yield {"type": "meta", "hits": hits, "top_score": top,
                               "refused": False, "model": model, "mode": mode, "k": k}
                    else:
                        collected.append(payload)
                        yield {"type": "token", "text": payload}
            except GenerationError as exc:
                if not collected:
                    yield {"type": "meta", "hits": hits, "top_score": top,
                           "refused": False, "model": "error", "mode": mode, "k": k}
                yield {"type": "error", "message": str(exc)}

        yield {
            "type": "done",
            "answer": "".join(collected),
            "model": model,
            "refused": refused,
            "top_score": top,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    def rebuild(self, chunk_size: int, chunk_overlap: int) -> dict:
        stats = build_index(chunk_size, chunk_overlap)
        fresh = type(self).load()
        self.index = fresh.index
        self.records = fresh.records
        self.manifest = fresh.manifest
        self.retriever = fresh.retriever
        self.generator = fresh.generator
        return {k: stats[k] for k in (
            "documents", "characters", "chunks", "average_chunk_length",
            "longest_chunk", "shortest_chunk", "dimension", "index_backend",
        )}

    def health(self) -> dict:
        return {
            "status": "ok",
            "chunks": self.index.ntotal,
            "documents": self.manifest["documents"],
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "index_backend": self.index.backend,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "threshold": settings.top_score_threshold,
        }
