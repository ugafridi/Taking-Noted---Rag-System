"""Offline ingestion: corpus -> chunks -> embeddings -> FAISS index + metadata.

Run with:  python -m rag.ingest
The printed numbers must match the notebook exactly:
13 files / 276,888 chars / 1,342 chunks / avg 254.7 / max 350 / min 8.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings
from rag.chunking import recursive_chunk
from rag.embeddings import embed
from rag.index_store import VectorIndex


def load_documents(corpus_dir: Path | None = None) -> dict[str, str]:
    corpus_dir = corpus_dir or settings.corpus_dir
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    return {p.name: p.read_text(encoding="utf-8") for p in files}


def build_index(chunk_size: int | None = None, chunk_overlap: int | None = None) -> dict:
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    out_dir = settings.index_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents()
    chunks: list[str] = []
    sources: list[str] = []
    per_doc: dict[str, int] = {}
    for name, text in docs.items():
        pieces = recursive_chunk(text, chunk_size, chunk_overlap)
        per_doc[name] = len(pieces)
        chunks.extend(pieces)
        sources.extend([name] * len(pieces))

    print(f"Embedding {len(chunks):,} chunks with {settings.embedding_model} ...")
    vectors = embed(chunks)
    index = VectorIndex.build(vectors, settings.embedding_dim)
    index.save(out_dir)

    tfidf = TfidfVectorizer().fit(chunks)
    tfidf_matrix = tfidf.transform(chunks)
    joblib.dump({"vectorizer": tfidf, "matrix": tfidf_matrix}, out_dir / "tfidf.joblib")

    records = [
        {"id": i, "source": sources[i], "text": chunks[i]}
        for i in range(len(chunks))
    ]
    (out_dir / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )

    digest = hashlib.sha256(
        "".join(f"{k}\0{v}" for k, v in docs.items()).encode("utf-8")
    ).hexdigest()
    stats = {
        "documents": len(docs),
        "characters": sum(len(t) for t in docs.values()),
        "chunks": len(chunks),
        "average_chunk_length": round(sum(map(len, chunks)) / len(chunks), 1),
        "longest_chunk": max(map(len, chunks)),
        "shortest_chunk": min(map(len, chunks)),
        "dimension": int(vectors.shape[1]),
        "index_backend": index.backend,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": settings.embedding_model,
        "per_document": per_doc,
        "content_hash": digest,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def main() -> None:
    s = build_index()
    print("\nDocuments:", s["documents"])
    print("Chunks:", s["chunks"])
    print(f"Average chunk length: {s['average_chunk_length']} characters")
    print("Longest chunk:", s["longest_chunk"], "characters")
    print("Shortest chunk:", s["shortest_chunk"], "characters")
    print("Embeddings shape:", f"({s['chunks']}, {s['dimension']})")
    print("Index backend:", s["index_backend"])
    print("Total characters:", f"{s['characters']:,}")


if __name__ == "__main__":
    main()
