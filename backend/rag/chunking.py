"""Recursive character chunker, ported verbatim from the Week 8 Day 2 notebook.

Separators are tried largest-first: paragraph, line, sentence, word. Overlap
copies the beginning of the next chunk onto the end of the previous one.
"""
from __future__ import annotations

SEPARATORS = ["\n\n", "\n", ". ", " "]


def recursive_chunk(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    def _split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= chunk_size:
            return [t]
        if not seps:
            step = chunk_size - chunk_overlap
            return [t[i:i + chunk_size] for i in range(0, len(t), step)]
        sep = seps[0]
        parts = t.split(sep)
        pieces = [p + sep if i < len(parts) - 1 else p for i, p in enumerate(parts)]
        chunks, current = [], ""
        for piece in pieces:
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current)
                if len(piece) > chunk_size:
                    chunks.extend(_split(piece, seps[1:]))
                    current = ""
                else:
                    current = piece
        if current:
            chunks.append(current)
        return chunks

    chunks = [c for c in _split(text, SEPARATORS) if c.strip()]
    out = []
    for i, c in enumerate(chunks):
        if i < len(chunks) - 1:
            out.append(c + chunks[i + 1][:chunk_overlap])
        else:
            out.append(c)
    return out
