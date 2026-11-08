"""Augmented prompt with grounding instruction, ported from notebook Step 5."""
from __future__ import annotations


def build_prompt(question: str, hits: list[dict]) -> str:
    context = ""
    for i, hit in enumerate(hits, start=1):
        context += f"\n[Source {i}: {hit['source']}]\n{hit['chunk']}\n"
    return (
        "Answer the question using ONLY the context below. "
        'If the answer is not in the context, say "I do not know". '
        "Do not guess. Cite the source file for each fact you use.\n"
        f"\nContext:{context}"
        f"\nQuestion: {question}\nAnswer:"
    )
