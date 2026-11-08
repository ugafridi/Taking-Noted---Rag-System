"""Local embeddings with all-MiniLM-L6-v2.

This is the notebook's exact mean-pooling implementation (AutoTokenizer +
AutoModel, max_length 256). Keeping it verbatim preserves the notebook's
cosine scores, e.g. 0.662 for "What is precision?".
"""
from __future__ import annotations

import threading

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from app.config import settings

_tokenizer = None
_model = None
_lock = threading.Lock()


def load_model() -> tuple[AutoTokenizer, AutoModel]:
    global _tokenizer, _model
    with _lock:
        if _model is None:
            _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
            _model = AutoModel.from_pretrained(settings.embedding_model)
            _model.eval()
    return _tokenizer, _model


def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    tokenizer, model = load_model()
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=settings.max_seq_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        mean = (output * mask).sum(1) / mask.sum(1)
        vectors.append(mean.numpy())
    return np.vstack(vectors).astype("float32")


def embed_one(text: str) -> np.ndarray:
    return embed([text])
