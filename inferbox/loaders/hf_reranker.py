"""Reranker loader using sentence-transformers CrossEncoder."""
import logging
import torch
from dataclasses import dataclass
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger("inferbox")


@dataclass
class RerankerModel:
    model: Any
    device: str


def load(config: ModelConfig) -> RerankerModel:
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CrossEncoder(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
        device=device,
    )
    return RerankerModel(model=model, device=device)


def unload(model: RerankerModel):
    del model.model
    torch.cuda.empty_cache()


def rerank(
    model: RerankerModel,
    config: ModelConfig,
    query: str,
    documents: list[str],
    top_k: int | None = None,
) -> list[dict]:
    pairs = [(query, doc) for doc in documents]

    scores = model.model.predict(pairs)

    # Convert numpy to python floats
    results = [{"index": i, "score": float(s)} for i, s in enumerate(scores)]
    results.sort(key=lambda x: x["score"], reverse=True)

    if top_k:
        results = results[:top_k]

    return results
