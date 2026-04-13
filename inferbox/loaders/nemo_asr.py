"""NeMo ASR model loader (e.g. Parakeet TDT 0.6B)."""
import logging
from dataclasses import dataclass
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger("inferbox")


@dataclass
class NemoASRModel:
    model: Any
    device: str


def load(config: ModelConfig) -> NemoASRModel:
    import nemo.collections.asr as nemo_asr
    import torch

    # NeMo's from_pretrained defaults to CPU. Move to GPU explicitly —
    # on a 3060, Parakeet TDT 0.6B runs ~30× faster on CUDA than CPU.
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=config.model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to(device)
    model.eval()
    logger.info(f"NeMo ASR {config.model_id} loaded on {device}")
    return NemoASRModel(model=model, device=device)


def unload(model: NemoASRModel):
    # Do NOT `del model.model` — race with live inference. See hf_embed.
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def transcribe(
    model: NemoASRModel,
    config: ModelConfig,
    audio_path: str,
    language: str | None = None,
) -> dict:
    result = model.model.transcribe([audio_path], timestamps=True)

    if not isinstance(result, list) or not result:
        return {"text": str(result) if result else "", "segments": []}

    # Newer NeMo returns a list of Hypothesis objects. The timestamp
    # attribute was renamed from `timestep` to `timestamp` and now
    # holds a dict keyed by granularity: {"segment": [...], "word": [...],
    # "char": [...]}. Each entry is itself a dict — but key names vary
    # by NeMo version (some use "segment"/"word", others "text"). Be
    # defensive.
    hyp = result[0]
    text = getattr(hyp, "text", "") or ""

    segments: list[dict] = []

    def _pull(entry: dict, *keys: str, default=""):
        for k in keys:
            if k in entry and entry[k] is not None:
                return entry[k]
        return default

    ts = getattr(hyp, "timestamp", None)
    if isinstance(ts, dict):
        seg_list = ts.get("segment") or []
        if not seg_list:
            seg_list = ts.get("word") or []
        for seg in seg_list:
            if not isinstance(seg, dict):
                continue
            segments.append({
                "text": _pull(seg, "segment", "word", "text"),
                "start": float(_pull(seg, "start", "start_offset", default=0.0) or 0.0),
                "end": float(_pull(seg, "end", "end_offset", default=0.0) or 0.0),
            })

    # Fallback: hyp.words (list) if timestamp didn't give us anything
    if not segments:
        words = getattr(hyp, "words", None) or []
        for w in words:
            if isinstance(w, dict):
                segments.append({
                    "text": _pull(w, "word", "text"),
                    "start": float(_pull(w, "start", default=0.0) or 0.0),
                    "end": float(_pull(w, "end", default=0.0) or 0.0),
                })

    return {"text": text, "segments": segments}
