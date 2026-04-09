"""NeMo ASR model loader (e.g. Parakeet TDT 0.6B)."""
import logging
from dataclasses import dataclass
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger("inferbox")


@dataclass
class NemoASRModel:
    model: Any


def load(config: ModelConfig) -> NemoASRModel:
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=config.model_id)
    return NemoASRModel(model=model)


def unload(model: NemoASRModel):
    del model.model
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

    # NeMo returns different structures depending on version
    if isinstance(result, list):
        # Newer NeMo returns list of Hypothesis objects
        hyp = result[0]
        if hasattr(hyp, "text"):
            text = hyp.text
        else:
            text = str(hyp)

        segments = []
        if hasattr(hyp, "timestep") and hyp.timestep:
            for seg in hyp.timestep.get("segment", []):
                segments.append({
                    "text": seg.get("text", ""),
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                })
        elif hasattr(hyp, "words") and hyp.words:
            for w in hyp.words:
                segments.append({
                    "text": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                })
    else:
        text = str(result)
        segments = []

    return {"text": text, "segments": segments}
