"""HuggingFace causal LM loader (e.g. GLM-OCR, general generation)."""
import logging
import torch
from dataclasses import dataclass
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger("inferbox")


@dataclass
class HFCausalModel:
    model: Any
    tokenizer: Any
    processor: Any
    device: str


def load(config: ModelConfig) -> HFCausalModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, config.dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, trust_remote_code=config.trust_remote_code
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    ).to(device).eval()

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            config.model_id, trust_remote_code=config.trust_remote_code
        )
    except Exception:
        pass

    return HFCausalModel(model=model, tokenizer=tokenizer, processor=processor, device=device)


def unload(model: HFCausalModel):
    # Do NOT `del model.XXX`. See hf_embed.unload for the race explanation:
    # deleting attributes on a live object crashes any in-flight request
    # that's holding a reference. Let GC reclaim after the manager drops
    # its own reference.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(
    model: HFCausalModel,
    config: ModelConfig,
    prompt: str | None = None,
    messages: list[dict] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stop: list[str] | None = None,
    images: list[str] | None = None,
) -> dict:
    if messages:
        # Use chat template if available
        try:
            text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    else:
        text = prompt or ""

    inputs = model.tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 0.01),
    }
    if stop:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnTokens(StoppingCriteria):
            def __init__(self, stop_ids):
                self.stop_ids = stop_ids

            def __call__(self, input_ids, scores, **kwargs):
                for stop_id in self.stop_ids:
                    if input_ids[0][-len(stop_id):].tolist() == stop_id:
                        return True
                return False

        stop_ids = [model.tokenizer.encode(s, add_special_tokens=False) for s in stop]
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens(stop_ids)])

    with torch.no_grad():
        output = model.model.generate(**inputs, **gen_kwargs)

    generated = output[0][input_len:]
    text_out = model.tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "text": text_out,
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": len(generated),
        },
    }
