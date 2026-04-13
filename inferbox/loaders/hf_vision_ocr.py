"""HuggingFace vision-language OCR loader (GLM-OCR, etc.).

Uses transformers' `AutoModelForImageTextToText` which is the correct
AutoModel class for vision models that produce text (OCR, image
captioning, VQA). This is *not* `AutoModelForCausalLM` — those are
text-only. transformers 5.x renamed `AutoModelForVision2Seq` to
`AutoModelForImageTextToText`.
"""
import base64
import io
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image

from ..config import ModelConfig

logger = logging.getLogger("inferbox")


@dataclass
class VisionOCRModel:
    model: Any
    processor: Any
    device: str


def _resolve_device(config: ModelConfig) -> str:
    if config.device:
        return config.device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load(config: ModelConfig) -> VisionOCRModel:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = _resolve_device(config)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    ).to(device).eval()

    logger.info(f"Vision-OCR {config.model_id} loaded on {device}")
    return VisionOCRModel(model=model, processor=processor, device=device)


def unload(model: VisionOCRModel):
    # NOTE: do NOT `del model.model` / `del model.processor` — if any in-flight
    # request is still using the reference, deleting attributes creates a
    # use-after-free race. See hf_embed.unload for details.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _decode_image(img_data: str) -> Image.Image:
    """Accept a base64 string (with optional data: prefix) or a raw file path."""
    if img_data.startswith("data:"):
        _, _, b64 = img_data.partition(",")
        img_data = b64
    try:
        raw = base64.b64decode(img_data)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        # Fall back to treating as path
        return Image.open(img_data).convert("RGB")


def generate(
    model: VisionOCRModel,
    config: ModelConfig,
    prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    stop: Optional[list[str]] = None,
    images: Optional[list[str]] = None,
    grammar=None,
) -> dict:
    """Run OCR/VQA over one or more images and return the decoded text.

    Accepts either `prompt`+`images`, or `messages` in chat format with
    embedded image parts. Returns {"text": ..., "usage": {...}}.
    """
    if not images and not messages:
        raise ValueError("vision OCR generate requires images (or messages with images)")

    pil_images: list[Image.Image] = []
    text_prompt = prompt or ""

    if messages:
        # Extract text + images from chat messages
        text_parts: list[str] = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("type")
                        if t == "text":
                            text_parts.append(part.get("text", ""))
                        elif t in ("image", "image_url"):
                            url = part.get("image_url", {}).get("url") if t == "image_url" else part.get("image")
                            if url:
                                pil_images.append(_decode_image(url))
        text_prompt = "\n".join(p for p in text_parts if p)

    if images:
        for img in images:
            pil_images.append(_decode_image(img))

    if not pil_images:
        raise ValueError("no images decoded for vision OCR generate")

    # Default OCR prompt if caller gave none
    if not text_prompt:
        text_prompt = "Transcribe all text visible in the image."

    # Use the processor's chat template with images embedded in content.
    # GLM-OCR (and most transformers 5.x vision LMs) expect the PIL image
    # objects to be passed *inside* each content part, not as a separate
    # `images=` kwarg — doing the latter produces an empty image-token
    # count and a "tokens: 0, features: N" mismatch at generation time.
    chat_content: list[dict] = []
    for img in pil_images:
        chat_content.append({"type": "image", "image": img})
    chat_content.append({"type": "text", "text": text_prompt})
    chat = [{"role": "user", "content": chat_content}]
    inputs = model.processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.model.generate(**inputs, **gen_kwargs)

    # Strip the input prompt tokens so we only decode the generated continuation
    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    generated = outputs[:, input_len:] if input_len else outputs

    text = model.processor.batch_decode(generated, skip_special_tokens=True)[0]

    return {
        "text": text,
        "usage": {
            "prompt_tokens": int(input_len),
            "completion_tokens": int(generated.shape[1]) if generated.ndim > 1 else 0,
        },
    }
