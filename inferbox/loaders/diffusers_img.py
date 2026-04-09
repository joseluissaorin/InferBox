"""Diffusers-based image generation loader.

Supports SDXL, SDXL Turbo, Flux Schnell, etc.
"""
import base64
import io
import logging
from dataclasses import dataclass
from typing import Any

import torch

from ..config import ModelConfig

logger = logging.getLogger("inferbox")


@dataclass
class DiffusersModel:
    pipe: Any
    device: str


def load(config: ModelConfig) -> DiffusersModel:
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None,
        ).to(device)
        # Memory optimisations
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        if device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass
    except torch.cuda.OutOfMemoryError:
        logger.warning("GPU OOM loading image model, retrying on CPU")
        torch.cuda.empty_cache()
        device = "cpu"
        pipe = AutoPipelineForText2Image.from_pretrained(
            config.model_id, torch_dtype=torch.float32,
        ).to(device)

    return DiffusersModel(pipe=pipe, device=device)


def unload(model: DiffusersModel):
    del model.pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_image(
    model: DiffusersModel,
    config: ModelConfig,
    prompt: str,
    negative_prompt: str | None = None,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int | None = None,
    n: int = 1,
) -> list[str]:
    """Generate n images, return as base64 PNG strings."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=model.device).manual_seed(seed)

    kwargs: dict = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": n,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if generator is not None:
        kwargs["generator"] = generator

    result = model.pipe(**kwargs)
    images = result.images

    out: list[str] = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out
