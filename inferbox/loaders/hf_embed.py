"""HuggingFace embedding model loader.

Supports text-only and multimodal (text + images), bitsandbytes quantization,
LoRA adapters via PEFT, and explicit device selection for multi-GPU.
"""
import base64
import io
import logging
import torch
from dataclasses import dataclass, field
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger("inferbox")


@dataclass
class HFEmbedModel:
    model: Any
    tokenizer: Any
    processor: Any
    device: str
    multimodal: bool
    active_adapter: str | None = None
    available_adapters: dict[str, str] = field(default_factory=dict)


def _make_quant_config(config: ModelConfig):
    """Build a BitsAndBytesConfig if quantize is set."""
    if not config.quantize:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning("bitsandbytes not available - falling back to fp16")
        return None
    if config.quantize == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if config.quantize == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    return None


def _resolve_device(config: ModelConfig) -> str:
    """Resolve device from config, defaulting to first available CUDA."""
    if config.device:
        return config.device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load(config: ModelConfig) -> HFEmbedModel:
    from transformers import AutoModel, AutoTokenizer, AutoProcessor

    device = _resolve_device(config)
    dtype = getattr(torch, config.dtype, torch.float16)
    quant_config = _make_quant_config(config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, trust_remote_code=config.trust_remote_code
    )

    # Try GPU; fallback to CPU on OOM
    try:
        kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
            kwargs["device_map"] = device
            model = AutoModel.from_pretrained(config.model_id, **kwargs).eval()
        else:
            model = AutoModel.from_pretrained(config.model_id, **kwargs).to(device).eval()
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"GPU OOM loading {config.model_id}, retrying on CPU")
        torch.cuda.empty_cache()
        device = "cpu"
        model = AutoModel.from_pretrained(
            config.model_id,
            torch_dtype=torch.float32,
            trust_remote_code=config.trust_remote_code,
        ).to(device).eval()

    multimodal = bool(config.options.get("multimodal"))
    processor = None
    if multimodal:
        try:
            processor = AutoProcessor.from_pretrained(
                config.model_id, trust_remote_code=config.trust_remote_code
            )
        except Exception as e:
            logger.warning(f"No processor for {config.model_id}: {e}")
            multimodal = False

    # Apply LoRA adapters via PEFT
    available_adapters = {}
    active_adapter = None
    if config.lora_adapters:
        try:
            from peft import PeftModel
            for adapter_name, adapter_path in config.lora_adapters.items():
                try:
                    if active_adapter is None:
                        model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)
                        active_adapter = adapter_name
                    else:
                        model.load_adapter(adapter_path, adapter_name=adapter_name)
                    available_adapters[adapter_name] = adapter_path
                    logger.info(f"Loaded LoRA adapter '{adapter_name}' from {adapter_path}")
                except Exception as e:
                    logger.warning(f"Failed to load LoRA adapter {adapter_name}: {e}")
        except ImportError:
            logger.warning("peft not installed - LoRA adapters skipped")

    return HFEmbedModel(
        model=model, tokenizer=tokenizer, processor=processor,
        device=device, multimodal=multimodal,
        active_adapter=active_adapter, available_adapters=available_adapters,
    )


def unload(model: HFEmbedModel):
    del model.model
    del model.tokenizer
    if model.processor:
        del model.processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _decode_image(b64: str):
    from PIL import Image
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def embed(
    model: HFEmbedModel,
    config: ModelConfig,
    texts: list[str],
    instruction: str | None = None,
    images: list[str] | None = None,
) -> list[list[float]]:
    if instruction:
        texts = [f"{instruction}{t}" for t in texts]

    # Multimodal path
    if images and model.multimodal and model.processor is not None:
        pil_images = [_decode_image(img) for img in images]
        if len(texts) < len(pil_images):
            texts = texts + [""] * (len(pil_images) - len(texts))

        try:
            inputs = model.processor(
                text=texts, images=pil_images,
                padding=True, truncation=True, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model.model(**inputs)

            attention_mask = inputs.get("attention_mask")
            hidden = outputs.last_hidden_state
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                embeddings = hidden.mean(dim=1)

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().float().numpy().tolist()
        except Exception as e:
            logger.error(f"Multimodal embed failed: {e}, falling back to text-only")

    # Text-only path
    inputs = model.tokenizer(
        texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.model(**inputs)

    attention_mask = inputs["attention_mask"]
    hidden = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    embeddings = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().float().numpy().tolist()
