import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import BaseModel


class ModelConfig(BaseModel):
    type: str  # embedding, reranker, generate, transcription, diarization, image_gen
    loader: str  # hf_embed, hf_causal, hf_reranker, gguf, nemo_asr, diffusers
    model_id: str
    model_file: str | None = None  # for GGUF
    dtype: str = "float16"
    trust_remote_code: bool = False
    vram_mb: int = 1000
    default_for: str | None = None
    options: dict = {}
    quantize: str | None = None  # auto, int8, int4
    device: str | None = None  # cuda:0, cuda:1, cpu (None = auto)
    lora_adapters: dict[str, str] = {}  # adapter_name -> hf path
    draft_model: str | None = None  # registry id of draft model for speculative decoding


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8811
    api_key: str = "change-me"  # Primary key (back-compat)
    api_keys: str = ""  # Comma-separated additional keys: "key1:label1,key2:label2"
    idle_timeout: int = 300
    total_vram_mb: int = 12000
    models_config: str = "models.yaml"
    hf_cache: str = ""
    preload: str = ""
    audit_log: str = ""  # Path to audit log file (JSONL)
    rate_limit_per_min: int = 600  # Per-key default rate limit
    drain_timeout: int = 30  # Seconds to wait for in-flight requests on shutdown
    enable_micro_batching: bool = True
    micro_batch_window_ms: int = 20
    micro_batch_max_size: int = 64

    model_config = {"env_prefix": "INFERBOX_", "env_file": ".env"}


def load_model_registry(path: str) -> dict[str, ModelConfig]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        data = yaml.safe_load(f)
    models = {}
    for name, cfg in data.get("models", {}).items():
        models[name] = ModelConfig(**cfg)
    return models


settings = Settings()
