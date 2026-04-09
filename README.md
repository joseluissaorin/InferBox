# InferBox

A self-hosted GPU inference server. Point any OpenAI-compatible client at it and use your own hardware for embeddings, reranking, text generation, transcription, or image generation.

InferBox is a thin FastAPI server that wraps HuggingFace Transformers, llama-cpp-python, NeMo ASR, and Diffusers behind one consistent REST API with a model manager that loads models on demand, unloads them when idle, and multiplexes many model types on a single GPU.

## Why

- **One endpoint for many model types** — embeddings, rerankers, LLMs, speech-to-text, image generation, all behind the same API key
- **Model manager** — loads models on demand, auto-unloads after idle timeout, evicts LRU to fit VRAM
- **OpenAI-compatible shim** — existing tools (LangChain, LlamaIndex, Aider, OpenAI SDK) work unchanged against `http://your-box:8811/v1`
- **Runs anywhere you have a CUDA GPU** — single box, docker container, or behind Tailscale
- **Project-agnostic** — drop it on a spare GPU and let all your projects share it

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/v1/health` | Health + GPU memory info + in-flight request count |
| `GET` | `/v1/models` | List models (OpenAI-compatible envelope) |
| `POST` | `/v1/models/{id}/load` | Preload a model |
| `POST` | `/v1/models/{id}/unload` | Unload a model |
| `GET` | `/v1/stats` | Request counts, p50/p95 latency per model+endpoint |
| `POST` | `/v1/embed` | Text and/or image embeddings (batch) |
| `POST` | `/v1/embeddings` | OpenAI-compatible embeddings |
| `POST` | `/v1/rerank` | Query + documents reranking |
| `POST` | `/v1/generate` | Text generation (supports streaming SSE) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non, grammar-constrained) |
| `POST` | `/v1/transcribe` | Audio file → text + segments |
| `POST` | `/v1/images/generations` | OpenAI-compatible image generation |
| `POST` | `/v1/chunk` | Text splitting (chars / sentences / paragraphs / tokens) |
| `POST` | `/v1/sessions` | Stateful chat sessions |
| `POST` | `/v1/admin/reload` | Hot-reload `models.yaml` |
| `GET` | `/metrics` | Prometheus exposition |
| `GET` | `/ui` | Live dashboard (models, stats, GPU memory) |

All `/v1/*` routes accept either `X-API-Key: ...` or `Authorization: Bearer ...`.

## Quick start

### Docker

```bash
docker compose up -d
```

Edit `docker-compose.yml` to set `INFERBOX_API_KEY` and your preload list first.

### Systemd + venv

```bash
cd ~/InferBox
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional extras depending on what you want to run:
pip install peft bitsandbytes diffusers opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

cp .env.example .env
# edit .env to set INFERBOX_API_KEY and INFERBOX_MODELS_CONFIG

# Run it
uvicorn inferbox.server:app --host 0.0.0.0 --port 8811
```

The included `inferbox.service` unit file can be copied to `/etc/systemd/system/` for a supervised install.

## Models

Declare models in `models.yaml`. Each entry specifies a `type`, a `loader`, the HuggingFace ID or GGUF path, and an estimated VRAM footprint.

```yaml
models:
  qwen3-vl-embed:
    type: embedding
    loader: hf_embed
    model_id: "Qwen/Qwen3-VL-Embedding-2B"
    dtype: float16
    trust_remote_code: true
    vram_mb: 4500
    default_for: embedding
    options:
      multimodal: true

  bge-reranker:
    type: reranker
    loader: hf_reranker
    model_id: "BAAI/bge-reranker-v2-m3"
    dtype: float16
    vram_mb: 600
    default_for: reranker

  qwen3-0.6b:
    type: generate
    loader: gguf
    model_id: "Qwen/Qwen3-0.6B-GGUF"
    model_file: "Qwen3-0.6B-Q8_0.gguf"
    vram_mb: 800
    default_for: generate
    options:
      n_gpu_layers: -1
      n_ctx: 8192

  sdxl-turbo:
    type: image_gen
    loader: diffusers_img
    model_id: "stabilityai/sdxl-turbo"
    dtype: float16
    vram_mb: 7000
    default_for: image_gen
```

### Available loaders

| Loader | What it runs |
|---|---|
| `hf_embed` | HuggingFace embedding models (incl. multimodal like Qwen3-VL-Embedding) |
| `hf_causal` | HuggingFace `AutoModelForCausalLM` (OCR, general generation) |
| `hf_reranker` | sentence-transformers `CrossEncoder` (e.g. bge-reranker, jina-reranker) |
| `gguf` | `llama-cpp-python` with KV cache reuse, streaming, grammars, speculative decoding |
| `nemo_asr` | NVIDIA NeMo ASR (e.g. Parakeet TDT) |
| `diffusers_img` | HuggingFace Diffusers `AutoPipelineForText2Image` (SDXL, Flux) |

## Usage examples

### OpenAI Python SDK (drop-in)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://your-box:8811/v1",
    api_key="your-inferbox-key",
)

# Embeddings
r = client.embeddings.create(
    model="qwen3-vl-embed",
    input=["hello", "world"],
)
print(len(r.data[0].embedding))  # 2048

# Chat
r = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Say hi"}],
    max_tokens=20,
)
print(r.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,
    max_tokens=50,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Raw curl

```bash
# Embed
curl -X POST http://your-box:8811/v1/embed \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"input": ["machine learning is cool"]}'

# Rerank
curl -X POST http://your-box:8811/v1/rerank \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "AI", "documents": ["neural networks", "the cat sat"]}'

# Generate (streaming)
curl -N -X POST http://your-box:8811/v1/generate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain caching in one sentence:", "max_tokens": 60, "stream": true}'

# JSON-constrained generation
curl -X POST http://your-box:8811/v1/chat/completions \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Return a JSON object with keys name and age for Alice, 30"}],
    "response_format": {"type": "json_object"},
    "max_tokens": 50
  }'
```

## Features

- **Micro-batching**: concurrent requests for the same model are coalesced into one forward pass (configurable window and batch size)
- **KV cache reuse** for GGUF models via `LlamaRAMCache`
- **Speculative decoding** with a draft model (llama.cpp)
- **Grammar-constrained generation**: JSON-object or JSON-schema response formats
- **Per-key rate limiting + audit log** (JSONL)
- **Circuit breaker** and retries in the client (see `backend/app/services/inferbox_client.py` of ScholarisWeb for a reference client)
- **Graceful shutdown** — refuses new requests during drain, waits for in-flight to finish, cleanly unloads all models
- **Hot reload** of `models.yaml` via `POST /v1/admin/reload`
- **OpenTelemetry tracing** (activates if `opentelemetry-sdk` is installed; falls back to console exporter if no OTLP endpoint is set)
- **Prometheus metrics** at `/metrics`
- **Dashboard** at `/ui` (dark theme, auto-refresh, shows GPU memory / loaded models / endpoint latency)
- **Multi-GPU aware** model manager (picks the device with most free VRAM, tracks per-device budgets)
- **BitsAndBytes int8/int4 quantization** on load (set `quantize: int4` on the model in `models.yaml`)
- **LoRA adapters** via PEFT (declare `lora_adapters: {name: hf_path}` in model config)
- **GPU OOM → CPU fallback** for the embedding loader
- **Image / multimodal embeddings** — send base64 images alongside text inputs to `/v1/embed`

## Configuration reference

See `.env.example` for all runtime environment variables.

## License

MIT — see LICENSE.
