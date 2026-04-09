import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import settings, load_model_registry
from .model_manager import ModelManager
from .auth import registry as key_registry, limiter, audit
from .tracing import init_tracing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("inferbox")

manager: ModelManager | None = None
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Drain state
_draining = False
_in_flight = 0
_drain_lock = asyncio.Lock()


async def verify_api_key(request: Request, key: str | None = Security(api_key_header)):
    """Verify API key (X-API-Key or Authorization: Bearer) and apply rate limit."""
    if _draining:
        raise HTTPException(status_code=503, detail="Server draining for shutdown")

    # Also accept Authorization: Bearer for OpenAI SDK compatibility
    if not key:
        auth_header = request.headers.get("authorization") or ""
        if auth_header.lower().startswith("bearer "):
            key = auth_header[7:].strip()

    if not key or not key_registry.is_valid(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if not limiter.check(key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    request.state.api_key = key
    request.state.api_key_label = key_registry.label_for(key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    init_tracing()
    registry = load_model_registry(settings.models_config)
    manager = ModelManager(registry, settings.total_vram_mb, settings.idle_timeout)
    manager.start_idle_checker()
    logger.info(f"InferBox started. {len(registry)} models registered.")

    if settings.preload:
        preload_ids = [m.strip() for m in settings.preload.split(",") if m.strip()]
        for model_id in preload_ids:
            if model_id not in registry:
                logger.warning(f"Preload: unknown model '{model_id}', skipping")
                continue
            try:
                logger.info(f"Preloading {model_id}...")
                await manager.load(model_id)
                logger.info(f"Preloaded {model_id}")
            except Exception as e:
                logger.warning(f"Failed to preload {model_id}: {e}")

    yield

    # Graceful drain
    global _draining
    _draining = True
    logger.info(f"Draining... (timeout {settings.drain_timeout}s)")
    deadline = time.time() + settings.drain_timeout
    while _in_flight > 0 and time.time() < deadline:
        await asyncio.sleep(0.1)
    if _in_flight > 0:
        logger.warning(f"Drain timeout: {_in_flight} requests still in flight")
    else:
        logger.info("Drain complete")

    manager.stop_idle_checker()
    for mid in list(manager.loaded.keys()):
        try:
            await manager.unload(mid)
        except Exception as e:
            logger.warning(f"Error unloading {mid} on shutdown: {e}")
    logger.info("InferBox shutting down.")


app = FastAPI(title="InferBox", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_inflight_and_audit(request: Request, call_next):
    """Tracks in-flight requests for graceful drain + writes audit log."""
    global _in_flight
    _in_flight += 1
    t0 = time.time()
    try:
        response = await call_next(request)
        latency_ms = int((time.time() - t0) * 1000)
        # Audit log
        label = getattr(request.state, "api_key_label", None) or "anon"
        audit.record(label, request.url.path, response.status_code, latency_ms)
        return response
    finally:
        _in_flight -= 1


def get_manager() -> ModelManager:
    return manager


@app.get("/v1/health")
async def health():
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "gpu": torch.cuda.get_device_name(0),
                "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6),
                "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6),
                "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / 1e6),
            }
    except ImportError:
        pass
    return {
        "status": "draining" if _draining else "ok",
        "models_loaded": len(manager.loaded) if manager else 0,
        "vram_used_mb": manager.used_vram() if manager else 0,
        "in_flight": _in_flight,
        "gpu": gpu_info,
    }


# Import and register route modules
from .routes import models, embed, generate, rerank, transcribe, chunk, sessions, openai_compat, dashboard, metrics, admin, images

app.include_router(models.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(embed.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(generate.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(rerank.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(transcribe.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(chunk.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(sessions.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(openai_compat.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(images.router, prefix="/v1", dependencies=[Depends(verify_api_key)])
app.include_router(admin.router, prefix="/v1/admin", dependencies=[Depends(verify_api_key)])
# Dashboard and metrics are unauthenticated for ease of use on LAN
app.include_router(dashboard.router)
app.include_router(metrics.router)
