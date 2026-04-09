from fastapi import APIRouter, HTTPException

from ..server import get_manager
from ..stats import get_stats

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models():
    """List models in OpenAI-compatible envelope.

    Returns both 'data' (OpenAI format) and 'models' (native format) for
    backward compatibility with existing clients and dashboard.
    """
    mgr = get_manager()
    items = mgr.status()
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": 0,
                "owned_by": "inferbox",
                "type": m["type"],
                "model_id": m["model_id"],
                "vram_mb": m["vram_mb"],
                "status": m["status"],
                "idle_seconds": m.get("idle_seconds"),
            }
            for m in items
        ],
        "models": items,  # back-compat
    }


@router.get("/stats")
async def stats():
    """Aggregate request stats per model and endpoint."""
    return get_stats()


@router.post("/models/{model_id}/load")
async def load_model(model_id: str):
    mgr = get_manager()
    if model_id not in mgr.registry:
        raise HTTPException(404, f"Unknown model: {model_id}")
    try:
        await mgr.load(model_id)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"status": "loaded", "model": model_id}


@router.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    mgr = get_manager()
    if model_id not in mgr.registry:
        raise HTTPException(404, f"Unknown model: {model_id}")
    await mgr.unload(model_id)
    return {"status": "unloaded", "model": model_id}
