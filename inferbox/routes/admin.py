"""Admin endpoints: hot reload models.yaml, etc."""
from fastapi import APIRouter, HTTPException

from ..server import get_manager
from ..config import settings, load_model_registry

router = APIRouter(tags=["admin"])


@router.post("/reload")
async def reload_models():
    """Re-read models.yaml and update the registry without restart."""
    mgr = get_manager()
    if mgr is None:
        raise HTTPException(503, "Manager not initialized")

    try:
        new_registry = load_model_registry(settings.models_config)
    except Exception as e:
        raise HTTPException(400, f"Failed to load config: {e}")

    added = []
    removed = []
    updated = []

    # Find removals
    for old_id in list(mgr.registry.keys()):
        if old_id not in new_registry:
            removed.append(old_id)
            if old_id in mgr.loaded:
                await mgr.unload(old_id)

    # Find additions and updates
    for new_id, new_cfg in new_registry.items():
        if new_id not in mgr.registry:
            added.append(new_id)
        elif mgr.registry[new_id].model_dump() != new_cfg.model_dump():
            updated.append(new_id)
            # If currently loaded, reload with new config
            if new_id in mgr.loaded:
                await mgr.unload(new_id)

    # Replace registry
    mgr.registry = new_registry
    # Rebuild defaults
    mgr.defaults = {}
    for name, cfg in new_registry.items():
        if cfg.default_for and cfg.default_for not in mgr.defaults:
            mgr.defaults[cfg.default_for] = name

    return {
        "added": added,
        "removed": removed,
        "updated": updated,
        "total": len(new_registry),
    }


@router.get("/audit/recent")
async def recent_audit(limit: int = 100):
    """Return the last N audit log entries."""
    if not settings.audit_log:
        return {"entries": [], "note": "audit log disabled"}
    try:
        from pathlib import Path
        import json
        path = Path(settings.audit_log)
        if not path.exists():
            return {"entries": []}
        with open(path) as f:
            lines = f.readlines()
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
        return {"entries": entries}
    except Exception as e:
        raise HTTPException(500, str(e))
