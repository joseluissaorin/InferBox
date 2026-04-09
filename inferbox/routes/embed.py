import asyncio
import time
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from ..server import get_manager
from ..stats import record_request
from ..batcher import registry as batcher_registry
from ..config import settings

router = APIRouter(tags=["embed"])


class EmbedRequest(BaseModel):
    model: str | None = None
    input: list[str] = []
    instruction: str | None = None
    images: list[str] | None = None
    batched: bool = True  # Allow micro-batching with other requests


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]


async def _make_runner(model_id: str, entry):
    """Build a runner that processes a batch of embed requests in one forward pass."""

    async def runner(payloads: list[dict]) -> list[list[list[float]]]:
        # Each payload is {"texts": [...], "instruction": ?, "images": [...]?}
        # Coalesce all texts into a single forward pass
        all_texts: list[str] = []
        all_images: list[str] | None = None
        boundaries: list[int] = []
        instruction = None

        for p in payloads:
            instruction = instruction or p.get("instruction")
            texts = p.get("texts") or []
            imgs = p.get("images") or []
            all_texts.extend(texts)
            if imgs:
                if all_images is None:
                    all_images = []
                all_images.extend(imgs)
            boundaries.append(len(texts))

        # Run forward pass once with the coalesced batch
        embeddings = await asyncio.to_thread(
            entry.loader_module.embed,
            entry.obj, entry.config,
            all_texts,
            instruction=instruction,
            images=all_images,
        )

        # Split back into per-request results
        results: list[list[list[float]]] = []
        idx = 0
        for n in boundaries:
            results.append(embeddings[idx:idx + n])
            idx += n
        return results

    return runner


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if not req.input and not req.images:
        raise HTTPException(400, "Either 'input' or 'images' is required")

    mgr = get_manager()
    try:
        model_id = mgr.resolve_model(req.model, "embedding")
    except KeyError as e:
        raise HTTPException(400, str(e))
    try:
        entry = await mgr.get(model_id)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    t0 = time.time()
    try:
        # Decide path: micro-batched if enabled and no instruction (instruction
        # would change the prompt prefix and break coalescing)
        use_batcher = (
            settings.enable_micro_batching
            and req.batched
            and not req.instruction
            and not req.images
        )

        if use_batcher:
            runner = await _make_runner(model_id, entry)
            batcher = await batcher_registry.get_or_create(model_id, "embed", runner)
            embeddings = await batcher.submit({
                "texts": req.input,
                "images": req.images,
                "instruction": req.instruction,
            })
        else:
            embeddings = await asyncio.to_thread(
                entry.loader_module.embed,
                entry.obj, entry.config,
                req.input,
                instruction=req.instruction,
                images=req.images,
            )

        record_request(model_id, "embed", time.time() - t0, success=True,
                       items=max(len(req.input), len(req.images or [])))
        return EmbedResponse(model=model_id, embeddings=embeddings)
    except Exception:
        record_request(model_id, "embed", time.time() - t0, success=False)
        raise
