"""Image generation endpoint (OpenAI-compatible)."""
import asyncio
import time
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from ..server import get_manager
from ..stats import record_request

router = APIRouter(tags=["images"])


class ImageGenRequest(BaseModel):
    model: str | None = None
    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 4
    guidance_scale: float = 0.0
    seed: int | None = None
    n: int = 1
    response_format: str = "b64_json"  # OpenAI compat


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None


class ImageGenResponse(BaseModel):
    created: int
    data: list[ImageData]
    model: str


@router.post("/images/generations", response_model=ImageGenResponse)
async def generate_images(req: ImageGenRequest):
    mgr = get_manager()
    try:
        model_id = mgr.resolve_model(req.model, "image_gen")
    except KeyError as e:
        raise HTTPException(400, str(e))

    t0 = time.time()
    try:
        # Diffusers pipelines share scheduler state; serialize.
        async with mgr.use(model_id, serialize=True) as entry:
            images_b64 = await asyncio.to_thread(
                entry.loader_module.generate_image,
                entry.obj, entry.config,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                n=req.n,
            )
        record_request(model_id, "images.generations", time.time() - t0, success=True, items=req.n)
        return ImageGenResponse(
            created=int(time.time()),
            model=model_id,
            data=[ImageData(b64_json=b) for b in images_b64],
        )
    except RuntimeError as e:
        record_request(model_id, "images.generations", time.time() - t0, success=False)
        raise HTTPException(503, str(e))
    except Exception:
        record_request(model_id, "images.generations", time.time() - t0, success=False)
        raise
