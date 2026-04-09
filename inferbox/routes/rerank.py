import asyncio
import time
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from ..server import get_manager
from ..stats import record_request

router = APIRouter(tags=["rerank"])


class RerankRequest(BaseModel):
    model: str | None = None
    query: str
    documents: list[str]
    top_k: int | None = None


class RerankResult(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


@router.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    mgr = get_manager()
    try:
        model_id = mgr.resolve_model(req.model, "reranker")
    except KeyError as e:
        raise HTTPException(400, str(e))
    try:
        entry = await mgr.get(model_id)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    t0 = time.time()
    try:
        results = await asyncio.to_thread(
            entry.loader_module.rerank,
            entry.obj,
            entry.config,
            req.query,
            req.documents,
            top_k=req.top_k,
        )
        record_request(model_id, "rerank", time.time() - t0, success=True,
                       items=len(req.documents))
        return RerankResponse(model=model_id, results=results)
    except Exception:
        record_request(model_id, "rerank", time.time() - t0, success=False)
        raise
