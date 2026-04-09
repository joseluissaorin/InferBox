import asyncio
import tempfile
import time
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from ..server import get_manager
from ..stats import record_request

router = APIRouter(tags=["transcribe"])


class Segment(BaseModel):
    text: str
    start: float
    end: float


class TranscribeResponse(BaseModel):
    model: str
    text: str
    segments: list[Segment] = []


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
):
    mgr = get_manager()
    try:
        model_id = mgr.resolve_model(model, "transcription")
    except KeyError as e:
        raise HTTPException(400, str(e))
    try:
        entry = await mgr.get(model_id)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    # Save upload to temp file
    suffix = f".{file.filename.split('.')[-1]}" if file.filename and "." in file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    t0 = time.time()
    try:
        result = await asyncio.to_thread(
            entry.loader_module.transcribe,
            entry.obj,
            entry.config,
            tmp_path,
            language=language,
        )
        record_request(model_id, "transcribe", time.time() - t0, success=True)
        return TranscribeResponse(model=model_id, **result)
    except Exception:
        record_request(model_id, "transcribe", time.time() - t0, success=False)
        raise
    finally:
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
