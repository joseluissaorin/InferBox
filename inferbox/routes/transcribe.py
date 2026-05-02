import asyncio
import os
import subprocess
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


def _normalize_audio(src_path: str, dst_path: str) -> None:
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-i", src_path,
            "-ac", "1", "-ar", "16000", "-f", "wav",
            dst_path,
        ],
        capture_output=True,
        timeout=300,
    )
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="replace").strip()[:500] or "ffmpeg failed"
        raise HTTPException(status_code=400, detail=f"audio decode failed: {msg}")


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

    suffix = f".{file.filename.split('.')[-1]}" if file.filename and "." in file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        upload_path = tmp.name

    norm_path = upload_path + ".16k.wav"
    t0 = time.time()
    try:
        await asyncio.to_thread(_normalize_audio, upload_path, norm_path)
        async with mgr.use(model_id, serialize=True) as entry:
            result = await asyncio.to_thread(
                entry.loader_module.transcribe,
                entry.obj,
                entry.config,
                norm_path,
                language=language,
            )
        record_request(model_id, "transcribe", time.time() - t0, success=True)
        return TranscribeResponse(model=model_id, **result)
    except RuntimeError as e:
        record_request(model_id, "transcribe", time.time() - t0, success=False)
        raise HTTPException(503, str(e))
    except Exception:
        record_request(model_id, "transcribe", time.time() - t0, success=False)
        raise
    finally:
        for p in (upload_path, norm_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
