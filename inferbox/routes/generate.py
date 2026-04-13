import asyncio
import json
import time
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..server import get_manager
from ..stats import record_request

router = APIRouter(tags=["generate"])


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str | None = None
    prompt: str | None = None
    messages: list[Message] | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    stop: list[str] | None = None
    images: list[str] | None = None  # base64 images for multimodal
    stream: bool = False


class GenerateResponse(BaseModel):
    model: str
    text: str
    usage: dict = {}


@router.post("/generate")
async def generate(req: GenerateRequest):
    if not req.prompt and not req.messages:
        raise HTTPException(400, "Either 'prompt' or 'messages' is required")

    mgr = get_manager()
    try:
        model_id = mgr.resolve_model(req.model, "generate")
    except KeyError as e:
        raise HTTPException(400, str(e))

    messages_dict = [m.model_dump() for m in req.messages] if req.messages else None

    if req.stream:
        # Streaming response via SSE. The `use(serialize=True)` context
        # holds both the in-flight counter (blocks eviction) and the
        # per-model inference lock (blocks concurrent generate calls —
        # required because llama-cpp/HF causal LMs are NOT thread-safe).
        async def event_stream():
            t0 = time.time()
            try:
                try:
                    ctx = mgr.use(model_id, serialize=True)
                    entry = await ctx.__aenter__()
                except RuntimeError as e:
                    yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
                    return
                try:
                    stream_fn = getattr(entry.loader_module, "generate_stream", None)
                    if stream_fn is None:
                        result = await asyncio.to_thread(
                            entry.loader_module.generate,
                            entry.obj, entry.config,
                            prompt=req.prompt, messages=messages_dict,
                            max_tokens=req.max_tokens, temperature=req.temperature,
                            stop=req.stop, images=req.images,
                        )
                        yield f"data: {json.dumps({'text': result['text'], 'done': True, 'usage': result.get('usage', {})})}\n\n"
                    else:
                        queue: asyncio.Queue = asyncio.Queue()
                        loop = asyncio.get_event_loop()

                        def producer():
                            try:
                                for token in stream_fn(
                                    entry.obj, entry.config,
                                    prompt=req.prompt, messages=messages_dict,
                                    max_tokens=req.max_tokens, temperature=req.temperature,
                                    stop=req.stop,
                                ):
                                    asyncio.run_coroutine_threadsafe(queue.put(("token", token)), loop)
                                asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
                            except Exception as e:
                                asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

                        asyncio.get_event_loop().run_in_executor(None, producer)

                        while True:
                            kind, payload = await queue.get()
                            if kind == "token":
                                yield f"data: {json.dumps({'text': payload, 'done': False})}\n\n"
                            elif kind == "done":
                                yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
                                break
                            elif kind == "error":
                                yield f"data: {json.dumps({'error': payload, 'done': True})}\n\n"
                                break

                    record_request(model_id, "generate", time.time() - t0, success=True)
                finally:
                    await ctx.__aexit__(None, None, None)
            except Exception as e:
                record_request(model_id, "generate", time.time() - t0, success=False)
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    t0 = time.time()
    try:
        async with mgr.use(model_id, serialize=True) as entry:
            result = await asyncio.to_thread(
                entry.loader_module.generate,
                entry.obj, entry.config,
                prompt=req.prompt, messages=messages_dict,
                max_tokens=req.max_tokens, temperature=req.temperature,
                stop=req.stop, images=req.images,
            )
        record_request(model_id, "generate", time.time() - t0, success=True)
        return GenerateResponse(model=model_id, **result)
    except RuntimeError as e:
        record_request(model_id, "generate", time.time() - t0, success=False)
        raise HTTPException(503, str(e))
    except Exception:
        record_request(model_id, "generate", time.time() - t0, success=False)
        raise
