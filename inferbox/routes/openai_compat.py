"""OpenAI-compatible API shim.

Exposes /v1/chat/completions and /v1/embeddings in OpenAI's schema,
delegating to InferBox's native loaders. Lets any tool that speaks the
OpenAI SDK (LangChain, LlamaIndex, Aider, etc.) use InferBox without
modification — just point base_url at http://host:8811/v1.
"""
import asyncio
import json
import time
import uuid
from typing import Any, Literal
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..server import get_manager
from ..stats import record_request

router = APIRouter(tags=["openai"])


# ============================================================================
# Chat Completions
# ============================================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] | str | None = None
    stream: bool = False
    response_format: ResponseFormat | None = None
    n: int = 1
    seed: int | None = None


class Choice(BaseModel):
    index: int
    message: dict
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    mgr = get_manager()

    # Resolve model: accept registry id or fall back to default
    if req.model in mgr.registry:
        model_id = req.model
    else:
        try:
            model_id = mgr.resolve_model(None, "generate")
        except KeyError:
            raise HTTPException(404, f"Model not found: {req.model}")

    messages_dict = [m.model_dump(exclude_none=True) for m in req.messages]
    stop = [req.stop] if isinstance(req.stop, str) else req.stop
    max_tokens = req.max_tokens or 512

    # Optional structured output
    grammar = None
    if req.response_format:
        if req.response_format.type == "json_object":
            grammar = "json"
        elif req.response_format.type == "json_schema" and req.response_format.json_schema:
            grammar = ("json_schema", req.response_format.json_schema)

    if req.stream:
        async def event_stream():
            t0 = time.time()
            cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            # Initial chunk with role
            first = {
                "id": cmpl_id, "object": "chat.completion.chunk",
                "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first)}\n\n"

            try:
                try:
                    ctx = mgr.use(model_id, serialize=True)
                    entry = await ctx.__aenter__()
                except RuntimeError as e:
                    err = {"error": {"message": str(e), "type": "server_error"}}
                    yield f"data: {json.dumps(err)}\n\n"
                    return
                try:
                    stream_fn = getattr(entry.loader_module, "generate_stream", None)
                    if stream_fn is None:
                        result = await asyncio.to_thread(
                            entry.loader_module.generate,
                            entry.obj, entry.config,
                            prompt=None, messages=messages_dict,
                            max_tokens=max_tokens, temperature=req.temperature,
                            stop=stop,
                        )
                        chunk = {
                            "id": cmpl_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_id,
                            "choices": [{"index": 0, "delta": {"content": result["text"]}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        queue: asyncio.Queue = asyncio.Queue()
                        loop = asyncio.get_event_loop()

                        def producer():
                            try:
                                for token in stream_fn(
                                    entry.obj, entry.config,
                                    prompt=None, messages=messages_dict,
                                    max_tokens=max_tokens, temperature=req.temperature,
                                    stop=stop,
                                ):
                                    asyncio.run_coroutine_threadsafe(queue.put(("token", token)), loop)
                                asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
                            except Exception as e:
                                asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

                        asyncio.get_event_loop().run_in_executor(None, producer)

                        while True:
                            kind, payload = await queue.get()
                            if kind == "token":
                                chunk = {
                                    "id": cmpl_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model_id,
                                    "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                            elif kind == "done":
                                break
                            elif kind == "error":
                                raise RuntimeError(payload)

                    final = {
                        "id": cmpl_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                    yield "data: [DONE]\n\n"
                    record_request(model_id, "chat.completions", time.time() - t0, success=True)
                finally:
                    await ctx.__aexit__(None, None, None)
            except Exception as e:
                record_request(model_id, "chat.completions", time.time() - t0, success=False)
                err = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    t0 = time.time()
    try:
        async with mgr.use(model_id, serialize=True) as entry:
            try:
                result = await asyncio.to_thread(
                    entry.loader_module.generate,
                    entry.obj, entry.config,
                    prompt=None, messages=messages_dict,
                    max_tokens=max_tokens, temperature=req.temperature,
                    stop=stop, grammar=grammar,
                )
            except TypeError:
                # Loader doesn't accept grammar param - retry without
                result = await asyncio.to_thread(
                    entry.loader_module.generate,
                    entry.obj, entry.config,
                    prompt=None, messages=messages_dict,
                    max_tokens=max_tokens, temperature=req.temperature,
                    stop=stop,
                )
        record_request(model_id, "chat.completions", time.time() - t0, success=True)

        usage_data = result.get("usage", {})
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=model_id,
            choices=[Choice(
                index=0,
                message={"role": "assistant", "content": result["text"]},
                finish_reason="stop",
            )],
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("prompt_tokens", 0) + usage_data.get("completion_tokens", 0),
            ),
        )
    except RuntimeError as e:
        record_request(model_id, "chat.completions", time.time() - t0, success=False)
        raise HTTPException(503, str(e))
    except Exception as e:
        record_request(model_id, "chat.completions", time.time() - t0, success=False)
        raise HTTPException(500, str(e))


# ============================================================================
# Embeddings
# ============================================================================

class EmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingItem]
    model: str
    usage: Usage


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def embeddings(req: EmbeddingsRequest):
    mgr = get_manager()
    if req.model in mgr.registry:
        model_id = req.model
    else:
        try:
            model_id = mgr.resolve_model(None, "embedding")
        except KeyError:
            raise HTTPException(404, f"Model not found: {req.model}")

    inputs = [req.input] if isinstance(req.input, str) else req.input

    t0 = time.time()
    try:
        async with mgr.use(model_id, serialize=False) as entry:
            result = await asyncio.to_thread(
                entry.loader_module.embed,
                entry.obj, entry.config, inputs,
            )
        record_request(model_id, "embeddings", time.time() - t0, success=True, items=len(inputs))
        if req.dimensions:
            result = [v[:req.dimensions] for v in result]
        return EmbeddingsResponse(
            data=[EmbeddingItem(index=i, embedding=v) for i, v in enumerate(result)],
            model=model_id,
            usage=Usage(prompt_tokens=sum(len(t.split()) for t in inputs), total_tokens=sum(len(t.split()) for t in inputs)),
        )
    except RuntimeError as e:
        record_request(model_id, "embeddings", time.time() - t0, success=False)
        raise HTTPException(503, str(e))
    except Exception as e:
        record_request(model_id, "embeddings", time.time() - t0, success=False)
        raise HTTPException(500, str(e))


# Note: /models is served by routes/models.py which now returns
# OpenAI-compatible {object: "list", data: [...]} envelope.
