"""Stateful chat sessions.

Each session keeps a message history server-side. The KV cache is
implicitly reused by llama-cpp via prompt caching when the prefix
matches a previous turn (managed in the gguf loader).
"""
import asyncio
import time
import uuid
from threading import Lock
from typing import Literal
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from ..server import get_manager
from ..stats import record_request

router = APIRouter(tags=["sessions"])


# In-memory session store
class _SessionStore:
    def __init__(self, max_sessions: int = 1000, ttl_s: int = 3600):
        self.max_sessions = max_sessions
        self.ttl_s = ttl_s
        self._sessions: dict[str, dict] = {}
        self._lock = Lock()

    def create(self, model: str, system: str | None = None) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._gc_expired()
            self._sessions[sid] = {
                "id": sid,
                "model": model,
                "messages": [{"role": "system", "content": system}] if system else [],
                "created": time.time(),
                "last_used": time.time(),
                "turn_count": 0,
            }
            if len(self._sessions) > self.max_sessions:
                # Evict oldest
                oldest = min(self._sessions.values(), key=lambda s: s["last_used"])
                del self._sessions[oldest["id"]]
        return sid

    def get(self, sid: str) -> dict | None:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess:
                sess["last_used"] = time.time()
            return sess

    def append(self, sid: str, role: str, content: str):
        with self._lock:
            if sid in self._sessions:
                self._sessions[sid]["messages"].append({"role": role, "content": content})
                self._sessions[sid]["last_used"] = time.time()
                self._sessions[sid]["turn_count"] += 1

    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._sessions.pop(sid, None) is not None

    def list_all(self) -> list[dict]:
        with self._lock:
            self._gc_expired()
            return [
                {
                    "id": s["id"],
                    "model": s["model"],
                    "turns": s["turn_count"],
                    "created": s["created"],
                    "last_used": s["last_used"],
                }
                for s in self._sessions.values()
            ]

    def _gc_expired(self):
        now = time.time()
        expired = [k for k, v in self._sessions.items() if now - v["last_used"] > self.ttl_s]
        for k in expired:
            del self._sessions[k]


store = _SessionStore()


class CreateSessionRequest(BaseModel):
    model: str | None = None
    system: str | None = None


class SessionTurnRequest(BaseModel):
    content: str
    max_tokens: int = 512
    temperature: float = 0.7
    stop: list[str] | None = None


class SessionResponse(BaseModel):
    id: str
    model: str
    turns: int
    created: float
    last_used: float


@router.post("/sessions", response_model=SessionResponse)
async def create_session(req: CreateSessionRequest):
    mgr = get_manager()
    try:
        model_id = mgr.resolve_model(req.model, "generate")
    except KeyError as e:
        raise HTTPException(400, str(e))
    sid = store.create(model_id, req.system)
    sess = store.get(sid)
    return SessionResponse(
        id=sid, model=model_id, turns=0,
        created=sess["created"], last_used=sess["last_used"],
    )


@router.get("/sessions")
async def list_sessions():
    return {"sessions": store.list_all()}


@router.get("/sessions/{sid}")
async def get_session(sid: str):
    sess = store.get(sid)
    if not sess:
        raise HTTPException(404, "Session not found")
    return sess


@router.delete("/sessions/{sid}")
async def delete_session(sid: str):
    if not store.delete(sid):
        raise HTTPException(404, "Session not found")
    return {"status": "deleted"}


@router.post("/sessions/{sid}/turn")
async def session_turn(sid: str, req: SessionTurnRequest):
    sess = store.get(sid)
    if not sess:
        raise HTTPException(404, "Session not found")

    mgr = get_manager()
    model_id = sess["model"]

    store.append(sid, "user", req.content)
    sess = store.get(sid)

    t0 = time.time()
    try:
        async with mgr.use(model_id, serialize=True) as entry:
            result = await asyncio.to_thread(
                entry.loader_module.generate,
                entry.obj, entry.config,
                prompt=None, messages=sess["messages"],
                max_tokens=req.max_tokens, temperature=req.temperature,
                stop=req.stop,
            )
        store.append(sid, "assistant", result["text"])
        record_request(model_id, "session.turn", time.time() - t0, success=True)
        return {
            "session_id": sid,
            "model": model_id,
            "response": result["text"],
            "turns": store.get(sid)["turn_count"],
        }
    except RuntimeError as e:
        record_request(model_id, "session.turn", time.time() - t0, success=False)
        raise HTTPException(503, str(e))
    except Exception as e:
        record_request(model_id, "session.turn", time.time() - t0, success=False)
        raise HTTPException(500, str(e))
