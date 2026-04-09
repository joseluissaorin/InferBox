"""Micro-batching coordinator.

Coalesces concurrent requests for the same model into single forward
passes. Each model gets one batcher per inference type (embed, rerank).
A background coroutine pulls items from the queue, accumulates them
until either the time window or max batch size, then runs them in one
shot and dispatches results to waiting callers.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .config import settings

logger = logging.getLogger("inferbox")


@dataclass
class _PendingItem:
    payload: Any
    future: asyncio.Future


@dataclass
class _Batcher:
    name: str
    runner: Callable[[list[Any]], Awaitable[list[Any]]]
    window_ms: int
    max_size: int
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None

    async def submit(self, payload: Any) -> Any:
        """Submit a single item, await its result."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        await self.queue.put(_PendingItem(payload=payload, future=fut))
        return await fut

    async def _loop(self):
        while True:
            try:
                first = await self.queue.get()
                items: list[_PendingItem] = [first]
                deadline = time.time() + self.window_ms / 1000.0

                # Drain queue up to max_size or window expiry
                while len(items) < self.max_size:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                        items.append(item)
                    except asyncio.TimeoutError:
                        break

                # Execute the batch
                payloads = [it.payload for it in items]
                try:
                    results = await self.runner(payloads)
                    if len(results) != len(items):
                        raise RuntimeError(f"Batcher result length mismatch: {len(results)} vs {len(items)}")
                    for it, res in zip(items, results):
                        if not it.future.done():
                            it.future.set_result(res)
                except Exception as e:
                    for it in items:
                        if not it.future.done():
                            it.future.set_exception(e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Batcher {self.name} loop error: {e}")
                await asyncio.sleep(0.1)

    def start(self):
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._loop())

    def stop(self):
        if self.task:
            self.task.cancel()


class BatcherRegistry:
    """Holds one batcher per (model_id, op) pair."""

    def __init__(self):
        self._batchers: dict[tuple[str, str], _Batcher] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        model_id: str,
        op: str,
        runner: Callable[[list[Any]], Awaitable[list[Any]]],
    ) -> _Batcher:
        key = (model_id, op)
        async with self._lock:
            if key not in self._batchers:
                b = _Batcher(
                    name=f"{model_id}/{op}",
                    runner=runner,
                    window_ms=settings.micro_batch_window_ms,
                    max_size=settings.micro_batch_max_size,
                )
                b.start()
                self._batchers[key] = b
            return self._batchers[key]

    def stop_all(self):
        for b in self._batchers.values():
            b.stop()
        self._batchers.clear()


registry = BatcherRegistry()
