import asyncio
import time
import logging
import importlib
from dataclasses import dataclass, field
from typing import Any

from .config import ModelConfig

logger = logging.getLogger("inferbox")


def _detect_devices() -> dict[str, int]:
    """Detect available CUDA devices and their VRAM in MB.
    Returns dict mapping 'cuda:N' to VRAM MB. Always includes 'cpu' with 0 MB.
    """
    devices: dict[str, int] = {"cpu": 0}
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices[f"cuda:{i}"] = round(props.total_memory / 1e6)
    except Exception:
        pass
    return devices


@dataclass
class LoadedModel:
    obj: Any
    config: ModelConfig
    loader_module: Any
    device: str
    last_used: float = field(default_factory=time.time)
    # Number of currently-running requests holding a reference to this
    # model. The idle checker and eviction logic MUST NOT unload a model
    # with in_flight > 0 — doing so races against live inference and
    # causes AttributeError crashes in loaders that mutate the object.
    in_flight: int = 0
    # Per-model inference lock. Most underlying model objects — notably
    # llama-cpp-python's `Llama` and HF `AutoModelForCausalLM` — are NOT
    # safe to call concurrently from multiple threads: the KV cache /
    # generation state is shared mutable state and concurrent access
    # causes IndexError, SIGSEGV, or silent garbage output. InferBox's
    # generate route uses `asyncio.to_thread`, so two in-flight requests
    # land on two thread-pool threads hitting the same Llama. The lock
    # serialises them. Embed is fine to parallelise at the batch layer.
    inference_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ModelManager:
    def __init__(self, registry: dict[str, ModelConfig], total_vram_mb: int, idle_timeout: int, pinned: set[str] | None = None):
        self.registry = registry
        self.idle_timeout = idle_timeout
        self.pinned: set[str] = set(pinned or ())
        self.loaded: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None

        # Per-device VRAM budgets (multi-GPU aware)
        self.devices = _detect_devices()
        if not any(d.startswith("cuda") for d in self.devices):
            # No GPU - use the configured total
            self.devices["cpu"] = total_vram_mb
        logger.info(f"Detected devices: {self.devices}")

        # Build default_for lookup
        self.defaults: dict[str, str] = {}
        for name, cfg in registry.items():
            if cfg.default_for and cfg.default_for not in self.defaults:
                self.defaults[cfg.default_for] = name

    @property
    def total_vram_mb(self) -> int:
        return sum(v for k, v in self.devices.items() if k != "cpu")

    def start_idle_checker(self):
        self._idle_task = asyncio.create_task(self._idle_checker())

    def stop_idle_checker(self):
        if self._idle_task:
            self._idle_task.cancel()

    def resolve_model(self, model_id: str | None, task_type: str) -> str:
        if model_id:
            if model_id not in self.registry:
                raise KeyError(f"Unknown model: {model_id}")
            return model_id
        if task_type in self.defaults:
            return self.defaults[task_type]
        raise KeyError(f"No default model for task type: {task_type}")

    def used_vram(self, device: str | None = None) -> int:
        if device is None:
            return sum(self.loaded[m].config.vram_mb for m in self.loaded if self.loaded[m].device != "cpu")
        return sum(self.loaded[m].config.vram_mb for m in self.loaded if self.loaded[m].device == device)

    def status(self) -> list[dict]:
        result = []
        now = time.time()
        for name, cfg in self.registry.items():
            entry = {
                "id": name,
                "type": cfg.type,
                "model_id": cfg.model_id,
                "vram_mb": cfg.vram_mb,
                "status": "loaded" if name in self.loaded else "unloaded",
            }
            if name in self.loaded:
                entry["idle_seconds"] = int(now - self.loaded[name].last_used)
                entry["device"] = self.loaded[name].device
            result.append(entry)
        return result

    def device_status(self) -> list[dict]:
        result = []
        for dev, total in self.devices.items():
            used = self.used_vram(dev)
            result.append({
                "device": dev,
                "vram_total_mb": total,
                "vram_used_mb": used,
                "loaded_models": [m for m in self.loaded if self.loaded[m].device == dev],
            })
        return result

    def _real_free_vram_mb(self, device: str) -> int | None:
        """Real free VRAM on ``device`` in MB, or None if the device
        isn't a CUDA device we can query.

        Uses ``torch.cuda.mem_get_info`` (NVML) for driver-level free
        bytes across all processes on the GPU. That's the right signal
        for BOTH failure modes:

          1. THIS process has "ghost" torch-allocator reservations
             from previously-unloaded models (the 2026-04-16 prod
             incident — inside a single long-running uvicorn).
          2. OTHER processes on the card are holding memory our
             internal ``memory_reserved`` won't see (co-hosted GPU
             workloads, GPU benchmarks, pytest runs).

        The previous implementation used ``memory_reserved`` (this-
        process only) which reported the wrong number when another
        process was occupying the card.
        """
        if not device.startswith("cuda"):
            return None
        try:
            import torch
            idx = int(device.split(":")[1]) if ":" in device else 0
            if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
                return None
            free_bytes, _total_bytes = torch.cuda.mem_get_info(idx)
            return max(0, int(free_bytes / 1_048_576))
        except Exception:
            return None

    def _pick_device(self, cfg: ModelConfig) -> str | None:
        """Pick the best device for a model.
        - If cfg.device is set, use it
        - Otherwise, pick the GPU with most free VRAM (or CPU as fallback)
        """
        if cfg.device:
            return cfg.device
        # Find GPU with most free space. We combine TWO signals:
        #   1. Configured-budget free space (total - sum(config.vram_mb))
        #   2. Real allocator free space (total - torch.cuda.memory_reserved)
        # and require BOTH to have headroom. Without #2, ghost VRAM
        # from previously-unloaded models makes the manager think
        # there's room when there isn't, and the actual load OOMs.
        best_device = None
        best_free = -1
        for dev, total in self.devices.items():
            if dev == "cpu":
                continue
            free_budget = total - self.used_vram(dev)
            free_real = self._real_free_vram_mb(dev)
            # Use the more conservative of the two
            free = min(free_budget, free_real) if free_real is not None else free_budget
            if free >= cfg.vram_mb and free > best_free:
                best_free = free
                best_device = dev
        return best_device

    async def get(self, model_id: str) -> LoadedModel:
        """Fetch a loaded model, loading it on demand.

        WARNING: using the returned entry directly is UNSAFE — if inference
        takes longer than `idle_timeout` seconds, the idle checker can
        evict the model out from under you. Prefer the `use()` async
        context manager, which increments an in-flight counter to block
        eviction while you hold the reference.
        """
        async with self._lock:
            if model_id in self.loaded:
                self.loaded[model_id].last_used = time.time()
                return self.loaded[model_id]

            cfg = self.registry[model_id]

            # Pick device or evict to make room
            device = self._pick_device(cfg)
            while device is None:
                evicted = self._find_eviction_candidate()
                if evicted is None:
                    raise RuntimeError(
                        f"Cannot load {model_id} ({cfg.vram_mb}MB): no device with enough free VRAM"
                    )
                await self._unload(evicted)
                device = self._pick_device(cfg)

            # Override config device if not explicitly set
            effective_cfg = cfg.model_copy(update={"device": device}) if not cfg.device else cfg

            logger.info(f"Loading model {model_id} ({cfg.model_id}) on {device}...")
            loader = self._get_loader(cfg.loader)
            obj = await asyncio.to_thread(loader.load, effective_cfg)
            self.loaded[model_id] = LoadedModel(
                obj=obj, config=cfg, loader_module=loader, device=device,
            )
            logger.info(f"Model {model_id} loaded on {device}. Device VRAM used: {self.used_vram(device)}MB")
            return self.loaded[model_id]

    def use(self, model_id: str, serialize: bool = False):
        """Async context manager: acquire a model for the duration of a request.

        Increments the `in_flight` counter on entry and decrements on exit.
        The idle checker and eviction logic skip any model with in_flight > 0,
        guaranteeing the loaded object stays alive for as long as the caller
        holds the reference — no matter how long inference takes.

        If `serialize=True`, also acquires the model's per-model inference
        lock, serialising concurrent callers. This is REQUIRED for loaders
        whose underlying object is not thread-safe (GGUF/llama-cpp, HF
        causal LMs with KV cache). Embed/rerank loaders can leave it False.

        Usage:
            async with manager.use("qwen3-0.6b", serialize=True) as entry:
                result = await asyncio.to_thread(
                    entry.loader_module.generate, entry.obj, ...
                )
        """
        mgr = self

        class _Ctx:
            def __init__(self):
                self.entry: LoadedModel | None = None
                self._lock_held = False

            async def __aenter__(self):
                self.entry = await mgr.get(model_id)
                # Mark in-flight under the lock so eviction sees it immediately
                async with mgr._lock:
                    self.entry.in_flight += 1
                    self.entry.last_used = time.time()
                if serialize:
                    await self.entry.inference_lock.acquire()
                    self._lock_held = True
                return self.entry

            async def __aexit__(self, exc_type, exc, tb):
                if self._lock_held and self.entry is not None:
                    self.entry.inference_lock.release()
                    self._lock_held = False
                if self.entry is not None:
                    async with mgr._lock:
                        self.entry.in_flight = max(0, self.entry.in_flight - 1)
                        self.entry.last_used = time.time()
                return False

        return _Ctx()

    async def load(self, model_id: str) -> None:
        await self.get(model_id)

    async def unload(self, model_id: str) -> None:
        async with self._lock:
            await self._unload(model_id)

    async def _unload(self, model_id: str) -> None:
        if model_id not in self.loaded:
            return
        entry = self.loaded.pop(model_id)
        logger.info(f"Unloading model {model_id}...")
        try:
            await asyncio.to_thread(entry.loader_module.unload, entry.obj)
        except Exception as e:
            logger.warning(f"Error unloading {model_id}: {e}")

        # Drop our local strong ref and nudge the GC + torch
        # caching allocator to actually release the VRAM. Without
        # this, torch.cuda.memory_reserved stays elevated for GB
        # after every unload and subsequent lazy loads OOM (prod
        # incident 2026-04-16).
        #
        # Importantly: do NOT set ``entry.obj = None`` on the
        # dataclass instance. Another in-flight request can be
        # holding a reference to this LoadedModel (via
        # mgr.use()) and will crash with ``AttributeError:
        # 'NoneType' object has no attribute '...'`` on its next
        # attribute access. We want the GC to collect the loader
        # object only once the last reference is dropped —
        # Python refcount takes care of it as soon as the last
        # caller exits its ``use()`` context.
        del entry

        def _flush():
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # reset_peak_memory_stats is cheap and makes the next
                # memory_reserved() read reflect current state rather
                # than historic peak.
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

        try:
            await asyncio.to_thread(_flush)
        except Exception as e:
            logger.warning(f"Post-unload VRAM flush failed for {model_id}: {e}")

        logger.info(f"Model {model_id} unloaded.")

    def _find_eviction_candidate(self) -> str | None:
        """Pick the oldest idle model that is NOT currently serving a request
        and NOT pinned (preload-pinned models are kept resident even under
        VRAM pressure — callers will get a load failure instead).
        """
        oldest_name = None
        oldest_time = float("inf")
        for name, entry in self.loaded.items():
            if entry.in_flight > 0:
                continue  # Never evict an in-use model
            if name in self.pinned:
                continue  # Pinned models survive VRAM pressure
            if entry.last_used < oldest_time:
                oldest_time = entry.last_used
                oldest_name = name
        return oldest_name

    def _get_loader(self, loader_name: str):
        module = importlib.import_module(f".loaders.{loader_name}", package="inferbox")
        return module

    async def _idle_checker(self):
        while True:
            await asyncio.sleep(60)
            now = time.time()
            async with self._lock:
                to_unload = [
                    mid for mid, entry in self.loaded.items()
                    if entry.in_flight == 0
                    and mid not in self.pinned
                    and now - entry.last_used > self.idle_timeout
                ]
                for mid in to_unload:
                    await self._unload(mid)
