"""ModelManager._unload actually releases real GPU VRAM.

This is the test that would have caught the 2026-04-16 prod incident
where intelligent-search was returning 500s because InferBox's
``/v1/generate`` couldn't load qwen3-0.6b. The book-keeping said
plenty of VRAM was free; the torch caching allocator was actually
holding ~5.7 GB of "ghost" reservations from models that had been
unloaded earlier.

The key insight the old tests missed: **state-accumulation bugs need
multi-cycle tests**. A single load+unload was fine for weeks; the
accumulation only became visible after dozens of cycles in prod.
These tests exercise the cycle pattern directly, using the REAL
``ModelManager`` + real loaders + real CUDA allocator (no fakes).

Skipped if CUDA isn't available — the bug is specifically about the
torch CUDA caching allocator, so a CPU run can't reproduce it.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Make ``inferbox`` importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: E402
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(
    not _HAS_CUDA,
    reason="CUDA not available; VRAM release test has nothing meaningful to measure",
)


# ---------- fixtures ----------


@pytest.fixture(scope="module")
def manager():
    """A real ModelManager with the same registry the prod server uses.

    We read ``models.yaml`` from the repo root so the config matches
    what's running on 192.168.1.102. If the user has edited the yaml
    the test picks up the latest values automatically.
    """
    import yaml
    from inferbox.config import ModelConfig
    from inferbox.model_manager import ModelManager

    yaml_path = Path(__file__).resolve().parents[1] / "models.yaml"
    raw = yaml.safe_load(yaml_path.read_text())
    registry = {
        name: ModelConfig(**cfg) for name, cfg in raw["models"].items()
    }

    total_vram = 0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram = int(props.total_memory / 1_048_576)

    mgr = ModelManager(
        registry=registry,
        total_vram_mb=total_vram,
        idle_timeout=3600,
    )
    yield mgr
    # Cleanup: unload everything this test loaded
    loaded_ids = list(mgr.loaded.keys())
    for mid in loaded_ids:
        asyncio.run(mgr.unload(mid))


# ---------- helpers ----------


def _reserved_mb() -> int:
    """Real GPU VRAM reserved by the torch caching allocator, in MB."""
    return int(torch.cuda.memory_reserved(0) / 1_048_576)


def _pick_small_model(manager) -> str:
    """Pick a model from the registry small enough that loading is
    quick but non-trivial (so we have a measurable VRAM delta).
    We prefer text-only loaders with ≤ 2 GB vram_mb."""
    for name, cfg in manager.registry.items():
        if cfg.vram_mb <= 2500 and cfg.type in ("embedding", "reranker"):
            return name
    # Fallback to whatever is smallest
    return min(manager.registry, key=lambda n: manager.registry[n].vram_mb)


# ---------- tests ----------


class TestUnloadReleasesRealVRAM:
    """A single load+unload must return real VRAM to baseline.

    Without the post-unload ``gc.collect + torch.cuda.empty_cache``
    fix this asserts fails — ``_reserved_mb`` stays elevated by
    roughly the model's size because the caching allocator holds
    onto its blocks.
    """

    def test_single_cycle(self, manager):
        model_id = _pick_small_model(manager)
        model_vram = manager.registry[model_id].vram_mb
        # Sanity: model isn't already loaded
        assert model_id not in manager.loaded

        baseline = _reserved_mb()

        # Real load via the real ModelManager + real loader
        asyncio.run(manager.load(model_id))
        loaded_reserved = _reserved_mb()
        # Lower-bound sanity: we actually allocated something
        assert loaded_reserved > baseline, (
            f"expected VRAM to go up after loading {model_id} "
            f"(baseline={baseline} MB, after_load={loaded_reserved} MB)"
        )

        # The load delta should be at least a fraction of the
        # configured vram_mb (CUDA context + weights). We don't demand
        # the full amount because some cfgs overestimate.
        load_delta = loaded_reserved - baseline
        assert load_delta >= model_vram * 0.3, (
            f"loaded_delta={load_delta} MB suspiciously small for "
            f"a model configured as {model_vram} MB"
        )

        # The actual thing under test: unload must release real VRAM
        asyncio.run(manager.unload(model_id))
        after_unload = _reserved_mb()

        # We don't require returning exactly to baseline (CUDA context
        # itself + cudnn handles can stick around), but we DO require
        # the model's own weights to be released. Tolerance: 500 MB
        # above baseline is acceptable CUDA context overhead; more
        # than that means the allocator kept the weights reserved.
        slack_mb = 500
        assert after_unload - baseline < slack_mb, (
            f"VRAM not released after unload of {model_id}: "
            f"baseline={baseline} MB, after_unload={after_unload} MB, "
            f"slack={slack_mb} MB. Ghost VRAM is accumulating."
        )


class TestRepeatedUnloadDoesNotAccumulate:
    """Many cycles of load+unload must not drift VRAM upward.

    This is the test shape that specifically catches the
    prod incident. A single cycle was always fine in dev; only after
    hours of cycles in prod did the allocator run out.
    """

    def test_five_cycles(self, manager):
        model_id = _pick_small_model(manager)
        assert model_id not in manager.loaded

        baseline = _reserved_mb()
        peaks: list[int] = []
        post_unload_readings: list[int] = []

        N_CYCLES = 5
        for i in range(N_CYCLES):
            asyncio.run(manager.load(model_id))
            peaks.append(_reserved_mb())
            asyncio.run(manager.unload(model_id))
            post_unload_readings.append(_reserved_mb())

        # Drift check: the last post-unload reading must be within
        # slack of the first post-unload reading. If the allocator
        # were accumulating ghost VRAM per cycle, this delta would
        # grow linearly with the cycle count.
        drift = post_unload_readings[-1] - post_unload_readings[0]
        drift_tolerance_mb = 200
        assert drift < drift_tolerance_mb, (
            f"VRAM drifted over {N_CYCLES} cycles: "
            f"first_post_unload={post_unload_readings[0]} MB, "
            f"last_post_unload={post_unload_readings[-1]} MB, "
            f"drift={drift} MB (tolerance={drift_tolerance_mb} MB). "
            f"post_unload_series={post_unload_readings}"
        )

        # Also: post-unload should stay near baseline the whole time,
        # not creep up cycle-by-cycle.
        max_post_unload = max(post_unload_readings)
        assert max_post_unload - baseline < 700, (
            f"Post-unload VRAM crept to {max_post_unload} MB from "
            f"baseline {baseline} MB across {N_CYCLES} cycles. Series: "
            f"{post_unload_readings}"
        )


class TestUnloadDoesNotInvalidateInFlightRef:
    """Unload must not corrupt references held by in-flight requests.

    Regression guard for a bug I shipped myself (commit 63c9920)
    where ``_unload`` set ``entry.obj = None`` on the dataclass to
    "help GC". That mutated the object other coroutines were still
    reading, causing ``AttributeError: 'NoneType' object has no
    attribute 'tokenizer'`` on embed/rerank calls that overlapped
    an eviction.

    The real contract: the ``use()`` context manager blocks eviction
    while ``in_flight > 0``. Tests should verify that even if
    something ELSE calls unload (explicit admin endpoint, etc.)
    during an in-flight request, the dataclass attributes that the
    request is reading stay valid. We simulate this by grabbing a
    reference to ``entry.obj`` before unload and checking it's
    still the same object after.
    """

    def test_entry_obj_survives_unload(self, manager):
        """Evicting a model must not null out other holders'
        references to its dataclass attributes."""
        model_id = _pick_small_model(manager)
        assert model_id not in manager.loaded

        asyncio.run(manager.load(model_id))
        entry = manager.loaded[model_id]
        held_obj = entry.obj  # simulate an in-flight request
        assert held_obj is not None

        asyncio.run(manager.unload(model_id))

        # Dataclass attributes must NOT have been mutated out from
        # under the in-flight caller. The local ``held_obj`` is
        # still a live Python reference; the original
        # ``entry.obj`` attribute must still point at it.
        assert entry.obj is held_obj, (
            "unload mutated entry.obj = None on the dataclass, which "
            "would crash any in-flight request reading entry.obj. "
            "This is the bug class the `use()` context manager is "
            "supposed to prevent; don't reintroduce it via defensive "
            "reference-nulling."
        )


class TestPickDeviceRespectsRealVRAM:
    """``_pick_device`` must refuse to assign a device when the real
    allocator has no room, even if the configured-budget accounting
    says there's space.

    We apply pressure via an in-process torch allocation to simulate
    the prod state. A pytest-scoped fresh CUDA context is not
    guaranteed here — the host may have other GPU processes running —
    so we compute the allocation size from
    ``torch.cuda.mem_get_info(0)`` (driver-level free) rather than
    ``memory_reserved`` (PyTorch-level reserved) so the test cooperates
    with whatever else is on the card.
    """

    def test_refuses_when_real_vram_exhausted(self, manager):
        from inferbox.config import ModelConfig

        # mem_get_info returns (free_bytes, total_bytes) across ALL
        # CUDA processes — the right source of truth when the host
        # has other services on the card.
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        if free_bytes < 1 * 1024 * 1024 * 1024:
            pytest.skip(
                f"not enough real free VRAM to set up the pressure "
                f"test ({free_bytes / 1e9:.2f} GB free)"
            )

        # Leave 256 MB free in the driver's view. Any model cfg
        # bigger than that should be refused by _pick_device.
        want_to_leave = 256 * 1024 * 1024
        to_allocate = max(0, free_bytes - want_to_leave)

        big = torch.empty(to_allocate, dtype=torch.uint8, device="cuda")
        try:
            # Verify the pressure is actually applied at the driver
            # level (not just PyTorch's allocator view)
            free_after, _ = torch.cuda.mem_get_info(0)
            assert free_after < 512 * 1024 * 1024, (
                f"expected pressure test to leave <512 MB driver-level "
                f"free, got {free_after / 1e6:.0f} MB"
            )

            # Under pressure, the real allocator is nearly full. Our
            # in-process ``torch.cuda.memory_reserved`` reflects this
            # because we just allocated via torch. A cfg for a 1500 MB
            # model should be refused: configured budget might say
            # "fine" but real allocator says no.
            cfg = ModelConfig(
                type="generate",
                loader="gguf",
                model_id="synthetic-test-model",
                vram_mb=1500,
            )

            # Without the fix: _pick_device returns cuda:0 (budget OK)
            # → subsequent load OOMs silently at Llama() or .to(cuda).
            # With the fix: _pick_device sees real reserved near total
            # and returns None → caller evicts or raises cleanly.
            device = manager._pick_device(cfg)
            assert device is None, (
                f"_pick_device returned {device!r} while real VRAM "
                f"was under pressure — it should refuse. This is the "
                f"regression that caused the 2026-04-16 prod incident."
            )
        finally:
            del big
            torch.cuda.empty_cache()

    def test_allows_when_real_vram_has_room(self, manager):
        """Sanity-check the pressure path: without any pressure,
        _pick_device should happily assign a device for a small
        model cfg — otherwise the pressure-test assertion isn't
        distinguishing signal from noise."""
        from inferbox.config import ModelConfig

        torch.cuda.empty_cache()
        cfg = ModelConfig(
            type="generate",
            loader="gguf",
            model_id="synthetic-test-model",
            vram_mb=500,
        )
        device = manager._pick_device(cfg)
        assert device is not None
        assert device.startswith("cuda")
