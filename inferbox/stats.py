"""Lightweight in-memory stats tracking for InferBox."""
import time
from collections import defaultdict, deque
from threading import Lock

# Sliding window of recent latencies per (model, endpoint)
_window_size = 1000
_lock = Lock()
_counts: dict[tuple[str, str], int] = defaultdict(int)
_errors: dict[tuple[str, str], int] = defaultdict(int)
_items: dict[tuple[str, str], int] = defaultdict(int)
_latencies: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=_window_size))
_started_at = time.time()


def record_request(model: str, endpoint: str, latency_s: float, success: bool, items: int = 1):
    """Record a request's latency and outcome."""
    key = (model, endpoint)
    with _lock:
        _counts[key] += 1
        _items[key] += items
        if not success:
            _errors[key] += 1
        _latencies[key].append(latency_s)


def get_stats() -> dict:
    """Return aggregate stats for all (model, endpoint) pairs."""
    with _lock:
        result = []
        for key in _counts:
            model, endpoint = key
            lats = list(_latencies[key])
            count = _counts[key]
            errors = _errors[key]
            items = _items[key]

            if lats:
                avg_ms = round(sum(lats) / len(lats) * 1000, 1)
                lats_sorted = sorted(lats)
                p50_ms = round(lats_sorted[len(lats_sorted) // 2] * 1000, 1)
                p95_ms = round(lats_sorted[int(len(lats_sorted) * 0.95)] * 1000, 1)
            else:
                avg_ms = p50_ms = p95_ms = 0

            result.append({
                "model": model,
                "endpoint": endpoint,
                "requests": count,
                "errors": errors,
                "items": items,
                "avg_ms": avg_ms,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
            })
        return {
            "uptime_seconds": int(time.time() - _started_at),
            "endpoints": sorted(result, key=lambda x: -x["requests"]),
        }
