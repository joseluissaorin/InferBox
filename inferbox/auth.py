"""Multi-key auth, rate limiting, and audit logging."""
import json
import time
from collections import defaultdict, deque
from threading import Lock
from pathlib import Path

from .config import settings


class _KeyRegistry:
    """Maps API keys to labels (for audit log + per-key rate limits)."""

    def __init__(self):
        self.keys: dict[str, str] = {}
        # Primary key
        if settings.api_key:
            self.keys[settings.api_key] = "primary"
        # Additional keys: "key1:label1,key2:label2"
        if settings.api_keys:
            for entry in settings.api_keys.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if ":" in entry:
                    k, label = entry.split(":", 1)
                    self.keys[k.strip()] = label.strip()
                else:
                    self.keys[entry] = entry[:8]

    def label_for(self, key: str) -> str | None:
        return self.keys.get(key)

    def is_valid(self, key: str) -> bool:
        return key in self.keys


registry = _KeyRegistry()


class _RateLimiter:
    """Sliding window rate limiter per key."""

    def __init__(self, requests_per_minute: int = 600):
        self.limit = requests_per_minute
        self.window_s = 60.0
        self._buckets: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> bool:
        """Returns True if request is allowed, False if rate-limited."""
        now = time.time()
        with self._lock:
            bucket = self._buckets[key]
            # Drop entries older than window
            cutoff = now - self.window_s
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True

    def remaining(self, key: str) -> int:
        with self._lock:
            return self.limit - len(self._buckets.get(key, []))


limiter = _RateLimiter(settings.rate_limit_per_min)


class _AuditLog:
    """Append-only JSONL audit log."""

    def __init__(self, path: str):
        self.path: Path | None = Path(path) if path else None
        self._lock = Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, key_label: str, endpoint: str, status: int, latency_ms: int, extra: dict | None = None):
        if not self.path:
            return
        entry = {
            "ts": time.time(),
            "key": key_label,
            "endpoint": endpoint,
            "status": status,
            "latency_ms": latency_ms,
        }
        if extra:
            entry.update(extra)
        try:
            with self._lock:
                with open(self.path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


audit = _AuditLog(settings.audit_log)
