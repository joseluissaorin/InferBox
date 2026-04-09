"""Prometheus metrics endpoint."""
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from ..stats import get_stats
from ..server import get_manager

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Expose stats in Prometheus exposition format."""
    lines: list[str] = []
    s = get_stats()

    lines.append("# HELP inferbox_uptime_seconds Server uptime in seconds")
    lines.append("# TYPE inferbox_uptime_seconds counter")
    lines.append(f"inferbox_uptime_seconds {s['uptime_seconds']}")

    lines.append("# HELP inferbox_requests_total Total requests by model and endpoint")
    lines.append("# TYPE inferbox_requests_total counter")
    lines.append("# HELP inferbox_request_errors_total Total errors")
    lines.append("# TYPE inferbox_request_errors_total counter")
    lines.append("# HELP inferbox_items_total Total items processed (texts, docs, etc.)")
    lines.append("# TYPE inferbox_items_total counter")
    lines.append("# HELP inferbox_request_latency_ms Request latency in ms")
    lines.append("# TYPE inferbox_request_latency_ms summary")

    for ep in s["endpoints"]:
        labels = f'model="{ep["model"]}",endpoint="{ep["endpoint"]}"'
        lines.append(f'inferbox_requests_total{{{labels}}} {ep["requests"]}')
        lines.append(f'inferbox_request_errors_total{{{labels}}} {ep["errors"]}')
        lines.append(f'inferbox_items_total{{{labels}}} {ep["items"]}')
        lines.append(f'inferbox_request_latency_ms{{{labels},quantile="0.5"}} {ep["p50_ms"]}')
        lines.append(f'inferbox_request_latency_ms{{{labels},quantile="0.95"}} {ep["p95_ms"]}')

    # Model loaded gauge
    mgr = get_manager()
    if mgr:
        lines.append("# HELP inferbox_model_loaded Models currently loaded (1=loaded, 0=not)")
        lines.append("# TYPE inferbox_model_loaded gauge")
        lines.append("# HELP inferbox_vram_used_mb VRAM used by loaded models")
        lines.append("# TYPE inferbox_vram_used_mb gauge")
        for entry in mgr.status():
            loaded = 1 if entry["status"] == "loaded" else 0
            lines.append(f'inferbox_model_loaded{{model="{entry["id"]}"}} {loaded}')
        lines.append(f"inferbox_vram_used_mb {mgr.used_vram()}")

    # GPU stats
    try:
        import torch
        if torch.cuda.is_available():
            lines.append("# HELP inferbox_gpu_memory_allocated_mb GPU memory allocated by torch")
            lines.append("# TYPE inferbox_gpu_memory_allocated_mb gauge")
            lines.append(f"inferbox_gpu_memory_allocated_mb {round(torch.cuda.memory_allocated(0) / 1e6)}")
    except Exception:
        pass

    return "\n".join(lines) + "\n"
