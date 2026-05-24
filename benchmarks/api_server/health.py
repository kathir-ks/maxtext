"""Health, readiness, and metrics endpoints for the MaxText API server.

These are mounted UNDER the same FastAPI app but without the bearer-token
dependency: probes (IAP, systemd ExecStartPost, monitoring) need to reach
them without credentials. They never reveal request bodies, tokens, or
prompts.

`ready_event` is a module-level threading.Event the server sets once
`MaxTextGenerator` construction completes. /ready returns 503 until then
and 200 after.

`metrics_state` is a small module-level dict updated from the request /
batching loop in maxtext_server.py. We avoid pulling in a metrics
library (prometheus_client etc.) to keep the dependency surface small.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any

from fastapi import APIRouter, Response, status


ready_event = threading.Event()

_metrics_lock = threading.Lock()
_metrics_state: dict[str, Any] = {
    "started_at": time.time(),
    "completions_total": 0,
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    "in_flight": 0,
    "queue_depth": 0,
    "last_batch_size": 0,
    "last_batch_finished_at": 0.0,
}
_recent_throughput: collections.deque[tuple[float, int]] = collections.deque(maxlen=2048)


def mark_ready() -> None:
  ready_event.set()


def record_queued() -> None:
  with _metrics_lock:
    _metrics_state["queue_depth"] += 1


def record_dequeued() -> None:
  with _metrics_lock:
    _metrics_state["queue_depth"] = max(0, _metrics_state["queue_depth"] - 1)
    _metrics_state["in_flight"] += 1


def record_batch_complete(num_prompts: int, prompt_tokens: int, completion_tokens: int) -> None:
  now = time.time()
  with _metrics_lock:
    _metrics_state["in_flight"] = max(0, _metrics_state["in_flight"] - num_prompts)
    _metrics_state["completions_total"] += num_prompts
    _metrics_state["prompt_tokens_total"] += prompt_tokens
    _metrics_state["completion_tokens_total"] += completion_tokens
    _metrics_state["last_batch_size"] = num_prompts
    _metrics_state["last_batch_finished_at"] = now
  _recent_throughput.append((now, completion_tokens))


def _avg_tok_per_s(window_s: float = 60.0) -> float:
  cutoff = time.time() - window_s
  total = 0
  earliest = None
  for ts, toks in list(_recent_throughput):
    if ts < cutoff:
      continue
    if earliest is None:
      earliest = ts
    total += toks
  if earliest is None:
    return 0.0
  span = max(1e-3, time.time() - earliest)
  return total / span


def _hbm_per_chip() -> list[dict[str, Any]]:
  try:
    import jax  # noqa: PLC0415
  except ImportError:
    return []
  out: list[dict[str, Any]] = []
  for d in jax.local_devices():
    if not hasattr(d, "memory_stats"):
      continue
    stats = d.memory_stats() or {}
    out.append({
        "id": d.id,
        "used_gb": round(stats.get("bytes_in_use", 0) / 1e9, 3),
        "peak_gb": round(stats.get("peak_bytes_in_use", 0) / 1e9, 3),
        "limit_gb": round(stats.get("bytes_limit", 0) / 1e9, 3),
    })
  return out


router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
  return {"status": "ok"}


@router.get("/ready")
def ready(response: Response) -> dict[str, Any]:
  if not ready_event.is_set():
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "loading", "ready": False}
  return {"status": "ok", "ready": True}


@router.get("/metrics")
def metrics() -> dict[str, Any]:
  with _metrics_lock:
    snap = dict(_metrics_state)
  snap["ready"] = ready_event.is_set()
  snap["uptime_s"] = round(time.time() - snap.pop("started_at"), 1)
  snap["avg_tok_per_s_60s"] = round(_avg_tok_per_s(60.0), 2)
  snap["hbm_per_chip"] = _hbm_per_chip()
  return snap
