"""Numpy ground-truth implementation of sparse top-k MoE forward.

Used by both ``v5e_grouped_gemm_test.py`` and
``moe_v5e_allgather_test.py`` to validate that the optimized Pallas
kernel + shard_map orchestration produce identical outputs.

Slow but correct. Loops over batch and top-k slots; no batching;
no sharding; no jit. CPU-runnable.

Reference MoE per token:

    for each (b, slot) in (B, K):
        e = top_k_idx[b, slot]
        gate = x[b] @ w0[e]                # (H,) @ (H, F) -> (F,)
        up   = x[b] @ w1[e]                # (H,) @ (H, F) -> (F,)
        mid  = silu(gate) * up             # (F,)
        down = mid @ wo[e]                 # (F,) @ (F, H) -> (H,)
        out[b] += top_k_w[b, slot] * down  # (H,)
"""

from __future__ import annotations

import numpy as np


def _silu(x: np.ndarray) -> np.ndarray:
  # silu(x) = x * sigmoid(x); use a numerically stable sigmoid to avoid
  # over/underflow in the bf16 → float64 conversion at test time.
  return x * (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(x.dtype)


def naive_sparse_moe(
    x: np.ndarray,
    w0: np.ndarray,
    w1: np.ndarray,
    wo: np.ndarray,
    top_k_idx: np.ndarray,
    top_k_w: np.ndarray,
) -> np.ndarray:
  """Compute the post-MoE activation for each token, by definition.

  Args:
    x:        (B, H)            — token activations
    w0:       (E, H, F)          — gate projection per expert
    w1:       (E, H, F)          — up projection per expert
    wo:       (E, F, H)          — down projection per expert
    top_k_idx: (B, K), int       — expert indices for each token
    top_k_w:   (B, K)            — gating weights (already normalized
                                   if norm_topk_prob is on)

  Returns:
    out: (B, H) — sum of contributions from the K experts each token
      selected, weighted by top_k_w.
  """
  if x.ndim != 2:
    raise ValueError(f"x must be (B, H); got shape {x.shape}")
  B, H = x.shape
  E, H_w, F = w0.shape
  if H_w != H:
    raise ValueError(f"x hidden dim {H} != w0 hidden dim {H_w}")
  if w1.shape != w0.shape:
    raise ValueError(f"w1.shape {w1.shape} != w0.shape {w0.shape}")
  if wo.shape != (E, F, H):
    raise ValueError(f"wo.shape must be (E={E}, F={F}, H={H}); got {wo.shape}")
  if top_k_idx.shape != top_k_w.shape:
    raise ValueError(f"top_k_idx {top_k_idx.shape} vs top_k_w {top_k_w.shape}")

  K = top_k_idx.shape[1]
  out = np.zeros_like(x)
  for b in range(B):
    for slot in range(K):
      e = int(top_k_idx[b, slot])
      gate = x[b] @ w0[e]
      up = x[b] @ w1[e]
      mid = _silu(gate) * up
      down = mid @ wo[e]
      out[b] += float(top_k_w[b, slot]) * down
  return out
