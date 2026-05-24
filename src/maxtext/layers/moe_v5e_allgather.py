"""MoE forward path that runs sparse compute on TPU v5e.

Replaces the inter-chip routing step of MaxText's existing
``megablox`` / ``sparse_matmul`` paths (which require
``ragged_all_to_all`` — unsupported on v5e ICI) with a
broadcast-and-locally-filter pattern using only ``all_gather``,
``psum``, and the ``v5e_grouped_gemm`` Pallas kernel.

Flow per layer at decode time::

    x_local                                       (B, H/TP)
        │ all_gather across "expert" axis
        ▼
    x_global                                      (EP*B, H/TP)
        │ static_pad_select based on top-k routing
        ▼
    (tokens, local_expert_ids, gate_weights)      (capacity, ...)
        │ v5e_grouped_gemm × 3 with SwiGLU between
        ▼
    contrib                                       (capacity, H/TP)
        │ scatter_add to (EP*B, H/TP)
        ▼
    partial_out                                   (EP*B, H/TP)
        │ psum across "expert" axis
        ▼
    output                                        (EP*B, H/TP)
        │ dynamic_slice to local TP shard
        ▼
    return                                        (B, H/TP)

Capacity is set to ``ceil(capacity_factor * EP*B * top_k / num_experts)``.
With ``B=8 EP=8 top_k=8 num_experts=256``, default capacity_factor=1.5
gives capacity ≈ 3 — small static-padded gather.

The math here is identical to a top-k sparse MoE forward; only the
*how it's distributed across chips* differs.

Wired into ``RoutedMoE.__call__`` (src/maxtext/layers/moe.py:~2268)
when ``cfg.routed_moe_path == "v5e_allgather"``.
"""

# TODO(phase-3): full implementation.

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def v5e_allgather_moe_forward(
    cfg,
    expert_parallelism_name,
    tensor_parallelism_name,
    inputs: jnp.ndarray,
    gate_logits: jnp.ndarray,
    pre_bias_logits: Optional[jnp.ndarray],
    w0_kernel: jnp.ndarray,
    w1_kernel: jnp.ndarray,
    wo_kernel: jnp.ndarray,
    w0_bias: Optional[jnp.ndarray],
    w1_bias: Optional[jnp.ndarray],
    wo_bias: Optional[jnp.ndarray],
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
  """Run the v5e-allgather sparse MoE forward.

  Args:
    cfg: pyconfig (for top_k, num_experts, capacity_factor, ...).
    expert_parallelism_name: mesh axis to all_gather/psum across (must
      equal ``RoutedMoE._expert_parallelism_name``).
    tensor_parallelism_name: mesh axis for hidden-dim tensor-parallel
      shard (must equal ``RoutedMoE._tensor_parallelism_name``).
    inputs: (B, S, H) — token activations, TP-sharded along H.
    gate_logits: (B, S, num_experts) — router logits.
    pre_bias_logits: optional sigmoid-bias pre-route signal (MiniMax-M2).
    w0_kernel, w1_kernel: (num_experts, H, F) gate/up projections,
      EP+TP sharded.
    wo_kernel: (num_experts, F, H) down projection, EP+TP sharded.
    w0_bias, w1_bias, wo_bias: optional biases.

  Returns:
    (output, lb_loss, bias_updates) — same shape contract as
    ``RoutedMoE.sparse_matmul``.

  Raises:
    NotImplementedError: Phase 3 hasn't been written yet.
  """
  raise NotImplementedError(
      "moe_v5e_allgather: orchestration scaffolded but not yet written. "
      "See Phase 3 in /home/kathirks_gc/.claude/plans/validated-roaming-crane.md."
  )


def static_pad_select(
    x_global: jnp.ndarray,
    top_k_indices: jnp.ndarray,
    top_k_weights: jnp.ndarray,
    local_expert_range: Tuple[int, int],
    capacity: int,
):
  """Static-shape gather of (token, local-expert) pairs that this chip owns.

  For each chip, scan top_k_indices for entries falling in
  ``local_expert_range``. Output a capacity-padded list of (token,
  local_expert, weight) tuples. Pad entries are marked with a sentinel
  ``expert_assign = -1`` and ``weight = 0`` so they contribute zero
  to downstream computation.

  This is the trick that converts the ragged routing problem into a
  fixed-shape kernel call. Phase 3 will implement.
  """
  raise NotImplementedError("static_pad_select: phase 3.")
