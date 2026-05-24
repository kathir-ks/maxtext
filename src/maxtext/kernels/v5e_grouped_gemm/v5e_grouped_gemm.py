"""Hand-tuned Pallas grouped-GEMM kernel for TPU v5e.

The MoE forward needs to multiply each token's activation through the
weight matrix of its selected expert. Across a batch the assignment is
irregular ("ragged"). v5e's ICI does not support ``ragged_all_to_all``,
which is what MaxText's existing ``megablox`` / ``sparse_matmul`` paths
use to ship tokens to the chips owning their experts.

This kernel sidesteps the issue by accepting an already-padded
``(capacity, ...)``-shaped tensor where every row is a distinct
(token, expert) pair. The orchestrator (``moe_v5e_allgather.py``)
builds that padded tensor via ``all_gather`` + static-shape masking
— operations that v5e *does* support.

Inside the kernel, we further pre-permute weights on the calling side
so the inner loop is a contiguous matmul (no in-kernel gather, which
is slow on v5e's smaller VMEM). Inner accumulation is float32 to
prevent bf16 drift; output is cast at the end.

Tile sizes default to multiples of 128 to match v5e MXU's native
128x128 bf16 tile. They should be tuned per (H, F) at Phase 2 time.
"""

# TODO(phase-2): import jax.experimental.pallas as pl, jax.numpy as jnp
# TODO(phase-2): write the actual kernel + driver here.

import jax.numpy as jnp


def v5e_grouped_gemm(
    x: jnp.ndarray,
    w: jnp.ndarray,
    expert_assign: jnp.ndarray,
    *,
    block_capacity: int = 128,
    block_f: int = 128,
    block_h: int = 128,
) -> jnp.ndarray:
  """Compute y[i] = x[i] @ w[expert_assign[i]] for each row i.

  Args:
    x: (capacity, H) — padded input tokens.
    w: (E_local, H, F) — local experts' weight matrix.
    expert_assign: (capacity,) — which local expert each token uses.
    block_capacity: row-block size; must divide capacity and be a
      multiple of 128.
    block_f: output-col-block size; must divide F and be a multiple of 128.
    block_h: contracting-dim block size; must divide H and be a multiple
      of 128.

  Returns:
    y: (capacity, F).

  Notes:
    - Inner accumulator is float32, downcast to ``x.dtype`` on store.
    - Pre-permute weights via ``w[expert_assign]`` on the caller side
      so the kernel sees a contiguous ``(capacity, H, F)`` slice.
    - Tile defaults are conservative; tune via ``jax.profiler`` traces.
  """
  raise NotImplementedError(
      "v5e_grouped_gemm: Pallas kernel is scaffolded but not yet written. "
      "See Phase 2 in /home/kathirks_gc/.claude/plans/validated-roaming-crane.md."
  )
