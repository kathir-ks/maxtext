"""Grouped GEMM for TPU v5e: per-row matmul with per-row expert selection.

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

Two implementations exposed:

  * ``v5e_grouped_gemm(..., backend="jax")`` — pure JAX baseline using
    ``jnp.take`` + ``jnp.einsum``. Always correct, always available.
    Used as the validation reference and as the runtime default on
    platforms where Pallas isn't profiled.

  * ``v5e_grouped_gemm(..., backend="pallas")`` — Pallas-backed
    implementation tuned for v5e MXU (128×128×128 bf16 tile). Same
    contract; faster on v5e once tile sizes are tuned.

Both share the same pre-permutation trick: ``w_gathered =
w[expert_assign]`` is computed at the outer JAX level (where XLA can
optimize it), so the kernel itself sees a contiguous
``(capacity, H, F)`` slice and never gathers inside the inner loop.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp


def _gemm_jax(
    x: jax.Array,
    w_gathered: jax.Array,
) -> jax.Array:
  """Reference per-row matmul. Pure JAX, no Pallas.

  Args:
    x:           (capacity, H)
    w_gathered:  (capacity, H, F)  — already gathered by expert_assign

  Returns:
    y:           (capacity, F)
  """
  # einsum keeps accumulation in the dtype of the inputs (or the
  # preferred-element-type if set). We accumulate in float32 then cast
  # back to x.dtype to dodge bf16 drift on long contracting dims.
  out_f32 = jnp.einsum(
      "ch,chf->cf",
      x.astype(jnp.float32),
      w_gathered.astype(jnp.float32),
      preferred_element_type=jnp.float32,
  )
  return out_f32.astype(x.dtype)


def _gemm_pallas(
    x: jax.Array,
    w_gathered: jax.Array,
    block_capacity: int,
    block_f: int,
    block_h: int,
) -> jax.Array:
  """Pallas-tuned per-row matmul for TPU v5e.

  Grid is ``(capacity // block_capacity, F // block_f)``. Each program
  instance computes a ``(block_capacity, block_f)`` output tile by
  iterating over the H dimension in ``block_h`` chunks.

  block_* must divide the full dims and ideally be multiples of 128
  (v5e MXU lane count). The contracting-dim block_h drives the
  accumulation loop; larger block_h = fewer kernel invocations but
  more VMEM pressure.

  Args:
    x:            (capacity, H)
    w_gathered:   (capacity, H, F)
    block_capacity: rows per program; multiple of 128.
    block_f:        output cols per program; multiple of 128.
    block_h:        contracting-dim chunk; multiple of 128.

  Returns:
    y:            (capacity, F)
  """
  # Local import so the JAX backend works on platforms without Pallas.
  from jax.experimental import pallas as pl  # pylint: disable=import-outside-toplevel
  from jax.experimental.pallas import tpu as pltpu  # pylint: disable=import-outside-toplevel  # noqa: F401

  capacity, H = x.shape
  capacity_w, H_w, F = w_gathered.shape
  if capacity != capacity_w or H != H_w:
    raise ValueError(
        f"shape mismatch: x={x.shape}, w_gathered={w_gathered.shape}")
  if capacity % block_capacity:
    raise ValueError(f"capacity={capacity} must be divisible by block_capacity={block_capacity}")
  if F % block_f:
    raise ValueError(f"F={F} must be divisible by block_f={block_f}")
  if H % block_h:
    raise ValueError(f"H={H} must be divisible by block_h={block_h}")

  def kernel(x_ref, w_ref, out_ref):
    # Accumulate in float32 inside VMEM, then store as input dtype.
    acc = jnp.zeros((block_capacity, block_f), dtype=jnp.float32)
    h_tiles = H // block_h
    for h in range(h_tiles):
      x_tile = x_ref[:, h * block_h:(h + 1) * block_h].astype(jnp.float32)
      w_tile = w_ref[:, h * block_h:(h + 1) * block_h, :].astype(jnp.float32)
      acc = acc + jnp.einsum("ch,chf->cf", x_tile, w_tile,
                              preferred_element_type=jnp.float32)
    out_ref[...] = acc.astype(out_ref.dtype)

  grid = (capacity // block_capacity, F // block_f)
  out = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[
          pl.BlockSpec((block_capacity, H),
                       lambda i, j: (i, 0)),
          pl.BlockSpec((block_capacity, H, block_f),
                       lambda i, j: (i, 0, j)),
      ],
      out_specs=pl.BlockSpec((block_capacity, block_f),
                              lambda i, j: (i, j)),
      out_shape=jax.ShapeDtypeStruct((capacity, F), x.dtype),
  )(x, w_gathered)
  return out


def v5e_grouped_gemm(
    x: jax.Array,
    w: jax.Array,
    expert_assign: jax.Array,
    *,
    backend: Literal["jax", "pallas"] = "jax",
    block_capacity: int = 128,
    block_f: int = 128,
    block_h: int = 128,
) -> jax.Array:
  """Compute ``y[i] = x[i] @ w[expert_assign[i]]`` for each row i.

  Args:
    x:             (capacity, H) — padded input tokens.
    w:             (E_local, H, F) — local experts' weight matrix.
    expert_assign: (capacity,) int — which local expert each token uses.
    backend:       "jax" (default, portable) or "pallas" (v5e-tuned).
    block_capacity, block_f, block_h: Pallas tile sizes. Ignored
        when ``backend == "jax"``.

  Returns:
    y:             (capacity, F)

  Notes:
    * The pre-permutation ``w[expert_assign]`` is performed at the JAX
      level so XLA can fuse / optimize the gather. The kernel sees a
      contiguous ``(capacity, H, F)`` slice.
    * Inner accumulation is float32 in both backends; output dtype
      matches ``x.dtype``.
  """
  if x.ndim != 2:
    raise ValueError(f"x must be 2-D (capacity, H); got shape {x.shape}")
  if w.ndim != 3:
    raise ValueError(f"w must be 3-D (E_local, H, F); got shape {w.shape}")
  if expert_assign.ndim != 1 or expert_assign.shape[0] != x.shape[0]:
    raise ValueError(
        f"expert_assign must be 1-D of length capacity={x.shape[0]}; "
        f"got shape {expert_assign.shape}")
  if x.shape[1] != w.shape[1]:
    raise ValueError(f"H mismatch: x.shape[1]={x.shape[1]}, w.shape[1]={w.shape[1]}")

  # Gather first — XLA optimizes this aggressively.
  w_gathered = jnp.take(w, expert_assign, axis=0)
  # Cast to x.dtype so downstream einsum reductions are dtype-consistent.
  w_gathered = w_gathered.astype(x.dtype)

  if backend == "jax":
    return _gemm_jax(x, w_gathered)
  if backend == "pallas":
    return _gemm_pallas(
        x, w_gathered,
        block_capacity=block_capacity,
        block_f=block_f,
        block_h=block_h,
    )
  raise ValueError(f"unknown backend {backend!r}; use 'jax' or 'pallas'")
