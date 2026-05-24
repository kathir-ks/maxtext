"""Unit tests for ``v5e_grouped_gemm`` Pallas kernel.

Run on a single TPU chip (or CPU stub once Pallas-CPU lands) with a
small ``(capacity, H, F)``. Validates against numpy by replaying the
per-(token, expert) matmul.
"""

import numpy as np
import pytest


def _per_token_reference(x_np, w_np, expert_assign_np):
  """y[i] = x[i] @ w[expert_assign[i]]"""
  out = np.zeros((x_np.shape[0], w_np.shape[2]), dtype=x_np.dtype)
  for i, e in enumerate(expert_assign_np):
    out[i] = x_np[i] @ w_np[int(e)]
  return out


@pytest.mark.skip(reason="Phase 2: kernel not yet implemented")
def test_v5e_grouped_gemm_matches_reference_small():
  """Validates kernel output against numpy reference at capacity=128, H=384, F=192.

  These dims match our v5e per-chip slice at TP=8 EP=8: H=3072/8=384,
  F=1536/8=192. Once the Pallas kernel is wired up this should pass.
  """
  pytest.importorskip("jax")
  import jax
  import jax.numpy as jnp
  from maxtext.kernels.v5e_grouped_gemm import v5e_grouped_gemm

  capacity, E, H, F = 128, 4, 384, 192
  rng = np.random.default_rng(0)
  x_np = rng.standard_normal((capacity, H)).astype(np.float32)
  w_np = rng.standard_normal((E, H, F)).astype(np.float32) * 0.05
  expert_assign_np = rng.integers(0, E, size=(capacity,)).astype(np.int32)

  expected = _per_token_reference(x_np, w_np, expert_assign_np)
  got = v5e_grouped_gemm(jnp.asarray(x_np), jnp.asarray(w_np),
                         jnp.asarray(expert_assign_np))
  np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skip(reason="Phase 2: tile-size tuning needs a v5e chip")
def test_v5e_grouped_gemm_mxu_utilization():
  """Profile the kernel and assert MXU utilization ≥ 80%.

  Run via:
      JAX_PLATFORMS=tpu pytest -k mxu_utilization -s
  Requires a v5e chip and ``jax.profiler``. Phase 2 will fill this in.
  """
  raise NotImplementedError("Phase 2: profiling harness pending")
