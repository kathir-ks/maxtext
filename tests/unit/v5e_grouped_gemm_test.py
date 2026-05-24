"""Unit tests for ``v5e_grouped_gemm``.

The JAX backend is portable (CPU/TPU) and runs in CI on the dev VM.
The Pallas backend is v5e-only and validated via the v5e profile
harness in Phase 2b.
"""

import numpy as np
import pytest


def _per_token_reference(x_np, w_np, expert_assign_np):
  """y[i] = x[i] @ w[expert_assign[i]]"""
  out = np.zeros((x_np.shape[0], w_np.shape[2]), dtype=x_np.dtype)
  for i, e in enumerate(expert_assign_np):
    out[i] = x_np[i] @ w_np[int(e)]
  return out


def test_jax_backend_matches_reference_small():
  """JAX backend, small case: capacity=8, E=4, H=16, F=8."""
  pytest.importorskip("jax")
  import jax.numpy as jnp
  from maxtext.kernels.v5e_grouped_gemm import v5e_grouped_gemm

  capacity, E, H, F = 8, 4, 16, 8
  rng = np.random.default_rng(0)
  x_np = rng.standard_normal((capacity, H)).astype(np.float32)
  w_np = rng.standard_normal((E, H, F)).astype(np.float32) * 0.05
  expert_assign_np = rng.integers(0, E, size=(capacity,)).astype(np.int32)

  expected = _per_token_reference(x_np, w_np, expert_assign_np)
  got = v5e_grouped_gemm(
      jnp.asarray(x_np), jnp.asarray(w_np),
      jnp.asarray(expert_assign_np), backend="jax")
  np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)


def test_jax_backend_matches_reference_v5e_shape():
  """JAX backend at the actual v5e per-chip dims (capacity=128, H=384, F=192).

  These match TP=8 EP=8 on v5e-64: H=3072/8=384, F=1536/8=192.
  Phase 2 correctness gate.
  """
  pytest.importorskip("jax")
  import jax.numpy as jnp
  from maxtext.kernels.v5e_grouped_gemm import v5e_grouped_gemm

  capacity, E, H, F = 128, 4, 384, 192
  rng = np.random.default_rng(0)
  x_np = rng.standard_normal((capacity, H)).astype(np.float32) * 0.1
  w_np = rng.standard_normal((E, H, F)).astype(np.float32) * 0.05
  expert_assign_np = rng.integers(0, E, size=(capacity,)).astype(np.int32)

  expected = _per_token_reference(x_np, w_np, expert_assign_np)
  got = v5e_grouped_gemm(
      jnp.asarray(x_np), jnp.asarray(w_np),
      jnp.asarray(expert_assign_np), backend="jax")
  # JAX einsum on CPU has different accumulation order than numpy `@`,
  # and on TPU may use reduced-precision matmul. ~1e-3 absolute drift
  # over a 384-element contracting dim is expected.
  np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-2, atol=2e-3)


def test_jax_backend_bf16_drift_bounded():
  """bf16 storage + fp32 accumulation: per-element drift stays small."""
  pytest.importorskip("jax")
  import jax.numpy as jnp
  from maxtext.kernels.v5e_grouped_gemm import v5e_grouped_gemm

  capacity, E, H, F = 8, 4, 128, 64
  rng = np.random.default_rng(1)
  x_np = (rng.standard_normal((capacity, H)) * 0.1).astype(np.float32)
  w_np = (rng.standard_normal((E, H, F)) * 0.05).astype(np.float32)
  expert_assign_np = rng.integers(0, E, size=(capacity,)).astype(np.int32)

  expected = _per_token_reference(x_np, w_np, expert_assign_np)
  got = v5e_grouped_gemm(
      jnp.asarray(x_np).astype(jnp.bfloat16),
      jnp.asarray(w_np).astype(jnp.bfloat16),
      jnp.asarray(expert_assign_np),
      backend="jax")
  np.testing.assert_allclose(
      np.asarray(got, dtype=np.float32), expected, rtol=2e-2, atol=2e-2)


def test_input_shape_validation():
  """Catch caller errors at the public API boundary."""
  pytest.importorskip("jax")
  import jax.numpy as jnp
  from maxtext.kernels.v5e_grouped_gemm import v5e_grouped_gemm

  rng = np.random.default_rng(0)
  x = jnp.asarray(rng.standard_normal((4, 8)).astype(np.float32))
  w = jnp.asarray(rng.standard_normal((2, 8, 4)).astype(np.float32))
  e = jnp.asarray(rng.integers(0, 2, size=(4,)).astype(np.int32))

  with pytest.raises(ValueError, match="x must be 2-D"):
    v5e_grouped_gemm(jnp.asarray([1.0, 2.0]), w, e, backend="jax")
  with pytest.raises(ValueError, match="w must be 3-D"):
    v5e_grouped_gemm(x, jnp.zeros((2, 8)), e, backend="jax")
  with pytest.raises(ValueError, match="expert_assign must be 1-D"):
    v5e_grouped_gemm(x, w, jnp.zeros((3,), dtype=jnp.int32), backend="jax")
  with pytest.raises(ValueError, match="H mismatch"):
    v5e_grouped_gemm(x, jnp.zeros((2, 16, 4)), e, backend="jax")
  with pytest.raises(ValueError, match="unknown backend"):
    v5e_grouped_gemm(x, w, e, backend="nonsense")  # type: ignore[arg-type]


@pytest.mark.skip(reason="Phase 2b: Pallas tile-size tuning needs a v5e chip")
def test_pallas_backend_matches_jax_on_v5e():
  """Pallas vs JAX agreement at v5e per-chip dims. Phase 2b on TPU."""
  raise NotImplementedError("Phase 2b: runs on v5e via profile harness")
