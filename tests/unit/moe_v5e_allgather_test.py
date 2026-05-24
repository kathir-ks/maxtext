"""Unit tests for the v5e-allgather MoE forward path.

Two-tier validation:
  1. CPU/single-chip: compare against ``naive_sparse_moe`` (numpy
     ground truth) for correctness.
  2. v6e small mesh: compare token streams against the existing
     megablox path; must match under greedy decoding.

Phase 1 lands tier-1 (CPU). Phase 3 lands tier-2 (multi-chip).
"""

import numpy as np
import pytest

from tests.unit.moe_v5e_naive_reference import naive_sparse_moe


def test_naive_against_known_values():
  """Hand-computed 2-expert, 2-token, H=4, F=2 toy.

  Phase 1 gate: this must pass before any kernel work begins.
  Locks in the math contract for ``naive_sparse_moe``.
  """
  # 2 tokens, 2 experts, H=4, F=2, top-1 routing (so just one expert
  # per token, weight=1.0).
  H, F, E, K = 4, 2, 2, 1
  x = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
  # Expert 0: w0 picks col 0 of x; w1 picks col 0; wo writes back to col 0.
  # Expert 1: w0 picks col 1; w1 picks col 1; wo writes back to col 1.
  w0 = np.zeros((E, H, F), dtype=np.float32)
  w0[0, 0, 0] = 1.0  # E0: gate[0] = x[0]
  w0[1, 1, 0] = 1.0  # E1: gate[0] = x[1]
  w1 = np.zeros((E, H, F), dtype=np.float32)
  w1[0, 0, 0] = 1.0  # E0: up[0] = x[0]
  w1[1, 1, 0] = 1.0  # E1: up[0] = x[1]
  wo = np.zeros((E, F, H), dtype=np.float32)
  wo[0, 0, 0] = 1.0  # E0: write to dim 0
  wo[1, 0, 1] = 1.0  # E1: write to dim 1

  # Token 0 -> expert 0 only; Token 1 -> expert 1 only.
  top_k_idx = np.array([[0], [1]], dtype=np.int32)
  top_k_w = np.ones((2, K), dtype=np.float32)

  out = naive_sparse_moe(x, w0, w1, wo, top_k_idx, top_k_w)

  # By hand for token 0:
  #   gate = (1,0,0,0) @ w0[0] = (1, 0)
  #   up   = (1,0,0,0) @ w1[0] = (1, 0)
  #   mid  = silu(1)*1 ≈ 0.7311 at dim 0, 0 at dim 1
  #   down = (0.7311, 0) @ wo[0] = (0.7311, 0, 0, 0)
  # so out[0] ≈ (0.7311, 0, 0, 0)
  silu_of_one = 1.0 / (1.0 + np.exp(-1.0))  # ≈ 0.7311
  expected_token_0 = np.array([silu_of_one, 0.0, 0.0, 0.0], dtype=np.float32)
  expected_token_1 = np.array([0.0, silu_of_one, 0.0, 0.0], dtype=np.float32)

  np.testing.assert_allclose(out[0], expected_token_0, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(out[1], expected_token_1, rtol=1e-5, atol=1e-5)


def test_naive_shape_validation():
  """Catches caller mistakes early."""
  rng = np.random.default_rng(0)
  with pytest.raises(ValueError, match="must be"):
    naive_sparse_moe(
        x=rng.standard_normal((4,)).astype(np.float32),  # wrong: 1-D
        w0=rng.standard_normal((2, 4, 8)).astype(np.float32),
        w1=rng.standard_normal((2, 4, 8)).astype(np.float32),
        wo=rng.standard_normal((2, 8, 4)).astype(np.float32),
        top_k_idx=np.zeros((1, 1), dtype=np.int32),
        top_k_w=np.ones((1, 1), dtype=np.float32),
    )


def test_naive_topk_2():
  """Verify per-token contributions sum correctly across multiple slots."""
  rng = np.random.default_rng(42)
  B, H, F, E, K = 3, 8, 4, 4, 2
  x = rng.standard_normal((B, H)).astype(np.float32)
  w0 = rng.standard_normal((E, H, F)).astype(np.float32) * 0.1
  w1 = rng.standard_normal((E, H, F)).astype(np.float32) * 0.1
  wo = rng.standard_normal((E, F, H)).astype(np.float32) * 0.1
  top_k_idx = rng.integers(0, E, size=(B, K)).astype(np.int32)
  # Random non-normalized weights to test that weighting works.
  top_k_w = rng.uniform(0.1, 0.9, size=(B, K)).astype(np.float32)

  out = naive_sparse_moe(x, w0, w1, wo, top_k_idx, top_k_w)

  # Recompute by hand and compare.
  expected = np.zeros_like(x)
  for b in range(B):
    for k in range(K):
      e = int(top_k_idx[b, k])
      gate = x[b] @ w0[e]
      up = x[b] @ w1[e]
      mid = gate * (1.0 / (1.0 + np.exp(-gate))) * up
      down = mid @ wo[e]
      expected[b] += float(top_k_w[b, k]) * down

  np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skip(reason="Phase 3: orchestrator not yet implemented")
def test_v5e_allgather_matches_megablox_v6e():
  """End-to-end: same prompt, same model, two paths, identical tokens.

  Tier 2 gate (Phase 3). Requires v6e access to run megablox reference.
  """
  raise NotImplementedError("Phase 3 cross-path comparison pending")
