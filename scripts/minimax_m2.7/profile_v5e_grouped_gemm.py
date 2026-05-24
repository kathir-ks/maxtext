"""Phase 2b harness: validate + profile the Pallas backend of ``v5e_grouped_gemm`` on TPU v5e.

Run this on a v5e chip (any size — uses a single host's local devices):

    cd ~/maxtext
    source ~/venv-maxtext-py312/bin/activate
    python scripts/minimax_m2.7/profile_v5e_grouped_gemm.py

What it does:

  1. Builds synthetic input at the v5e per-chip dims for MiniMax-M2.7
     (capacity=128, E_local=32, H=384, F=192). The numbers come from
     TP=8 EP=8 sharding of the production config.
  2. Calls the JAX backend, then the Pallas backend with default tiles.
  3. Asserts outputs agree within a tight tolerance.
  4. Times both via ``time.monotonic()`` after a warmup, reports
     ms/call and a rough TFLOPS/chip estimate.
  5. Iterates over a small set of tile sizes and reports which one wins.

Exits non-zero on correctness failure.

This is the Phase 2b validation gate. The actual ``80% MXU utilization``
goal in the project plan is verified by inspecting the per-tile timings
this script prints versus the theoretical peak ``v5e bf16 ≈ 197
TFLOPS/chip``.
"""

import statistics
import sys
import time
from typing import List, Tuple

import numpy as np


def _percent(t_ms: float, flops: int, peak_tflops_per_chip: float = 197.0) -> float:
  """Rough MXU utilization estimate."""
  tflops = (flops / 1e12) / (t_ms / 1000.0)
  return tflops / peak_tflops_per_chip * 100.0


def main():
  import jax
  import jax.numpy as jnp

  # Make sure we're actually on a TPU.
  devs = jax.devices()
  kind = devs[0].device_kind if devs else "(none)"
  print(f"jax devices: {len(devs)} of kind {kind}")
  if "TPU" not in kind:
    print("WARNING: not running on a TPU; Pallas backend will be a no-op test.")

  # Synthetic v5e per-chip workload.
  capacity, E_local, H, F = 128, 32, 384, 192
  flops_per_call = 2 * capacity * H * F  # ~9.4 MFLOPs

  rng = np.random.default_rng(0)
  x_np = (rng.standard_normal((capacity, H)) * 0.1).astype(np.float32)
  w_np = (rng.standard_normal((E_local, H, F)) * 0.05).astype(np.float32)
  ea_np = rng.integers(0, E_local, size=(capacity,)).astype(np.int32)

  # Cast to bf16 to match production decode.
  x = jnp.asarray(x_np).astype(jnp.bfloat16)
  w = jnp.asarray(w_np).astype(jnp.bfloat16)
  ea = jnp.asarray(ea_np)

  from maxtext.kernels.v5e_grouped_gemm import v5e_grouped_gemm

  # --- Correctness: Pallas vs JAX ---
  print("\n[correctness] running JAX backend...")
  jax_out = jax.jit(lambda a, b, c: v5e_grouped_gemm(a, b, c, backend="jax"))(x, w, ea)
  jax.block_until_ready(jax_out)

  print("[correctness] running Pallas backend with default tiles...")
  try:
    pallas_out = jax.jit(
        lambda a, b, c: v5e_grouped_gemm(a, b, c, backend="pallas",
                                          block_capacity=128, block_f=128, block_h=128)
    )(x, w, ea)
    jax.block_until_ready(pallas_out)
  except Exception as exc:  # pragma: no cover -- diagnostic
    print(f"[correctness] Pallas backend FAILED to compile: {type(exc).__name__}: {exc}")
    sys.exit(2)

  max_abs = float(jnp.abs(jax_out.astype(jnp.float32) - pallas_out.astype(jnp.float32)).max())
  print(f"[correctness] max |jax - pallas| = {max_abs:.4e}")
  if max_abs > 5e-2:
    print("[correctness] FAIL: drift too large")
    sys.exit(1)
  print("[correctness] PASS")

  # --- Timing: JAX backend ---
  def time_call(fn, name, n=50):
    # Warmup.
    out = fn(x, w, ea); jax.block_until_ready(out)
    out = fn(x, w, ea); jax.block_until_ready(out)
    times_ms: List[float] = []
    for _ in range(n):
      t0 = time.monotonic()
      out = fn(x, w, ea); jax.block_until_ready(out)
      times_ms.append((time.monotonic() - t0) * 1000.0)
    med = statistics.median(times_ms)
    print(f"  [{name}] median {med:.3f} ms (n={n}); "
          f"~{_percent(med, flops_per_call):.1f}% peak bf16 v5e MXU "
          f"({flops_per_call/1e9 * (1000.0/med):.2f} GFLOPS/s)")
    return med

  print("\n[timing] JAX backend")
  jax_fn = jax.jit(lambda a, b, c: v5e_grouped_gemm(a, b, c, backend="jax"))
  jax_ms = time_call(jax_fn, "jax")

  # --- Tile-size sweep on Pallas backend ---
  print("\n[timing] Pallas backend tile sweep")
  best: Tuple[Tuple[int, int, int], float] = ((128, 128, 128), float("inf"))
  for bc, bf, bh in [
      (128, 128, 128),
      (128, 192, 128),   # F is 192, this is the full F
      (128, 192, 384),   # full H+F in one shot (small workload)
      (64, 128, 128),
      (64, 192, 128),
      (32, 192, 128),
  ]:
    # Skip combos that don't divide.
    if capacity % bc or F % bf or H % bh:
      print(f"  skip ({bc=}, {bf=}, {bh=}): not divisible")
      continue
    pallas_fn = jax.jit(
        lambda a, b, c, bc=bc, bf=bf, bh=bh: v5e_grouped_gemm(
            a, b, c, backend="pallas", block_capacity=bc, block_f=bf, block_h=bh)
    )
    try:
      t_ms = time_call(pallas_fn, f"pallas {bc=:3d} {bf=:3d} {bh=:3d}")
      if t_ms < best[1]:
        best = ((bc, bf, bh), t_ms)
    except Exception as exc:  # pragma: no cover
      print(f"  ({bc=}, {bf=}, {bh=}): FAILED ({type(exc).__name__}: {exc})")

  print(f"\n[summary] best pallas tile: {best[0]} at {best[1]:.3f} ms "
        f"(~{_percent(best[1], flops_per_call):.1f}% peak)")
  print(f"[summary] vs jax: {jax_ms:.3f} ms ({jax_ms/best[1]:.2f}x slower)")


if __name__ == "__main__":
  main()
