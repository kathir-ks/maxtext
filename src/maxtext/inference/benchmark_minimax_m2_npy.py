"""Benchmark wrapper for MiniMax-M2.7 using per-host distributed .npy shards.

Companion to ``decode_minimax_m2_npy``: reuses the load_params monkey-patch
to feed sharded params built from local .npy files into ``MaxEngine``, then
hands off to MaxText's standard ``inference_microbenchmark`` for prefill
latency + steady-state decode tok/s in standardized JSON.

Usage::

    python -m maxtext.inference.benchmark_minimax_m2_npy \\
        --npy_dir /mnt/dtmpfs/minimax-m2.7-npy-distributed \\
        src/maxtext/configs/base.yml \\
        model_name=minimax-m2.7 \\
        ...

The same ``install_load_params_patch`` works unmodified — both
``inference_microbenchmark.run_benchmarks`` and ``decode.main`` call
``engine.load_params(rng)`` on the MaxEngine class.
"""

import argparse
import pathlib
import sys


def main():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--npy_dir", required=True)
  parser.add_argument("--help", "-h", action="store_true")
  args, rest = parser.parse_known_args()
  if args.help:
    print(__doc__)
    return

  npy_dir = pathlib.Path(args.npy_dir).expanduser().resolve()
  has_replicated = (npy_dir / "manifest.json").exists()
  has_distributed = any(npy_dir.glob("manifest.p*.json"))
  if not (has_replicated or has_distributed):
    raise SystemExit(f"no manifest.json or manifest.p*.json found under {npy_dir}")

  # Install the load_params monkey-patch from the decode wrapper. Both
  # decode.main and inference_microbenchmark.run_benchmarks call
  # MaxEngine.load_params(rng) — so the patch transparently intercepts
  # for either path.
  from maxtext.inference import decode_minimax_m2_npy as _decode_npy
  _decode_npy.install_load_params_patch(npy_dir)

  # Hand off to MaxText's standard microbenchmark; the argv slot we
  # prepend is ignored by absl/pyconfig but is required to keep the
  # arg shape consistent with `python -m maxtext.inference.decode`.
  from maxtext.inference import inference_microbenchmark
  inference_microbenchmark.main(["benchmark_minimax_m2_npy"] + rest)


if __name__ == "__main__":
  main()
