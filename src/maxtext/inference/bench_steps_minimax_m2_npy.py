"""Minimal generate-loop benchmark for MiniMax-M2.7 on the .npy decode path.

Built because ``inference_microbenchmark`` trips a multi-host KV-cache
sharding mismatch when combined with our custom
``make_array_from_single_device_arrays`` load_params monkey-patch.

What this script measures:
  - prefill latency (ms) for one short prompt
  - steady-state decode step time (ms / token), averaged across
    ``warmup_steps + measure_steps`` generate calls
  - derived: pod-wide tok/s, per-chip tok/s

All values are written to a JSON file on worker 0 (other workers run the
exact same code for jax.distributed consensus but only worker 0 dumps).

Usage::

    python -m maxtext.inference.bench_steps_minimax_m2_npy \\
        --npy_dir /mnt/dtmpfs/minimax-m2.7-npy-distributed \\
        --warmup_steps 4 \\
        --measure_steps 32 \\
        --out_json $HOME/bench_steps.json \\
        src/maxtext/configs/base.yml \\
        model_name=minimax-m2.7 \\
        per_device_batch_size=1 \\
        ...
"""

import argparse
import json
import os
import pathlib
import statistics
import sys
import time


def main():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--npy_dir", required=True)
  parser.add_argument("--out_json", default="")
  parser.add_argument("--warmup_steps", type=int, default=4)
  parser.add_argument("--measure_steps", type=int, default=32)
  parser.add_argument("--prompt", default="The capital of France is")
  parser.add_argument("--help", "-h", action="store_true")
  args, rest = parser.parse_known_args()
  if args.help:
    print(__doc__)
    return

  npy_dir = pathlib.Path(args.npy_dir).expanduser().resolve()
  if not any(npy_dir.glob("manifest.p*.json")) and not (npy_dir / "manifest.json").exists():
    raise SystemExit(f"no manifest under {npy_dir}")

  # Install our load_params monkey-patch before importing MaxEngine.
  from maxtext.inference import decode_minimax_m2_npy as _dnpy
  _dnpy.install_load_params_patch(npy_dir)

  import jax
  jax.distributed.initialize()
  from maxtext.configs import pyconfig
  from maxtext.inference.maxengine import maxengine

  config = pyconfig.initialize(["bench_steps"] + rest)

  proc_idx = jax.process_index()
  proc_count = jax.process_count()
  n_chips = len(jax.devices())
  is_main = (proc_idx == 0)

  def log(msg):
    if is_main:
      print(f"[bench-steps] {msg}", flush=True)

  log(f"processes={proc_count} chips={n_chips} chips_per_host={len(jax.local_devices())}")

  t_engine_start = time.monotonic()
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load = jax.random.split(rng)
  params = engine.load_params(rng_load)
  t_loaded = time.monotonic()
  log(f"engine + load_params: {t_loaded - t_engine_start:.1f} s")

  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  has_chat_template = False
  try:
    has_chat_template = bool(getattr(tokenizer_model.tokenizer, "chat_template", False))
  except AttributeError:
    pass
  is_bos = config.add_bos and not has_chat_template
  tokens, true_length = tokenizer_model.encode(
      args.prompt, is_bos=is_bos, prefill_lengths=[config.max_prefill_predict_length])

  # Prefill (timed, includes compile of prefill graph).
  t_prefill_start = time.monotonic()
  rng, rng_prefill = jax.random.split(rng)
  prefill_result, first_token = engine.prefill(
      params=params, padded_tokens=tokens, true_length=true_length,
      rng=rng_prefill, slot=0)
  # Force materialization.
  jax.block_until_ready(prefill_result)
  t_prefill_done = time.monotonic()
  prefill_ms = (t_prefill_done - t_prefill_start) * 1000.0
  log(f"prefill: {prefill_ms:.1f} ms (length={tokens.size}, true_length={true_length})")

  rng, rng_init_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_init_decode)
  decode_state = engine.insert(prefill_result, decode_state, slot=0)
  jax.block_until_ready(decode_state)
  log("insert done")

  # Warmup: pay for the generate-step compile so it's not in the measurement.
  t_warm_start = time.monotonic()
  for _ in range(args.warmup_steps):
    rng, rng_g = jax.random.split(rng)
    decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_g)
  jax.block_until_ready(sampled_tokens)
  t_warm_done = time.monotonic()
  log(f"warmup ({args.warmup_steps} steps): {(t_warm_done - t_warm_start) * 1000.0:.1f} ms")

  # Measure: time each step individually to detect outliers.
  step_times_ms = []
  for _ in range(args.measure_steps):
    rng, rng_g = jax.random.split(rng)
    t0 = time.monotonic()
    decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_g)
    jax.block_until_ready(sampled_tokens)
    t1 = time.monotonic()
    step_times_ms.append((t1 - t0) * 1000.0)

  # Use the median of the measured steps to ignore one-off stalls
  # (occasional GC pauses, etc).
  step_p50 = statistics.median(step_times_ms)
  step_p90 = sorted(step_times_ms)[int(0.9 * len(step_times_ms))]
  step_mean = statistics.mean(step_times_ms)
  step_min = min(step_times_ms)
  step_max = max(step_times_ms)

  # Pod-wide tokens-per-second: at batch=N, every step produces N tokens.
  global_batch = int(config.per_device_batch_size * n_chips)
  tok_per_s_pod = global_batch * 1000.0 / step_p50
  tok_per_s_chip = tok_per_s_pod / n_chips

  log(f"step_ms p50={step_p50:.2f}  mean={step_mean:.2f}  p90={step_p90:.2f}  min={step_min:.2f}  max={step_max:.2f}")
  log(f"global_batch={global_batch}  tok/s_pod={tok_per_s_pod:.2f}  tok/s/chip={tok_per_s_chip:.3f}")

  if is_main and args.out_json:
    out = {
        "model_name": config.model_name,
        "per_device_batch_size": float(config.per_device_batch_size),
        "global_batch_size": global_batch,
        "n_chips": n_chips,
        "n_processes": proc_count,
        "max_prefill_predict_length": int(config.max_prefill_predict_length),
        "max_target_length": int(config.max_target_length),
        "quantization": str(getattr(config, "quantization", "") or ""),
        "quantize_kvcache": bool(getattr(config, "quantize_kvcache", False)),
        "capacity_factor": float(getattr(config, "capacity_factor", -1.0)),
        "ici_tensor_parallelism": int(config.ici_tensor_parallelism),
        "ici_expert_parallelism": int(config.ici_expert_parallelism),
        "attention": str(config.attention),
        "megablox": bool(getattr(config, "megablox", True)),
        "sparse_matmul": bool(getattr(config, "sparse_matmul", True)),
        "prefill_ms": prefill_ms,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "step_ms_p50": step_p50,
        "step_ms_p90": step_p90,
        "step_ms_mean": step_mean,
        "step_ms_min": step_min,
        "step_ms_max": step_max,
        "tok_per_s_pod": tok_per_s_pod,
        "tok_per_s_chip": tok_per_s_chip,
        "step_times_ms": step_times_ms,
    }
    pathlib.Path(args.out_json).write_text(json.dumps(out, indent=2))
    log(f"wrote {args.out_json}")


if __name__ == "__main__":
  main()
