# MiniMax-M2.7 on TPU v5e-64 — runbook

Terse operational doc for re-running the distributed pipeline on
`node-v5e-64-europe-west4-b` (or any v5e-64 in europe-west4-b). Aim:
fresh slice to first decode in under 30 min wall-time when the JAX
compile cache is warm.

## Prerequisites

- A v5e-64 TPU node, **READY** state (check via
  `gcloud compute tpus tpu-vm describe <name> --zone=<zone> --format='value(state)'`).
- `gcloud` configured for the right project; SSH key fanned out
  (`gcloud compute tpus tpu-vm ssh ... --worker=all --command='hostname'`
  once is enough to push keys).
- HuggingFace anonymous access works for `MiniMaxAI/MiniMax-M2.7`
  (no token required; only ~30 files/host needed per window).

## Environment matrix

| Var | Default | Meaning |
|---|---|---|
| `BRANCH` | `feature/minimax-m2.7` | Branch the bootstrap pulls |
| `HF_DOWNLOAD_WORKERS` | `8` | snapshot_download max_workers (per host) |
| `PREFETCH_WINDOW` | `8` | Layers per prefetch chunk; caps dtmpfs use |
| `JAX_CACHE_DIR` | `/mnt/dtmpfs/jax_cache_v5e64` | JAX compile cache |
| `QUANT` | `''` (bf16) | `''`, `int8`, `int4`, `intmp` |
| `KV_QUANT` | `false` | `true` enables int8/int4 KV-cache quant |
| `BATCH` | `1` | per_device_batch_size |
| `CAPACITY` | `2.0` | MoE capacity_factor |
| `MAX_TARGET_LENGTH` | `64` | Decode max output (small for smoke) |
| `PROMPT` | `"The capital of France is"` | Test prompt |

## Bring-up sequence

```bash
NODE=node-v5e-64-europe-west4-b
ZONE=europe-west4-b

# 0) (one-time per repo) SCP the bootstrap script. The first bootstrap
#    has nothing to fetch from git so we ship the script directly.
gcloud compute tpus tpu-vm scp \
  scripts/minimax_m2.7/bootstrap_tpu.sh ${NODE}:~/bootstrap_tpu.sh \
  --zone=${ZONE} --worker=all

# 1) Bootstrap: clone + venv + maxtext/.[tpu] + torch (cpu) + JetStream
gcloud compute tpus tpu-vm ssh ${NODE} --zone=${ZONE} --worker=all \
  --command='bash ~/bootstrap_tpu.sh'

# 2) Mount dtmpfs (120 GB) — gone on every slice lifecycle event
gcloud compute tpus tpu-vm ssh ${NODE} --zone=${ZONE} --worker=all \
  --command='sudo bash ~/maxtext/scripts/minimax_m2.7/setup_dtmpfs.sh /mnt/dtmpfs 120G'

# 3) Distributed convert. Idempotent: re-runs skip already-converted
#    hosts and just barrier-wait.
gcloud compute tpus tpu-vm ssh ${NODE} --zone=${ZONE} --worker=all \
  --command='bash ~/maxtext/scripts/minimax_m2.7/convert_distributed.sh'

# 4) Decode
gcloud compute tpus tpu-vm ssh ${NODE} --zone=${ZONE} --worker=all \
  --command='bash ~/maxtext/scripts/minimax_m2.7/decode_distributed.sh'

# 5) Sweep benchmarks (from your dev machine)
bash scripts/minimax_m2.7/sweep.sh
```

## Expected wall-time

| Phase | Cold | Warm jax_cache |
|---|---|---|
| Bootstrap (fresh) | ~5-10 min | ~1 min |
| dtmpfs mount | <30 s | <30 s |
| Distributed convert | ~50 min | skip-and-barrier ~10 s if shards exist |
| First decode (compile) | ~12 min | ~30 s |
| Single benchmark cell | ~2-3 min | ~2-3 min |
| **Total to first decode** | **~65 min** | **~3 min** |

## Failure modes

| Symptom | Cause | Recovery |
|---|---|---|
| OOM at batch=8 | KV cache + activations exceed 16 GB/chip HBM | Drop to batch=4 or enable `kv_quant=true` |
| `HF 429 Too Many Requests` | snapshot_download saturated HF Hub | Set `HF_DOWNLOAD_WORKERS=4`; retry. HF rate-limits at >~64 concurrent per IP. |
| `BarrierError... Shutdown` mid-convert | Stragglers crashed after fast workers exited; pre-barrier fix | Re-run convert. Idempotent skip kicks in on the already-done hosts; only stragglers re-do work. |
| `No space left on device` on dtmpfs | dtmpfs too small or prefetch window too large | Set `PREFETCH_WINDOW=4` or remount dtmpfs at 140 G. |
| Compile-cache poisoning (`InvalidArgumentError` from XLA) | Stale cache entry from a different code rev | `gcloud ssh --worker=all --command='rm -rf /mnt/dtmpfs/jax_cache_v5e64'` |
| `Ragged all-to-all is currently not supported` | MoE knobs default to megablox/sparse_matmul; v5e ICI lacks the routing | Confirm `megablox=false sparse_matmul=false capacity_factor=2.0` are passed (decode_distributed.sh does this by default) |
| Worker process abandoned, dtmpfs full | Old python lingering after preempt | `sudo pkill -9 -f 'convert_minimax_m2\\|launch_decode'` on the affected worker; remount dtmpfs |
| v5e-64 slice preempted | Spot interruption | Wait + poll for READY (10-60 min). Then re-run from step 1. `$HOME` survives most lifecycle events but dtmpfs does not. |

## Sanity checks

After each phase, expect:

- **Bootstrap**: `python -c 'import jax, torch, jetstream'` succeeds on every worker.
- **dtmpfs**: `df -h /mnt/dtmpfs` shows 120 G capacity on every worker.
- **Convert**: each worker has `/mnt/dtmpfs/minimax-m2.7-npy-distributed/manifest.p*.json`
  plus 64 `.npy` files totalling ~27 GB.
- **Decode**: every worker's `~/decode_dist.log` contains the same
  `Input ... -> ...` line at roughly the same UTC second.

## Architectural pointers

- The decode wrapper monkey-patches `MaxEngine.load_params` so any
  caller (`decode.main`, `inference_microbenchmark.main`) transparently
  picks up the .npy params via `make_array_from_single_device_arrays`.
- The convert+decode use the same `(tensor=8, expert=8)` mesh; the
  benchmark wrapper inherits these from `pyconfig.initialize`.
- KV cache memory is the binding constraint at `batch>4`; see
  `pagedattn_num_pages` and `quantize_kvcache` for headroom.
