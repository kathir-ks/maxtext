# MiniMax-M2 / M2.7 on MaxText (v5e / v6e)

This guide describes how to run MiniMax-M2.7 (230B total, 10B active params)
end-to-end on a single TPU VM with **zero out-of-VM data transfer**: the HF
weights and the MaxText checkpoint both live on a RAM-backed `dtmpfs` mount
on the same machine that runs the TPU workload.

> Why dtmpfs?
>
> A regular GCS download from `huggingface.co` to a same-region disk is fine,
> but copying the converted MaxText checkpoint out to a *different-region*
> GCS bucket can run into hundreds of dollars in egress (see project memory
> notes). Keeping everything in tmpfs on a same-region TPU VM avoids egress
> entirely and also makes the conversion step faster (RAM-speed reads).

## Architecture summary

| Field | Value |
|---|---|
| Total parameters | 230 B |
| Active parameters | 10 B |
| Layers | 62 (all MoE) |
| Hidden size | 3072 |
| Attention | GQA — 48 query heads, 8 KV heads, `head_dim=128` |
| QK-norm | per-layer RMSNorm over `(heads × head_dim)` (Global RMSNorm) |
| RoPE | partial, `rotary_dim=64` of `head_dim=128`, `theta=5e6` |
| Routing | sigmoid + learnable bias, top-8 of 256 experts, no shared experts |
| Per-expert FFN | SwiGLU, intermediate 1536 |
| Vocab | 200 064 |
| Context | 196 608 (M2.7) / 204 800 (M2) |
| Quantization (HF) | `float8_e4m3fn`, block 128×128 |
| MTP | 3 modules of 1 transformer layer (training only) |

The MaxText config (`src/maxtext/configs/models/minimax-m2.7.yml`) layers the
DeepSeek-V3 routing knobs (`routed_score_func=sigmoid`, `routed_bias=True`,
`norm_topk_prob=false`, `shared_experts=0`) on top of Qwen3-MoE attention
(`use_qk_norm=true`) and sets `partial_rotary_factor=0.5` for partial RoPE.
The `MINIMAX_M2` decoder block (`src/maxtext/models/minimax_m2.py`) subclasses
`Qwen3MoeDecoderLayer`; partial RoPE flows through `AttentionWithNorm` from the
config field.

## End-to-end pipeline (single TPU VM)

These scripts live in `scripts/minimax_m2.7/`. The TPU VM must have:

* at least ~700 GB of free RAM if you size the tmpfs accordingly (default), or
* swap the staging directory for a same-region GCS bucket once you have the
  weights converted.

### 1. Mount dtmpfs (one-time, per VM)

```bash
sudo scripts/minimax_m2.7/setup_dtmpfs.sh /mnt/dtmpfs 700G
```

The script is idempotent: if `/mnt/dtmpfs` is already a mountpoint it just
prints the current mount and exits.

### 2. Download HF weights and convert to MaxText

```bash
# Activate the MaxText venv (Python 3.12)
source ~/venv-maxtext-py312/bin/activate
pip install huggingface_hub safetensors

scripts/minimax_m2.7/download_and_convert.sh
```

Environment overrides:

| var | default | meaning |
|---|---|---|
| `REPO` | `MiniMaxAI/MiniMax-M2.7` | HF repo to pull |
| `MODEL_TAG` | `minimax-m2.7` | MaxText model name (`minimax-m2` or `minimax-m2.7`) |
| `DTMPFS_ROOT` | `/mnt/dtmpfs` | dtmpfs root |
| `HF_REVISION` | `main` | HF revision pin |
| `SIMULATED_CPUS` | `16` | how many shards to write |

The converter dequantizes FP8 → BF16 on the fly (HF stores 128×128 block-scaled
`float8_e4m3fn` weights, same scheme as DeepSeek V3). No intermediate BF16
safetensors are written to disk.

### 3. Decode

```bash
# v5e (e.g. v5e-16)
scripts/minimax_m2.7/decode_v5e.sh

# v6e (e.g. v6e-64)
scripts/minimax_m2.7/decode_v6e.sh
```

Override `BATCH`, `MAX_TARGET_LENGTH`, `ICI_TENSOR`, `ICI_EXPERT`, and
`PROMPT` via env vars. The scripts read the MaxText checkpoint and the
HF tokenizer from the same dtmpfs paths.

## Multi-host TPU launch

For v5e/v6e slices larger than one host, the `setsid`/`pkill`/`ssh -A`
pattern documented in `feedback_ssh_launch` (see project memory) is the
recommended way to fan the decode command out across workers in parallel.
Note that dtmpfs is **per-host RAM**: each worker needs its own copy of
the checkpoint, either by:

1. running `download_and_convert.sh` on every host (RAM-only, fastest, no
   GCS hop), or
2. converting once to a same-region GCS bucket and pointing
   `load_parameters_path` there.

Option (1) keeps the no-out-of-VM-transfer guarantee verbatim. Option (2)
involves an in-region GCS round-trip but lets a single conversion serve N
hosts.

## Memory budget cheat sheet

* HF FP8 download: ~230 GB
* HF FP8 + working numpy float16 (one full tree in RAM during conversion):
  ~230 GB + ~460 GB ≈ **~690 GB peak** during conversion.
* MaxText scanned checkpoint (BF16 on disk via Orbax): ~460 GB

The numpy float16 footprint of the entire scanned tree is the dominant
cost. The converter currently pre-allocates one giant zero array per
weight type (matching the existing `convert_qwen3_moe.py` pattern), so
it cannot stream — pick a conversion host with enough RAM.

| TPU host class | Approx host RAM | Fits conversion? |
|---|---|---|
| v5e single host | ~64 GB | No |
| v6e-1 | ~64 GB | No |
| v6e-8 | ~512 GB | Tight; works if no other big tenants |
| v6e-16 / v6e-64 (per host) | ~1024 GB | Yes |

Fallbacks if you can't get a 500+ GB host:
1. point `--maxtext_model_path` at a **same-region** GCS bucket so
   Orbax streams the output as it shards, then point `decode_*.sh`
   at the bucket. This adds an in-region round-trip but stays inside
   the no-cross-region-egress constraint.
2. ship the FP8 weights as-is (no conversion) and convert in chunks
   per-layer — TODO, not implemented yet.

## TPU slice sizing for inference

The model footprint dominates HBM. Approximate weight sizes for the
230 B-parameter MiniMax-M2.7:

| Precision | Total weights | Per-chip min (v5e, 16 GB) | Per-chip min (v6e, 32 GB) |
|---|---|---|---|
| BF16 | ~460 GB | ≥ 32 chips | ≥ 16 chips |
| INT8 | ~230 GB | ≥ 16 chips | ≥ 8 chips |
| INT4 | ~115 GB | ≥ 8 chips | ≥ 4 chips |

Recommended minimum slices (leaves headroom for KV cache, activations,
and runtime allocator overhead):

| Slice | Total HBM | OK for BF16? | OK for INT8? | OK for INT4? |
|---|---|---|---|---|
| v5e-16 | 256 GB | no | tight | yes |
| v5e-64 | 1024 GB | yes | yes | yes |
| v6e-8 | 256 GB | no | tight | yes |
| v6e-16 | 512 GB | tight | yes | yes |
| v6e-64 | 2048 GB | yes | yes | yes |

Caveat: MaxText's `quantization=int8` is post-load dynamic
quantization — the BF16 weights are materialized first and only
then quantized. So even if the *runtime* fits in int8, the **load
step** still needs BF16 headroom. To run on smaller slices you
either need a pre-quantized checkpoint or you need to widen the
slice for the load and re-shard. See
`feedback_no_local_conversion` in project memory for the
in-region-conversion pattern that avoids cross-region egress.

`scripts/minimax_m2.7/decode_v5e.sh` and `decode_v6e.sh` default to
`weight_dtype=bfloat16` and `quantization=` (none). Override with
`BATCH=` and `ICI_*` env vars to match your actual slice.

## Tokenizer notes

MiniMax-M2's `tokenizer_config.json` declares no `pad_token`.
MaxText's inference engine (`maxengine.py`) handles this by falling
back to `unk_token_id`, then `eos_token_id`, so batched decode still
works. You will see a one-line warning in the log; that's expected.

## Fast path: v5e-64 in ~30 min cold-start

The current production setup uses the **distributed dtmpfs path** on
v5e-64. Each host stores only its 1/16 of the model (~27 GB) on local
dtmpfs; no host ever holds the whole 460 GB, so the 188 GB v5e host RAM
fits comfortably.

```bash
# 1) Bootstrap every TPU worker (fast-paths if already installed)
gcloud compute tpus tpu-vm ssh node-v5e-64-europe-west4-b \
  --zone=europe-west4-b --worker=all \
  --command='bash ~/maxtext/scripts/minimax_m2.7/bootstrap_tpu.sh'

# 2) Mount dtmpfs (one-time per slice lifecycle)
gcloud compute tpus tpu-vm ssh node-v5e-64-europe-west4-b \
  --zone=europe-west4-b --worker=all \
  --command='sudo bash ~/maxtext/scripts/minimax_m2.7/setup_dtmpfs.sh /mnt/dtmpfs 120G'

# 3) Distributed convert (windowed prefetch + parallel HF download)
gcloud compute tpus tpu-vm ssh node-v5e-64-europe-west4-b \
  --zone=europe-west4-b --worker=all \
  --command='bash ~/maxtext/scripts/minimax_m2.7/convert_distributed.sh'

# 4) Decode (first call compiles ~12 min, subsequent calls ~30 s via JAX cache)
gcloud compute tpus tpu-vm ssh node-v5e-64-europe-west4-b \
  --zone=europe-west4-b --worker=all \
  --command='bash ~/maxtext/scripts/minimax_m2.7/decode_distributed.sh'

# 5) Sweep benchmarks (env-var driven, aggregated CSV)
bash scripts/minimax_m2.7/sweep.sh
```

Wall times:
- Bootstrap: ~5-10 min cold, ~1 min warm (fast-path skip).
- Convert: ~50 min (windowed prefetch from HF; bottlenecked by per-layer
  dequant, not download).
- First decode: ~12 min JIT compile (cached under
  `/mnt/dtmpfs/jax_cache_v5e64`); subsequent decodes ~30 s.
- Per-host dtmpfs: ~30 GB shards + ~30 GB rolling FP8 window = ~60 GB
  peak; fits in 120 GB tmpfs.

v5e constraints to be aware of: `attention=dot_product`, `megablox=false`,
`sparse_matmul=false`. v5e ICI routing doesn't support ragged-all-to-all
so the MoE falls back to capacity-bounded dense matmul routing.

## Measured throughput

### v6e-64 (`node-v6e-64-europe-west4-a`, 2026-05-22)

Sweep harness: `scripts/minimax_m2.7/bench_steps.sh` →
`src/maxtext/inference/bench_steps_minimax_m2_npy.py` (prefill +
warmup + 32 timed generate steps with `jax.block_until_ready` around
each). Raw JSONs are in `benchmarks/v6e_minimax_m2_7_2026_05_22/`.
v6e supports `megablox=true sparse_matmul=true` so the MoE actually
reads only the 8 active experts per token instead of all 256.

| Config | Per-device batch | Global batch | max_target | step_ms p50 | **tok/s pod-wide** | tok/s/chip | Use case |
|---|---|---|---|---|---|---|---|
| bf16 + bf16-kv | 1 | 64 | 256 | 74 | 860 | 13.4 | Single-stream latency |
| bf16 + bf16-kv | 4 | 256 | 256 | 124 | 2,060 | 32.2 | Small-batch latency |
| bf16 + bf16-kv | 8 | 512 | 256 | 182 | 2,818 | 44.0 | |
| bf16 + int8-kv | 8 | 512 | 256 | 158 | 3,234 | 50.5 | |
| bf16 + int8-kv | 16 | 1024 | 256 | 251 | 4,087 | 63.9 | |
| bf16 + int8-kv | 32 | 2048 | 256 | 451 | 4,539 | 70.9 | Balanced |
| bf16 + int4-kv | 64 | 4096 | 256 | 708 | 5,787 | 90.4 | |
| bf16 + int4-kv | 64 | 4096 | 128 | 698 | 5,866 | 91.7 | |
| **bf16 + int4-kv** | **96** | **6144** | **128** | **920** | **6,678** | **104.3** | **Max throughput** |

Cells beyond batch=96 OOM at v6e's 32 GB/chip HBM (KV cache + activations
+ 7.2 GB weights/chip). Cells with `quantization=int8` weight-quant
crash with a qwix Pallas-kernel assertion (`v=192 bv=1024 s=192`) because
the per-chip MLP dim (192) doesn't divide the kernel's tile block (1024).
Fixing that should unlock another ~1.5-2x headroom by halving weight
bandwidth. Open issue tracked below.

Best single-stream tok/s (interactive chat speed): **~13 tok/s** at
batch=1.
Best aggregate throughput (serving many concurrent users):
**~6,700 tok/s** at batch=96 (6144 slots pod-wide).

### v5e-64 (`node-v5e-64-europe-west4-b`, 2026-05-20, for reference)

v5e's ICI doesn't support ragged-all-to-all → MoE falls back to dense
matmul over all 256 experts, costing ~25× more memory bandwidth per
token. Measured baseline only.

| Config | Batch | Prefill | Decoded tokens | Total wall | tok/s pod-wide | tok/s/chip |
|---|---|---|---|---|---|---|
| bf16, capacity_factor=2.0 | 1 | 8 | 120 | 46 s | ~5 | ~0.08 |

Notes / caveats:
- Wall time includes pyconfig + engine init + abstract_state + JIT
  cache-hit (~20 s total). The pure-decode steady-state is ~25 s for
  120 tokens, so the "real" generation throughput is ~5 tok/s pod-wide.
- v5e ICI routing forces `megablox=false sparse_matmul=false
  capacity_factor=2.0`, which falls back to dense matmul over experts —
  the dominant compute cost for MoE inference. v6e would do much better
  here once it's available.
- Quantization sweep (int8 / int4 / kv_quant) and batch-scaling cells
  were attempted via `inference_microbenchmark` but hit a KV-cache
  sharding mismatch when the AR cache is initialized with our
  `make_array_from_single_device_arrays` param path; see "Open issues"
  below.

## Open issues

- **int8 weight quantization** fails on v6e-64 with a qwix Pallas
  block-spec assertion `v=192 bv=1024 s=192`. The per-chip MLP
  intermediate dim (1536/8 = 192) doesn't divide qwix's default
  Pallas tile block (1024). Likely fix: tune `wi_tile_*_mlp_dim` /
  `wo_tile_*_embed_dim` to a divisor of 192 (e.g. 192 or 64) and
  retry. Should unlock another ~1.5-2× peak throughput by halving
  weight bandwidth.
- `inference_microbenchmark.run_benchmarks` calls `engine.aot_compile`
  which freezes a `decode_state_layouts` and then expects every
  subsequent `engine.insert` / `engine.generate` to use the same
  sharding. With our custom `decode_minimax_m2_npy` load_params path
  the engine-allocated cache buffers report a sharding mismatch on
  multi-host. Worked around by `bench_steps_minimax_m2_npy.py` which
  does its own prefill+generate loop without aot_compile.
- ICI mesh swaps (e.g. `tensor=4, expert=16`) require re-converting
  the per-host shards since the .npy layout is baked-in to the mesh.
  Re-conversion is ~30 min on v6e-64.

## File map

| File | Purpose |
|---|---|
| `src/maxtext/common/common_types.py` | `MINIMAX_M2` enum |
| `src/maxtext/models/minimax_m2.py` | `MiniMaxM2DecoderLayer` (subclass of Qwen3 MoE layer) |
| `src/maxtext/models/qwen3.py` | `AttentionWithNorm` patched to forward `partial_rotary_factor` |
| `src/maxtext/layers/attentions.py` | `GlobalRMSNorm` enabled for MINIMAX_M2 (per-layer QK norm spans heads) |
| `src/maxtext/layers/decoders.py` | Linen-path dispatch for MINIMAX_M2 |
| `src/maxtext/layers/nnx_decoders.py` | NNX-path dispatch for MINIMAX_M2 |
| `src/maxtext/configs/models/minimax-m2.yml` | M2 config (context 204 800) |
| `src/maxtext/configs/models/minimax-m2.7.yml` | M2.7 config (context 196 608) |
| `src/maxtext/configs/types.py` | Validators extended for `partial_rotary_factor` and fully-MoE base_mlp_dim |
| `src/maxtext/checkpoint_conversion/standalone_scripts/convert_minimax_m2.py` | HF → MaxText converter (single-host, for ≥1024 GB hosts) |
| `src/maxtext/checkpoint_conversion/standalone_scripts/convert_minimax_m2_streaming.py` | Single-host streaming converter (memmap-backed .npy) |
| `src/maxtext/checkpoint_conversion/standalone_scripts/convert_minimax_m2_distributed.py` | Distributed 16-process converter; windowed prefetch + barrier-on-exit |
| `src/maxtext/inference/decode_minimax_m2_npy.py` | Decode wrapper; monkey-patches MaxEngine.load_params for .npy shards |
| `src/maxtext/inference/benchmark_minimax_m2_npy.py` | Benchmark wrapper; same patch + inference_microbenchmark for tok/s |
| `scripts/minimax_m2.7/bootstrap_tpu.sh` | One-shot worker bootstrap (idempotent fast-path on re-runs) |
| `scripts/minimax_m2.7/setup_dtmpfs.sh` | Mount /mnt/dtmpfs |
| `scripts/minimax_m2.7/download_and_convert.sh` | Legacy single-host pipeline |
| `scripts/minimax_m2.7/convert_distributed.sh` | Distributed convert driver |
| `scripts/minimax_m2.7/decode_distributed.sh` | Distributed decode driver (env-var configurable) |
| `scripts/minimax_m2.7/benchmark_distributed.sh` | Per-cell benchmark driver |
| `scripts/minimax_m2.7/sweep.sh` | Iterates the Phase-2 config sweep + aggregates CSV |
| `scripts/minimax_m2.7/decode_{v5e,v6e}.sh` | Single-host decode (legacy) |
