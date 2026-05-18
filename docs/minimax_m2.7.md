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

## Tokenizer notes

MiniMax-M2's `tokenizer_config.json` declares no `pad_token`.
MaxText's inference engine (`maxengine.py`) handles this by falling
back to `unk_token_id`, then `eos_token_id`, so batched decode still
works. You will see a one-line warning in the log; that's expected.

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
| `src/maxtext/checkpoint_conversion/standalone_scripts/convert_minimax_m2.py` | HF → MaxText converter (with on-the-fly FP8 dequant) |
| `scripts/minimax_m2.7/setup_dtmpfs.sh` | Mount /mnt/dtmpfs |
| `scripts/minimax_m2.7/download_and_convert.sh` | Stage HF weights and convert in place |
| `scripts/minimax_m2.7/decode_{v5e,v6e}.sh` | Launch decode |
