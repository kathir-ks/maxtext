<!--
 Copyright 2023-2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

(activation_extraction_usage)=

# Activation Extraction — Usage

This page covers how to **run** the extraction tool and **consume** its
output. For how it works internally, see the [design doc](design.md).

## Quick start

The minimal command is:

```bash
python -m maxtext.tools.extract_activations.main \
  src/maxtext/configs/extract_activations.yml \
  model_name=qwen3-30b-a3b \
  load_parameters_path=gs://your-bucket/path/items/0 \
  activation_extraction_output_path=gs://your-bucket/sae-data/v1 \
  activation_extraction_dataset_path=gs://your-bucket/pile_subset.jsonl
```

`src/maxtext/configs/extract_activations.yml` is an example preset that
inherits `base.yml`, turns on extraction, and sets sensible defaults
(bf16 storage, 2M-token shards, residual-stream hook, JSONL backend). On
the CLI you override the four things that change per run: model name,
checkpoint path, output path, and dataset path.

## Configuration surface

All extraction-specific keys live under the `activation_extraction_*`
prefix and are read by both the model layers (for the `sow` injection)
and the runner (for the output pipeline).

| Key | Default | Description |
|---|---|---|
| `activation_extraction_enabled` | `False` | Master switch. Must be `True` for the tool. |
| `activation_extraction_layers` | `[]` | Layer indices to write. Empty → `[N//4, N//2, 3N//4]`. Negative indices count from the top. |
| `activation_extraction_hooks` | `["residual_post"]` | Subset of `{residual_post, mlp_out, attn_out}`. |
| `activation_extraction_output_path` | `""` | Local or `gs://` directory. **Required.** |
| `activation_extraction_shard_size_tokens` | `2_000_000` | Rows per safetensors shard. ~4 GB at `d_model=2048` bf16. |
| `activation_extraction_output_dtype` | `"bfloat16"` | One of `bfloat16` / `float16` / `float32`. |
| `activation_extraction_max_tokens` | `0` | Stop after this many tokens per layer (`0` = no limit). |
| `activation_extraction_skip_bos` | `True` | Drop position 0 of every sequence. BOS has an uninformative residual. |
| `activation_extraction_dataset` | `"jsonl"` | Backend: `jsonl` / `hf` / `pretokenized`. |
| `activation_extraction_dataset_path` | `""` | Path or HF dataset name. |
| `activation_extraction_dataset_split` | `"train"` | For HF datasets only. |
| `activation_extraction_text_key` | `"text"` | JSON/HF field with the input text. |
| `activation_extraction_max_documents` | `0` | Hard cap on docs read across all hosts. |

All of these can be set in YAML, via CLI `key=value`, or via `M_KEY=value`
environment variables (the standard MaxText pyconfig pattern).

## Dataset backends

| `..._dataset` | `..._dataset_path` | Notes |
|---|---|---|
| `jsonl` | local file or `gs://` URI | One JSON object per line; reads `text` or `tokens`/`token_ids`. |
| `hf` | HuggingFace dataset name | Streaming; requires the `datasets` Python package. |
| `pretokenized` | path to `.npy` of `[num_docs, seq_len]` int32 | Bypasses the tokenizer; useful for reproducibility. |

Across `P` JAX processes, documents are partitioned by `doc_id % P`, so
every document is read exactly once and the merged output is
position-for-position identical to a single-host run.

## Choosing layers and hooks

- **Default (`activation_extraction_layers=[]`)** picks three layers at
  the model's quartile boundaries — the most common "scope"-style release
  pattern (Gemma Scope, Qwen Scope).
- **A single middle layer** is the most common starting point for a new
  SAE study. For a 48-layer model, `activation_extraction_layers=[24]`.
- **Three hook types**:

  | Hook | What it captures |
  |---|---|
  | `residual_post` | Post-block residual stream. **Default; matches Qwen-Scope / Gemma-Scope.** |
  | `mlp_out` | MLP output *before* the residual add. |
  | `attn_out` | Attention output *before* the residual add. |

  For MoE models (Qwen3-30B-A3B, DeepSeek, MiniMax), `residual_post` is
  the natural choice — it captures the residual *after* routed and
  shared experts have summed back in.

## Output layout

```
{output_path}/
├── host_00/                # one dir per JAX process (when P > 1)
│   ├── manifest.json
│   ├── residual_post/
│   │   └── layer_0012/
│   │       ├── shard_00000.safetensors
│   │       ├── shard_00001.safetensors
│   │       └── metadata.json
│   └── …
├── host_01/
└── …
```

Each shard is a safetensors file with four tensors:

| Tensor | Shape | dtype |
|---|---|---|
| `activations` | `(N, d_model)` | `bfloat16` / `float16` / `float32` |
| `token_ids`   | `(N,)`         | `int32` |
| `positions`   | `(N,)`         | `int32` |
| `doc_ids`     | `(N,)`         | `int64` |

When the runner is executed with a single process, the per-host subdir
is omitted and shards are written directly under `{output_path}/`.

## Reading shards

```python
from maxtext.tools.extract_activations import load_merged

data = load_merged(
    "gs://your-bucket/sae-data/v1",
    hook="residual_post",
    layer_idx=12,
)
acts = data["activations"]    # (N, d_model)
toks = data["token_ids"]      # (N,)
poss = data["positions"]      # (N,)
dids = data["doc_ids"]        # (N,)
```

`load_merged` transparently concatenates all shards from every
`host_NN/` subdirectory and sorts rows lexicographically by
`(doc_id, position)`. The result is identical regardless of how many
hosts wrote the data (this is the
[multi-host correctness contract](design.md#multi-host-correctness)).

For very large datasets prefer the per-shard iterator:

```python
from maxtext.tools.extract_activations import list_host_dirs, iter_shard_paths
from safetensors.numpy import load_file

for host in list_host_dirs("gs://your-bucket/sae-data/v1"):
    for shard in iter_shard_paths(host, hook="residual_post", layer_idx=12):
        d = load_file(str(shard))
        # feed d["activations"] into your trainer
```

## Integrating with SAE training libraries

The output format matches the de-facto SAE training convention. Adapters
to popular libraries are intentionally thin.

**EleutherAI `sparsify`.** Each shard already carries
`activations` and (effectively) `tokens` tensors. Point the
`CacheLoader` at `{output_path}/host_*/<hook>/layer_<idx>/`:

```python
from sparsify.data import CacheLoader   # external library
loader = CacheLoader(
    "gs://your-bucket/sae-data/v1/host_00/residual_post/layer_0012",
    activations_key="activations",
    tokens_key="token_ids",
)
```

**`sae_lens`.** Use a `CachedActivationsLoader` wrapping `load_merged`:

```python
import torch
from maxtext.tools.extract_activations import load_merged

data = load_merged("/path/to/output", hook="residual_post", layer_idx=12)
acts = torch.from_numpy(data["activations"].astype("float32"))
# Feed acts into sae_lens.ActivationsStore with a fixed buffer.
```

## Multi-host execution

For multi-host runs (TPU v5e/v6e pods, GPU clusters), every host runs
the same command and JAX coordinates them. The MaxText runtime
initialises JAX distributed mode automatically (via
`maxtext.utils.max_utils.maybe_initialize_jax_distributed_system`); you
do not need to set up barriers or coordination yourself.

A typical 16-worker v5litepod-64 launch (single command, fanned out
across workers by your launcher):

```bash
python -m maxtext.tools.extract_activations.main \
  src/maxtext/configs/extract_activations.yml \
  model_name=qwen3-30b-a3b \
  load_parameters_path=gs://your-bucket/checkpoints/qwen3-30b-a3b/items/0 \
  activation_extraction_output_path=gs://your-bucket/sae-data/qwen3-30b-a3b/v1 \
  activation_extraction_dataset=hf \
  activation_extraction_dataset_path=monology/pile-uncopyrighted \
  activation_extraction_layers="[12,24,36]" \
  activation_extraction_shard_size_tokens=2000000 \
  per_device_batch_size=1 \
  max_prefill_predict_length=2048
```

Each worker writes to its own `host_NN/` subdirectory under the shared
output path; no cross-host gather is needed at write time. Use
`load_merged` (or the merged-iterator helpers) to consume the result.

## Limitations (v1)

- **Pipeline parallelism is not supported.** The runner asserts
  `not config.using_pipeline_parallelism`. Activation gather across
  pipeline stages is non-trivial and unnecessary for the
  inference-style configurations this tool targets.
- **Per-expert MoE activations are not exposed.** The aggregate post-MoE
  residual matches the Qwen-Scope convention. Per-expert hooks are a
  candidate v2 feature.
- **Single document per JIT invocation.** Throughput is improved by
  raising `per_device_batch_size` once a vectorised prefill is added in
  a follow-up.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `activation_extraction_enabled must be True` | Forgot to enable it on the CLI. | Pass `activation_extraction_enabled=True` or use the `extract_activations.yml` preset. |
| `bfloat16 output requires the `ml_dtypes` package` | Optional dep missing. | `pip install ml_dtypes` (it's already a transitive dep of JAX). |
| `pipeline parallelism is not supported` | `ici_pipeline_parallelism > 1`. | Use FSDP + TP only for v1. |
| `hook tuple length L != num_layers N` | Layer-count mismatch between sown intermediates and the configured model. | Confirm `num_decoder_layers` matches the checkpoint (look at the model YAML). |
| OOM during prefill | Sequence × batch × `d_model` × `num_hooks` too large. | Lower `per_device_batch_size`, drop `mlp_out`/`attn_out` and keep only `residual_post`, or reduce `max_prefill_predict_length`. |
| Empty `host_NN/` directories for some workers | `activation_extraction_max_documents` capped before each worker saw any. | Raise the cap, or remove it. |

## Related

- [Design doc](design.md) — internals, scan/MoE handling, multi-host
  correctness contract, performance back-of-envelope.
- [Optimization guide](../optimization.md) — sharding and performance
  tuning that applies to the prefill path used here.
