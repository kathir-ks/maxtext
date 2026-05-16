# Activation Extraction

Offline tool that runs a MaxText model in prefill mode, captures
intermediate activations (post-block residual stream, MLP output,
attention output), and writes them to safetensors shards for SAE
training and mechanistic interpretability research.

Design docs:

* High-level: [`docs/guides/activation_extraction/hld.md`](../../../../docs/guides/activation_extraction/hld.md)
* Low-level: [`docs/guides/activation_extraction/lld.md`](../../../../docs/guides/activation_extraction/lld.md)

## Quick start

```bash
python -m maxtext.tools.extract_activations.main \
  src/maxtext/configs/extract_activations.yml \
  model_name=qwen3-30b-a3b \
  load_parameters_path=gs://your-bucket/path/items/0 \
  activation_extraction_output_path=gs://your-bucket/sae-data/v1 \
  activation_extraction_dataset=jsonl \
  activation_extraction_dataset_path=gs://your-bucket/pile_subset.jsonl \
  per_device_batch_size=1 \
  max_prefill_predict_length=2048
```

## Output layout

```
{output_path}/
├── host_00/                # one dir per JAX process when run multi-host
│   ├── manifest.json
│   ├── residual_post/
│   │   └── layer_0012/
│   │       ├── shard_00000.safetensors    # acts (N,D) bf16 + token_ids/positions/doc_ids
│   │       ├── shard_00001.safetensors
│   │       └── metadata.json
│   └── ...
├── host_01/
└── ...
```

Each shard is a safetensors file with four tensors:

| key | shape | dtype |
|---|---|---|
| `activations` | `(N, d_model)` | `bfloat16` / `float16` / `float32` |
| `token_ids` | `(N,)` | `int32` |
| `positions` | `(N,)` | `int32` |
| `doc_ids` | `(N,)` | `int64` |

## Reading the output

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

`load_merged` transparently concatenates all shards from all
`host_NN/` subdirectories and sorts rows by `(doc_id, position)`. The
result is identical regardless of how many hosts were used during
extraction (this is verified by
`tests/integration/extract_activations/multihost_equivalence_test.py`).

## Supported hooks

| Hook | What it captures |
|---|---|
| `residual_post` | The post-block residual stream (default; Qwen-Scope / Gemma-Scope convention) |
| `mlp_out` | The MLP output **before** the residual add |
| `attn_out` | The attention output **before** the residual add |

For MoE models (Qwen3-30B-A3B, DeepSeek, MiniMax) the `residual_post`
hook captures the residual stream **after** the routed-and-shared
expert sum has been folded back in — uniform with dense models.

## Dataset backends

| `activation_extraction_dataset` | `activation_extraction_dataset_path` | Notes |
|---|---|---|
| `jsonl` | local or `gs://` path | one JSON object per line; needs `text` or `tokens`/`token_ids` |
| `hf` | HF dataset name | streaming; requires the `datasets` package |
| `pretokenized` | `.npy` of shape `[num_docs, seq_len]` | int32 tokens already chunked |

Documents are partitioned across JAX processes by `doc_id % P` so every
document is read exactly once and the merged output is identical to a
single-host run.

## Multi-host correctness

The headline guarantee is that:

> For a fixed dataset and checkpoint, the merged shards across all hosts
> (any `P`) sort to a bit-identical sequence of rows as the same dataset
> processed on a single host.

This is verified by the integration test:

```bash
pytest tests/integration/extract_activations/multihost_equivalence_test.py
```

which runs the loop with `P=1`, `P=2`, and `P=4` on identical inputs
and asserts that `load_merged(...)` returns the same `activations`,
`token_ids`, `positions`, and `doc_ids` in all three cases.

## Limitations (v1)

* Pipeline parallelism is not supported. The runner asserts
  `not config.using_pipeline_parallelism`.
* Per-expert MoE activations are not exposed. The `residual_post` hook
  captures the aggregate post-MoE residual, matching Qwen-Scope.
* The current prefill path processes one document per process per JIT
  invocation; throughput is improved by raising
  `per_device_batch_size` once a vectorised prefill is added in v2.
