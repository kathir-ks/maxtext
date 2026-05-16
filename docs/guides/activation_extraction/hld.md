# Activation Extraction — High-Level Design

## 1. Goal

Add a first-class capability to MaxText for **extracting intermediate model
activations** from trained checkpoints so they can be consumed by

* Sparse Autoencoder (SAE) training pipelines
  (EleutherAI `sparsify`, `sae_lens`, custom).
* Mechanistic interpretability tooling (`Delphi`, `Neuronpedia`, ad-hoc probes).
* Steering / activation patching / circuit-discovery experiments.

The capability must:

* Work for every model family currently supported by MaxText
  (Llama 2/3/4, Mistral, Gemma 2/3/4, Qwen 2.5 / 3 / 3-MoE, DeepSeek v2/v3,
  MiniMax, GPT-OSS, …) without per-model code.
* Work for dense and Mixture-of-Experts models with the same hook.
* Run efficiently on TPU pods (v5e, v5p, v6e, single-slice and multi-slice)
  using the existing MaxText sharding/checkpointing stack.
* Produce output that is **drop-in compatible** with the established
  open-source SAE training format
  (safetensors shards of flattened `(N, d_model)` activations + token-id
  sidecar).

It must **not**:

* Slow down or change behaviour of training / inference paths when the
  extraction flag is off.
* Introduce a parallel model implementation that has to be kept in sync.
* Require pickling or any unsafe deserialisation in the output format.

## 2. Non-goals (v1)

* Pipeline parallelism (`ici_pipeline_parallelism > 1`).  Activation gather
  across pipeline stages is non-trivial and is not needed for the current
  inference workloads.
* Per-expert MoE activations.  Standard practice (Qwen-Scope) is to hook the
  post-block residual stream uniformly across dense and MoE models.
* On-the-fly shuffling.  Trainers already maintain an
  `ActivationsStore` shuffle buffer; we emit in generation order.
* Training-time activation logging.  Extraction is a separate offline tool.

## 3. Approach (one paragraph)

We extend MaxText’s existing Flax `sow("intermediates", …)` machinery —
already used in `DecoderLayer` for activation-statistics — to optionally
emit **full activation tensors** (`layer_output`, `mlp_lnx`,
`attention_lnx`) when a config flag is set.  A new entry-point
`src/maxtext/tools/extract_activations/main.py` reuses `MaxEngine` to load
parameters, set up the device mesh and tokenize input, runs prefill-mode
forward passes with `mutable=["intermediates"]`, gathers across hosts using
`jax.experimental.multihost_utils.process_allgather`, and writes
**safetensors** shards of bf16 activations together with token-id /
position / document-id sidecars.  Output is layer-keyed and shard-paginated
to ~2 GB so that EleutherAI `sparsify` and `sae_lens` consume it directly.

## 4. Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Hook mechanism | Flax `sow()` | Works inside `nn.scan` (`variable_axes={"intermediates":0}`), zero cost when off, no parallel model |
| Default hook point | Post-block residual (`layer_output`) | Anthropic / OpenAI / Gemma Scope / Qwen Scope convention; uniform across dense and MoE |
| Forward pass mode | `MODEL_MODE_PREFILL` | Identical layer outputs to `TRAIN`, matches inference distribution, no KV-cache complexity |
| Output format | Safetensors, flattened `(N, d_model)` bf16, per-layer dir | Drop-in for `sparsify` and `sae_lens`; safe deserialisation |
| Sidecar | `token_ids` (int32), `positions` (int32), `doc_ids` (int64) | Enables Delphi / Neuronpedia context reconstruction |
| Multi-host gather | `multihost_utils.process_allgather` | Already used by MaxText; no socket-barrier server needed |
| CLI pattern | `pyconfig.initialize(argv)` | Same as `inference/decode.py`; YAML + CLI + env vars |
| Model loading | Reuse `MaxEngine` | Free Orbax restore, sharding, tokenizer, abstract-params |
| MoE | Same hook as dense | Qwen-Scope convention; per-expert SAE rare; revisit in v2 |

## 5. Output layout

```
{extract_output_path}/
├── manifest.json                  # model, dtype, d_model, hooks, total_tokens
├── layer_12/
│   ├── shard_00000.safetensors    # activations (N, d_model) bf16
│   │                               # token_ids   (N,)         int32
│   │                               # positions   (N,)         int32
│   │                               # doc_ids     (N,)         int64
│   ├── shard_00001.safetensors
│   └── metadata.json              # per-layer counts and per-shard sizes
├── layer_24/
│   └── …
└── layer_36/
```

This layout maps cleanly to:

* EleutherAI `sparsify` — its `CacheLoader` reads `activations` + `tokens`
  from per-hook safetensors shards.
* `sae_lens` `CachedActivationsLoader` — a thin adapter maps `activations`
  to `acts` and exposes the manifest as `cfg`.

## 6. Module layout

```
src/maxtext/
├── tools/extract_activations/
│   ├── __init__.py
│   ├── main.py        # CLI entry, pyconfig.initialize → Runner.run
│   ├── runner.py      # batched prefill loop, gather, dispatch to writer
│   ├── writer.py      # sharded safetensors writer + manifest
│   ├── datasets.py    # JSONL + HF + pre-tokenized backends
│   └── hooks.py       # shared constants for hook names / collection key
├── layers/decoders.py # +20 LoC: guarded sow() for full tensors
├── configs/
│   ├── types.py       # +5 fields under ActivationExtractionConfig
│   ├── base.yml       # +5 default values
│   └── extract_activations.yml   # example preset
└── inference/maxengine/maxengine.py   # +1 LoC: include 'intermediates' in mutable when flag is on
```

## 7. Data flow (single host, dense model)

```
prompt batch
    │
    ▼
tokenize (HF / sp / tiktoken)        ──▶ (padded_tokens, true_length, token_ids)
    │
    ▼
MaxEngine.prefill(...)               ──▶ logits, intermediates, kv_cache
    │
    ▼
intermediates[scan]["residual_post"] shape [L, B, T, D]  (scan stacks)
    │  pick wanted layer indices
    ▼
per-layer arrays [B, T, D] bf16
    │  mask padding tokens (true_length)
    │  flatten to [N_tokens, D]
    ▼
ShardedSafetensorsWriter.append(layer_idx, acts, token_ids, positions, doc_ids)
    │  buffer until shard_size_tokens reached
    ▼
shard_K.safetensors  +  metadata.json
```

## 8. Multi-host correctness contract

When extraction is run with `P` hosts on input dataset `D`, the union of
all shards across all per-layer directories must, after sorting by
`(doc_id, position)`, be **bit-identical** to the output of a single-host
run on the same `D` with the same checkpoint and same sharding intent.

This is the property tested by the multihost equivalence test
(`tests/integration/extract_activations_multihost_test.py`).

## 9. Risks

| Risk | Mitigation |
|---|---|
| `sow()` + `with_sharding_constraint` interaction inside `nn.scan` | Unit test on toy 2-layer model exercising both code paths |
| MaxEngine `prefill` API takes a single 1-D `padded_tokens` today | Runner loops over the batch axis on each host; throughput acceptable for v1, vectorised prefill in v2 |
| Quantised checkpoints (int4 MiniMax) | Activations remain in bf16/fp32 inside the residual stream; smoke-test on the int4 MiniMax M2.5 checkpoint |
| Disk / GCS bandwidth on v6e-64 | bf16 halves bytes vs fp32; sharded streaming + GCS multi-part uploads keep up |
| Pipeline parallelism with sown activations | Explicitly unsupported in v1; runner asserts `not config.using_pipeline_parallelism` |
| MoE expert routing changes outputs | Hook is *after* the routed+shared sum, so the residual is post-MoE; this is the documented Qwen-Scope behaviour |

## 10. Out-of-scope follow-ups

* Per-expert MoE activations
* Pipeline-parallel-aware gather
* On-the-fly streaming directly into a co-located SAE trainer
* Pre-shuffled output (currently in generation order; trainer shuffles)
* fp8 activation storage
