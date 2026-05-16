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

(activation_extraction_design)=

# Activation Extraction — Design

This page documents the internals of the
`maxtext.tools.extract_activations` tool. For how to **use** the tool,
see the [usage guide](usage.md).

## Goals and non-goals

The extraction tool exists to feed external SAE-training and
mech-interp pipelines from MaxText checkpoints with as little code
divergence from inference as possible. Concretely:

**Goals.**

- Capture intermediate activations (post-block residual stream, MLP
  output, attention output) for every model family MaxText supports —
  dense and Mixture-of-Experts — with **one** code path per layer.
- Run on the existing inference infrastructure (MaxEngine, Orbax,
  pyconfig) without forking a parallel forward pass.
- Produce output that drops straight into
  [EleutherAI `sparsify`](https://github.com/EleutherAI/sparsify) and
  [`sae_lens`](https://github.com/jbloomAus/SAELens).
- Be **preemption-resilient** (one host crashing does not corrupt the
  rest of the data).
- **Zero HLO impact** when extraction is disabled: training and
  inference paths must compile to the same code with the flag off.

**Non-goals (v1).**

- Pipeline parallelism (`ici_pipeline_parallelism > 1`). Cross-stage
  activation gather is non-trivial and unnecessary for the current
  inference-style configurations.
- Per-expert MoE activations. We follow the Qwen-Scope convention of
  treating the post-block residual uniformly across dense and MoE
  models.
- On-the-fly shuffling. Trainers already maintain a shuffle buffer; we
  emit shards in generation order.
- Training-time activation logging. Extraction is offline.

## Approach in one paragraph

We extend MaxText's existing Flax `sow("intermediates", …)` machinery
— already used in `DecoderLayer` for activation statistics — to
optionally emit **full activation tensors**
(`layer_output`, `mlp_lnx`, `attention_lnx`) when a config flag is set.
A new entry point `src/maxtext/tools/extract_activations/main.py`
reuses `MaxEngine` to load parameters, set up the device mesh, and
build a tokenizer, then runs prefill-mode forward passes with
`mutable=["intermediates"]`, and writes **safetensors** shards of bf16
activations together with token-id / position / document-id sidecars.
Each JAX process writes its own `host_NN/` subdirectory; offline merge
sorts rows by `(doc_id, position)` to produce a bit-deterministic view
that is identical to a single-host run on the same dataset.

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Hook mechanism | Flax `sow()` | Works inside `nn.scan` (`variable_axes={"intermediates":0}`), zero cost when off, no parallel model |
| Default hook point | Post-block residual (`layer_output`) | Anthropic / OpenAI / Gemma Scope / Qwen Scope convention; uniform across dense and MoE |
| Forward pass mode | `MODEL_MODE_PREFILL` | Identical layer outputs to `TRAIN`, matches inference distribution, no KV-cache complexity |
| Output format | Safetensors, `(N, d_model)` bf16, per-`(hook, layer)` dir | Drop-in for `sparsify` and `sae_lens`; safe deserialisation |
| Sidecar arrays | `token_ids` (int32), `positions` (int32), `doc_ids` (int64) | Enables Delphi / Neuronpedia context reconstruction |
| Multi-host layout | Per-host `host_NN/` subdir | No cross-host gather needed; preemption-resilient; trivial offline merge |
| CLI pattern | `pyconfig.initialize(argv)` | Same as `inference/decode.py`; YAML + CLI + env vars |
| Model loading | Reuse `MaxEngine` | Free Orbax restore, sharding, tokenizer, abstract-params |
| MoE | Same hook as dense | Qwen-Scope convention; per-expert SAE rare in published work |

## Module layout

```
src/maxtext/
├── tools/extract_activations/
│   ├── __init__.py           # public re-exports
│   ├── main.py               # CLI entry: pyconfig.initialize → Runner.run
│   ├── runner.py             # ExtractionLoop (pure) + Runner (MaxEngine-wired)
│   ├── writer.py             # ShardedSafetensorsWriter
│   ├── datasets.py           # JSONL / HF / pretokenized backends
│   ├── merge.py              # load_merged() + iter helpers
│   └── hooks.py              # maybe_sow_activations + hook name constants
├── layers/decoders.py        # one-line maybe_sow_activations call
├── models/{llama2,llama4,qwen3,gemma,gemma2,gemma4,mistral,mixtral,
│           gpt3,gpt_oss,olmo3}.py
│                             # same one-liner per model's DecoderLayer
├── configs/types.py          # ActivationExtraction pydantic mixin
├── configs/base.yml          # default values (all extraction off)
└── configs/extract_activations.yml  # example preset
```

## Hook injection

The shared helper in `src/maxtext/tools/extract_activations/hooks.py`
is called from every `DecoderLayer` implementation right after the
existing `record_internal_nn_metrics` block:

```python
from maxtext.tools.extract_activations.hooks import maybe_sow_activations

maybe_sow_activations(
    self,
    config=cfg,
    layer_output=layer_output,
    mlp_out=mlp_lnx,
    attn_out=attention_lnx,
)
```

The helper short-circuits when `cfg.activation_extraction_enabled` is
False, so the guarded branches are eliminated by JAX during tracing.
Under `nn.scan` (the common case for MaxText), the scan wrapper sets
`variable_axes={"intermediates": 0}`, so multiple sows of the same
name across the scanned layers automatically stack as
`[num_layers, …]` along axis 0 — exactly the layout the runner
expects.

## Runner architecture

The runner is split into two parts:

1. **`ExtractionLoop`** — pure data plane. Takes an injected
   `forward_fn`, a tokenizer callable, a dataset backend, and a
   writer; orchestrates `tokenize → forward → write` without any
   dependency on MaxEngine, JetStream, or a real TPU. This is the
   class exercised by every unit and integration test.
2. **`Runner`** — production wiring. Loads pyconfig, initialises JAX
   distributed mode (via the standard `max_utils` helper), builds a
   `MaxEngine`, materialises sharded parameters, builds a tokenizer,
   constructs a JIT'd `forward_fn` that calls
   `engine.model.apply(..., mutable=["intermediates"])`, and hands all
   of that to `ExtractionLoop`.

The split makes the loop fully testable without TPUs, real
checkpoints, or JetStream — and means changes to MaxEngine internals
do not silently break the loop.

### Forward pass

The forward function used in production calls `engine.model.apply` in
`MODEL_MODE_PREFILL`, with `mutable=["intermediates"]` added so the
sown activations come back in the side dict. We do **not** reuse
`engine.prefill` because that function is entangled with sampling,
multimodal inputs, paged-attention bookkeeping, and the
"first generated token" path — all of which extraction does not need.
A standalone JIT trace lets the extraction binary skip them entirely.

```python
@jax.jit
def _forward(_unused, padded_tokens, true_length):
    input_tokens = padded_tokens[None, :]
    positions = jnp.arange(input_tokens.shape[1])[None, :]
    seg = (jnp.arange(input_tokens.shape[1]) < true_length)[None, :].astype(jnp.int32)
    _logits, new_vars = model.apply(
        params, input_tokens, positions,
        decoder_segment_ids=seg,
        enable_dropout=False,
        model_mode=MODEL_MODE_PREFILL,
        rngs={"params": jax.random.PRNGKey(0)},
        mutable=["intermediates"],
        true_length=true_length,
    )
    flat = {}
    _collect_by_hook(new_vars["intermediates"], wanted, flat)
    return flat
```

### Intermediates walker

Flax stores sown intermediates differently depending on whether the
decoder is scanned:

- `scan_layers=True`:
  `intermediates["layers"]["residual_post"]` is a length-1 tuple whose
  element has shape `[num_layers, B, T, D]`.
- `scan_layers=False`: each `layers_<i>` submodule has its own
  `["residual_post"]` length-1 tuple of `[B, T, D]` arrays.

`_collect_by_hook` recursively walks the dict, collecting all matches
in numerical layer order. The downstream helper `select_layer_axis`
accepts every concrete layout (single stacked array, length-1 tuple
wrapping a stacked array, tuple of per-layer arrays) and returns a
uniform `[K, B, T, D]` view of just the requested layers.

## Output format

```
{output_path}/
├── host_00/                    # one dir per JAX process (multi-host)
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

Each shard contains four safetensors tensors:

| Tensor | Shape | dtype |
|---|---|---|
| `activations` | `(N, d_model)` | configured output dtype |
| `token_ids`   | `(N,)`         | `int32` |
| `positions`   | `(N,)`         | `int32` |
| `doc_ids`     | `(N,)`         | `int64` |

`positions` is the offset of the token in its sequence (0-indexed) and
`doc_ids` is the global document index assigned by the dataset
backend. Together they form a primary key into the source text —
enough for downstream tools (Delphi, Neuronpedia) to reconstruct
context.

Shards are written **atomically**: the writer streams to a `.partial`
sibling and then renames; on GCS this is a copy-and-delete handled by
`etils.epath`, on local disks it is an atomic rename. There is no
in-flight state on disk that a reader could see as a torn write.

## Multi-host correctness

The headline correctness property is:

> For a fixed dataset `D`, a fixed checkpoint `C`, and the same
> sharding intent, the union of shards produced by running with `P`
> processes on `D` is **bit-identical** as a multiset of
> `(doc_id, position) → (activation, token_id)` rows to the output of
> a single-process run on `D`.

The mechanisms that guarantee this:

1. Document IDs are **global** — the dataset backend assigns
   `doc_id = i` to the `i`-th input document, independent of which
   process reads it.
2. The host-aware iterator partitions documents by `doc_id mod P`, so
   every document is read by exactly one process and the union is
   exactly `D`.
3. Activations are written in iteration order; the merged read
   (`load_merged`) sorts by `(doc_id, position)` lexicographically,
   producing a canonical ordering that is independent of `P`.
4. JAX SPMD is deterministic for the same global batch on the same
   mesh — so the activation values themselves do not depend on process
   count.

This is the property exercised by
`tests/integration/extract_activations/multihost_equivalence_test.py`,
which runs the loop with `P=1`, `P=2`, and `P=4` on identical input
and asserts row-for-row equality of every per-layer shard.

## Scan vs unscanned equivalence

A second correctness property is that the merged output is identical
between `scan_layers=True` and `scan_layers=False` models with the
same parameters. This is tested by
`tests/integration/extract_activations/real_forward_test.py`, which
builds a toy `nn.scan`-wrapped decoder and an unscanned reference,
copies parameters between them, and asserts the merged shards are
numerically equal.

This matters because production checkpoints are typically saved with
`scan_layers=True` (it saves memory at training time), but
interpretability tooling sometimes prefers an unscanned forward for
easier inspection.

## MoE handling

For MoE models (Qwen3-30B-A3B, DeepSeek, MiniMax M2.5), the hook
captures the residual stream **after** routed and shared experts have
been summed back in:

```
layer_output = inputs + attention_lnx + (routed_experts + shared_experts)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        captured by residual_post
```

This is the convention adopted by Qwen-Scope (the canonical SAE
release for an MoE model) and treats the residual stream uniformly
across dense and MoE families. Per-expert hooks are an explicit
non-goal for v1; "Switch SAEs" (Mukherjee et al. 2024) put MoE
structure inside the SAE itself, which is the modern alternative to
extracting per-expert activations.

## Performance envelope

For Qwen3-30B-A3B (d=2048, 48 layers, bf16, 3 layers × 1 hook):

- **Per-token storage**: 1 hook × 3 layers × 2048 × 2 B = **12 KB / token**.
- **200 M tokens** ⇒ 2.4 TB.
- At a sustained GCS write of 1 GB/s ⇒ ~40 minutes of pure I/O.
- Prefill throughput on v6e-64 at `batch=8`, `seq=2048` ≈ 2 800 tok/s
  ⇒ ~20 hours of compute.

The pipeline is compute-bound, not I/O-bound — which is the expected
shape and means raising `per_device_batch_size` (once vectorised
prefill lands) directly translates into wall-clock speedup.

## Risks and follow-ups

| Risk | Status |
|---|---|
| `sow()` + `with_sharding_constraint` inside `nn.scan` | Exercised by `hooks_test.py` on a scanned 4-layer toy model |
| MaxEngine API drift breaking the JIT'd forward | `ExtractionLoop` is decoupled from MaxEngine; only `Runner` is affected |
| Quantised checkpoints (int4 MiniMax) | Residual stream is still bf16/fp32 inside the model; smoke-test pending on real checkpoint |
| GCS bandwidth on v6e-64 | bf16 halves bytes vs fp32; sharded streaming + multi-part uploads keep up |
| Pipeline parallelism | Explicitly unsupported in v1; runner asserts `not config.using_pipeline_parallelism` |
| MoE per-expert hooks | v2 candidate |
| Vectorised prefill (batch > 1 per JIT) | v2 candidate; major throughput improvement |
| Resume from partial run | v1.5 candidate; per-layer `metadata.json` already records counts |

## See also

- [Usage guide](usage.md) — running the tool and consuming output.
- [Optimization guide](../optimization.md) — applies to the prefill
  path.
- [Distillation guide](../distillation.md) — another consumer of
  MaxText's `sow("intermediates", …)` machinery.
