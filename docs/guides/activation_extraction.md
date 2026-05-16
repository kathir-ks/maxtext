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

(activation_extraction)=

# Activation Extraction

MaxText ships an offline tool for capturing **intermediate activations** from
any trained model — post-block residual stream, MLP output, or attention
output — for use in Sparse Autoencoder (SAE) training and mechanistic
interpretability research.

The output format is drop-in compatible with the established open-source SAE
training stack ([EleutherAI `sparsify`](https://github.com/EleutherAI/sparsify),
[`sae_lens`](https://github.com/jbloomAus/SAELens)): safetensors shards of
flattened `(N, d_model)` activations with `token_ids` / `positions` / `doc_ids`
sidecars, one directory per `(hook, layer)` pair.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 🚀 Usage
:link: activation_extraction/usage
:link-type: doc

Quick start, CLI invocation, output layout, reading back shards, and integration
with `sparsify` / `sae_lens`.
:::

:::{grid-item-card} 🧩 Design
:link: activation_extraction/design
:link-type: doc

How the tool hooks into model internals via Flax `sow`, the multi-host write
layout, MoE handling, and the correctness contract verified by the multi-host
equivalence test.
:::

::::

## At a glance

- **Works for every MaxText model family** — Llama 2/3/4, Qwen 2.5 / 3 / 3-MoE,
  Mistral, Mixtral, Gemma / 2 / 3 / 4, DeepSeek, MiniMax, GPT-3, GPT-OSS, Olmo.
- **Zero impact when disabled.** When `activation_extraction_enabled=False`
  the inference and training HLO is unchanged (the guarded `sow` calls are
  dead-code eliminated after Flax tracing).
- **Dense and Mixture-of-Experts.** The `residual_post` hook captures the
  post-block residual *after* routed + shared experts have been folded in,
  uniform across dense and MoE models (the Qwen-Scope / Gemma-Scope
  convention).
- **Preemption resilient.** Each JAX process writes its own `host_NN/`
  subdirectory; a crashing host does not lose the others' work. Merged with
  {py:func}`~maxtext.tools.extract_activations.load_merged` at read time.
- **Multi-host equivalence guarantee.** For the same dataset and checkpoint,
  the merged shards are bit-identical regardless of process count — verified
  by `tests/integration/extract_activations/multihost_equivalence_test.py`.

## Related

- [Optimization guide](optimization.md) — sharding strategies and performance
  tuning that apply to the prefill path used by extraction.
- [Distillation guide](distillation.md) — another use of MaxText's
  `sow("intermediates", ...)` machinery; uses the same scan-aware mechanism
  this tool extends.

```{toctree}
---
hidden:
maxdepth: 1
---
activation_extraction/usage.md
activation_extraction/design.md
```
