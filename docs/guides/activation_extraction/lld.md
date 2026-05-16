# Activation Extraction — Low-Level Design

This document covers the concrete interfaces, data structures, and
algorithms.  Companion: [`hld.md`](./hld.md).

## 1. Config additions

`src/maxtext/configs/types.py` adds a nested `ActivationExtractionConfig`
to the top-level `HyperParameters`:

```python
class ActivationExtractionConfig(BaseModel):
    enabled: bool = Field(
        False,
        description="When True, DecoderLayer sows full activation tensors "
                    "into the 'intermediates' collection and MaxEngine "
                    "captures them.")
    layers: list[int] = Field(
        [],
        description="Layer indices to keep when writing to disk. "
                    "Negative indices count from the top of the stack. "
                    "Empty means [num_decoder_layers // 4, "
                    "num_decoder_layers // 2, 3 * num_decoder_layers // 4].")
    hooks: list[str] = Field(
        ["residual_post"],
        description="Hook points to capture per layer. "
                    "Subset of {'residual_post', 'mlp_out', 'attn_out'}.")
    output_path: str = Field(
        "",
        description="Local directory or gs:// path for shard output. "
                    "Required when enabled is True.")
    shard_size_tokens: int = Field(
        2_000_000,
        description="Tokens per safetensors shard. ~4 GB at d_model=2048 bf16.")
    output_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        "bfloat16", description="Storage dtype.")
    max_tokens: int = Field(
        0,
        description="Stop after this many tokens have been written per layer "
                    "(0 = no limit).")
    skip_bos: bool = Field(
        True,
        description="If True, drop the first token of each sequence "
                    "(BOS has uninformative residual).")
```

Plus the corresponding entries with defaults in `configs/base.yml` under a
top-level `activation_extraction:` key.

## 2. Hook injection in `DecoderLayer`

In `src/maxtext/layers/decoders.py` we add **one** guarded block right
after the existing `record_internal_nn_metrics` block (around line 217):

```python
if cfg.activation_extraction.enabled:
  hooks = set(cfg.activation_extraction.hooks)
  if "residual_post" in hooks:
    self.sow("intermediates", "residual_post", layer_output)
  if "mlp_out" in hooks:
    self.sow("intermediates", "mlp_out", mlp_lnx)
  if "attn_out" in hooks:
    self.sow("intermediates", "attn_out", attention_lnx)
```

Behaviour under `scan_layers=True`: because the scan wrapper in
`decoders.py` declares
`variable_axes={"intermediates": 0, ...}`,
the runner sees a single stacked tensor of shape
`[num_layers, B, T, D]` per hook name.  Under `scan_layers=False` the
intermediates dict has per-layer keys `layers_<idx>/<hook>`.  The runner
handles both shapes (see `runner._select_layers`).

The hook is a no-op (the dead-code branch is eliminated by JAX after
tracing) when the flag is off, so the inference and training paths are
unaffected.

The full residual stream activation inherits the same logical sharding
that `layer_output` already has (`activation_batch, activation_length,
activation_embed`); we do not re-shard it.

## 3. MaxEngine integration

In `src/maxtext/inference/maxengine/maxengine.py` we modify exactly one
call site in `_prefill_jit` (and `_prefill_jit_batched`, if added):

```python
mutable_keys = ["cache"]
if self.config.activation_extraction.enabled:
  mutable_keys.append("intermediates")
flat_logits, new_vars = self.model.apply(
    input_params, input_tokens, positions,
    enable_dropout=False, model_mode=MODEL_MODE_PREFILL,
    rngs={"params": new_rng},
    mutable=mutable_keys,
)
```

The runner reads `new_vars["intermediates"]` and processes it.  We also
add a new public method:

```python
def prefill_with_activations(
    self,
    *,
    params,
    padded_tokens: jax.Array,    # [B, T_pad] or [T_pad]
    true_lengths: jax.Array,     # [B]
    rng: PRNGKeyType,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Return: (intermediates_dict, logits). Skips sampling/cache export."""
```

This method bypasses the autoregressive sampling logic that
`engine.prefill` does for the “first generated token”.  It is JIT-cached
separately from `_prefill_jit` so that the activation extraction binary
does not pay the cost of the sampling code path.

## 4. `ShardedSafetensorsWriter`

```python
class ShardedSafetensorsWriter:
    def __init__(
        self,
        output_path: str,
        layers: list[int],
        hooks: list[str],
        d_model: int,
        shard_size_tokens: int,
        dtype: jnp.dtype,
        process_index: int,
        model_name: str,
    ): ...

    def append(
        self,
        hook: str,
        layer_idx: int,
        activations: np.ndarray,  # [N, D]
        token_ids: np.ndarray,    # [N]
        positions: np.ndarray,    # [N]
        doc_ids: np.ndarray,      # [N]
    ) -> None: ...

    def finalize(self) -> None:
        """Flush any partial shard, write manifest.json + per-layer
        metadata.json, close any open GCS handles."""
```

Behaviour:

* Only `process_index == 0` writes; other processes treat all calls as
  no-ops.  (The runner already gathered to host 0 before calling.)
* Each `(hook, layer_idx)` pair has its own buffer, shard counter and
  output directory `output_path/<hook>/layer_<idx>/`.
* A shard is flushed when its accumulated rows reach `shard_size_tokens`.
* Each shard file is a safetensors file with four tensors:
  `activations` (dtype = `output_dtype`, shape `[N, D]`),
  `token_ids` (int32, `[N]`), `positions` (int32, `[N]`),
  `doc_ids` (int64, `[N]`).
* Filenames are zero-padded `shard_00000.safetensors`,
  `shard_00001.safetensors`, …  Writes are atomic
  (write to `.tmp`, then rename / GCS-copy).
* GCS paths are handled via `tf.io.gfile` or the existing
  `epath` (`etils.epath`) used by MaxText for Orbax.
* `manifest.json` keys:
  `{"model_name", "d_model", "dtype", "hooks", "layers",
    "total_tokens_per_layer", "shards_per_layer", "created_at",
    "maxtext_commit_sha"}`.

## 4a. Per-host write layout (revised in v1 implementation)

After exploring the cross-host gather approach we settled on a simpler,
preemption-resilient design: **every host writes its own
``host_NN/`` subdirectory** rather than gathering everything to host 0.

  * No JAX-distributed dependency in the writer.
  * A crashing host does not lose the others' work.
  * The merged dataset is recovered offline by ``load_merged()``, which
    concatenates all ``host_NN/<hook>/layer_NNNN/`` shards and sorts by
    ``(doc_id, position)``.

This is bit-equivalent to the gather approach for downstream SAE
training because the sort order is canonical, and is the design used in
production by the user's existing ``activation-extract`` project.

## 5. Runner algorithm

```python
def Runner.run(self):
    self._init_jax_distributed()
    config = self.config
    engine = MaxEngine(config)
    params = engine.load_params(self.rng_load)
    tokenizer = engine.build_tokenizer(engine.get_tokenizer())

    layers_to_keep = self._resolve_layers(config)
    writer = ShardedSafetensorsWriter(
        output_path=config.activation_extraction.output_path,
        layers=layers_to_keep,
        hooks=config.activation_extraction.hooks,
        d_model=config.emb_dim,
        shard_size_tokens=config.activation_extraction.shard_size_tokens,
        dtype=_jnp_dtype(config.activation_extraction.output_dtype),
        process_index=jax.process_index(),
        model_name=config.model_name,
    )

    dataset = self._build_dataset(tokenizer)
    tokens_written = 0
    for batch in dataset.iter_host_local(jax.process_index(),
                                         jax.process_count()):
        padded, true_lens, token_ids, doc_ids = self._pack(batch, tokenizer)
        intermediates, _ = engine.prefill_with_activations(
            params=params,
            padded_tokens=padded,
            true_lengths=true_lens,
            rng=self.rng_step,
        )
        for hook in config.activation_extraction.hooks:
            stacked = self._gather_hook(intermediates, hook)   # [L, B, T, D]
            picked = stacked[jnp.array(layers_to_keep)]        # [K, B, T, D]
            gathered = multihost_utils.process_allgather(
                picked, tiled=True)                            # full batch
            if jax.process_index() == 0:
                acts_np = np.asarray(gathered)
                for k, layer_idx in enumerate(layers_to_keep):
                    self._dispatch_one(
                        writer, hook, layer_idx,
                        acts_np[k],
                        token_ids_full, positions_full, doc_ids_full,
                        true_lens_full,
                    )
        tokens_written += int(true_lens_full.sum())
        if (config.activation_extraction.max_tokens and
            tokens_written >= config.activation_extraction.max_tokens):
            break
    writer.finalize()
    multihost_utils.sync_global_devices("extract_activations:done")
```

`_dispatch_one` masks padding (`pos < true_len`), optionally drops BOS
(`pos > 0` when `skip_bos`), and flattens `[B, T, D] → [N, D]`.

The reason we `process_allgather` *every* batch is so that host 0 sees a
deterministic, contiguous view of the global batch in the same order the
runner enumerated documents.  Document IDs are assigned by the dataset
loader (globally monotonic), and the host-aware iterator emits disjoint
slices, so concatenated output across all hosts is identical to a single
host running the full dataset.

## 6. Dataset backends

```python
class DatasetBackend(Protocol):
    def iter_host_local(self, process_index: int,
                        process_count: int) -> Iterator[list[Document]]: ...

@dataclass
class Document:
    doc_id: int
    text: str | None       # for tokenizer-on-the-fly
    token_ids: list[int] | None  # for pre-tokenized
```

v1 backends:

1. `JsonlBackend(path)` — one JSON object per line with `text` or `tokens`.
   `doc_id` = global line index.  Host `p` of `P` reads lines where
   `line_idx % P == p`.
2. `HuggingFaceBackend(name, split, text_key)` — wraps
   `datasets.load_dataset(..., streaming=True)`.  Same modulo-`P` sharding.
3. `PreTokenizedBackend(path)` — `.npy` file of `int32` tokens already
   chunked into `max_prefill_predict_length` rows.

## 7. Multi-host correctness

The contract is: for a fixed dataset `D`, a fixed checkpoint `C`, and the
same sharding intent (FSDP×TP product over `P` processes), the union of
shards produced by `(D, C, P=1)` and `(D, C, P=2, 4, …)` are equal as
multisets when keyed by `(doc_id, position)`.

The mechanisms that guarantee this:

* `doc_id` is the global document index (not host-local).
* Host-aware iterator uses modulo `P` sharding — every document is read
  by exactly one host.
* `process_allgather` is order-preserving along the gather axis when
  `tiled=True`, so host 0 sees `[h0_docs ++ h1_docs ++ …]` per batch.
* Writer dispatches in iteration order; shards are named in increasing
  order; final sort by `(doc_id, position)` yields a deterministic
  ordering identical to single-host.

The integration test
`tests/integration/extract_activations_multihost_test.py` exercises this
on a tiny 4-layer model.

## 8. Test plan

### 8.1 Unit (`tests/unit/extract_activations/`)

* `test_writer_roundtrip.py`
  Append known activations to a writer with `shard_size_tokens=100`,
  read back all shards, assert shape / dtype / data equality and that
  shards do not exceed the configured token budget.

* `test_writer_manifest.py`
  Manifest fields populated correctly; per-layer counts match shards.

* `test_writer_atomic.py`
  Crash mid-write (raise inside `append`); on next run the partial
  `.tmp` is cleaned and no half-shard appears.

* `test_hook_injection_dense.py`
  Build a toy 4-layer dense Decoder, run a forward pass with
  `activation_extraction.enabled=True`, assert intermediates contains
  `residual_post`, `mlp_out`, `attn_out`, and that shapes match
  `[L, B, T, D]` (scan) or per-layer keys (no-scan).

* `test_hook_off_is_noop.py`
  With `enabled=False`, intermediates collection is empty and the
  forward-pass HLO is unchanged from baseline (compared by HLO hash).

* `test_layer_selection.py`
  `_resolve_layers` handles empty default, negative indices, and
  out-of-range error.

* `test_dataset_jsonl_sharding.py`
  P=4 processes each read disjoint, complete subsets; union is the full
  file.

* `test_skip_bos_and_padding_mask.py`
  Padding positions never appear in output; BOS skipped iff configured.

* `test_dtype_cast.py`
  Output dtype matches config; bf16 round-trip is lossless for values
  representable in bf16.

### 8.2 Integration (`tests/integration/`)

* `extract_activations_single_host_test.py`
  Run extraction on a tiny model + 64 sequences on a single host;
  assert shard contents match a direct golden forward pass that
  recomputes the residual stream manually.

* `extract_activations_multihost_test.py`  **(key correctness test)**

  Same tiny model, same input set, two runs:

  1. Single-process: `P=1`.
  2. Two-process: `P=2` launched via
     `subprocess.Popen` with `--num_processes 2`.

  After both runs, load every shard from both output dirs, sort by
  `(doc_id, position)`, and assert:

  ```python
  np.testing.assert_array_equal(single_acts, merged_acts)
  np.testing.assert_array_equal(single_token_ids, merged_token_ids)
  ```

  We rely on JAX SPMD determinism: the same global batch on the same
  mesh produces bitwise-identical activations regardless of process
  count.  The test runs on CPU mesh `(2,)` so it is hermetic.

* `extract_activations_moe_test.py`
  Run extraction on a 2-layer MoE configuration; assert
  `residual_post` is post-block (matches manual recompute that adds
  the routed-expert output to inputs).

* `extract_activations_scan_vs_unscan_test.py`
  Same model with `scan_layers=True` vs `False`; assert the merged
  output is identical.

### 8.3 Smoke (`tests/end_to_end/extract_activations/`)

* Bash script that runs against a real GCS-resident checkpoint of a
  small open-weights model (e.g. Llama 3 1B), 100 prompts, writes
  to a temp GCS prefix, and verifies manifest counts.

## 9. CLI invocation

```bash
python -m maxtext.tools.extract_activations.main \
  src/maxtext/configs/base.yml \
  model_name=qwen3-30b-a3b \
  load_parameters_path=gs://arc-data-europe-west4/checkpoints/qwen3-30b-a3b/items/0 \
  activation_extraction.enabled=True \
  activation_extraction.layers="[12,24,36]" \
  activation_extraction.hooks="[residual_post]" \
  activation_extraction.output_path=gs://arc-data-europe-west4/sae/qwen3-30b-a3b/v1 \
  activation_extraction.shard_size_tokens=2000000 \
  dataset_type=hf \
  hf_path=monology/pile-uncopyrighted \
  per_device_batch_size=1 \
  max_prefill_predict_length=2048
```

## 10. Failure modes and recovery

* **OOM during gather**: drop `per_device_batch_size` or
  `max_prefill_predict_length`; alternatively set
  `activation_extraction.hooks=[residual_post]` only.
* **Partial shard on preemption**: `manifest.json` is written only on
  `finalize`; on resume, the runner can skip already-finished shards by
  consulting per-layer `metadata.json`.  (Resume implementation lives
  in v1.5.)
* **GCS rate limits**: writer uses sequential shard uploads with
  exponential backoff already baked into `epath`.

## 11. Performance back-of-envelope

For Qwen3-30B-A3B (d=2048, 48 layers, bf16, 3 hooks × 3 layers):

* Per-token activation bytes on disk: `3 hooks × 3 layers × 2048 × 2 B`
  = **36 KB / token**.
* 200 M tokens ⇒ 7.2 TB.
* At a sustained GCS write of 1 GB/s ⇒ ~2 hours of pure I/O.
* Prefill throughput on v6e-64 at batch=8, seq=2048 ≈ 2 800 tok/s ⇒
  ~20 hours of compute.

Compute-bound, not I/O-bound — matches expectations.
