# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Activation extraction runner.

This module contains the data-plane logic that is independent of the
MaxText inference engine, plus the orchestrator that wires MaxEngine
and the writer together. The ``ExtractionLoop`` class is fully testable
without JetStream, a TPU, or a real checkpoint: callers inject a
``forward_fn`` and a tokenizer.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from maxtext.tools.extract_activations import hooks as hook_constants
from maxtext.tools.extract_activations.datasets import (
    DatasetBackend,
    Document,
    build_backend,
)
from maxtext.tools.extract_activations.writer import ShardedSafetensorsWriter

logger = logging.getLogger(__name__)


# A forward-pass callable: (params, padded_tokens [T_pad], true_length:int)
# -> dict[hook_name, jax.Array of shape [num_layers, 1, T_pad, d_model]].
ForwardFn = Callable[[Any, jax.Array, int], dict[str, jax.Array]]


@dataclasses.dataclass
class ExtractionConfig:
  """Subset of MaxText config that the loop actually needs.

  Keeping this small makes tests easy to construct without spinning up
  the full ``pyconfig`` machinery.
  """
  hooks: list[str]
  layers_to_keep: list[int]
  d_model: int
  num_layers: int
  max_prefill_predict_length: int
  output_path: str
  shard_size_tokens: int
  output_dtype: str = "bfloat16"
  max_tokens: int = 0
  skip_bos: bool = True
  model_name: str = ""


# ----- pure helpers -----


def resolve_layers(requested: list[int], num_layers: int) -> list[int]:
  """Expand defaults and negative indices; validate the result."""
  if not requested:
    return [num_layers // 4, num_layers // 2, (3 * num_layers) // 4]
  out: list[int] = []
  for li in requested:
    norm = li if li >= 0 else num_layers + li
    if norm < 0 or norm >= num_layers:
      raise ValueError(
          f"layer index {li} out of range for num_layers={num_layers}"
      )
    out.append(norm)
  # Preserve user order, drop duplicates.
  seen = set()
  unique = []
  for li in out:
    if li in seen:
      continue
    seen.add(li)
    unique.append(li)
  return unique


def select_layer_axis(
    intermediates_for_hook,
    *,
    layer_indices: list[int],
    num_layers: int,
) -> jax.Array:
  """Convert sow output to a uniform ``[K, B, T, D]`` tensor.

  Accepts any of the layouts Flax can produce for a sown intermediate:

    * single array of shape ``[L, B, T, D]`` (the scan case).
    * single array of shape ``[1, L, B, T, D]`` (some Flax versions add
      a length-1 leading axis around scanned intermediates).
    * tuple ``(arr,)`` with ``arr`` of shape ``[L, B, T, D]`` (the most
      common scan case — Flax wraps each sow in a length-1 tuple).
    * tuple of ``L`` arrays of shape ``[B, T, D]`` (the unscanned case
      where the walker collected per-layer sows).

  Args:
    intermediates_for_hook: one of the layouts above.
    layer_indices: which layer indices to keep, in order.
    num_layers: total number of decoder layers (for shape validation).

  Returns:
    Array of shape ``[len(layer_indices), B, T, D]``.
  """
  if isinstance(intermediates_for_hook, (list, tuple)):
    # Length-1 tuple: either wraps a stacked [L, B, T, D] array (scan)
    # or wraps another tuple. Peel.
    if len(intermediates_for_hook) == 1:
      inner = intermediates_for_hook[0]
      if isinstance(inner, (list, tuple)):
        return select_layer_axis(
            inner, layer_indices=layer_indices, num_layers=num_layers,
        )
      arr = jnp.asarray(inner)
      if arr.ndim >= 4 and arr.shape[0] == num_layers:
        stacked = arr
      else:
        raise ValueError(
            f"length-1 hook tuple wraps array of shape {arr.shape}; "
            f"expected leading dim {num_layers}"
        )
    else:
      arrs = [jnp.asarray(x) for x in intermediates_for_hook]
      if len(arrs) != num_layers:
        raise ValueError(
            f"hook tuple length {len(arrs)} != num_layers {num_layers}"
        )
      stacked = jnp.stack(arrs, axis=0)
  else:
    stacked = jnp.asarray(intermediates_for_hook)
    if stacked.ndim == 5 and stacked.shape[0] == 1:
      stacked = stacked[0]
    if stacked.shape[0] != num_layers:
      raise ValueError(
          f"stacked intermediates have shape {stacked.shape}; "
          f"expected leading dim {num_layers}"
      )
  return stacked[jnp.array(layer_indices)]


def mask_and_flatten(
    activations: np.ndarray,  # [B, T, D]
    token_ids: np.ndarray,    # [B, T]
    true_lengths: np.ndarray, # [B]
    doc_ids: np.ndarray,      # [B]
    *,
    skip_bos: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Drop padding (and optionally BOS), then flatten the batch axis.

  Returns: (acts[N,D], token_ids[N], positions[N], doc_ids[N]).
  """
  if activations.ndim != 3:
    raise ValueError(f"activations must be 3-D, got {activations.shape}")
  B, T, D = activations.shape
  if token_ids.shape != (B, T):
    raise ValueError(f"token_ids shape {token_ids.shape} != ({B},{T})")
  if true_lengths.shape != (B,):
    raise ValueError(f"true_lengths shape {true_lengths.shape} != ({B},)")
  if doc_ids.shape != (B,):
    raise ValueError(f"doc_ids shape {doc_ids.shape} != ({B},)")
  pos_grid = np.arange(T, dtype=np.int32)[None, :]            # [1, T]
  doc_grid = np.broadcast_to(doc_ids[:, None], (B, T))         # [B, T]
  valid = pos_grid < true_lengths[:, None]                     # [B, T]
  if skip_bos:
    valid = valid & (pos_grid > 0)
  flat_mask = valid.reshape(-1)
  flat_acts = activations.reshape(B * T, D)[flat_mask]
  flat_tids = token_ids.reshape(-1)[flat_mask]
  flat_pos = np.broadcast_to(pos_grid, (B, T)).reshape(-1)[flat_mask]
  flat_did = doc_grid.reshape(-1)[flat_mask]
  return flat_acts, flat_tids, flat_pos, flat_did


def pad_sequence(
    token_ids: list[int],
    *,
    max_length: int,
    pad_id: int,
) -> tuple[np.ndarray, int]:
  """Pad/truncate a list of token ids to ``max_length``. Returns
  ``(padded, true_length)``."""
  true_length = min(len(token_ids), max_length)
  arr = np.full((max_length,), pad_id, dtype=np.int32)
  arr[:true_length] = token_ids[:true_length]
  return arr, true_length


# ----- the loop itself -----


class ExtractionLoop:
  """Pure data-plane: orchestrates tokenize → forward → gather → write.

  No MaxEngine / JetStream dependencies. Test-friendly.
  """

  def __init__(
      self,
      *,
      config: ExtractionConfig,
      forward_fn: ForwardFn,
      tokenize_fn: Callable[[str], list[int]],
      pad_id: int,
      dataset: DatasetBackend,
      writer: ShardedSafetensorsWriter,
      process_index: int,
      process_count: int,
  ):
    self._config = config
    self._forward_fn = forward_fn
    self._tokenize_fn = tokenize_fn
    self._pad_id = pad_id
    self._dataset = dataset
    self._writer = writer
    self._process_index = process_index
    self._process_count = process_count
    self._tokens_seen_per_layer = 0

  def run(self) -> dict[str, int]:
    """Run the extraction loop to completion. Returns counters."""
    cfg = self._config
    docs_iter = self._dataset.iter_for_process(
        self._process_index, self._process_count
    )
    seen = 0
    for doc in docs_iter:
      tokens = self._materialize_tokens(doc)
      padded, true_length = pad_sequence(
          tokens, max_length=cfg.max_prefill_predict_length, pad_id=self._pad_id
      )
      intermediates = self._forward_fn(None, jnp.asarray(padded), true_length)

      # Per-host write: each process dispatches its own document directly
      # to its own writer (which is configured with per_host_subdir=True
      # when process_count > 1). The merged dataset is the union of all
      # per-host directories.
      gathered = self._gather_one_doc(
          intermediates=intermediates,
          padded_tokens=padded,
          true_length=true_length,
          doc_id=doc.doc_id,
      )
      self._dispatch(gathered)
      if (
          cfg.max_tokens
          and self._tokens_seen_per_layer >= cfg.max_tokens
      ):
        break
      seen += 1
    self._writer.finalize()
    return {
        "documents_seen": seen,
        "tokens_written_per_layer": self._tokens_seen_per_layer,
    }

  # ----- per-step helpers -----

  def _materialize_tokens(self, doc: Document) -> list[int]:
    if doc.token_ids is not None:
      return doc.token_ids
    return self._tokenize_fn(doc.text or "")

  def _gather_one_doc(
      self,
      *,
      intermediates: dict[str, jax.Array],
      padded_tokens: np.ndarray,
      true_length: int,
      doc_id: int,
  ) -> dict[str, np.ndarray]:
    """Assemble one doc's activations as host-local numpy arrays.

    Each host writes its own disjoint slice (``host_NN`` subdirectory),
    so we do not cross-host gather here. The merged dataset is the
    concatenation of all per-host directories, which the
    :func:`merge_per_host_outputs` helper performs offline.

    The forward returns ``[K, 1, T, D]`` for each hook. We return
    matching numpy views for downstream masking + writing.
    """
    cfg = self._config
    out: dict[str, np.ndarray] = {}
    token_ids = np.asarray(padded_tokens)[None, :]                # [1, T]
    true_lens = np.asarray([true_length], dtype=np.int32)         # [1]
    doc_ids = np.asarray([doc_id], dtype=np.int64)                # [1]
    for hook in cfg.hooks:
      stacked = select_layer_axis(
          intermediates[hook],
          layer_indices=cfg.layers_to_keep,
          num_layers=cfg.num_layers,
      )  # [K, 1, T, D]
      out[hook] = np.asarray(stacked)
    out["__token_ids__"] = token_ids
    out["__true_lengths__"] = true_lens
    out["__doc_ids__"] = doc_ids
    return out

  def _dispatch(self, gathered: dict[str, np.ndarray]) -> None:
    cfg = self._config
    token_ids = gathered["__token_ids__"]      # [B, T]
    true_lens = gathered["__true_lengths__"]   # [B]
    doc_ids = gathered["__doc_ids__"]          # [B]
    for hook in cfg.hooks:
      stacked = gathered[hook]  # [K, B, T, D]
      for k, layer_idx in enumerate(cfg.layers_to_keep):
        acts_btd = stacked[k]  # [B, T, D]
        flat_acts, flat_tids, flat_pos, flat_did = mask_and_flatten(
            acts_btd, token_ids, true_lens, doc_ids, skip_bos=cfg.skip_bos
        )
        if flat_acts.shape[0] == 0:
          continue
        self._writer.append(
            hook=hook,
            layer_idx=layer_idx,
            activations=flat_acts,
            token_ids=flat_tids,
            positions=flat_pos,
            doc_ids=flat_did,
        )
        if hook == cfg.hooks[0] and layer_idx == cfg.layers_to_keep[0]:
          self._tokens_seen_per_layer += int(flat_acts.shape[0])


# ----- production wiring (MaxEngine-backed) -----


def _build_forward_fn_from_maxengine(engine, params) -> tuple[ForwardFn, int]:
  """Return a JIT-compiled forward function that runs prefill and yields
  the ``intermediates`` collection plus the pad id from the tokenizer.
  """
  from maxtext.common.common_types import MODEL_MODE_PREFILL  # local import

  config = engine.config
  model = engine.model

  @jax.jit
  def _forward(_unused, padded_tokens: jax.Array, true_length: int):
    del _unused
    input_tokens = jnp.expand_dims(padded_tokens, 0)  # [1, T]
    positions = jnp.expand_dims(
        jnp.arange(input_tokens.shape[1], dtype=jnp.int32), 0
    )
    start_to_n = jnp.arange(input_tokens.shape[1], dtype=jnp.int32)
    ones_to_keep = start_to_n < true_length
    seg = jnp.expand_dims(ones_to_keep.astype(jnp.int32), 0)
    _logits, new_vars = model.apply(
        params,
        input_tokens,
        positions,
        decoder_segment_ids=seg,
        enable_dropout=False,
        model_mode=MODEL_MODE_PREFILL,
        rngs={"params": jax.random.PRNGKey(0)},
        mutable=[hook_constants.INTERMEDIATES_COLLECTION],
        true_length=true_length,
    )
    # Walk the intermediates pytree to flatten by hook name.
    inter = new_vars[hook_constants.INTERMEDIATES_COLLECTION]
    flat: dict[str, jax.Array] = {}
    _collect_by_hook(inter, set(config.activation_extraction_hooks), flat)
    return flat

  return _forward


def _collect_by_hook(tree, wanted: set[str], out: dict[str, Any]) -> None:
  """Recursively walk Flax intermediates to find sown hook tensors.

  The Flax 'intermediates' collection has two shapes here:

    * ``scan_layers=True``: ``intermediates['layers']['residual_post']``
      is a length-1 tuple whose element has shape ``[num_layers, ...]``.
      A single occurrence of the hook key.
    * ``scan_layers=False``: each layer module
      (``layers_0``, ``layers_1``, …) has its own
      ``['residual_post']`` length-1 tuple of per-layer arrays.

  This walker collects *all* hits per hook name in encounter order. The
  caller (``select_layer_axis``) then handles either single-stacked-array
  inputs or tuples of per-layer arrays uniformly.
  """
  collected: dict[str, list[Any]] = {h: [] for h in wanted}
  _walk(tree, wanted, collected, parent_key="")
  for hook, items in collected.items():
    if not items:
      continue
    if len(items) == 1:
      out[hook] = items[0]
    else:
      # Each ``items[i]`` is what Flax stored, usually a length-1 tuple
      # ``(arr,)``. Unwrap and present as a per-layer tuple.
      unwrapped = [(it[0] if isinstance(it, tuple) and len(it) == 1 else it)
                   for it in items]
      out[hook] = tuple(unwrapped)


def _walk(tree, wanted: set[str], collected: dict[str, list[Any]],
          parent_key: str) -> None:
  if isinstance(tree, dict):
    # Sort keys so that ``layers_0`` precedes ``layers_1`` etc. — Flax dicts
    # are insertion-ordered but we don't rely on it.
    items = sorted(
        tree.items(),
        key=lambda kv: _layer_sort_key(kv[0]),
    )
    for k, v in items:
      if k in wanted:
        collected[k].append(v)
      else:
        _walk(v, wanted, collected, parent_key=k)


def _layer_sort_key(name: str):
  """Sort ``layers_<i>`` numerically, anything else lexicographically."""
  if name.startswith("layers_"):
    try:
      return (0, int(name[len("layers_"):]))
    except ValueError:
      return (0, name)
  if name == "layers":
    return (0, -1)
  return (1, name)


class Runner:
  """Production runner: wires MaxEngine, the writer, and the loop."""

  def __init__(self, config):
    self._config = config

  def run(self) -> None:
    cfg = self._config
    if not cfg.activation_extraction_enabled:
      raise RuntimeError("activation_extraction_enabled is False.")
    if not cfg.activation_extraction_output_path:
      raise RuntimeError("activation_extraction_output_path is required.")
    if cfg.using_pipeline_parallelism:
      raise NotImplementedError(
          "pipeline parallelism is not supported by extract_activations (v1)."
      )

    # Local imports keep the unit-tested loop independent of MaxEngine.
    from maxtext.utils import max_utils
    from maxtext.inference.maxengine.maxengine import MaxEngine

    max_utils.maybe_initialize_jax_distributed_system(cfg.get_keys())
    engine = MaxEngine(cfg)
    params = engine.load_params(jax.random.PRNGKey(0))
    tokenizer = engine.build_tokenizer(engine.get_tokenizer())
    pad_id = getattr(tokenizer, "pad_id", None) or getattr(
        getattr(tokenizer, "tokenizer", None), "pad_token_id", 0
    )

    num_layers = cfg.num_decoder_layers
    layers_to_keep = resolve_layers(cfg.activation_extraction_layers, num_layers)
    extraction_config = ExtractionConfig(
        hooks=list(cfg.activation_extraction_hooks),
        layers_to_keep=layers_to_keep,
        d_model=cfg.emb_dim,
        num_layers=num_layers,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        output_path=cfg.activation_extraction_output_path,
        shard_size_tokens=cfg.activation_extraction_shard_size_tokens,
        output_dtype=cfg.activation_extraction_output_dtype,
        max_tokens=cfg.activation_extraction_max_tokens,
        skip_bos=cfg.activation_extraction_skip_bos,
        model_name=cfg.model_name,
    )

    writer = ShardedSafetensorsWriter(
        output_path=extraction_config.output_path,
        layers=extraction_config.layers_to_keep,
        hooks=extraction_config.hooks,
        d_model=extraction_config.d_model,
        shard_size_tokens=extraction_config.shard_size_tokens,
        output_dtype=extraction_config.output_dtype,
        process_index=jax.process_index(),
        model_name=extraction_config.model_name,
    )

    forward_fn = _build_forward_fn_from_maxengine(engine, params)

    def _tokenize(text: str) -> list[int]:
      out = tokenizer.encode(text)
      if isinstance(out, tuple):
        return list(out[0])
      return list(out)

    dataset = build_backend(cfg)
    loop = ExtractionLoop(
        config=extraction_config,
        forward_fn=forward_fn,
        tokenize_fn=_tokenize,
        pad_id=int(pad_id),
        dataset=dataset,
        writer=writer,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
    )
    counters = loop.run()
    if jax.process_index() == 0:
      logger.info("extract_activations: done %s", counters)
    multihost_utils.sync_global_devices("extract_activations:done")


__all__ = [
    "ExtractionConfig",
    "ExtractionLoop",
    "Runner",
    "resolve_layers",
    "select_layer_axis",
    "mask_and_flatten",
    "pad_sequence",
]
