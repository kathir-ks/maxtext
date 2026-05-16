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
"""Sharded safetensors writer for activation extraction.

Writes one directory per ``(hook, layer_idx)`` pair. Each shard is a
safetensors file with four tensors:

  - ``activations``: shape ``(N, d_model)``, dtype = ``output_dtype``
  - ``token_ids``:   shape ``(N,)``,         dtype int32
  - ``positions``:   shape ``(N,)``,         dtype int32
  - ``doc_ids``:     shape ``(N,)``,         dtype int64

Atomicity: each shard is written to ``.partial`` then renamed.
GCS paths are supported via ``etils.epath`` (already a MaxText dependency).
Writes happen only on ``process_index == 0``; on other ranks all
methods are no-ops.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
from typing import Iterable

import numpy as np
from etils import epath
from safetensors.numpy import save_file as st_save_file

logger = logging.getLogger(__name__)

# Numpy doesn't natively support bfloat16; we view it as uint16 for storage,
# then re-interpret on the consumer side. safetensors supports bf16 directly
# if we use the ml_dtypes shim, but we keep numpy-only here for portability.
try:
  import ml_dtypes  # type: ignore

  _HAS_ML_DTYPES = True
  _BF16_NP_DTYPE = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover
  _HAS_ML_DTYPES = False
  _BF16_NP_DTYPE = None


_DTYPE_NAME_TO_NP = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": _BF16_NP_DTYPE,  # may be None on systems without ml_dtypes
}


def _np_dtype_for(name: str) -> np.dtype:
  if name == "bfloat16" and not _HAS_ML_DTYPES:
    raise ImportError(
        "bfloat16 output requires the `ml_dtypes` package "
        "(pip install ml_dtypes)."
    )
  return np.dtype(_DTYPE_NAME_TO_NP[name])


@dataclasses.dataclass
class _BufferKey:
  hook: str
  layer_idx: int

  def __hash__(self) -> int:  # noqa: D401
    return hash((self.hook, self.layer_idx))


@dataclasses.dataclass
class _Buffer:
  activations: list[np.ndarray] = dataclasses.field(default_factory=list)
  token_ids: list[np.ndarray] = dataclasses.field(default_factory=list)
  positions: list[np.ndarray] = dataclasses.field(default_factory=list)
  doc_ids: list[np.ndarray] = dataclasses.field(default_factory=list)
  total_rows: int = 0
  shards_written: int = 0
  total_tokens_written: int = 0

  def add(
      self,
      acts: np.ndarray,
      token_ids: np.ndarray,
      positions: np.ndarray,
      doc_ids: np.ndarray,
  ) -> None:
    self.activations.append(acts)
    self.token_ids.append(token_ids)
    self.positions.append(positions)
    self.doc_ids.append(doc_ids)
    self.total_rows += int(acts.shape[0])

  def clear(self) -> None:
    self.activations.clear()
    self.token_ids.clear()
    self.positions.clear()
    self.doc_ids.clear()
    self.total_rows = 0


class ShardedSafetensorsWriter:
  """Buffer activations per hook+layer and flush as safetensors shards.

  Args:
    output_path: local directory or ``gs://`` URI.
    layers: layer indices that will be written.
    hooks: hook names that will be written.
    d_model: hidden size, recorded in the manifest.
    shard_size_tokens: rows per shard; the writer flushes once a buffer
      reaches this count.
    output_dtype: storage dtype, one of ``"bfloat16"``, ``"float16"``,
      ``"float32"``.
    process_index: JAX process index; only rank 0 writes.
    model_name: recorded in the manifest.
    extra_manifest: extra fields merged into ``manifest.json``.
  """

  def __init__(
      self,
      *,
      output_path: str,
      layers: Iterable[int],
      hooks: Iterable[str],
      d_model: int,
      shard_size_tokens: int,
      output_dtype: str = "bfloat16",
      process_index: int = 0,
      process_count: int = 1,
      model_name: str = "",
      extra_manifest: dict | None = None,
      per_host_subdir: bool | None = None,
  ):
    """Args:
      output_path: shared base path. Each host writes to a subdirectory
        ``host_<process_index:02d>`` underneath (when ``per_host_subdir``
        is True) so that hosts never collide. Default is to write to
        a per-host subdir whenever ``process_count > 1``.
      process_index: this host's process index.
      process_count: total number of processes; used to decide whether to
        partition by host.
      per_host_subdir: override the default per-host-subdir behaviour.
    """
    if shard_size_tokens <= 0:
      raise ValueError(f"shard_size_tokens must be positive, got {shard_size_tokens}")
    if not output_path:
      raise ValueError("output_path is required.")
    self._process_index = int(process_index)
    self._process_count = int(process_count)
    if per_host_subdir is None:
      per_host_subdir = self._process_count > 1
    # Every host is a writer in the per-host-subdir layout.
    # In the single-shared-dir layout only host 0 writes.
    self._is_writer = per_host_subdir or self._process_index == 0
    base = epath.Path(output_path)
    self._base_path = base
    if per_host_subdir:
      self._output_path = base / f"host_{self._process_index:02d}"
    else:
      self._output_path = base
    self._per_host_subdir = per_host_subdir
    self._layers = list(layers)
    self._hooks = list(hooks)
    self._d_model = int(d_model)
    self._shard_size_tokens = int(shard_size_tokens)
    self._np_dtype = _np_dtype_for(output_dtype)
    self._output_dtype_name = output_dtype
    self._model_name = model_name
    self._extra_manifest = dict(extra_manifest or {})
    self._buffers: dict[_BufferKey, _Buffer] = {
        _BufferKey(h, l): _Buffer() for h in self._hooks for l in self._layers
    }
    self._finalized = False
    if self._is_writer:
      self._output_path.mkdir(parents=True, exist_ok=True)
      for key in self._buffers:
        self._dir_for(key).mkdir(parents=True, exist_ok=True)

  # ---------- public API ----------

  def append(
      self,
      *,
      hook: str,
      layer_idx: int,
      activations: np.ndarray,
      token_ids: np.ndarray,
      positions: np.ndarray,
      doc_ids: np.ndarray,
  ) -> None:
    """Add a batch of rows for one (hook, layer) and flush if the
    accumulated row count exceeds ``shard_size_tokens``.

    All four arrays must have the same first dimension N.
    ``activations`` has shape ``(N, d_model)``; the other three have
    shape ``(N,)``.
    """
    if not self._is_writer:
      return
    self._validate_inputs(activations, token_ids, positions, doc_ids)
    key = _BufferKey(hook, layer_idx)
    if key not in self._buffers:
      raise KeyError(
          f"({hook!r}, {layer_idx}) not in configured "
          f"hooks={self._hooks} layers={self._layers}"
      )
    buf = self._buffers[key]
    buf.add(
        np.asarray(activations, dtype=self._np_dtype),
        np.asarray(token_ids, dtype=np.int32),
        np.asarray(positions, dtype=np.int32),
        np.asarray(doc_ids, dtype=np.int64),
    )
    while buf.total_rows >= self._shard_size_tokens:
      self._flush_one_shard(key, target_rows=self._shard_size_tokens)

  def finalize(self) -> None:
    """Flush any partial buffers and write the manifest."""
    if self._finalized:
      return
    self._finalized = True
    if not self._is_writer:
      return
    for key, buf in self._buffers.items():
      if buf.total_rows > 0:
        self._flush_one_shard(key, target_rows=buf.total_rows)
      self._write_per_layer_metadata(key, buf)
    self._write_manifest()

  # ---------- internals ----------

  def _validate_inputs(
      self,
      activations: np.ndarray,
      token_ids: np.ndarray,
      positions: np.ndarray,
      doc_ids: np.ndarray,
  ) -> None:
    if activations.ndim != 2:
      raise ValueError(f"activations must be 2-D, got shape {activations.shape}")
    if activations.shape[1] != self._d_model:
      raise ValueError(
          f"activations have d_model={activations.shape[1]}, "
          f"writer configured for d_model={self._d_model}"
      )
    n = activations.shape[0]
    for name, arr in (("token_ids", token_ids), ("positions", positions), ("doc_ids", doc_ids)):
      if arr.shape != (n,):
        raise ValueError(f"{name} shape {arr.shape} != ({n},)")

  def _dir_for(self, key: _BufferKey) -> epath.Path:
    return self._output_path / key.hook / f"layer_{key.layer_idx:04d}"

  def _shard_path(self, key: _BufferKey, shard_idx: int) -> epath.Path:
    return self._dir_for(key) / f"shard_{shard_idx:05d}.safetensors"

  def _flush_one_shard(self, key: _BufferKey, target_rows: int) -> None:
    buf = self._buffers[key]
    # Concatenate everything, then split off `target_rows` for this shard,
    # leaving the remainder in the buffer for the next one. This guarantees
    # shard sizes are exactly `shard_size_tokens` (except the final one).
    acts = np.concatenate(buf.activations, axis=0)
    tids = np.concatenate(buf.token_ids, axis=0)
    poss = np.concatenate(buf.positions, axis=0)
    dids = np.concatenate(buf.doc_ids, axis=0)
    take = min(target_rows, acts.shape[0])
    shard_acts = acts[:take]
    shard_tids = tids[:take]
    shard_poss = poss[:take]
    shard_dids = dids[:take]
    remainder_acts = acts[take:]
    remainder_tids = tids[take:]
    remainder_poss = poss[take:]
    remainder_dids = dids[take:]

    shard_path = self._shard_path(key, buf.shards_written)
    self._write_shard_atomic(
        shard_path,
        {
            "activations": shard_acts,
            "token_ids": shard_tids,
            "positions": shard_poss,
            "doc_ids": shard_dids,
        },
    )
    buf.shards_written += 1
    buf.total_tokens_written += take
    buf.clear()
    if remainder_acts.shape[0] > 0:
      buf.add(remainder_acts, remainder_tids, remainder_poss, remainder_dids)

  def _write_shard_atomic(self, target: epath.Path, tensors: dict[str, np.ndarray]) -> None:
    tmp = target.parent / (target.name + ".partial")
    # safetensors.save_file writes synchronously; epath handles GCS by buffering
    # to a local temp file under the hood for gs:// paths.
    st_save_file(tensors, os.fspath(tmp))
    # On GCS, epath rename is a copy+delete; on local, it's atomic.
    tmp.rename(target)
    logger.info(
        "extract_activations: wrote shard %s (%d rows, %s)",
        target, tensors["activations"].shape[0], tensors["activations"].dtype,
    )

  def _write_per_layer_metadata(self, key: _BufferKey, buf: _Buffer) -> None:
    meta_path = self._dir_for(key) / "metadata.json"
    payload = {
        "hook": key.hook,
        "layer_idx": key.layer_idx,
        "d_model": self._d_model,
        "dtype": self._output_dtype_name,
        "total_tokens": buf.total_tokens_written,
        "num_shards": buf.shards_written,
        "shard_size_tokens_target": self._shard_size_tokens,
    }
    meta_path.write_text(json.dumps(payload, indent=2))

  def _write_manifest(self) -> None:
    manifest_path = self._output_path / "manifest.json"
    payload = {
        "model_name": self._model_name,
        "d_model": self._d_model,
        "dtype": self._output_dtype_name,
        "hooks": self._hooks,
        "layers": self._layers,
        "shard_size_tokens_target": self._shard_size_tokens,
        "process_index": self._process_index,
        "process_count": self._process_count,
        "per_host_subdir": self._per_host_subdir,
        "total_tokens_per_layer": {
            f"{key.hook}/layer_{key.layer_idx:04d}": buf.total_tokens_written
            for key, buf in self._buffers.items()
        },
        "shards_per_layer": {
            f"{key.hook}/layer_{key.layer_idx:04d}": buf.shards_written
            for key, buf in self._buffers.items()
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **self._extra_manifest,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    logger.info("extract_activations: wrote manifest %s", manifest_path)


__all__ = ["ShardedSafetensorsWriter"]
