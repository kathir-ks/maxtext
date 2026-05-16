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
"""Read back per-host shard directories produced by the writer.

When the extraction runs with ``process_count > 1`` each host writes to
``output_path/host_<idx>/`` independently. This module provides helpers
to enumerate shards from one host or merge across hosts.
"""
from __future__ import annotations

import json
from typing import Iterator

import numpy as np
from etils import epath
from safetensors.numpy import load_file


def list_host_dirs(output_path: str) -> list[epath.Path]:
  """Return ``host_NN`` directories under ``output_path``.

  If there are none (single-shared-dir layout), returns ``[output_path]``.
  """
  base = epath.Path(output_path)
  hosts = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("host_"))
  return hosts if hosts else [base]


def iter_shard_paths(
    host_dir: str | epath.Path, hook: str, layer_idx: int
) -> Iterator[epath.Path]:
  """Yield shard paths for one (hook, layer) under a single host dir, in order."""
  d = epath.Path(host_dir) / hook / f"layer_{layer_idx:04d}"
  if not d.exists():
    return
  for p in sorted(d.iterdir()):
    if p.name.endswith(".safetensors"):
      yield p


def load_merged(
    output_path: str,
    *,
    hook: str,
    layer_idx: int,
    sort_by: tuple[str, ...] = ("doc_ids", "positions"),
) -> dict[str, np.ndarray]:
  """Load all shards from all host dirs for one (hook, layer).

  Returns a dict with concatenated tensors, sorted lexicographically by
  the columns named in ``sort_by``. This is the canonical merged view
  used by tests and downstream SAE training pipelines.
  """
  acts_list: list[np.ndarray] = []
  tids_list: list[np.ndarray] = []
  poss_list: list[np.ndarray] = []
  dids_list: list[np.ndarray] = []
  for host in list_host_dirs(output_path):
    for shard in iter_shard_paths(host, hook, layer_idx):
      d = load_file(str(shard))
      acts_list.append(d["activations"])
      tids_list.append(d["token_ids"])
      poss_list.append(d["positions"])
      dids_list.append(d["doc_ids"])
  if not acts_list:
    raise FileNotFoundError(
        f"No shards found under {output_path} for hook={hook!r} "
        f"layer={layer_idx}"
    )
  acts = np.concatenate(acts_list, axis=0)
  tids = np.concatenate(tids_list, axis=0)
  poss = np.concatenate(poss_list, axis=0)
  dids = np.concatenate(dids_list, axis=0)
  cols = {"doc_ids": dids, "positions": poss, "token_ids": tids}
  keys = tuple(cols[k] for k in reversed(sort_by))
  order = np.lexsort(keys)
  return {
      "activations": acts[order],
      "token_ids": tids[order],
      "positions": poss[order],
      "doc_ids": dids[order],
  }


def load_manifest(output_path: str, *, host_index: int | None = None) -> dict:
  """Load the manifest for a specific host (or the only host)."""
  base = epath.Path(output_path)
  if host_index is None:
    hosts = list_host_dirs(output_path)
    candidate = hosts[0]
  else:
    candidate = base / f"host_{host_index:02d}"
  manifest_path = candidate / "manifest.json"
  return json.loads(manifest_path.read_text())


__all__ = [
    "list_host_dirs",
    "iter_shard_paths",
    "load_merged",
    "load_manifest",
]
