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
"""Multi-host equivalence test for activation extraction.

This is the headline correctness test for the extraction tool. It runs
extraction twice on the *exact same* dataset:

  1. Single-host:  ``process_count = 1``
  2. Two-host:     ``process_count = 2``  (process 0 sees even doc-ids,
                                            process 1 sees odd doc-ids)

After both runs, we merge all shards (across hosts and shards), sort by
``(doc_id, position)``, and assert that the two views are bit-identical.

This proves three properties simultaneously:

  * No documents are dropped, duplicated, or reordered.
  * Token-ids and positions line up with activations row-for-row.
  * The per-host writer layout matches the canonical single-host layout
    after merging.

The forward pass is a deterministic synthetic function — we are not
testing JAX SPMD here, we are testing the extraction tool's data plane.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np

from maxtext.tools.extract_activations.datasets import JsonlBackend
from maxtext.tools.extract_activations.merge import load_merged
from maxtext.tools.extract_activations.runner import (
    ExtractionConfig,
    ExtractionLoop,
)
from maxtext.tools.extract_activations.writer import ShardedSafetensorsWriter


NUM_LAYERS = 6
D_MODEL = 4
LAYERS_TO_KEEP = [1, 3, 5]
HOOKS = ["residual_post"]
MAX_SEQ = 16


def _synthetic_forward():
  """Deterministic forward that encodes ``(layer, position, dim, token)``
  in each entry so any swap or reshape is easy to detect.
  """

  def fwd(_params, padded_tokens, true_length):
    del _params, true_length
    T = padded_tokens.shape[0]
    grid = np.zeros((NUM_LAYERS, 1, T, D_MODEL), dtype=np.float32)
    for l in range(NUM_LAYERS):
      for t in range(T):
        for d in range(D_MODEL):
          grid[l, 0, t, d] = (
              l * 1e6 + t * 1e3 + d * 10 + int(padded_tokens[t])
          )
    return {"residual_post": jnp.asarray(grid)}

  return fwd


def _build_jsonl(path: str, num_docs: int, rng: np.random.Generator):
  with open(path, "w") as f:
    for i in range(num_docs):
      length = int(rng.integers(low=2, high=12))
      tokens = rng.integers(low=1, high=1000, size=length).tolist()
      f.write(json.dumps({"tokens": tokens}) + "\n")


def _run_loop(
    *,
    jsonl_path: str,
    out_dir: str,
    process_index: int,
    process_count: int,
):
  writer = ShardedSafetensorsWriter(
      output_path=out_dir,
      layers=LAYERS_TO_KEEP,
      hooks=HOOKS,
      d_model=D_MODEL,
      shard_size_tokens=11,  # small to force multi-shard
      output_dtype="float32",
      process_index=process_index,
      process_count=process_count,
      model_name="synthetic",
  )
  cfg = ExtractionConfig(
      hooks=HOOKS,
      layers_to_keep=LAYERS_TO_KEEP,
      d_model=D_MODEL,
      num_layers=NUM_LAYERS,
      max_prefill_predict_length=MAX_SEQ,
      output_path=out_dir,
      shard_size_tokens=11,
      output_dtype="float32",
      skip_bos=False,
      model_name="synthetic",
  )
  loop = ExtractionLoop(
      config=cfg,
      forward_fn=_synthetic_forward(),
      tokenize_fn=lambda t: [],
      pad_id=0,
      dataset=JsonlBackend(jsonl_path),
      writer=writer,
      process_index=process_index,
      process_count=process_count,
  )
  return loop.run()


class MultihostEquivalenceTest(unittest.TestCase):
  """The promised property: merged P=2 output == single P=1 output."""

  def test_two_host_merge_equals_single_host(self):
    rng = np.random.default_rng(seed=0xC0FFEE)
    num_docs = 17  # deliberately odd so the modulo-partition is uneven
    with tempfile.TemporaryDirectory() as tmp:
      ds = os.path.join(tmp, "data.jsonl")
      _build_jsonl(ds, num_docs, rng)

      # ----- single-host reference -----
      ref_dir = os.path.join(tmp, "ref")
      counters_ref = _run_loop(
          jsonl_path=ds, out_dir=ref_dir,
          process_index=0, process_count=1,
      )
      self.assertEqual(counters_ref["documents_seen"], num_docs)

      # ----- two-host run (sequential simulation in this process) -----
      multi_dir = os.path.join(tmp, "multi")
      counters0 = _run_loop(
          jsonl_path=ds, out_dir=multi_dir,
          process_index=0, process_count=2,
      )
      counters1 = _run_loop(
          jsonl_path=ds, out_dir=multi_dir,
          process_index=1, process_count=2,
      )
      total = counters0["documents_seen"] + counters1["documents_seen"]
      self.assertEqual(total, num_docs)
      # Partition must be disjoint and complete.
      self.assertEqual(
          counters0["documents_seen"], (num_docs + 1) // 2
      )
      self.assertEqual(
          counters1["documents_seen"], num_docs // 2
      )

      # ----- merge and compare -----
      for layer in LAYERS_TO_KEEP:
        ref = load_merged(ref_dir, hook="residual_post", layer_idx=layer)
        merged = load_merged(multi_dir, hook="residual_post", layer_idx=layer)
        for key in ("activations", "token_ids", "positions", "doc_ids"):
          np.testing.assert_array_equal(
              ref[key], merged[key],
              err_msg=f"layer={layer} key={key} drifted between P=1 and P=2",
          )

  def test_per_host_directories_are_disjoint(self):
    """Each host's output dir contains only its own doc ids."""
    rng = np.random.default_rng(seed=42)
    num_docs = 10
    with tempfile.TemporaryDirectory() as tmp:
      ds = os.path.join(tmp, "data.jsonl")
      _build_jsonl(ds, num_docs, rng)
      multi_dir = os.path.join(tmp, "multi")
      _run_loop(
          jsonl_path=ds, out_dir=multi_dir,
          process_index=0, process_count=2,
      )
      _run_loop(
          jsonl_path=ds, out_dir=multi_dir,
          process_index=1, process_count=2,
      )
      host0 = load_merged(
          os.path.join(multi_dir, "host_00"),
          hook="residual_post",
          layer_idx=LAYERS_TO_KEEP[0],
      )
      host1 = load_merged(
          os.path.join(multi_dir, "host_01"),
          hook="residual_post",
          layer_idx=LAYERS_TO_KEEP[0],
      )
      ids0 = set(host0["doc_ids"].tolist())
      ids1 = set(host1["doc_ids"].tolist())
      self.assertEqual(ids0, set(range(0, num_docs, 2)))
      self.assertEqual(ids1, set(range(1, num_docs, 2)))
      self.assertEqual(ids0 & ids1, set())

  def test_four_host_merge_equals_single_host(self):
    """Scale up to P=4 to catch any P-dependence."""
    rng = np.random.default_rng(seed=7)
    num_docs = 23
    with tempfile.TemporaryDirectory() as tmp:
      ds = os.path.join(tmp, "data.jsonl")
      _build_jsonl(ds, num_docs, rng)
      ref_dir = os.path.join(tmp, "ref")
      multi_dir = os.path.join(tmp, "p4")
      _run_loop(
          jsonl_path=ds, out_dir=ref_dir,
          process_index=0, process_count=1,
      )
      for pi in range(4):
        _run_loop(
            jsonl_path=ds, out_dir=multi_dir,
            process_index=pi, process_count=4,
        )
      for layer in LAYERS_TO_KEEP:
        ref = load_merged(ref_dir, hook="residual_post", layer_idx=layer)
        merged = load_merged(multi_dir, hook="residual_post", layer_idx=layer)
        np.testing.assert_array_equal(ref["activations"], merged["activations"])
        np.testing.assert_array_equal(ref["doc_ids"], merged["doc_ids"])
        np.testing.assert_array_equal(ref["positions"], merged["positions"])
        np.testing.assert_array_equal(ref["token_ids"], merged["token_ids"])


if __name__ == "__main__":
  unittest.main()
