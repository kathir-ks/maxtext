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
"""End-to-end test for the single-host ExtractionLoop.

Uses a deterministic synthetic ``forward_fn`` that returns a known
activation pattern, so we can assert that the writer receives the same
values that the forward function emitted.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np
import safetensors.numpy as st_np

from maxtext.tools.extract_activations.datasets import JsonlBackend
from maxtext.tools.extract_activations.runner import (
    ExtractionConfig,
    ExtractionLoop,
)
from maxtext.tools.extract_activations.writer import ShardedSafetensorsWriter


def _make_synthetic_forward(num_layers: int, d_model: int):
  """Returns activations[hook][L,1,T,D] where each row encodes
  (layer_idx, batch_idx=0, token_position, dim_idx) so any drift in the
  pipeline is easy to spot.
  """

  def fwd(_params, padded_tokens, true_length):
    del _params, true_length
    T = padded_tokens.shape[0]
    grid = np.zeros((num_layers, 1, T, d_model), dtype=np.float32)
    for l in range(num_layers):
      for t in range(T):
        for d in range(d_model):
          grid[l, 0, t, d] = (
              l * 1_000_000 + t * 1_000 + d * 1 + int(padded_tokens[t])
          )
    return {"residual_post": jnp.asarray(grid)}

  return fwd


def _write_jsonl(path: str, lines):
  with open(path, "w") as f:
    for obj in lines:
      f.write(json.dumps(obj) + "\n")


class LoopEndToEndTest(unittest.TestCase):

  def test_writes_expected_rows(self):
    """Run the loop on 4 short docs, then read back the shards and
    reconstruct the (doc_id, position) -> activation mapping. Each row
    must match the synthetic forward's output exactly."""
    num_layers = 6
    d_model = 4
    layers_to_keep = [1, 4]
    max_seq = 8
    with tempfile.TemporaryDirectory() as tmp:
      ds_path = os.path.join(tmp, "data.jsonl")
      _write_jsonl(
          ds_path,
          [
              {"tokens": [10, 11, 12]},
              {"tokens": [20, 21]},
              {"tokens": [30]},
              {"tokens": [40, 41, 42, 43]},
          ],
      )
      out_dir = os.path.join(tmp, "out")
      writer = ShardedSafetensorsWriter(
          output_path=out_dir,
          layers=layers_to_keep,
          hooks=["residual_post"],
          d_model=d_model,
          shard_size_tokens=3,  # small to force multiple shards
          output_dtype="float32",
          process_index=0,
          model_name="toy",
      )
      cfg = ExtractionConfig(
          hooks=["residual_post"],
          layers_to_keep=layers_to_keep,
          d_model=d_model,
          num_layers=num_layers,
          max_prefill_predict_length=max_seq,
          output_path=out_dir,
          shard_size_tokens=3,
          output_dtype="float32",
          max_tokens=0,
          skip_bos=False,
          model_name="toy",
      )
      loop = ExtractionLoop(
          config=cfg,
          forward_fn=_make_synthetic_forward(num_layers, d_model),
          tokenize_fn=lambda t: [],
          pad_id=0,
          dataset=JsonlBackend(ds_path),
          writer=writer,
          process_index=0,
          process_count=1,
      )
      counters = loop.run()
      self.assertEqual(counters["documents_seen"], 4)
      self.assertEqual(counters["tokens_written_per_layer"], 3 + 2 + 1 + 4)

      # Read back layer 1 and reconstruct.
      shard_dir = os.path.join(out_dir, "residual_post", "layer_0001")
      shards = sorted(
          f for f in os.listdir(shard_dir) if f.endswith(".safetensors")
      )
      rows_acts = []
      rows_doc = []
      rows_pos = []
      rows_tok = []
      for s in shards:
        d_ = st_np.load_file(os.path.join(shard_dir, s))
        rows_acts.append(d_["activations"])
        rows_doc.append(d_["doc_ids"])
        rows_pos.append(d_["positions"])
        rows_tok.append(d_["token_ids"])
      acts = np.concatenate(rows_acts, axis=0)
      docs = np.concatenate(rows_doc, axis=0)
      poss = np.concatenate(rows_pos, axis=0)
      toks = np.concatenate(rows_tok, axis=0)

      # Sort canonically by (doc_id, position) to enable comparison.
      order = np.lexsort((poss, docs))
      acts, docs, poss, toks = acts[order], docs[order], poss[order], toks[order]

      # Expected docs and tokens (in lex order).
      expected_docs = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3], dtype=np.int64)
      expected_pos = np.array([0, 1, 2, 0, 1, 0, 0, 1, 2, 3], dtype=np.int32)
      expected_tok = np.array(
          [10, 11, 12, 20, 21, 30, 40, 41, 42, 43], dtype=np.int32
      )
      np.testing.assert_array_equal(docs, expected_docs)
      np.testing.assert_array_equal(poss, expected_pos)
      np.testing.assert_array_equal(toks, expected_tok)

      # Activation values: for layer_idx=1 and dim d, expected[d] = 1*1e6 + t*1e3 + d + token.
      for row in range(acts.shape[0]):
        t = poss[row]
        tok = toks[row]
        for d in range(d_model):
          self.assertEqual(
              acts[row, d], 1 * 1_000_000 + t * 1_000 + d + int(tok),
              f"row {row} dim {d}",
          )

  def test_skip_bos_drops_position_zero(self):
    num_layers = 2
    d_model = 2
    with tempfile.TemporaryDirectory() as tmp:
      ds_path = os.path.join(tmp, "data.jsonl")
      _write_jsonl(ds_path, [{"tokens": [10, 11, 12]}])
      out_dir = os.path.join(tmp, "out")
      writer = ShardedSafetensorsWriter(
          output_path=out_dir,
          layers=[0],
          hooks=["residual_post"],
          d_model=d_model,
          shard_size_tokens=100,
          output_dtype="float32",
      )
      cfg = ExtractionConfig(
          hooks=["residual_post"],
          layers_to_keep=[0],
          d_model=d_model,
          num_layers=num_layers,
          max_prefill_predict_length=8,
          output_path=out_dir,
          shard_size_tokens=100,
          output_dtype="float32",
          skip_bos=True,
      )
      loop = ExtractionLoop(
          config=cfg,
          forward_fn=_make_synthetic_forward(num_layers, d_model),
          tokenize_fn=lambda t: [],
          pad_id=0,
          dataset=JsonlBackend(ds_path),
          writer=writer,
          process_index=0,
          process_count=1,
      )
      loop.run()
      data = st_np.load_file(
          os.path.join(out_dir, "residual_post", "layer_0000", "shard_00000.safetensors")
      )
      # 3 tokens minus BOS = 2 rows
      self.assertEqual(data["activations"].shape, (2, d_model))
      np.testing.assert_array_equal(data["positions"], [1, 2])

  def test_max_tokens_stops_early(self):
    num_layers = 2
    d_model = 2
    with tempfile.TemporaryDirectory() as tmp:
      ds_path = os.path.join(tmp, "data.jsonl")
      _write_jsonl(
          ds_path,
          [{"tokens": [1, 2, 3]} for _ in range(10)],
      )
      out_dir = os.path.join(tmp, "out")
      writer = ShardedSafetensorsWriter(
          output_path=out_dir, layers=[0], hooks=["residual_post"],
          d_model=d_model, shard_size_tokens=100, output_dtype="float32",
      )
      cfg = ExtractionConfig(
          hooks=["residual_post"],
          layers_to_keep=[0],
          d_model=d_model,
          num_layers=num_layers,
          max_prefill_predict_length=8,
          output_path=out_dir,
          shard_size_tokens=100,
          output_dtype="float32",
          max_tokens=7,  # should stop after 3 docs (9 rows, first 3 docs give 9 rows > 7)
          skip_bos=False,
      )
      loop = ExtractionLoop(
          config=cfg,
          forward_fn=_make_synthetic_forward(num_layers, d_model),
          tokenize_fn=lambda t: [],
          pad_id=0,
          dataset=JsonlBackend(ds_path),
          writer=writer,
          process_index=0,
          process_count=1,
      )
      counters = loop.run()
      self.assertLess(counters["documents_seen"], 10)
      self.assertGreaterEqual(counters["tokens_written_per_layer"], 7)


if __name__ == "__main__":
  unittest.main()
