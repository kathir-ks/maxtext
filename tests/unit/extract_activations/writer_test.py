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
"""Unit tests for ShardedSafetensorsWriter."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pytest

import ml_dtypes  # noqa: F401  (used for bfloat16 dtype)
from safetensors.numpy import load_file

from maxtext.tools.extract_activations.writer import ShardedSafetensorsWriter


D = 8


def _rand_batch(n: int, d: int, seed: int = 0):
  rng = np.random.default_rng(seed)
  acts = rng.standard_normal((n, d)).astype(np.float32)
  tids = rng.integers(0, 32000, size=(n,), dtype=np.int32)
  pos = np.arange(n, dtype=np.int32)
  did = np.full((n,), seed, dtype=np.int64)
  return acts, tids, pos, did


class WriterRoundTripTest(unittest.TestCase):

  def test_writer_writes_exact_shard_sizes_and_roundtrips_fp32(self):
    with tempfile.TemporaryDirectory() as tmp:
      writer = ShardedSafetensorsWriter(
          output_path=tmp,
          layers=[0, 1],
          hooks=["residual_post"],
          d_model=D,
          shard_size_tokens=100,
          output_dtype="float32",
          process_index=0,
          model_name="toy",
      )
      # Append three batches totalling 250 rows -> two full shards + one partial.
      a, t, p, d = _rand_batch(120, D, seed=1)
      writer.append(hook="residual_post", layer_idx=0,
                    activations=a, token_ids=t, positions=p, doc_ids=d)
      a2, t2, p2, d2 = _rand_batch(80, D, seed=2)
      writer.append(hook="residual_post", layer_idx=0,
                    activations=a2, token_ids=t2, positions=p2, doc_ids=d2)
      a3, t3, p3, d3 = _rand_batch(50, D, seed=3)
      writer.append(hook="residual_post", layer_idx=0,
                    activations=a3, token_ids=t3, positions=p3, doc_ids=d3)
      writer.finalize()

      shard_dir = os.path.join(tmp, "residual_post", "layer_0000")
      shards = sorted(
          f for f in os.listdir(shard_dir) if f.endswith(".safetensors")
      )
      self.assertEqual(len(shards), 3)
      sizes = []
      for s in shards:
        d_ = load_file(os.path.join(shard_dir, s))
        sizes.append(d_["activations"].shape[0])
        self.assertEqual(d_["activations"].shape[1], D)
        self.assertEqual(d_["token_ids"].dtype, np.int32)
        self.assertEqual(d_["positions"].dtype, np.int32)
        self.assertEqual(d_["doc_ids"].dtype, np.int64)
      self.assertEqual(sizes, [100, 100, 50])

      # Concatenate all shards and compare against the original input.
      cat_acts = np.concatenate(
          [load_file(os.path.join(shard_dir, s))["activations"]
           for s in shards],
          axis=0,
      )
      orig = np.concatenate([a, a2, a3], axis=0)
      np.testing.assert_array_equal(cat_acts, orig)

      # Manifest must reflect the actual counts.
      manifest = json.loads(open(os.path.join(tmp, "manifest.json")).read())
      self.assertEqual(manifest["d_model"], D)
      self.assertEqual(manifest["dtype"], "float32")
      key = "residual_post/layer_0000"
      self.assertEqual(manifest["total_tokens_per_layer"][key], 250)
      self.assertEqual(manifest["shards_per_layer"][key], 3)

  def test_writer_bfloat16_storage_is_lossless_for_representable_values(self):
    with tempfile.TemporaryDirectory() as tmp:
      # Powers of two are exactly representable in bf16.
      acts = np.array(
          [[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]],
          dtype=np.float32,
      ).repeat(8, axis=0)
      tids = np.arange(8, dtype=np.int32)
      pos = np.arange(8, dtype=np.int32)
      dids = np.zeros((8,), dtype=np.int64)
      w = ShardedSafetensorsWriter(
          output_path=tmp, layers=[0], hooks=["residual_post"],
          d_model=D, shard_size_tokens=1000,
          output_dtype="bfloat16",
      )
      w.append(hook="residual_post", layer_idx=0,
               activations=acts, token_ids=tids, positions=pos, doc_ids=dids)
      w.finalize()
      shard = os.path.join(tmp, "residual_post", "layer_0000", "shard_00000.safetensors")
      data = load_file(shard)
      restored = data["activations"].astype(np.float32)
      np.testing.assert_array_equal(restored, acts)

  def test_writer_rejects_shape_mismatch(self):
    with tempfile.TemporaryDirectory() as tmp:
      w = ShardedSafetensorsWriter(
          output_path=tmp, layers=[0], hooks=["residual_post"],
          d_model=D, shard_size_tokens=1000, output_dtype="float32",
      )
      with self.assertRaises(ValueError):
        w.append(hook="residual_post", layer_idx=0,
                 activations=np.zeros((4, D + 1), dtype=np.float32),
                 token_ids=np.zeros((4,), dtype=np.int32),
                 positions=np.zeros((4,), dtype=np.int32),
                 doc_ids=np.zeros((4,), dtype=np.int64))
      with self.assertRaises(ValueError):
        w.append(hook="residual_post", layer_idx=0,
                 activations=np.zeros((4, D), dtype=np.float32),
                 token_ids=np.zeros((3,), dtype=np.int32),
                 positions=np.zeros((4,), dtype=np.int32),
                 doc_ids=np.zeros((4,), dtype=np.int64))

  def test_writer_rejects_unknown_hook_or_layer(self):
    with tempfile.TemporaryDirectory() as tmp:
      w = ShardedSafetensorsWriter(
          output_path=tmp, layers=[0], hooks=["residual_post"],
          d_model=D, shard_size_tokens=1000, output_dtype="float32",
      )
      with self.assertRaises(KeyError):
        w.append(hook="mlp_out", layer_idx=0,
                 activations=np.zeros((1, D), dtype=np.float32),
                 token_ids=np.zeros((1,), dtype=np.int32),
                 positions=np.zeros((1,), dtype=np.int32),
                 doc_ids=np.zeros((1,), dtype=np.int64))
      with self.assertRaises(KeyError):
        w.append(hook="residual_post", layer_idx=99,
                 activations=np.zeros((1, D), dtype=np.float32),
                 token_ids=np.zeros((1,), dtype=np.int32),
                 positions=np.zeros((1,), dtype=np.int32),
                 doc_ids=np.zeros((1,), dtype=np.int64))

  def test_non_writer_processes_noop(self):
    with tempfile.TemporaryDirectory() as tmp:
      w = ShardedSafetensorsWriter(
          output_path=tmp, layers=[0], hooks=["residual_post"],
          d_model=D, shard_size_tokens=10, output_dtype="float32",
          process_index=3,
      )
      a, t, p, d = _rand_batch(100, D)
      w.append(hook="residual_post", layer_idx=0,
               activations=a, token_ids=t, positions=p, doc_ids=d)
      w.finalize()
      # No files should have been created at all.
      self.assertEqual(os.listdir(tmp), [])

  def test_partial_file_is_renamed_atomically(self):
    with tempfile.TemporaryDirectory() as tmp:
      w = ShardedSafetensorsWriter(
          output_path=tmp, layers=[0], hooks=["residual_post"],
          d_model=D, shard_size_tokens=10, output_dtype="float32",
      )
      a, t, p, d = _rand_batch(25, D)
      w.append(hook="residual_post", layer_idx=0,
               activations=a, token_ids=t, positions=p, doc_ids=d)
      w.finalize()
      shard_dir = os.path.join(tmp, "residual_post", "layer_0000")
      partials = [f for f in os.listdir(shard_dir) if f.endswith(".partial")]
      self.assertEqual(partials, [])  # all renamed
      shards = [f for f in os.listdir(shard_dir) if f.endswith(".safetensors")]
      self.assertEqual(len(shards), 3)

  def test_writer_rejects_zero_shard_size(self):
    with self.assertRaises(ValueError):
      ShardedSafetensorsWriter(
          output_path="/tmp/x", layers=[0], hooks=["residual_post"],
          d_model=D, shard_size_tokens=0, output_dtype="float32",
      )


if __name__ == "__main__":
  unittest.main()
