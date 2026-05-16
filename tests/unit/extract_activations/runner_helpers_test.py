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
"""Unit tests for the pure helpers in runner.py."""
from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.tools.extract_activations.runner import (
    mask_and_flatten,
    pad_sequence,
    resolve_layers,
    select_layer_axis,
)


class ResolveLayersTest(unittest.TestCase):

  def test_empty_default_three_quartiles(self):
    self.assertEqual(resolve_layers([], 48), [12, 24, 36])
    self.assertEqual(resolve_layers([], 4), [1, 2, 3])

  def test_explicit_indices_preserve_order(self):
    self.assertEqual(resolve_layers([5, 1, 9], 12), [5, 1, 9])

  def test_negative_indices_count_from_top(self):
    self.assertEqual(resolve_layers([-1, -2], 12), [11, 10])

  def test_dedup_preserves_first_occurrence(self):
    self.assertEqual(resolve_layers([3, 5, 3, 7], 12), [3, 5, 7])

  def test_out_of_range_raises(self):
    with self.assertRaises(ValueError):
      resolve_layers([100], 12)
    with self.assertRaises(ValueError):
      resolve_layers([-100], 12)


class PadSequenceTest(unittest.TestCase):

  def test_pads_with_pad_id(self):
    arr, tl = pad_sequence([1, 2, 3], max_length=8, pad_id=0)
    np.testing.assert_array_equal(arr, [1, 2, 3, 0, 0, 0, 0, 0])
    self.assertEqual(tl, 3)

  def test_truncates_when_too_long(self):
    arr, tl = pad_sequence([1, 2, 3, 4, 5], max_length=3, pad_id=0)
    np.testing.assert_array_equal(arr, [1, 2, 3])
    self.assertEqual(tl, 3)


class SelectLayerAxisTest(unittest.TestCase):

  def test_scan_layout_picks_indices(self):
    # [L, B, T, D] with L=4
    raw = jnp.arange(4 * 1 * 2 * 3, dtype=jnp.float32).reshape(4, 1, 2, 3)
    out = select_layer_axis(raw, layer_indices=[0, 2], num_layers=4)
    self.assertEqual(out.shape, (2, 1, 2, 3))
    np.testing.assert_array_equal(out[0], raw[0])
    np.testing.assert_array_equal(out[1], raw[2])

  def test_unwraps_leading_singleton(self):
    raw = jnp.arange(4 * 1 * 2 * 3, dtype=jnp.float32).reshape(1, 4, 1, 2, 3)
    out = select_layer_axis(raw, layer_indices=[1, 3], num_layers=4)
    self.assertEqual(out.shape, (2, 1, 2, 3))

  def test_tuple_unscanned_layout(self):
    arrs = tuple(jnp.full((1, 2, 3), float(i)) for i in range(4))
    out = select_layer_axis(arrs, layer_indices=[1, 3], num_layers=4)
    self.assertEqual(out.shape, (2, 1, 2, 3))
    self.assertTrue(jnp.all(out[0] == 1.0))
    self.assertTrue(jnp.all(out[1] == 3.0))

  def test_layer_count_mismatch_raises(self):
    raw = jnp.zeros((3, 1, 2, 3))
    with self.assertRaises(ValueError):
      select_layer_axis(raw, layer_indices=[0], num_layers=4)


class MaskAndFlattenTest(unittest.TestCase):

  def test_drops_padding_only(self):
    B, T, D = 2, 4, 2
    acts = np.arange(B * T * D, dtype=np.float32).reshape(B, T, D)
    tids = np.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=np.int32)
    true_lens = np.array([3, 2], dtype=np.int32)
    doc_ids = np.array([7, 9], dtype=np.int64)
    a, t, p, d = mask_and_flatten(
        acts, tids, true_lens, doc_ids, skip_bos=False
    )
    # batch 0 contributes positions 0,1,2; batch 1 contributes positions 0,1
    self.assertEqual(a.shape, (5, D))
    np.testing.assert_array_equal(t, [10, 11, 12, 20, 21])
    np.testing.assert_array_equal(p, [0, 1, 2, 0, 1])
    np.testing.assert_array_equal(d, [7, 7, 7, 9, 9])

  def test_skip_bos_drops_position_zero(self):
    B, T, D = 1, 3, 2
    acts = np.arange(B * T * D, dtype=np.float32).reshape(B, T, D)
    tids = np.array([[10, 11, 12]], dtype=np.int32)
    true_lens = np.array([3], dtype=np.int32)
    doc_ids = np.array([0], dtype=np.int64)
    a, t, p, d = mask_and_flatten(
        acts, tids, true_lens, doc_ids, skip_bos=True
    )
    self.assertEqual(a.shape, (2, D))
    np.testing.assert_array_equal(p, [1, 2])

  def test_returns_empty_when_all_padded(self):
    a, t, p, d = mask_and_flatten(
        np.zeros((2, 4, 3), dtype=np.float32),
        np.zeros((2, 4), dtype=np.int32),
        np.zeros((2,), dtype=np.int32),  # all true_lens are 0
        np.zeros((2,), dtype=np.int64),
        skip_bos=False,
    )
    self.assertEqual(a.shape, (0, 3))

  def test_shape_validation_raises(self):
    with self.assertRaises(ValueError):
      mask_and_flatten(
          np.zeros((4, 3), dtype=np.float32),  # 2-D, should be 3-D
          np.zeros((1, 4), dtype=np.int32),
          np.zeros((1,), dtype=np.int32),
          np.zeros((1,), dtype=np.int64),
          skip_bos=False,
      )


if __name__ == "__main__":
  unittest.main()
