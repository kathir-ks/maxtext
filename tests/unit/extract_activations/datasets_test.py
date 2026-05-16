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
"""Unit tests for dataset backends."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from maxtext.tools.extract_activations.datasets import (
    JsonlBackend,
    PreTokenizedBackend,
)


class JsonlBackendTest(unittest.TestCase):

  def _write_jsonl(self, dir_: str, lines: list[dict]) -> str:
    path = os.path.join(dir_, "data.jsonl")
    with open(path, "w") as f:
      for obj in lines:
        f.write(json.dumps(obj) + "\n")
    return path

  def test_sharding_is_partition(self):
    """Across P processes the union of per-process iterators must equal
    the input dataset, with no duplicates and no omissions."""
    with tempfile.TemporaryDirectory() as tmp:
      path = self._write_jsonl(
          tmp, [{"text": f"doc {i}"} for i in range(33)]
      )
      P = 4
      seen_per_process: list[set[int]] = []
      seen_text: dict[int, str] = {}
      for p in range(P):
        backend = JsonlBackend(path, text_key="text")
        ids = set()
        for doc in backend.iter_for_process(p, P):
          ids.add(doc.doc_id)
          seen_text[doc.doc_id] = doc.text
        seen_per_process.append(ids)
      # Disjoint
      for i in range(P):
        for j in range(i + 1, P):
          self.assertEqual(
              seen_per_process[i] & seen_per_process[j], set(),
              f"processes {i} and {j} overlap",
          )
      # Covering
      union = set().union(*seen_per_process)
      self.assertEqual(union, set(range(33)))
      # doc_id reflects the original line number
      for i in range(33):
        self.assertEqual(seen_text[i], f"doc {i}")

  def test_token_ids_field_supported(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = self._write_jsonl(
          tmp,
          [{"tokens": [1, 2, 3]}, {"token_ids": [4, 5]}],
      )
      backend = JsonlBackend(path)
      docs = list(backend.iter_for_process(0, 1))
      self.assertEqual(docs[0].token_ids, [1, 2, 3])
      self.assertEqual(docs[1].token_ids, [4, 5])

  def test_max_documents_caps(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = self._write_jsonl(
          tmp, [{"text": str(i)} for i in range(20)]
      )
      backend = JsonlBackend(path, max_documents=5)
      docs = list(backend.iter_for_process(0, 1))
      self.assertEqual(len(docs), 5)

  def test_missing_keys_raises(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = self._write_jsonl(tmp, [{"other": "x"}])
      backend = JsonlBackend(path)
      with self.assertRaises(ValueError):
        list(backend.iter_for_process(0, 1))


class PreTokenizedBackendTest(unittest.TestCase):

  def test_roundtrip_and_sharding(self):
    with tempfile.TemporaryDirectory() as tmp:
      arr = np.arange(40, dtype=np.int32).reshape(10, 4)
      path = os.path.join(tmp, "data.npy")
      np.save(path, arr)
      backend = PreTokenizedBackend(path)
      docs = list(backend.iter_for_process(1, 3))
      # doc ids: 1, 4, 7
      self.assertEqual([d.doc_id for d in docs], [1, 4, 7])
      self.assertEqual(docs[0].token_ids, [4, 5, 6, 7])


if __name__ == "__main__":
  unittest.main()
