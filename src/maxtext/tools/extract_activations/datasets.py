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
"""Dataset backends for activation extraction.

A backend yields ``Document`` objects. Documents are partitioned across
JAX processes by global doc-id modulo process count, so every document
is read by exactly one process and the union of per-process streams is
exactly the original dataset in the original order.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Iterator, Protocol

from etils import epath


@dataclasses.dataclass(frozen=True)
class Document:
  """One unit of input.

  Exactly one of ``text`` and ``token_ids`` must be set.
  """
  doc_id: int
  text: str | None = None
  token_ids: list[int] | None = None


class DatasetBackend(Protocol):
  """Backend contract."""

  def iter_for_process(
      self, process_index: int, process_count: int
  ) -> Iterator[Document]:
    """Yield the subset of documents that belong to this process.

    The partitioning must satisfy: across all processes 0..P-1, every
    document with index ``i`` is yielded by exactly the one process with
    ``i % P == process_index``.
    """
    ...


class JsonlBackend:
  """One JSON object per line; expects a ``text`` key (configurable)."""

  def __init__(self, path: str, text_key: str = "text", max_documents: int = 0):
    self._path = epath.Path(path)
    self._text_key = text_key
    self._max_documents = int(max_documents)

  def iter_for_process(
      self, process_index: int, process_count: int
  ) -> Iterator[Document]:
    count = 0
    with self._path.open("r") as f:
      for i, line in enumerate(f):
        line = line.strip()
        if not line:
          continue
        if i % process_count != process_index:
          continue
        obj = json.loads(line)
        text = obj.get(self._text_key)
        token_ids = obj.get("token_ids") or obj.get("tokens")
        if text is None and token_ids is None:
          raise ValueError(
              f"Line {i} has neither {self._text_key!r} nor token_ids/tokens."
          )
        yield Document(doc_id=i, text=text, token_ids=token_ids)
        count += 1
        if self._max_documents and count >= self._max_documents:
          break


class HuggingFaceBackend:
  """Streaming HF dataset. Requires ``datasets`` to be installed."""

  def __init__(
      self,
      name: str,
      split: str = "train",
      text_key: str = "text",
      max_documents: int = 0,
  ):
    self._name = name
    self._split = split
    self._text_key = text_key
    self._max_documents = int(max_documents)

  def iter_for_process(
      self, process_index: int, process_count: int
  ) -> Iterator[Document]:
    # Lazy import so a missing optional dep does not break unit tests.
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(self._name, split=self._split, streaming=True)
    count = 0
    for i, row in enumerate(ds):
      if i % process_count != process_index:
        continue
      text = row.get(self._text_key)
      if text is None:
        raise ValueError(f"HF row {i} missing key {self._text_key!r}.")
      yield Document(doc_id=i, text=text)
      count += 1
      if self._max_documents and count >= self._max_documents:
        break


class PreTokenizedBackend:
  """Numpy ``.npy`` of pre-tokenized rows, shape ``[num_docs, seq_len]``.

  Useful for reproducibility (deterministic tokenization separated from
  extraction).
  """

  def __init__(self, path: str, max_documents: int = 0):
    self._path = epath.Path(path)
    self._max_documents = int(max_documents)

  def iter_for_process(
      self, process_index: int, process_count: int
  ) -> Iterator[Document]:
    import numpy as np

    # epath supports gs://; np.load needs a local path, so read bytes through epath.
    with self._path.open("rb") as f:
      arr = np.load(f)
    if arr.ndim != 2:
      raise ValueError(f"PreTokenizedBackend expects 2-D array, got {arr.shape}")
    count = 0
    for i in range(arr.shape[0]):
      if i % process_count != process_index:
        continue
      tokens = arr[i].tolist()
      yield Document(doc_id=i, token_ids=tokens)
      count += 1
      if self._max_documents and count >= self._max_documents:
        break


def build_backend(config) -> DatasetBackend:
  """Construct a backend from a MaxText HyperParameters config."""
  kind = config.activation_extraction_dataset
  path = config.activation_extraction_dataset_path
  text_key = config.activation_extraction_text_key
  max_docs = config.activation_extraction_max_documents
  if not path:
    raise ValueError(
        "activation_extraction_dataset_path must be set when extraction is enabled."
    )
  if kind == "jsonl":
    return JsonlBackend(path, text_key=text_key, max_documents=max_docs)
  if kind == "hf":
    return HuggingFaceBackend(
        name=path,
        split=config.activation_extraction_dataset_split,
        text_key=text_key,
        max_documents=max_docs,
    )
  if kind == "pretokenized":
    return PreTokenizedBackend(path, max_documents=max_docs)
  raise ValueError(f"Unknown activation_extraction_dataset={kind!r}")


__all__ = [
    "Document",
    "DatasetBackend",
    "JsonlBackend",
    "HuggingFaceBackend",
    "PreTokenizedBackend",
    "build_backend",
]
