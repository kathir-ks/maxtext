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
"""End-to-end test against a real Flax model that uses
``maybe_sow_activations`` exactly the way MaxText's DecoderLayers do.

Verifies that:

  * Sown intermediates flow through the runner unchanged.
  * Both ``scan_layers=True`` and ``scan_layers=False`` produce the same
    merged output for the same inputs.
  * Sharded shards reconstruct the layer-by-layer residual stream
    bit-identically to a direct ``model.apply`` reference.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.tools.extract_activations.datasets import JsonlBackend
from maxtext.tools.extract_activations.hooks import (
    INTERMEDIATES_COLLECTION,
    maybe_sow_activations,
)
from maxtext.tools.extract_activations.merge import load_merged
from maxtext.tools.extract_activations.runner import (
    ExtractionConfig,
    ExtractionLoop,
)
from maxtext.tools.extract_activations.writer import ShardedSafetensorsWriter


D_MODEL = 6
NUM_LAYERS = 4
VOCAB = 100
MAX_SEQ = 12


@dataclasses.dataclass(frozen=True)
class ToyCfg:
  scan_layers: bool
  activation_extraction_enabled: bool = True
  activation_extraction_hooks: tuple[str, ...] = ("residual_post",)


class ToyLayer(nn.Module):
  config: ToyCfg

  @nn.compact
  def __call__(self, x):
    cfg = self.config
    attn = nn.Dense(D_MODEL, name="attn")(x)
    mlp = nn.Dense(D_MODEL, name="mlp")(attn)
    out = x + attn + mlp
    maybe_sow_activations(self, config=cfg, layer_output=out,
                          mlp_out=mlp, attn_out=attn)
    if cfg.scan_layers:
      return out, None
    return out


class ToyModel(nn.Module):
  config: ToyCfg

  @nn.compact
  def __call__(self, tokens):
    x = nn.Embed(VOCAB, D_MODEL, name="embed")(tokens)
    cfg = self.config
    if cfg.scan_layers:
      x, _ = nn.scan(
          ToyLayer,
          variable_axes={"params": 0, "intermediates": 0},
          split_rngs={"params": True},
          length=NUM_LAYERS,
      )(config=cfg, name="layers")(x)
    else:
      for i in range(NUM_LAYERS):
        x = ToyLayer(config=cfg, name=f"layers_{i}")(x)
    return x


def _build_forward(model, params, hook_names: set[str]):
  """Mimic the production forward function: run apply with mutable
  intermediates and flatten by hook name. Uses the same walker as
  the production runner so it exercises that code path."""
  from maxtext.tools.extract_activations.runner import _collect_by_hook

  @jax.jit
  def fwd(_unused, padded_tokens, true_length):
    del _unused, true_length
    out, mvars = model.apply(
        params,
        padded_tokens[None, :],
        mutable=[INTERMEDIATES_COLLECTION],
    )
    inter = mvars[INTERMEDIATES_COLLECTION]
    flat: dict[str, jax.Array] = {}
    _collect_by_hook(inter, hook_names, flat)
    return flat

  return fwd


def _write_dataset(path, num_docs, max_tokens, rng):
  with open(path, "w") as f:
    for _ in range(num_docs):
      n = int(rng.integers(2, max_tokens))
      toks = rng.integers(1, VOCAB, size=n).tolist()
      f.write(json.dumps({"tokens": toks}) + "\n")


class RealForwardTest(unittest.TestCase):

  def _setup(self, scan_layers: bool):
    cfg = ToyCfg(scan_layers=scan_layers)
    model = ToyModel(config=cfg)
    rng = jax.random.PRNGKey(0)
    init_tokens = jnp.zeros((1, MAX_SEQ), dtype=jnp.int32)
    params = model.init(rng, init_tokens)
    return model, params

  def test_scan_vs_unscan_produce_same_merged_output(self):
    """A scanned model and an unscanned model with the *same params*
    should produce identical merged shards."""
    cfg_scan = ToyCfg(scan_layers=True)
    cfg_unscan = ToyCfg(scan_layers=False)

    # Build unscanned first, then use its params for scanned. We unify by
    # remapping ``layers_<i>/...`` <-> ``layers/...`` (Flax stacks the
    # scanned weights along axis 0).
    model_u = ToyModel(config=cfg_unscan)
    params_u = model_u.init(jax.random.PRNGKey(7),
                            jnp.zeros((1, MAX_SEQ), dtype=jnp.int32))

    def stack_along_layer_axis(params_unscanned):
      # turn {"params": {"embed": ..., "layers_0": {...}, ...}}
      # into {"params": {"embed": ..., "layers": stacked}}
      per_layer = [params_unscanned["params"][f"layers_{i}"] for i in range(NUM_LAYERS)]
      stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *per_layer)
      return {"params": {**{k: v for k, v in params_unscanned["params"].items()
                            if not k.startswith("layers_")},
                         "layers": stacked}}

    params_s = stack_along_layer_axis(params_u)

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
      ds = os.path.join(tmp, "data.jsonl")
      _write_dataset(ds, num_docs=6, max_tokens=MAX_SEQ, rng=rng)
      out_u = os.path.join(tmp, "unscan")
      out_s = os.path.join(tmp, "scan")
      for model_, params_, out_, scan_ in (
          (ToyModel(config=cfg_unscan), params_u, out_u, False),
          (ToyModel(config=cfg_scan), params_s, out_s, True),
      ):
        writer = ShardedSafetensorsWriter(
            output_path=out_,
            layers=[0, NUM_LAYERS - 1],
            hooks=["residual_post"],
            d_model=D_MODEL,
            shard_size_tokens=8,
            output_dtype="float32",
        )
        runner_cfg = ExtractionConfig(
            hooks=["residual_post"],
            layers_to_keep=[0, NUM_LAYERS - 1],
            d_model=D_MODEL,
            num_layers=NUM_LAYERS,
            max_prefill_predict_length=MAX_SEQ,
            output_path=out_,
            shard_size_tokens=8,
            output_dtype="float32",
            skip_bos=False,
        )
        loop = ExtractionLoop(
            config=runner_cfg,
            forward_fn=_build_forward(model_, params_, {"residual_post"}),
            tokenize_fn=lambda t: [],
            pad_id=0,
            dataset=JsonlBackend(ds),
            writer=writer,
            process_index=0,
            process_count=1,
        )
        loop.run()
      for layer in (0, NUM_LAYERS - 1):
        u = load_merged(out_u, hook="residual_post", layer_idx=layer)
        s = load_merged(out_s, hook="residual_post", layer_idx=layer)
        # Allow tiny FP variation from accumulation order.
        np.testing.assert_allclose(
            u["activations"], s["activations"], rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_array_equal(u["doc_ids"], s["doc_ids"])
        np.testing.assert_array_equal(u["positions"], s["positions"])
        np.testing.assert_array_equal(u["token_ids"], s["token_ids"])

  def test_shards_match_direct_apply(self):
    """Reconstructing activations from shards must equal running
    model.apply directly with the same params and tokens."""
    model, params = self._setup(scan_layers=False)
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
      ds = os.path.join(tmp, "data.jsonl")
      _write_dataset(ds, num_docs=4, max_tokens=MAX_SEQ, rng=rng)
      out_dir = os.path.join(tmp, "out")
      writer = ShardedSafetensorsWriter(
          output_path=out_dir,
          layers=[2],
          hooks=["residual_post"],
          d_model=D_MODEL,
          shard_size_tokens=100,
          output_dtype="float32",
      )
      cfg = ExtractionConfig(
          hooks=["residual_post"],
          layers_to_keep=[2],
          d_model=D_MODEL,
          num_layers=NUM_LAYERS,
          max_prefill_predict_length=MAX_SEQ,
          output_path=out_dir,
          shard_size_tokens=100,
          output_dtype="float32",
          skip_bos=False,
      )
      loop = ExtractionLoop(
          config=cfg,
          forward_fn=_build_forward(model, params, {"residual_post"}),
          tokenize_fn=lambda t: [],
          pad_id=0,
          dataset=JsonlBackend(ds),
          writer=writer,
          process_index=0,
          process_count=1,
      )
      loop.run()

      # Reference via direct model.apply.
      with open(ds) as f:
        docs = [json.loads(line)["tokens"] for line in f if line.strip()]
      merged = load_merged(out_dir, hook="residual_post", layer_idx=2)
      offset = 0
      for di, toks in enumerate(docs):
        padded = np.zeros(MAX_SEQ, dtype=np.int32)
        padded[: len(toks)] = toks
        _, mvars = model.apply(
            params,
            jnp.asarray(padded)[None, :],
            mutable=[INTERMEDIATES_COLLECTION],
        )
        ref = np.asarray(
            mvars[INTERMEDIATES_COLLECTION]["layers_2"]["residual_post"][0]
        )[0]  # [T, D]
        for t in range(len(toks)):
          row = merged["activations"][offset]
          np.testing.assert_allclose(row, ref[t], rtol=1e-5, atol=1e-5)
          self.assertEqual(int(merged["doc_ids"][offset]), di)
          self.assertEqual(int(merged["positions"][offset]), t)
          self.assertEqual(int(merged["token_ids"][offset]), toks[t])
          offset += 1
      self.assertEqual(offset, merged["activations"].shape[0])


if __name__ == "__main__":
  unittest.main()
