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
"""Tests for the maybe_sow_activations helper using a toy Flax model.

The toy model mimics the structural shape of a MaxText decoder layer
(input -> attention-like -> mlp-like -> residual) so that the same
``maybe_sow_activations`` call works identically. This isolates the hook
behaviour from MaxText's full model loading machinery.
"""
from __future__ import annotations

import dataclasses
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.tools.extract_activations.hooks import (
    HOOK_ATTN_OUT,
    HOOK_MLP_OUT,
    HOOK_RESIDUAL_POST,
    INTERMEDIATES_COLLECTION,
    maybe_sow_activations,
)


@dataclasses.dataclass(frozen=True)
class ToyConfig:
  activation_extraction_enabled: bool
  activation_extraction_hooks: tuple[str, ...]
  num_layers: int = 4
  d_model: int = 8
  scan_layers: bool = False


class ToyDecoderLayer(nn.Module):
  config: ToyConfig

  @nn.compact
  def __call__(self, x):
    cfg = self.config
    attention_lnx = nn.Dense(cfg.d_model, name="attn")(x)
    mlp_lnx = nn.Dense(cfg.d_model, name="mlp")(attention_lnx)
    layer_output = x + attention_lnx + mlp_lnx
    maybe_sow_activations(
        self,
        config=cfg,
        layer_output=layer_output,
        mlp_out=mlp_lnx,
        attn_out=attention_lnx,
    )
    if cfg.scan_layers:
      return layer_output, None
    return layer_output


class ToyDecoder(nn.Module):
  config: ToyConfig

  @nn.compact
  def __call__(self, x):
    cfg = self.config
    if cfg.scan_layers:
      scan_fn = nn.scan(
          ToyDecoderLayer,
          variable_axes={"params": 0, "intermediates": 0},
          split_rngs={"params": True},
          length=cfg.num_layers,
      )
      x, _ = scan_fn(config=cfg, name="layers")(x)
      return x
    else:
      for i in range(cfg.num_layers):
        x = ToyDecoderLayer(config=cfg, name=f"layers_{i}")(x)
      return x


class HookInjectionTest(unittest.TestCase):

  def _init_and_call(self, cfg: ToyConfig, capture: bool):
    model = ToyDecoder(config=cfg)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((1, 3, cfg.d_model))
    variables = model.init(rng, x)
    mutable = [INTERMEDIATES_COLLECTION] if capture else False
    return model.apply(variables, x, mutable=mutable)

  def test_disabled_produces_no_intermediates(self):
    cfg = ToyConfig(activation_extraction_enabled=False,
                    activation_extraction_hooks=(HOOK_RESIDUAL_POST,))
    out, mvars = self._init_and_call(cfg, capture=True)
    self.assertNotIn(INTERMEDIATES_COLLECTION, mvars)

  def test_enabled_residual_only_unscanned(self):
    cfg = ToyConfig(
        activation_extraction_enabled=True,
        activation_extraction_hooks=(HOOK_RESIDUAL_POST,),
        scan_layers=False,
    )
    out, mvars = self._init_and_call(cfg, capture=True)
    inter = mvars[INTERMEDIATES_COLLECTION]
    # Without scan we expect per-layer keys.
    self.assertEqual(
        sorted(inter.keys()),
        [f"layers_{i}" for i in range(cfg.num_layers)],
    )
    for i in range(cfg.num_layers):
      self.assertIn(HOOK_RESIDUAL_POST, inter[f"layers_{i}"])
      self.assertNotIn(HOOK_MLP_OUT, inter[f"layers_{i}"])

  def test_enabled_all_hooks_scanned_stacks_along_axis_0(self):
    cfg = ToyConfig(
        activation_extraction_enabled=True,
        activation_extraction_hooks=(HOOK_RESIDUAL_POST, HOOK_MLP_OUT, HOOK_ATTN_OUT),
        scan_layers=True,
    )
    out, mvars = self._init_and_call(cfg, capture=True)
    inter = mvars[INTERMEDIATES_COLLECTION]
    # With scan we expect a single "layers" submodule whose intermediates
    # are stacked along axis 0.
    self.assertIn("layers", inter)
    for hook in (HOOK_RESIDUAL_POST, HOOK_MLP_OUT, HOOK_ATTN_OUT):
      val = inter["layers"][hook]
      # Flax wraps sown values in a tuple of length 1.
      if isinstance(val, tuple):
        val = val[0]
      self.assertEqual(val.shape, (cfg.num_layers, 1, 3, cfg.d_model))

  def test_disabled_path_is_jit_compatible(self):
    """The off-path must compile cleanly even when wrapped in jit."""
    cfg = ToyConfig(activation_extraction_enabled=False,
                    activation_extraction_hooks=())
    model = ToyDecoder(config=cfg)
    x = jnp.ones((1, 3, cfg.d_model))
    variables = model.init(jax.random.PRNGKey(0), x)
    jitted = jax.jit(lambda v, xx: model.apply(v, xx))
    out = jitted(variables, x)
    self.assertEqual(out.shape, (1, 3, cfg.d_model))


if __name__ == "__main__":
  unittest.main()
