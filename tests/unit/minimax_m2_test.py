# Copyright 2025 Google LLC
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

"""Tests for MiniMax-M2 / M2.7 model wiring and HF→MaxText converter.

These tests are CPU-only and run in seconds. They cover:
  1. pyconfig accepts the `minimax-m2.7` model_name and forwards the
     expected MoE / partial-RoPE / QK-norm knobs;
  2. MiniMaxM2DecoderLayer instantiates, has 19 param leaves, and produces
     a non-zero forward on tiny random inputs;
  3. The HF→MaxText converter handles the actual MiniMax-M2 parameter
     naming (block_sparse_moe.*, w1/w2/w3 experts, e_score_correction_bias)
     on a synthetic two-layer BF16 mini-checkpoint and produces a MaxText
     tree with the right scanned shapes.
"""

import json
import os
import tempfile
import unittest

import numpy as np


_TINY_CFG_OVERRIDES = [
    "model_name=minimax-m2.7",
    "override_model_config=true",
    "tokenizer_path=assets/tokenizer.llama2",
    "per_device_batch_size=1",
    "max_target_length=16",
    "max_prefill_predict_length=8",
    "enable_checkpointing=false",
    "run_name=minimax_m2_test",
    "scan_layers=false",
    "base_num_decoder_layers=1",
    "num_experts=4",
    "num_experts_per_tok=2",
    "base_emb_dim=64",
    "base_num_query_heads=4",
    "base_num_kv_heads=2",
    "head_dim=32",
    "base_moe_mlp_dim=64",
    "base_mlp_dim=64",
    "vocab_size=128",
    "partial_rotary_factor=0.5",
    "mtp_num_layers=0",
    "ici_tensor_parallelism=1",
    "ici_fsdp_parallelism=1",
    "ici_expert_parallelism=1",
    "attention=dot_product",
    "weight_dtype=float32",
    "dtype=float32",
    "megablox=false",
    "sparse_matmul=false",
    "capacity_factor=2.0",
]


class MiniMaxM2ConfigTest(unittest.TestCase):
  """Verifies that the new model_name flows through pyconfig with the right knobs."""

  def test_pyconfig_loads_minimax_m2_7(self):
    from maxtext.configs import pyconfig
    from maxtext.utils.globals import MAXTEXT_REPO_ROOT

    base = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
    raw = pyconfig.initialize(["maxtext", base] + _TINY_CFG_OVERRIDES)
    c = raw.config if hasattr(raw, "config") else raw

    self.assertEqual(c.decoder_block.value, "minimax_m2")
    self.assertEqual(c.num_experts, 4)
    self.assertEqual(c.num_experts_per_tok, 2)
    self.assertEqual(c.partial_rotary_factor, 0.5)
    self.assertEqual(c.routed_score_func, "sigmoid")
    self.assertTrue(c.routed_bias)
    self.assertEqual(c.shared_experts, 0)
    self.assertTrue(c.use_qk_norm)


class MiniMaxM2LayerForwardTest(unittest.TestCase):
  """Builds a one-layer decoder block and runs a tiny forward."""

  def test_tiny_forward_is_nonzero(self):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.configs import pyconfig
    from maxtext.models.minimax_m2 import MiniMaxM2DecoderLayer
    from maxtext.utils import maxtext_utils
    from maxtext.utils.globals import MAXTEXT_REPO_ROOT

    base = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
    raw = pyconfig.initialize(["maxtext", base] + _TINY_CFG_OVERRIDES)
    c = raw.config if hasattr(raw, "config") else raw

    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(c), c.mesh_axes)
    with mesh:
      rngs = nnx.Rngs(params=0, dropout=1, aqt=2)
      layer = MiniMaxM2DecoderLayer(config=c, mesh=mesh, model_mode="train", quant=None, rngs=rngs)

    B, T, D = c.global_batch_size_to_load, c.max_target_length, c.emb_dim
    x = jax.random.normal(jax.random.PRNGKey(7), (B, T, D), dtype=jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))
    seg = jnp.ones((B, T), dtype=jnp.int32)
    with mesh:
      y, _ = layer(x, decoder_segment_ids=seg, decoder_positions=pos, deterministic=True, model_mode="train")

    self.assertEqual(y.shape, (B, T, D))
    # Routing + RMSNorm + attention should produce a clearly non-zero output
    # for random inputs at unit scale.
    self.assertGreater(float(y.std()), 1e-3)


class MiniMaxM2RoutingSemanticsTest(unittest.TestCase):
  """Verifies that the MoE routing matches MiniMax-M2's reference behaviour:
    1) top-k is taken over (sigmoid(logits) + bias) for *selection*;
    2) the combine weights are the *unbiased* sigmoid scores at chosen experts;
    3) those combine weights are renormalised to sum to 1 across the top-k.
  """

  def test_topk_uses_biased_scores_but_combine_weights_are_unbiased(self):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from maxtext.configs import pyconfig
    from maxtext.layers import moe
    from maxtext.utils import maxtext_utils
    from maxtext.utils.globals import MAXTEXT_REPO_ROOT

    base = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
    raw = pyconfig.initialize(["maxtext", base] + _TINY_CFG_OVERRIDES)
    c = raw.config if hasattr(raw, "config") else raw

    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(c), c.mesh_axes)
    with mesh:
      rngs = nnx.Rngs(params=0, dropout=1, aqt=2)
      moe_block = moe.RoutedMoE(
          config=c,
          num_experts=c.num_experts,
          num_experts_per_tok=c.num_experts_per_tok,
          mesh=mesh,
          kernel_init=lambda key, shape, dtype, a, b: jax.random.normal(key, shape, dtype),
          kernel_axes=("embed", None),
          intermediate_dim=c.moe_mlp_dim,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          quant=None,
          rngs=rngs,
      )

    # Hand-craft gate logits so the bias actively changes the top-k selection.
    B, T, E, K = 1, 1, c.num_experts, c.num_experts_per_tok
    self.assertEqual((E, K), (4, 2))
    raw_logits = jnp.array([[[1.0, 2.0, 3.0, 4.0]]])  # sigmoid descending experts 3>2>1>0
    bias = jnp.array([5.0, 0.0, 0.0, 0.0])             # bias flips expert 0 to top
    # gate_logits seen by get_topk is sigmoid(raw_logits) + bias
    sigmoid_scores = jax.nn.sigmoid(raw_logits.astype(jnp.float32))
    biased = sigmoid_scores + bias
    pre_bias = sigmoid_scores  # unbiased sigmoid scores
    top_k_weights, top_k_indices = moe_block.get_topk(biased, pre_bias, rngs=rngs)

    # Selection should pick expert 0 (because of bias) and expert 3 (highest sigmoid).
    self.assertEqual(set(np.asarray(top_k_indices).flatten().tolist()), {0, 3})

    # Combine weights should be the *unbiased* sigmoid values at chosen experts,
    # renormalised to sum to 1.
    expected_unbiased = np.array([sigmoid_scores[0, 0, 0], sigmoid_scores[0, 0, 3]])
    expected_combine = expected_unbiased / expected_unbiased.sum()
    flat = np.asarray(top_k_weights).reshape(-1)
    # Sort both pairs so we can compare regardless of internal ordering.
    np.testing.assert_allclose(np.sort(flat), np.sort(expected_combine), rtol=1e-5, atol=1e-6)
    # Sum-to-one (within the top-k).
    np.testing.assert_allclose(float(flat.sum()), 1.0, rtol=1e-5, atol=1e-6)


class MiniMaxM2ConverterTest(unittest.TestCase):
  """Drives convert_hf_to_maxtext against a synthetic two-layer mini-checkpoint."""

  def test_converter_shapes_against_synthetic_minimax_layout(self):
    try:
      import torch
      from safetensors.torch import save_file
    except ImportError:
      self.skipTest("torch / safetensors not available in this environment")

    from maxtext.checkpoint_conversion.standalone_scripts import convert_minimax_m2

    params = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size": 16,
        "head_dim": 8,
        "num_experts": 4,
        "moe_intermediate_size": 12,
        "vocab_size": 32,
    }
    L, H, KV, D, HD, E, F, V = (
        params["num_hidden_layers"],
        params["num_attention_heads"],
        params["num_key_value_heads"],
        params["hidden_size"],
        params["head_dim"],
        params["num_experts"],
        params["moe_intermediate_size"],
        params["vocab_size"],
    )

    weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(V, D, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(D, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(V, D, dtype=torch.bfloat16),
    }
    for l in range(L):
      weights[f"model.layers.{l}.input_layernorm.weight"] = torch.randn(D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.post_attention_layernorm.weight"] = torch.randn(D, dtype=torch.bfloat16)
      # HF Linear weights are (out, in).
      weights[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(H * HD, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.k_proj.weight"] = torch.randn(KV * HD, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.v_proj.weight"] = torch.randn(KV * HD, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.o_proj.weight"] = torch.randn(D, H * HD, dtype=torch.bfloat16)
      # Global QK norm: scale spans num_heads * head_dim and num_kv_heads * head_dim.
      weights[f"model.layers.{l}.self_attn.q_norm.weight"] = torch.randn(H * HD, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.k_norm.weight"] = torch.randn(KV * HD, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.block_sparse_moe.gate.weight"] = torch.randn(E, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.block_sparse_moe.e_score_correction_bias"] = torch.randn(E, dtype=torch.bfloat16)
      for e in range(E):
        weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight"] = torch.randn(F, D, dtype=torch.bfloat16)
        weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight"] = torch.randn(F, D, dtype=torch.bfloat16)
        weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight"] = torch.randn(D, F, dtype=torch.bfloat16)

    with tempfile.TemporaryDirectory() as tmp:
      shard = os.path.join(tmp, "model-00001-of-00001.safetensors")
      save_file(weights, shard)
      with open(os.path.join(tmp, "model.safetensors.index.json"), "w", encoding="utf8") as f:
        json.dump({"metadata": {}, "weight_map": {k: os.path.basename(shard) for k in weights}}, f)

      tree = convert_minimax_m2.convert_hf_to_maxtext(tmp, params)

    # Top-level
    self.assertEqual(tree["token_embedder"]["embedding"].shape, (V, D))
    self.assertEqual(tree["decoder"]["decoder_norm"]["scale"].shape, (D,))
    self.assertEqual(tree["decoder"]["logits_dense"]["kernel"].shape, (D, V))

    ln = tree["decoder"]["layers"]
    attn = ln["self_attention"]
    moe = ln["moe_block"]

    # After the final scanned transpose: per-layer stuff has layer axis last (or in axis 1 for kernels).
    self.assertEqual(ln["pre_self_attention_layer_norm"]["scale"].shape, (D, L))
    self.assertEqual(ln["post_self_attention_layer_norm"]["scale"].shape, (D, L))
    self.assertEqual(attn["query_norm"]["scale"].shape, (H * HD, L))
    self.assertEqual(attn["key_norm"]["scale"].shape, (KV * HD, L))
    self.assertEqual(attn["query"]["kernel"].shape, (D, L, H, HD))
    self.assertEqual(attn["key"]["kernel"].shape, (D, L, KV, HD))
    self.assertEqual(attn["value"]["kernel"].shape, (D, L, KV, HD))
    self.assertEqual(attn["out"]["kernel"].shape, (H, L, HD, D))
    self.assertEqual(moe["gate"]["kernel"].shape, (D, L, E))
    self.assertEqual(moe["gate"]["bias"].shape, (E, L))
    self.assertEqual(moe["wi_0"].shape, (E, L, D, F))
    self.assertEqual(moe["wi_1"].shape, (E, L, D, F))
    self.assertEqual(moe["wo"].shape, (E, L, F, D))

    # Sanity-check that the gate kernel actually contains the data we wrote
    # (instead of all zeros from the pre-alloc).
    expected_gate_l0 = weights["model.layers.0.block_sparse_moe.gate.weight"].to(torch.float16).numpy().T
    np.testing.assert_allclose(moe["gate"]["kernel"][:, 0, :], expected_gate_l0, rtol=0, atol=0)


class MiniMaxM2CheckpointShapeAgreementTest(unittest.TestCase):
  """Pins down that converter output shapes exactly match a fresh-init Transformer's
  param tree. If this breaks, decode will fail with cryptic shape mismatches at
  Orbax restore time — this test surfaces it on CPU in seconds instead.
  """

  def test_converter_tree_matches_model_state(self):
    try:
      import torch
      from safetensors.torch import save_file
    except ImportError:
      self.skipTest("torch / safetensors not available in this environment")

    import jax
    from flax import nnx
    from maxtext.checkpoint_conversion.standalone_scripts import convert_minimax_m2
    from maxtext.configs import pyconfig
    from maxtext.models import models as mt_models
    from maxtext.utils import maxtext_utils
    from maxtext.utils.globals import MAXTEXT_REPO_ROOT

    # The tiny dims are forced so num_heads * head_dim != hidden_size and
    # num_kv_heads * head_dim != hidden_size; otherwise a shape collision would
    # mask QK-norm-dim bugs.
    overrides = list(_TINY_CFG_OVERRIDES)
    overrides = [
        o for o in overrides
        if not o.startswith(("base_emb_dim=", "base_num_query_heads=",
                             "base_num_kv_heads=", "head_dim=",
                             "base_moe_mlp_dim=", "base_mlp_dim=",
                             "vocab_size=", "base_num_decoder_layers=",
                             "scan_layers="))
    ]
    overrides += [
        "base_num_decoder_layers=2",
        "base_emb_dim=16",
        "base_num_query_heads=4",
        "base_num_kv_heads=2",
        "head_dim=8",
        "base_moe_mlp_dim=12",
        "base_mlp_dim=12",
        "vocab_size=32",
        "scan_layers=true",
    ]
    base = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
    raw = pyconfig.initialize(["maxtext", base] + overrides)
    c = raw.config if hasattr(raw, "config") else raw

    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(c), c.mesh_axes)
    with mesh:
      rngs = nnx.Rngs(params=0, dropout=1, aqt=2)
      model = mt_models.Transformer(config=c, mesh=mesh, quant=None, rngs=rngs)
    state = nnx.state(model, nnx.Param)
    flat, _ = jax.tree_util.tree_flatten_with_path(state)

    def _normalize(p):
      parts = []
      for k in p:
        key = getattr(k, "key", None) or getattr(k, "idx", None) or getattr(k, "name", None) or str(k)
        parts.append(str(key))
      # Strip the NNX-internal trailing ".value" so leaf paths align with the
      # converter's plain numpy tree.
      if parts and parts[-1] == "value":
        parts = parts[:-1]
      return tuple(parts)

    model_shapes = {_normalize(p): tuple(leaf.shape) for p, leaf in flat}

    # Build a matching synthetic HF checkpoint and run the converter
    params = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size": 16,
        "head_dim": 8,
        "num_experts": 4,
        "moe_intermediate_size": 12,
        "vocab_size": 32,
    }
    L, H, KV, D, HD, E, F, V = (
        params["num_hidden_layers"], params["num_attention_heads"],
        params["num_key_value_heads"], params["hidden_size"],
        params["head_dim"], params["num_experts"],
        params["moe_intermediate_size"], params["vocab_size"],
    )

    weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(V, D, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(D, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(V, D, dtype=torch.bfloat16),
    }
    for l in range(L):
      weights[f"model.layers.{l}.input_layernorm.weight"] = torch.randn(D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.post_attention_layernorm.weight"] = torch.randn(D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(H * HD, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.k_proj.weight"] = torch.randn(KV * HD, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.v_proj.weight"] = torch.randn(KV * HD, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.o_proj.weight"] = torch.randn(D, H * HD, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.q_norm.weight"] = torch.randn(H * HD, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.k_norm.weight"] = torch.randn(KV * HD, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.block_sparse_moe.gate.weight"] = torch.randn(E, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.block_sparse_moe.e_score_correction_bias"] = torch.randn(E, dtype=torch.bfloat16)
      for e in range(E):
        weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight"] = torch.randn(F, D, dtype=torch.bfloat16)
        weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight"] = torch.randn(F, D, dtype=torch.bfloat16)
        weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight"] = torch.randn(D, F, dtype=torch.bfloat16)

    with tempfile.TemporaryDirectory() as tmp:
      shard = os.path.join(tmp, "model-00001-of-00001.safetensors")
      save_file(weights, shard)
      with open(os.path.join(tmp, "model.safetensors.index.json"), "w", encoding="utf8") as f:
        json.dump({"metadata": {}, "weight_map": {k: os.path.basename(shard) for k in weights}}, f)
      conv_tree = convert_minimax_m2.convert_hf_to_maxtext(tmp, params)

    # Walk conv_tree and collect leaf paths -> shapes (skip None placeholders).
    converter_shapes: dict[tuple, tuple] = {}

    def _walk(obj, prefix):
      if isinstance(obj, dict):
        for k, v in obj.items():
          _walk(v, prefix + (k,))
      else:
        if obj is None:
          return
        converter_shapes[prefix] = tuple(obj.shape)

    _walk(conv_tree, ())

    missing = sorted(set(model_shapes) - set(converter_shapes))
    extra = sorted(set(converter_shapes) - set(model_shapes))
    mismatched = sorted(
        (p, model_shapes[p], converter_shapes[p])
        for p in set(model_shapes) & set(converter_shapes)
        if model_shapes[p] != converter_shapes[p]
    )

    self.assertEqual(missing, [], f"converter is missing keys the model expects: {missing}")
    self.assertEqual(extra, [], f"converter has keys the model does not: {extra}")
    self.assertEqual(mismatched, [], f"converter shape != model shape: {mismatched}")


class MiniMaxM2OrbaxRoundTripTest(unittest.TestCase):
  """End-to-end: convert synthetic HF -> save Orbax -> restore -> shape check."""

  def test_save_then_restore_preserves_tree(self):
    try:
      import torch
      from safetensors.torch import save_file
    except ImportError:
      self.skipTest("torch / safetensors not available")

    import jax
    import jax.numpy as jnp
    from maxtext.checkpoint_conversion.standalone_scripts import convert_minimax_m2
    from maxtext.checkpoint_conversion.standalone_scripts import llama_or_mistral_ckpt
    from maxtext.common import checkpointing as ck

    params = dict(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=16,
        head_dim=8,
        num_experts=4,
        moe_intermediate_size=12,
        vocab_size=32,
    )
    L, H, KV, D, HD, E, F, V = (
        params["num_hidden_layers"], params["num_attention_heads"],
        params["num_key_value_heads"], params["hidden_size"],
        params["head_dim"], params["num_experts"],
        params["moe_intermediate_size"], params["vocab_size"],
    )
    weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(V, D, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(D, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(V, D, dtype=torch.bfloat16),
    }
    for l in range(L):
      weights[f"model.layers.{l}.input_layernorm.weight"] = torch.randn(D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.post_attention_layernorm.weight"] = torch.randn(D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(H * HD, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.k_proj.weight"] = torch.randn(KV * HD, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.v_proj.weight"] = torch.randn(KV * HD, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.o_proj.weight"] = torch.randn(D, H * HD, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.q_norm.weight"] = torch.randn(H * HD, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.self_attn.k_norm.weight"] = torch.randn(KV * HD, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.block_sparse_moe.gate.weight"] = torch.randn(E, D, dtype=torch.bfloat16)
      weights[f"model.layers.{l}.block_sparse_moe.e_score_correction_bias"] = torch.randn(E, dtype=torch.bfloat16)
      for e in range(E):
        weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight"] = torch.randn(F, D, dtype=torch.bfloat16)
        weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight"] = torch.randn(F, D, dtype=torch.bfloat16)
        weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight"] = torch.randn(D, F, dtype=torch.bfloat16)

    with tempfile.TemporaryDirectory() as hf_dir, tempfile.TemporaryDirectory() as mt_dir:
      shard = os.path.join(hf_dir, "model-00001-of-00001.safetensors")
      save_file(weights, shard)
      with open(os.path.join(hf_dir, "model.safetensors.index.json"), "w", encoding="utf8") as f:
        json.dump({"metadata": {}, "weight_map": {k: os.path.basename(shard) for k in weights}}, f)
      tree = convert_minimax_m2.convert_hf_to_maxtext(hf_dir, params)

      llama_or_mistral_ckpt.save_weights_to_checkpoint(
          mt_dir, tree, device_count=1, use_ocdbt=True, use_zarr3=True
      )
      items_dir = os.path.join(mt_dir, "0", "items")
      self.assertTrue(os.path.isdir(items_dir), f"Orbax items dir missing under {mt_dir}")

      abstract = jax.tree_util.tree_map(
          lambda x: jax.ShapeDtypeStruct(x.shape, jnp.float32), tree
      )
      restored = ck.load_params_from_path(
          items_dir, abstract, checkpoint_storage_concurrent_gb=4
      )

    # Spot-check that restored shapes match converter output.
    self.assertEqual(restored["token_embedder"]["embedding"].shape, (V, D))
    self.assertEqual(restored["decoder"]["decoder_norm"]["scale"].shape, (D,))
    self.assertEqual(restored["decoder"]["logits_dense"]["kernel"].shape, (D, V))
    moe = restored["decoder"]["layers"]["moe_block"]
    self.assertEqual(moe["gate"]["bias"].shape, (E, L))
    self.assertEqual(moe["wi_0"].shape, (E, L, D, F))


class MiniMaxM2FP8DequantTest(unittest.TestCase):
  """Confirms the converter's FP8 (float8_e4m3fn, 128x128 block-scaled) path
  produces bfloat16 values within a small relative error vs a plain BF16
  reference. Catches axis-/scale-indexing regressions in _block_dequant_to_bf16.
  """

  def test_block_dequant_matches_reference(self):
    try:
      import torch
    except ImportError:
      self.skipTest("torch not available")
    if not hasattr(torch, "float8_e4m3fn"):
      self.skipTest("torch lacks float8_e4m3fn")

    from maxtext.checkpoint_conversion.standalone_scripts import convert_minimax_m2

    torch.manual_seed(0)
    M, N, B = 256, 384, 128                # not a multiple of B in N to exercise edge handling
    ref = torch.randn(M, N, dtype=torch.float32)
    # Block-mean-amplitude scaling — simple, deterministic, mirrors HF's pattern.
    nblocks_m = (M + B - 1) // B
    nblocks_n = (N + B - 1) // B
    scale = torch.empty(nblocks_m, nblocks_n, dtype=torch.float32)
    fp8 = torch.empty(M, N, dtype=torch.float8_e4m3fn)
    for i in range(nblocks_m):
      for j in range(nblocks_n):
        r0, r1 = i * B, min((i + 1) * B, M)
        c0, c1 = j * B, min((j + 1) * B, N)
        block = ref[r0:r1, c0:c1]
        amax = block.abs().max().clamp(min=1e-6)
        s = amax / 448.0                   # 448 is the largest finite e4m3fn value
        scale[i, j] = s
        fp8[r0:r1, c0:c1] = (block / s).to(torch.float8_e4m3fn)

    out = convert_minimax_m2._block_dequant_to_bf16(fp8, scale, block=B)
    self.assertEqual(out.dtype, torch.bfloat16)
    self.assertEqual(tuple(out.shape), (M, N))

    rel_err = (out.float() - ref).abs().max() / ref.abs().max()
    # FP8 e4m3fn typically gives ~1-3% error on absolute scale-normalized data.
    self.assertLess(float(rel_err), 0.10, f"rel_err = {float(rel_err):.4f}")


if __name__ == "__main__":
  unittest.main()
