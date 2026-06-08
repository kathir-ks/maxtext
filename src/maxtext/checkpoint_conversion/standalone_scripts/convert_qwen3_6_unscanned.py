# Copyright 2026 Google LLC
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

"""Convert weights from Qwen3.6 (dense, hybrid GDN+GatedAttention) HF safetensors
to a MaxText unscanned Orbax checkpoint.

Currently text-only: vision_tower and MTP head weights are skipped.

Example:

python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_qwen3_6_unscanned \\
    --base-model-path /dev/shm/qwen3.6-27b \\
    --maxtext-model-path /dev/shm/qwen3.6-27b-mt \\
    --model-size qwen3.6-27b
"""

# pylint: disable=g-line-too-long
import argparse
import gc
import logging
import os
import pathlib

os.environ["JAX_PLATFORMS"] = "cpu"

import ml_dtypes
import psutil
import numpy as np
from safetensors import safe_open
import torch
from tqdm import tqdm
from typing import Any, Dict

from maxtext.checkpoint_conversion.standalone_scripts.llama_or_mistral_ckpt import save_weights_to_checkpoint
from maxtext.inference.inference_utils import str2bool
from maxtext.utils import max_logging


MODEL_PARAMS_DICT = {
    "qwen3.6-27b": {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        # Dense MLP (num_experts == 1 path in qwen3_5.py)
        "num_experts": 1,
        "intermediate_size": 17408,
        # Gated Attention (GA) params: applied on layers where (l + 1) % 4 == 0.
        "head_dim": 256,
        "ga_num_q_heads": 24,
        "ga_num_kv_heads": 4,
        # Gated DeltaNet (GDN) params: applied on the other layers.
        "gdn_num_value_heads": 48,
        "gdn_num_key_heads": 16,
        "gdn_key_head_dim": 128,
        "gdn_value_head_dim": 128,
        "gdn_conv_kernel_dim": 4,
        "inhomogeneous_layer_cycle_interval": 4,
    },
}

# numpy doesn't have native bf16; ml_dtypes is the standard bridge.
CAST_DTYPE = ml_dtypes.bfloat16


def _pt_to_np(pt_weight, cast_dtype=None, transpose=False):
  if cast_dtype:
    np_weight = pt_weight.to(torch.float32).numpy().astype(cast_dtype)
  else:
    np_weight = pt_weight.to(torch.float32).numpy()
  if transpose:
    np_weight = np_weight.transpose()
  return np_weight


def _fuse_qkv_z(in_proj_qkv_pt, in_proj_z_pt, num_k_heads: int, head_k_dim: int,
                num_v_heads: int, head_v_dim: int, hidden_size: int) -> np.ndarray:
  """Re-interleave HF's head-major (Q|K|V) + Z projections into MaxText's
  per-K-head-group fused in_proj_qkvz weight.

  HF storage in `in_proj_qkv.weight` (out, in) is head-major:
    rows[0 : key_dim]                    = all Q heads stacked
    rows[key_dim : 2*key_dim]            = all K heads stacked
    rows[2*key_dim : 2*key_dim+value_dim]= all V slots stacked
  And `in_proj_z.weight` is value_dim rows for Z.

  MaxText expects `in_proj_qkvz.weight` (in, out) where the flat out dim
  reshapes to (num_k_heads, 2*head_k_dim + 2 * v_per_k * head_v_dim) with the
  per-K-head block laid out as [Q_h, K_h, V_h_block, Z_h_block].
  """
  key_dim = num_k_heads * head_k_dim
  value_dim = num_v_heads * head_v_dim
  assert num_v_heads % num_k_heads == 0
  v_per_k = num_v_heads // num_k_heads

  qkv = _pt_to_np(in_proj_qkv_pt, cast_dtype=CAST_DTYPE)  # (2*key_dim + value_dim, hidden)
  z = _pt_to_np(in_proj_z_pt, cast_dtype=CAST_DTYPE)      # (value_dim, hidden)
  assert qkv.shape == (2 * key_dim + value_dim, hidden_size), (qkv.shape, key_dim, value_dim, hidden_size)
  assert z.shape == (value_dim, hidden_size), (z.shape, value_dim, hidden_size)

  q = qkv[:key_dim].reshape(num_k_heads, head_k_dim, hidden_size)
  k = qkv[key_dim : 2 * key_dim].reshape(num_k_heads, head_k_dim, hidden_size)
  # V heads are flat; GQA convention groups consecutive V heads per K head.
  v = qkv[2 * key_dim :].reshape(num_k_heads, v_per_k * head_v_dim, hidden_size)
  z = z.reshape(num_k_heads, v_per_k * head_v_dim, hidden_size)

  per_kh = np.concatenate([q, k, v, z], axis=1)  # (H_k, 2*D_k + 2*V_per_K*D_v, hidden)
  fused = per_kh.reshape(num_k_heads * (2 * head_k_dim + 2 * v_per_k * head_v_dim), hidden_size)
  return fused.transpose()  # (hidden, fused_out)


def _fuse_b_a(in_proj_b_pt, in_proj_a_pt, num_k_heads: int, num_v_heads: int,
              hidden_size: int) -> np.ndarray:
  """HF stores `b` and `a` flat over V slots (one scalar per V slot per token).
  MaxText fused `in_proj_ba` lays out per K-head group as [B_group, A_group]."""
  v_per_k = num_v_heads // num_k_heads
  b = _pt_to_np(in_proj_b_pt, cast_dtype=CAST_DTYPE).reshape(num_k_heads, v_per_k, hidden_size)
  a = _pt_to_np(in_proj_a_pt, cast_dtype=CAST_DTYPE).reshape(num_k_heads, v_per_k, hidden_size)
  per_kh = np.concatenate([b, a], axis=1)  # (H_k, 2*v_per_k, hidden)
  fused = per_kh.reshape(num_k_heads * 2 * v_per_k, hidden_size)
  return fused.transpose()  # (hidden, 2*num_v_heads)


def _dense_mlp_pytree() -> Dict[str, Any]:
  return {
      "wi_0": {"kernel": None},
      "wi_1": {"kernel": None},
      "wo": {"kernel": None},
  }


def create_unscanned_layer_pytree(layer_idx: int) -> Dict[str, Any]:
  """Per-layer parameter skeleton. Dense MLP is identical across all layers;
  attention sub-tree differs between Gated Attention (every 4th layer) and GDN."""
  base = {
      "input_layernorm": {"scale": None},
      "post_attention_layernorm": {"scale": None},
      "mlp": _dense_mlp_pytree(),
  }
  if (layer_idx + 1) % 4 == 0:
    base["attention"] = {
        "attention": {
            "query": {"kernel": None},
            "key": {"kernel": None},
            "value": {"kernel": None},
            "out": {"kernel": None},
            "query_norm": {"scale": None},
            "key_norm": {"scale": None},
        },
    }
  else:
    base["attention"] = {
        "A_log": None,
        "conv1d": {"kernel": None},
        "dt_bias": None,
        "in_proj_ba": {"kernel": None},
        "in_proj_qkvz": {"kernel": None},
        "norm": {"rms_norm": {"scale": None}},
        "out_proj": {"kernel": None},
    }
  return base


# HF keys we keep (everything else — vision_tower.*, mtp.*, etc. — is dropped).
def _is_kept_key(key: str) -> bool:
  if key.startswith("model.language_model."):
    # Multimodal Qwen3.6 nests text weights under language_model.*; normalize.
    return True
  if key.startswith("model.") and not key.startswith("model.vision_tower"):
    return True
  if key.startswith("lm_head."):
    return True
  return False


def _strip_lm_prefix(key: str) -> str:
  """Some Qwen3.6 checkpoints nest text weights under model.language_model.*.
  Normalize back to model.* so the rest of the script doesn't branch."""
  prefix = "model.language_model."
  if key.startswith(prefix):
    return "model." + key[len(prefix):]
  return key


def convert_hf_to_maxtext(base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process):
  """Convert HF safetensors → nested numpy dict matching the MaxText param tree."""

  num_layers = model_params["num_hidden_layers"]
  hidden_size = model_params["hidden_size"]
  ga_num_q_heads = model_params["ga_num_q_heads"]
  ga_num_kv_heads = model_params["ga_num_kv_heads"]
  head_dim = model_params["head_dim"]
  intermediate_size = model_params["intermediate_size"]

  max_logging.log(f"Loading base model from {base_model_path}")
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
  if not ckpt_paths:
    raise FileNotFoundError(f"No safetensors found at {base_model_path}")
  chkpt_vars: Dict[str, torch.Tensor] = {}

  dropped_prefixes: Dict[str, int] = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint shard {i+1}/{len(ckpt_paths)}: {ckpt_path.name}")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        if not _is_kept_key(key):
          prefix = key.split(".")[0:3]
          dropped_prefixes[".".join(prefix)] = dropped_prefixes.get(".".join(prefix), 0) + 1
          continue
        chkpt_vars[_strip_lm_prefix(key)] = f.get_tensor(key)

  if dropped_prefixes:
    max_logging.log(f"Skipped HF weight groups (vision/mtp/etc): {dropped_prefixes}")
  max_logging.log(f"Loaded {len(chkpt_vars)} text-model tensors. Memory: {mem_info.memory_info().rss / 1024**3:.1f} GB")

  jax_weights: Dict[str, Any] = {
      "token_embedder": {"embedding": None},
      "decoder": {
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
  }
  for l in range(num_layers):
    jax_weights["decoder"][f"layers_{l}"] = create_unscanned_layer_pytree(l)

  # Non-layer weights.
  max_logging.log("Populating non-layer weights")
  jax_weights["token_embedder"]["embedding"] = _pt_to_np(
      chkpt_vars["model.embed_tokens.weight"], cast_dtype=CAST_DTYPE
  )
  jax_weights["decoder"]["decoder_norm"]["scale"] = _pt_to_np(
      chkpt_vars["model.norm.weight"], cast_dtype=CAST_DTYPE
  )
  jax_weights["decoder"]["logits_dense"]["kernel"] = _pt_to_np(
      chkpt_vars["lm_head.weight"], cast_dtype=CAST_DTYPE
  ).transpose()

  # Per-layer attention + MLP weights.
  max_logging.log("Processing attention layers (GA + GDN) and dense MLPs")
  for l in tqdm(range(num_layers), desc="layers", leave=False):
    layer = jax_weights["decoder"][f"layers_{l}"]

    layer["input_layernorm"]["scale"] = _pt_to_np(
        chkpt_vars[f"model.layers.{l}.input_layernorm.weight"], cast_dtype=CAST_DTYPE
    )
    layer["post_attention_layernorm"]["scale"] = _pt_to_np(
        chkpt_vars[f"model.layers.{l}.post_attention_layernorm.weight"], cast_dtype=CAST_DTYPE
    )

    # Dense MLP: gate_proj=wi_0, up_proj=wi_1, down_proj=wo. (silu, linear) SwiGLU.
    layer["mlp"]["wi_0"]["kernel"] = _pt_to_np(
        chkpt_vars[f"model.layers.{l}.mlp.gate_proj.weight"], cast_dtype=CAST_DTYPE
    ).transpose()
    layer["mlp"]["wi_1"]["kernel"] = _pt_to_np(
        chkpt_vars[f"model.layers.{l}.mlp.up_proj.weight"], cast_dtype=CAST_DTYPE
    ).transpose()
    layer["mlp"]["wo"]["kernel"] = _pt_to_np(
        chkpt_vars[f"model.layers.{l}.mlp.down_proj.weight"], cast_dtype=CAST_DTYPE
    ).transpose()

    if (l + 1) % 4 == 0:
      # Gated full attention. q_proj contains [query, gate] concatenated along the
      # head-feature dim, so its head-dim is head_dim * 2.
      gated = layer["attention"]["attention"]
      gated["query"]["kernel"] = (
          _pt_to_np(chkpt_vars[f"model.layers.{l}.self_attn.q_proj.weight"], cast_dtype=CAST_DTYPE)
          .transpose()
          .reshape(hidden_size, ga_num_q_heads, head_dim * 2)
      )
      gated["key"]["kernel"] = (
          _pt_to_np(chkpt_vars[f"model.layers.{l}.self_attn.k_proj.weight"], cast_dtype=CAST_DTYPE)
          .transpose()
          .reshape(hidden_size, ga_num_kv_heads, head_dim)
      )
      gated["value"]["kernel"] = (
          _pt_to_np(chkpt_vars[f"model.layers.{l}.self_attn.v_proj.weight"], cast_dtype=CAST_DTYPE)
          .transpose()
          .reshape(hidden_size, ga_num_kv_heads, head_dim)
      )
      gated["out"]["kernel"] = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.self_attn.o_proj.weight"], cast_dtype=CAST_DTYPE
      ).transpose()
      gated["query_norm"]["scale"] = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.self_attn.q_norm.weight"], cast_dtype=CAST_DTYPE
      )
      gated["key_norm"]["scale"] = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.self_attn.k_norm.weight"], cast_dtype=CAST_DTYPE
      )
    else:
      # Gated DeltaNet (linear attention). HF Qwen3.6 splits the original
      # Qwen3-Next fused projections back into four (in_proj_qkv / _z / _b / _a);
      # we re-interleave per K-head group to match MaxText's expected layout.
      lin = layer["attention"]
      lin["A_log"] = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.linear_attn.A_log"], cast_dtype=CAST_DTYPE
      )
      lin["conv1d"]["kernel"] = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.linear_attn.conv1d.weight"], cast_dtype=CAST_DTYPE
      ).transpose(2, 1, 0)
      lin["dt_bias"] = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.linear_attn.dt_bias"], cast_dtype=CAST_DTYPE
      )
      lin["in_proj_qkvz"]["kernel"] = _fuse_qkv_z(
          chkpt_vars[f"model.layers.{l}.linear_attn.in_proj_qkv.weight"],
          chkpt_vars[f"model.layers.{l}.linear_attn.in_proj_z.weight"],
          num_k_heads=model_params["gdn_num_key_heads"],
          head_k_dim=model_params["gdn_key_head_dim"],
          num_v_heads=model_params["gdn_num_value_heads"],
          head_v_dim=model_params["gdn_value_head_dim"],
          hidden_size=hidden_size,
      )
      lin["in_proj_ba"]["kernel"] = _fuse_b_a(
          chkpt_vars[f"model.layers.{l}.linear_attn.in_proj_b.weight"],
          chkpt_vars[f"model.layers.{l}.linear_attn.in_proj_a.weight"],
          num_k_heads=model_params["gdn_num_key_heads"],
          num_v_heads=model_params["gdn_num_value_heads"],
          hidden_size=hidden_size,
      )
      lin["norm"]["rms_norm"]["scale"] = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.linear_attn.norm.weight"], cast_dtype=CAST_DTYPE
      )
      lin["out_proj"]["kernel"] = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.linear_attn.out_proj.weight"], cast_dtype=CAST_DTYPE
      ).transpose()

    gc.collect()

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  del chkpt_vars
  gc.collect()
  return jax_weights


def convert_to_jax_weights(base_model_path: str, model_size: str):
  if model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError(f"Model '{model_size}' is not supported.")
  model_params = MODEL_PARAMS_DICT[model_size]
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return convert_hf_to_maxtext(base_model_path, model_size, model_params, mem_info)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-model-path", type=str, required=True)
  parser.add_argument("--maxtext-model-path", type=str, required=True)
  parser.add_argument("--model-size", type=str, required=True, choices=list(MODEL_PARAMS_DICT.keys()))
  parser.add_argument("--simulated-cpu-devices-count", type=int, required=False, default=16)
  parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
  parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
  args = parser.parse_args()

  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"

  save_weights_to_checkpoint(
      args.maxtext_model_path,
      convert_to_jax_weights(args.base_model_path, args.model_size),
      args.simulated_cpu_devices_count,
      args.use_ocdbt,
      args.use_zarr3,
  )
  max_logging.log(f"Successfully saved base_weights to {args.maxtext_model_path}.")
