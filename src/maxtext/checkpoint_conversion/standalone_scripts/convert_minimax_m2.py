"""
Copyright 2025 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

r"""Convert MiniMax-M2 / M2.7 HuggingFace weights to a MaxText checkpoint.

The MiniMax-M2 family ships FP8 (float8_e4m3fn, 128x128 block-scaled) weights on
the Hugging Face Hub. This converter:
  1. streams safetensors files into CPU memory one at a time;
  2. dequantizes FP8 tensors to bfloat16 on the fly (using `<name>_scale_inv`);
  3. maps HF parameter names to MaxText's scanned MoE layout;
  4. drops the MTP modules (training-only, not used for inference);
  5. writes an Orbax checkpoint via the standard MaxText path.

Run on a host with enough RAM to hold the dequantized model (M2.7: ~460 GB
in BF16, ~230 GB in FP8). On TPU VMs in the same region as your GCS bucket
this avoids any cross-region transfer.

Example:

  python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_minimax_m2 \
      --base_model_path /mnt/dtmpfs/minimax-m2.7-hf \
      --maxtext_model_path /mnt/dtmpfs/minimax-m2.7-maxtext \
      --model_size minimax-m2.7

GCS path also works for --maxtext_model_path (e.g. gs://<bucket>/<path>). To
satisfy the "no out-of-VM transfer" constraint, only use buckets in the same
region as the conversion VM.
"""

import argparse
import gc
import json
import os
import pathlib

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from maxtext.checkpoint_conversion.standalone_scripts import llama_or_mistral_ckpt
from maxtext.inference.inference_utils import str2bool
from maxtext.utils import max_logging


MODEL_PARAMS_DICT = {
    "minimax-m2": {
        "num_hidden_layers": 62,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "hidden_size": 3072,
        "head_dim": 128,
        "num_experts": 256,
        "moe_intermediate_size": 1536,
        "vocab_size": 200064,
    },
    "minimax-m2.7": {
        "num_hidden_layers": 62,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "hidden_size": 3072,
        "head_dim": 128,
        "num_experts": 256,
        "moe_intermediate_size": 1536,
        "vocab_size": 200064,
    },
}


def _block_dequant_to_bf16(x: torch.Tensor, s: torch.Tensor, block: int = 128) -> torch.Tensor:
  """Dequantizes a block-scaled FP8 tensor (M, N) to bfloat16 using scale (Mb, Nb)."""
  assert x.dim() == 2 and s.dim() == 2, "Expect 2D weight + 2D scale_inv"
  M, N = x.shape
  x = x.to(torch.float32)
  y = torch.empty_like(x, dtype=torch.bfloat16)
  for i in range(0, M, block):
    for j in range(0, N, block):
      r0, r1 = i, min(i + block, M)
      c0, c1 = j, min(j + block, N)
      scale = s[i // block, j // block]
      y[r0:r1, c0:c1] = (x[r0:r1, c0:c1] * scale).to(torch.bfloat16)
  return y


def _materialize(weight_map: dict, base_path: str, name: str, cache: dict) -> torch.Tensor:
  """Load `name` from its shard (using weight_map index) with file-level cache."""
  fname = weight_map[name]
  if fname not in cache:
    fpath = os.path.join(base_path, fname)
    cache[fname] = safe_open(fpath, framework="pt", device="cpu")
  return cache[fname].get_tensor(name)


def _get_dequantized(weight_map: dict, base_path: str, name: str, cache: dict) -> torch.Tensor:
  """Return BF16 tensor for `name`, dequantizing if it was stored as FP8."""
  t = _materialize(weight_map, base_path, name, cache)
  if t.element_size() == 1:  # FP8
    scale = _materialize(weight_map, base_path, f"{name}_scale_inv", cache)
    return _block_dequant_to_bf16(t, scale)
  return t.to(torch.bfloat16)


def convert_hf_to_maxtext(base_model_path: str, model_params: dict) -> dict:
  """Reads HF safetensors and returns a MaxText-shaped weights tree (numpy float16)."""
  num_layers = model_params["num_hidden_layers"]
  num_experts = model_params["num_experts"]
  hidden_size = model_params["hidden_size"]
  num_heads = model_params["num_attention_heads"]
  num_kv_heads = model_params["num_key_value_heads"]
  head_dim = model_params["head_dim"]
  ffn_dim = model_params["moe_intermediate_size"]
  vocab_size = model_params["vocab_size"]

  index_file = os.path.join(base_model_path, "model.safetensors.index.json")
  with open(index_file, "rt", encoding="utf8") as f:
    weight_map = json.load(f)["weight_map"]

  # Fail fast if the HF download is incomplete. Filling missing slots with
  # zeros silently would produce a checkpoint that decodes garbage on TPU —
  # the kind of bug that takes hours to chase down.
  expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
  for l in range(num_layers):
    expected |= {
        f"model.layers.{l}.input_layernorm.weight",
        f"model.layers.{l}.post_attention_layernorm.weight",
        f"model.layers.{l}.self_attn.q_proj.weight",
        f"model.layers.{l}.self_attn.k_proj.weight",
        f"model.layers.{l}.self_attn.v_proj.weight",
        f"model.layers.{l}.self_attn.o_proj.weight",
        f"model.layers.{l}.self_attn.q_norm.weight",
        f"model.layers.{l}.self_attn.k_norm.weight",
        f"model.layers.{l}.block_sparse_moe.gate.weight",
        f"model.layers.{l}.block_sparse_moe.e_score_correction_bias",
    }
    for e in range(num_experts):
      expected |= {
          f"model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight",
          f"model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight",
          f"model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight",
      }
  missing = sorted(expected - set(weight_map))
  if missing:
    sample = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
    raise RuntimeError(
        f"HF checkpoint at {base_model_path} is missing {len(missing)} expected weights "
        f"(first few: {sample}). Re-download or pass the correct path."
    )

  # Verify all shard files referenced by the index actually exist on disk.
  shards = sorted({weight_map[k] for k in expected})
  missing_files = [s for s in shards if not os.path.isfile(os.path.join(base_model_path, s))]
  if missing_files:
    raise RuntimeError(
        f"HF checkpoint at {base_model_path} is missing safetensors shard files: {missing_files}"
    )

  cache: dict = {}

  def get(name: str) -> np.ndarray:
    return _get_dequantized(weight_map, base_model_path, name, cache).to(torch.float16).numpy()

  # Pre-allocate the scanned MaxText tree
  maxtext = {
      "decoder": {
          "layers": {
              "pre_self_attention_layer_norm": {
                  "scale": np.zeros((num_layers, hidden_size), dtype=np.float16)
              },
              "post_self_attention_layer_norm": {
                  "scale": np.zeros((num_layers, hidden_size), dtype=np.float16)
              },
              "self_attention": {
                  "query": {
                      "kernel": np.zeros((num_layers, hidden_size, num_heads, head_dim), dtype=np.float16)
                  },
                  "key": {
                      "kernel": np.zeros((num_layers, hidden_size, num_kv_heads, head_dim), dtype=np.float16)
                  },
                  "value": {
                      "kernel": np.zeros((num_layers, hidden_size, num_kv_heads, head_dim), dtype=np.float16)
                  },
                  "out": {
                      "kernel": np.zeros((num_layers, num_heads, head_dim, hidden_size), dtype=np.float16)
                  },
                  "query_norm": {
                      "scale": np.zeros((num_layers, num_heads * head_dim), dtype=np.float16)
                  },
                  "key_norm": {
                      "scale": np.zeros((num_layers, num_kv_heads * head_dim), dtype=np.float16)
                  },
              },
              "moe_block": {
                  "gate": {
                      "kernel": np.zeros((num_layers, hidden_size, num_experts), dtype=np.float16),
                      "bias": np.zeros((num_layers, num_experts), dtype=np.float16),
                  },
                  "wi_0": np.zeros((num_experts, num_layers, hidden_size, ffn_dim), dtype=np.float16),
                  "wi_1": np.zeros((num_experts, num_layers, hidden_size, ffn_dim), dtype=np.float16),
                  "wo": np.zeros((num_experts, num_layers, ffn_dim, hidden_size), dtype=np.float16),
              },
          },
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  max_logging.log("Loading embedding / final norm / lm_head ...")
  maxtext["token_embedder"]["embedding"] = get("model.embed_tokens.weight")
  maxtext["decoder"]["decoder_norm"]["scale"] = get("model.norm.weight")
  # HF stores lm_head as (vocab, hidden). MaxText expects (hidden, vocab).
  maxtext["decoder"]["logits_dense"]["kernel"] = get("lm_head.weight").T
  assert maxtext["token_embedder"]["embedding"].shape == (vocab_size, hidden_size)

  ln = maxtext["decoder"]["layers"]
  attn = ln["self_attention"]
  moe = ln["moe_block"]

  for l in tqdm(range(num_layers), desc="Layers"):
    ln["pre_self_attention_layer_norm"]["scale"][l] = get(f"model.layers.{l}.input_layernorm.weight")
    ln["post_self_attention_layer_norm"]["scale"][l] = get(f"model.layers.{l}.post_attention_layernorm.weight")

    # Attention projections — HF stores Linear weight as (out, in); MaxText wants (in, n_heads, head_dim).
    attn["query"]["kernel"][l] = (
        get(f"model.layers.{l}.self_attn.q_proj.weight").T.reshape(hidden_size, num_heads, head_dim)
    )
    attn["key"]["kernel"][l] = (
        get(f"model.layers.{l}.self_attn.k_proj.weight").T.reshape(hidden_size, num_kv_heads, head_dim)
    )
    attn["value"]["kernel"][l] = (
        get(f"model.layers.{l}.self_attn.v_proj.weight").T.reshape(hidden_size, num_kv_heads, head_dim)
    )
    attn["out"]["kernel"][l] = (
        get(f"model.layers.{l}.self_attn.o_proj.weight").T.reshape(num_heads, head_dim, hidden_size)
    )
    # MiniMax-M2 uses GlobalRMSNorm: scale is (num_heads * head_dim,) and (num_kv_heads * head_dim,).
    attn["query_norm"]["scale"][l] = get(f"model.layers.{l}.self_attn.q_norm.weight")
    attn["key_norm"]["scale"][l] = get(f"model.layers.{l}.self_attn.k_norm.weight")

    # MoE — note expert weight naming uses w1/w2/w3.
    moe["gate"]["kernel"][l] = get(f"model.layers.{l}.block_sparse_moe.gate.weight").T
    moe["gate"]["bias"][l] = get(f"model.layers.{l}.block_sparse_moe.e_score_correction_bias")

    for e in range(num_experts):
      # w1: gate_proj (silu side) -> wi_0; w3: up_proj (linear side) -> wi_1; w2: down_proj -> wo
      moe["wi_0"][e, l] = get(f"model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight").T
      moe["wi_1"][e, l] = get(f"model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight").T
      moe["wo"][e, l] = get(f"model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight").T

    # Drop file handles to keep the FD count and mmap pressure low.
    cache.clear()
    gc.collect()

  # Final transpose: scanned MaxText layout interleaves (layer, ...) for params other than MoE-expert tensors.
  max_logging.log("Transposing layer-axis to scanned format ...")
  ln["pre_self_attention_layer_norm"]["scale"] = np.transpose(ln["pre_self_attention_layer_norm"]["scale"], (1, 0))
  ln["post_self_attention_layer_norm"]["scale"] = np.transpose(ln["post_self_attention_layer_norm"]["scale"], (1, 0))
  attn["query_norm"]["scale"] = np.transpose(attn["query_norm"]["scale"], (1, 0))
  attn["key_norm"]["scale"] = np.transpose(attn["key_norm"]["scale"], (1, 0))
  attn["query"]["kernel"] = np.transpose(attn["query"]["kernel"], (1, 0, 2, 3))
  attn["key"]["kernel"] = np.transpose(attn["key"]["kernel"], (1, 0, 2, 3))
  attn["value"]["kernel"] = np.transpose(attn["value"]["kernel"], (1, 0, 2, 3))
  attn["out"]["kernel"] = np.transpose(attn["out"]["kernel"], (1, 0, 2, 3))
  moe["gate"]["kernel"] = np.transpose(moe["gate"]["kernel"], (1, 0, 2))
  moe["gate"]["bias"] = np.transpose(moe["gate"]["bias"], (1, 0))

  gc.collect()
  return maxtext


def main(args):
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"

  if args.model_size not in MODEL_PARAMS_DICT:
    raise ValueError(f"Model size '{args.model_size}' not found in MODEL_PARAMS_DICT.")

  model_params = MODEL_PARAMS_DICT[args.model_size]
  max_logging.log(f"Converting MiniMax model: {args.model_size}")
  jax_weights = convert_hf_to_maxtext(args.base_model_path, model_params)
  max_logging.log(f"Saving MaxText checkpoint to {args.maxtext_model_path}")
  llama_or_mistral_ckpt.save_weights_to_checkpoint(
      args.maxtext_model_path,
      jax_weights,
      args.simulated_cpu_devices_count,
      args.use_ocdbt,
      args.use_zarr3,
  )
  max_logging.log("Checkpoint saved successfully.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert MiniMax-M2 family HF weights to a MaxText checkpoint.")
  parser.add_argument("--base_model_path", type=str, required=True,
                      help="Path to the HF MiniMax-M2 checkpoint (FP8 or BF16 safetensors).")
  parser.add_argument("--maxtext_model_path", type=str, required=True,
                      help="Destination for the MaxText checkpoint (local path or gs:// URI).")
  parser.add_argument("--model_size", type=str, required=True, choices=list(MODEL_PARAMS_DICT.keys()),
                      help="Which preset to use.")
  parser.add_argument("--simulated_cpu_devices_count", type=int, default=16,
                      help="Number of simulated CPU devices for the Orbax save step.")
  parser.add_argument("--use-ocdbt", type=str2bool, default=True, help="Use OCDBT format for saving.")
  parser.add_argument("--use-zarr3", type=str2bool, default=True, help="Use Zarr3 format for saving.")

  main(parser.parse_args())
