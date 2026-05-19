"""
Copyright 2025 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
"""

r"""Streaming HF -> per-leaf .npy converter for MiniMax-M2 / M2.7.

Memory-efficient companion to ``convert_minimax_m2``. Instead of allocating
the full ~460 GB scanned tree in process RAM, each big leaf is written
through ``numpy.lib.format.open_memmap`` directly onto dtmpfs, layer by
layer, with a per-layer working set of ~3-7 GB. As layers finish, the
safetensors shards they consumed are unlinked, so the 215 GB FP8 footprint
shrinks while the BF16 output grows; peak RAM stays under 500 GB on a
708 GB v6e host.

The output is **not** an Orbax checkpoint. It is a directory of .npy
files keyed by their MaxText leaf path (e.g.
``decoder.layers.moe_block.wi_0.npy``). Use
``decode_minimax_m2_npy.py`` to load these at decode time without holding
the full tree on any one process.

Usage::

    python -m maxtext.checkpoint_conversion.standalone_scripts.convert_minimax_m2_streaming \
        --base_model_path /mnt/dtmpfs/minimax-m2.7-hf \
        --output_dir /mnt/dtmpfs/minimax-m2.7-npy \
        --model_size minimax-m2.7

Set ``MINIMAX_PROGRESSIVE_FP8_CLEANUP=1`` to unlink consumed FP8 shards
mid-stream (default on).
"""

import argparse
import collections
import gc
import json
import os
import pathlib
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from maxtext.utils import max_logging


MODEL_PARAMS_DICT = {
    "minimax-m2": dict(num_hidden_layers=62, num_attention_heads=48, num_key_value_heads=8,
                       hidden_size=3072, head_dim=128, num_experts=256,
                       moe_intermediate_size=1536, vocab_size=200064),
    "minimax-m2.7": dict(num_hidden_layers=62, num_attention_heads=48, num_key_value_heads=8,
                        hidden_size=3072, head_dim=128, num_experts=256,
                        moe_intermediate_size=1536, vocab_size=200064),
}


def _block_dequant_to_bf16(x: torch.Tensor, s: torch.Tensor, block: int = 128) -> torch.Tensor:
  """Dequantize block-scaled FP8 (M, N) → bfloat16 using scale (Mb, Nb)."""
  assert x.dim() == 2 and s.dim() == 2
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


def _get_tensor_bf16(weight_map: Dict[str, str], base_path: str, name: str,
                    handle_cache: Dict[str, "safe_open"]) -> torch.Tensor:
  fname = weight_map[name]
  if fname not in handle_cache:
    handle_cache[fname] = safe_open(os.path.join(base_path, fname), framework="pt", device="cpu")
  t = handle_cache[fname].get_tensor(name)
  if t.element_size() == 1:  # FP8
    scale = handle_cache[fname].get_tensor(f"{name}_scale_inv") if (
        f"{name}_scale_inv" in weight_map and weight_map[f"{name}_scale_inv"] == fname
    ) else _get_tensor_bf16(weight_map, base_path, f"{name}_scale_inv", handle_cache)
    return _block_dequant_to_bf16(t, scale)
  return t.to(torch.bfloat16)


def _np_from_torch(t: torch.Tensor) -> np.ndarray:
  # MaxText loader will read these back as float16 / bfloat16; we store as
  # float16 to halve disk/RAM while keeping the same byte width as bfloat16.
  return t.to(torch.float32).numpy().astype(np.float16)


def _build_file_layer_map(weight_map: Dict[str, str], num_layers: int) -> Dict[str, Set[int]]:
  """Return file → set of layer indices whose tensors live in that file.

  Tensors not associated with a numbered layer (embeddings, final norm,
  lm_head) are associated with layer -1 (sentinel).
  """
  out: Dict[str, Set[int]] = collections.defaultdict(set)
  for name, fname in weight_map.items():
    if name.startswith("model.layers."):
      l = int(name.split(".")[2])
      out[fname].add(l)
    else:
      out[fname].add(-1)
  return out


def _file_is_safe_to_unlink(layers_in_file: Set[int], current_layer: int) -> bool:
  """A file is safe to unlink once every layer it contains has been processed."""
  return all(l == -1 or l < current_layer for l in layers_in_file) and (-1 not in layers_in_file)


# ----- Leaf shape book keeping --------------------------------------------------

def leaf_shapes(p: dict) -> Dict[str, Tuple[Tuple[int, ...], str, str]]:
  """Map MaxText leaf path -> (shape, np_dtype, sharding-axis label).

  Shapes match MaxText's *post-transpose* scanned layout (see the final
  np.transpose pass at the bottom of convert_hf_to_maxtext in
  convert_minimax_m2.py): the layer axis lives at position 1 for every
  per-layer leaf, NOT position 0. The MaxText TP sharding annotations
  shard the non-layer hidden axis, so this is the layout the engine
  actually expects.
  """
  L, H = p["num_hidden_layers"], p["hidden_size"]
  Hq, Hkv, D = p["num_attention_heads"], p["num_key_value_heads"], p["head_dim"]
  E, F = p["num_experts"], p["moe_intermediate_size"]
  V = p["vocab_size"]
  return {
      # ----- non-layered -----
      "token_embedder.embedding": ((V, H), "float16", "vocab"),
      "decoder.decoder_norm.scale": ((H,), "float16", "hidden"),
      "decoder.logits_dense.kernel": ((H, V), "float16", "vocab"),
      # ----- per-layer norm / attention (layer axis = position 1) -----
      "decoder.layers.pre_self_attention_layer_norm.scale": ((H, L), "float16", "hidden-layer"),
      "decoder.layers.post_self_attention_layer_norm.scale": ((H, L), "float16", "hidden-layer"),
      "decoder.layers.self_attention.query.kernel": ((H, L, Hq, D), "float16", "hidden-layer-q"),
      "decoder.layers.self_attention.key.kernel": ((H, L, Hkv, D), "float16", "hidden-layer-kv"),
      "decoder.layers.self_attention.value.kernel": ((H, L, Hkv, D), "float16", "hidden-layer-kv"),
      "decoder.layers.self_attention.out.kernel": ((Hq, L, D, H), "float16", "q-layer-d-hidden"),
      "decoder.layers.self_attention.query_norm.scale": ((Hq * D, L), "float16", "q-layer"),
      "decoder.layers.self_attention.key_norm.scale": ((Hkv * D, L), "float16", "kv-layer"),
      # ----- per-layer MoE (gate kernel: (H, L, E); gate bias: (E, L); experts: (E, L, ...)) -----
      "decoder.layers.moe_block.gate.kernel": ((H, L, E), "float16", "hidden-layer-experts"),
      "decoder.layers.moe_block.gate.bias": ((E, L), "float16", "experts-layer"),
      "decoder.layers.moe_block.wi_0": ((E, L, H, F), "float16", "experts-layer"),
      "decoder.layers.moe_block.wi_1": ((E, L, H, F), "float16", "experts-layer"),
      "decoder.layers.moe_block.wo": ((E, L, F, H), "float16", "experts-layer"),
  }


def open_memmap(path: pathlib.Path, shape, dtype: str) -> np.ndarray:
  path.parent.mkdir(parents=True, exist_ok=True)
  arr = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.dtype(dtype), shape=tuple(shape))
  return arr


def main(args):
  if args.model_size not in MODEL_PARAMS_DICT:
    raise SystemExit(f"unknown model_size {args.model_size!r}")
  p = MODEL_PARAMS_DICT[args.model_size]

  base = args.base_model_path
  out_dir = pathlib.Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  with open(os.path.join(base, "model.safetensors.index.json"), "rt", encoding="utf8") as f:
    weight_map = json.load(f)["weight_map"]

  num_layers = p["num_hidden_layers"]
  num_experts = p["num_experts"]
  H, Hq, Hkv, D = p["hidden_size"], p["num_attention_heads"], p["num_key_value_heads"], p["head_dim"]
  V = p["vocab_size"]
  F = p["moe_intermediate_size"]

  # ----- Pre-flight: every expected tensor must be in the index. -----
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
    raise RuntimeError(
        f"HF checkpoint at {base} is missing {len(missing)} expected weights "
        f"(first few: {missing[:5]})"
    )

  # ----- Open memmaps for every leaf -----
  shapes = leaf_shapes(p)
  arrs: Dict[str, np.ndarray] = {}
  total_bytes = 0
  for leaf, (shape, dtype, _) in shapes.items():
    arrs[leaf] = open_memmap(out_dir / f"{leaf}.npy", shape, dtype)
    total_bytes += np.prod(shape) * np.dtype(dtype).itemsize
  max_logging.log(f"Opened {len(arrs)} memmaps on dtmpfs, total target = {total_bytes/1e9:.1f} GB BF16")

  # ----- Streaming conversion -----
  handle_cache: Dict[str, "safe_open"] = {}
  file_layer_map = _build_file_layer_map(weight_map, num_layers)
  progressive_cleanup = os.environ.get("MINIMAX_PROGRESSIVE_FP8_CLEANUP", "1") == "1"

  def fetch(name: str) -> torch.Tensor:
    return _get_tensor_bf16(weight_map, base, name, handle_cache)

  # Non-layered weights (small, do them first while RAM is plentiful)
  max_logging.log("Writing embed / final norm / lm_head ...")
  arrs["token_embedder.embedding"][:] = _np_from_torch(fetch("model.embed_tokens.weight"))
  arrs["decoder.decoder_norm.scale"][:] = _np_from_torch(fetch("model.norm.weight"))
  # HF lm_head is (V, H); MaxText wants (H, V).
  arrs["decoder.logits_dense.kernel"][:] = _np_from_torch(fetch("lm_head.weight").T)

  for l in tqdm(range(num_layers), desc="Layers"):
    # Norms — layer is axis 1.
    arrs["decoder.layers.pre_self_attention_layer_norm.scale"][:, l] = _np_from_torch(
        fetch(f"model.layers.{l}.input_layernorm.weight"))
    arrs["decoder.layers.post_self_attention_layer_norm.scale"][:, l] = _np_from_torch(
        fetch(f"model.layers.{l}.post_attention_layernorm.weight"))

    # Attention projections — HF (out, in); MaxText (H, L, n_heads, head_dim) etc.
    arrs["decoder.layers.self_attention.query.kernel"][:, l, :, :] = _np_from_torch(
        fetch(f"model.layers.{l}.self_attn.q_proj.weight").T.reshape(H, Hq, D))
    arrs["decoder.layers.self_attention.key.kernel"][:, l, :, :] = _np_from_torch(
        fetch(f"model.layers.{l}.self_attn.k_proj.weight").T.reshape(H, Hkv, D))
    arrs["decoder.layers.self_attention.value.kernel"][:, l, :, :] = _np_from_torch(
        fetch(f"model.layers.{l}.self_attn.v_proj.weight").T.reshape(H, Hkv, D))
    arrs["decoder.layers.self_attention.out.kernel"][:, l, :, :] = _np_from_torch(
        fetch(f"model.layers.{l}.self_attn.o_proj.weight").T.reshape(Hq, D, H))
    arrs["decoder.layers.self_attention.query_norm.scale"][:, l] = _np_from_torch(
        fetch(f"model.layers.{l}.self_attn.q_norm.weight"))
    arrs["decoder.layers.self_attention.key_norm.scale"][:, l] = _np_from_torch(
        fetch(f"model.layers.{l}.self_attn.k_norm.weight"))

    # MoE gate: kernel shape (H, L, E); bias shape (E, L).
    arrs["decoder.layers.moe_block.gate.kernel"][:, l, :] = _np_from_torch(
        fetch(f"model.layers.{l}.block_sparse_moe.gate.weight").T)
    arrs["decoder.layers.moe_block.gate.bias"][:, l] = _np_from_torch(
        fetch(f"model.layers.{l}.block_sparse_moe.e_score_correction_bias"))

    # MoE experts: (E, L, ...) — layer is already at axis 1.
    for e in range(num_experts):
      arrs["decoder.layers.moe_block.wi_0"][e, l] = _np_from_torch(
          fetch(f"model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight").T)
      arrs["decoder.layers.moe_block.wi_1"][e, l] = _np_from_torch(
          fetch(f"model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight").T)
      arrs["decoder.layers.moe_block.wo"][e, l] = _np_from_torch(
          fetch(f"model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight").T)

    # Flush the memmaps for this layer's writes to dtmpfs and drop torch tensors.
    for arr in arrs.values():
      arr.flush()
    handle_cache_keys = list(handle_cache)
    for fname in handle_cache_keys:
      if progressive_cleanup and _file_is_safe_to_unlink(file_layer_map[fname], current_layer=l + 1):
        # Close handle (releases mmap) and unlink file.
        del handle_cache[fname]
        gc.collect()
        try:
          os.unlink(os.path.join(base, fname))
        except OSError:
          pass
    gc.collect()

  # Final flush + close
  for arr in arrs.values():
    arr.flush()
  handle_cache.clear()
  gc.collect()

  # Write a manifest so the loader knows what's there.
  manifest = {
      "model_size": args.model_size,
      "params": {leaf: {"shape": list(shape), "dtype": dtype}
                 for leaf, (shape, dtype, _) in shapes.items()},
  }
  with open(out_dir / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
  max_logging.log(f"Wrote manifest to {out_dir / 'manifest.json'}")
  max_logging.log("Streaming conversion done.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_model_path", required=True,
                      help="HF MiniMax-M2 checkpoint dir on dtmpfs.")
  parser.add_argument("--output_dir", required=True,
                      help="Destination dir on dtmpfs for per-leaf .npy files.")
  parser.add_argument("--model_size", required=True,
                      choices=list(MODEL_PARAMS_DICT.keys()))
  main(parser.parse_args())
