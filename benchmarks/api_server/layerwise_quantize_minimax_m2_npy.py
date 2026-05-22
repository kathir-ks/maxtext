"""Layer-by-layer AQT INT4 quantization for MiniMax-M2.7 from the streaming
converter's .npy pile, with CPU-only execution and an O(1-layer) memory
working set.

Why a new script?
- `src/maxtext/utils/layerwise_quantization.py` is hard-gated to the DEEPSEEK
  decoder block and reads Orbax (not .npy).
- The single-shot `intmp` path keeps the full BF16 model in HBM/RAM during the
  AQT forward pass, which OOMs on v4-8 (128 GB HBM) *and* on this host's
  400 GB RAM (proven empirically: 403 GB RSS before OOM-kill).
- We need: keep only ONE layer's params resident at a time, quantize via the
  Linen layer's own `apply(mutable=True)` (same machinery MaxEngine uses),
  accumulate the AQT pytree on the host, save Orbax at the end.

Per-layer working set:
  attention proj + 256 expert weights = ~7 GB BF16 per layer. We free between
  layers so memory stays flat around ~10-15 GB process RSS for the whole run.

Usage:
  JAX_PLATFORMS=cpu python -m benchmarks.api_server.layerwise_quantize_minimax_m2_npy \
      --npy_dir /mnt/disk1/minimax-m2.7-npy \
      src/maxtext/configs/base.yml \
      model_name=minimax-m2.7 \
      tokenizer_path=/mnt/dtmpfs/minimax-m2.7-hf \
      tokenizer_type=huggingface \
      ici_tensor_parallelism=1 \
      ici_expert_parallelism=1 \
      ici_fsdp_parallelism=1 \
      ici_autoregressive_parallelism=1 \
      quantization=intmp \
      quant_cfg_path=src/maxtext/configs/quantization/int4_weight_only.json \
      max_target_length=128 \
      max_prefill_predict_length=32 \
      per_device_batch_size=1 \
      scan_layers=false \
      attention=dot_product \
      megablox=false \
      sparse_matmul=false \
      weight_dtype=bfloat16 \
      save_quantized_params_path=/mnt/dtmpfs/minimax-m2.7-int4-orbax \
      async_checkpointing=false \
      checkpoint_storage_use_ocdbt=false \
      checkpoint_storage_use_zarr3=false \
      enable_single_controller=true
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import os
import pathlib
import sys
from typing import Any, Sequence


# Leaf-name → (shape-tuple, dtype, sharding-label) — must match the streaming
# converter's `leaf_shapes`. The layer axis is at position 1 for every per-layer
# leaf. We use this map to slice the .npy files.
NPY_LEAF_LAYER_AXIS = 1


def _per_layer_npy_paths(npy_dir: pathlib.Path) -> dict[str, str]:
  """Map MaxText leaf name → .npy filename (no path, in npy_dir)."""
  return {
      # ---- layer-axis-1 per-layer leaves ----
      "decoder.layers.pre_self_attention_layer_norm.scale":
          "decoder.layers.pre_self_attention_layer_norm.scale.npy",
      "decoder.layers.post_self_attention_layer_norm.scale":
          "decoder.layers.post_self_attention_layer_norm.scale.npy",
      "decoder.layers.self_attention.query.kernel":
          "decoder.layers.self_attention.query.kernel.npy",
      "decoder.layers.self_attention.key.kernel":
          "decoder.layers.self_attention.key.kernel.npy",
      "decoder.layers.self_attention.value.kernel":
          "decoder.layers.self_attention.value.kernel.npy",
      "decoder.layers.self_attention.out.kernel":
          "decoder.layers.self_attention.out.kernel.npy",
      "decoder.layers.self_attention.query_norm.scale":
          "decoder.layers.self_attention.query_norm.scale.npy",
      "decoder.layers.self_attention.key_norm.scale":
          "decoder.layers.self_attention.key_norm.scale.npy",
      "decoder.layers.moe_block.gate.kernel":
          "decoder.layers.moe_block.gate.kernel.npy",
      "decoder.layers.moe_block.gate.bias":
          "decoder.layers.moe_block.gate.bias.npy",
      "decoder.layers.moe_block.wi_0":
          "decoder.layers.moe_block.wi_0.npy",
      "decoder.layers.moe_block.wi_1":
          "decoder.layers.moe_block.wi_1.npy",
      "decoder.layers.moe_block.wo":
          "decoder.layers.moe_block.wo.npy",
      # ---- non-layered ----
      "token_embedder.embedding":
          "token_embedder.embedding.npy",
      "decoder.decoder_norm.scale":
          "decoder.decoder_norm.scale.npy",
      "decoder.logits_dense.kernel":
          "decoder.logits_dense.kernel.npy",
  }


def _maxtext_leaf_keys_for_layer() -> dict[str, tuple[str, ...]]:
  """For a single MiniMax-M2 layer (Qwen3-MoE style), the nested params dict
  expected by `layer.apply` has these top-level paths (relative to the layer's
  own params subtree). The .npy leaves listed above are the per-layer subset.

  This is the conventional layout for a Qwen3MoeDecoderLayer:
    self_attention/{query,key,value,out}/kernel
    self_attention/{query_norm,key_norm}/scale
    pre_self_attention_layer_norm/scale
    post_self_attention_layer_norm/scale
    moe_block/gate/kernel
    moe_block/gate/bias
    moe_block/wi_0
    moe_block/wi_1
    moe_block/wo
  """
  return {
      "decoder.layers.pre_self_attention_layer_norm.scale":
          ("pre_self_attention_layer_norm", "scale"),
      "decoder.layers.post_self_attention_layer_norm.scale":
          ("post_self_attention_layer_norm", "scale"),
      "decoder.layers.self_attention.query.kernel":
          ("self_attention", "query", "kernel"),
      "decoder.layers.self_attention.key.kernel":
          ("self_attention", "key", "kernel"),
      "decoder.layers.self_attention.value.kernel":
          ("self_attention", "value", "kernel"),
      "decoder.layers.self_attention.out.kernel":
          ("self_attention", "out", "kernel"),
      "decoder.layers.self_attention.query_norm.scale":
          ("self_attention", "query_norm", "scale"),
      "decoder.layers.self_attention.key_norm.scale":
          ("self_attention", "key_norm", "scale"),
      "decoder.layers.moe_block.gate.kernel":
          ("moe_block", "gate", "kernel"),
      "decoder.layers.moe_block.gate.bias":
          ("moe_block", "gate", "bias"),
      "decoder.layers.moe_block.wi_0":
          ("moe_block", "wi_0"),
      "decoder.layers.moe_block.wi_1":
          ("moe_block", "wi_1"),
      "decoder.layers.moe_block.wo":
          ("moe_block", "wo"),
  }


def _put_nested(d: dict, path: tuple[str, ...], value: Any) -> None:
  """Insert value at the nested path, creating dicts on the way."""
  cur = d
  for k in path[:-1]:
    cur = cur.setdefault(k, {})
  cur[path[-1]] = value


def load_layer_params_from_npy(
    npy_dir: pathlib.Path, layer_idx: int, dtype):
  """Read one MoE layer's params from the .npy pile, return as nested dict.

  Reads slice [:, layer_idx, ...] from each layer-axis-1 .npy file. Each
  returned leaf is a jax.Array on the default device (CPU here).
  """
  import jax
  import jax.numpy as jnp
  import numpy as np

  paths = _per_layer_npy_paths(npy_dir)
  keys = _maxtext_leaf_keys_for_layer()
  out: dict[str, Any] = {}
  for leaf, fname in paths.items():
    if leaf not in keys:
      continue  # non-layered, handled separately
    np_arr = np.load(npy_dir / fname, mmap_mode="r")
    layer_slice = np.asarray(np_arr[(slice(None), layer_idx, ...)])
    jax_arr = jnp.asarray(layer_slice).astype(dtype)
    _put_nested(out, keys[leaf], jax_arr)
    del np_arr, layer_slice
  return out


def load_nonlayered_from_npy(npy_dir: pathlib.Path, dtype):
  """Read the per-leaf non-layered weights (embeddings, final norm, lm head)."""
  import jax.numpy as jnp
  import numpy as np

  def _load(fname):
    np_arr = np.load(npy_dir / fname, mmap_mode="r")
    return jnp.asarray(np.asarray(np_arr)).astype(dtype)

  return {
      "token_embedder": {"embedding": _load("token_embedder.embedding.npy")},
      "decoder": {
          "decoder_norm": {"scale": _load("decoder.decoder_norm.scale.npy")},
          "logits_dense": {"kernel": _load("decoder.logits_dense.kernel.npy")},
      },
  }


def main() -> None:
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  os.environ["TPU_NAME"] = ""

  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--npy_dir", required=True)
  parser.add_argument("--help", "-h", action="store_true")
  args, rest = parser.parse_known_args()
  if args.help:
    print(__doc__)
    return

  npy_dir = pathlib.Path(args.npy_dir).expanduser().resolve()
  if not (npy_dir / "manifest.json").exists():
    raise SystemExit(f"no manifest.json under {npy_dir}")

  # MUST initialise pyconfig (which calls jax.distributed.initialize) BEFORE
  # any jax call that would materialise the backend.
  from maxtext.configs import pyconfig
  config = pyconfig.initialize(["layerwise_quantize_minimax_m2_npy"] + rest)

  assert config.scan_layers is False, (
      "scan_layers must be false for per-layer quantization — otherwise the "
      "abstract state will have layer-stacked leaves that the .npy loader "
      "can't populate one layer at a time."
  )
  assert config.save_quantized_params_path, (
      "save_quantized_params_path must be set so the AQT checkpoint is saved"
  )

  import jax
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  import jax.numpy as jnp
  from flax import nnx
  from flax.linen import partitioning as nn_partitioning
  from tqdm import tqdm
  from maxtext.common import common_types
  from maxtext.layers import quantizations
  from maxtext.models import minimax_m2
  from maxtext.utils import max_logging, max_utils, maxtext_utils
  # Reuse the path-extraction utility from the DeepSeek-flavoured tool.
  from maxtext.utils.layerwise_quantization import remove_quantized_params

  print(f"[lw-quant] devices: {jax.devices()}", flush=True)
  max_utils.print_system_information()

  # Mesh
  devices_array = maxtext_utils.create_device_mesh(config=config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  # Quantization config (AQT) — quant_mode=convert means apply() builds AQT vars.
  quant = quantizations.configure_quantization(config)
  quant.quant_mode = quantizations.get_quant_mode("convert")

  rng = jax.random.PRNGKey(1234)
  rng, rng_quant = jax.random.split(rng)
  model_mode = common_types.MODEL_MODE_PREFILL
  weight_dtype = jnp.bfloat16  # config.weight_dtype is a str like "bfloat16"

  # Build one Linen-wrapped MiniMax-M2 layer to reuse across all 62 indices.
  layer = minimax_m2.MiniMaxM2DecoderLayerToLinen(
      config=config, mesh=mesh, quant=quant,
      model_mode=model_mode, rngs=nnx.Rngs(rng),
  )

  def model_apply(params, _rng):
    # Same shape as DeepSeek layerwise quantization's apply: residual input,
    # None decoder_positions, zeros segment_ids, deterministic=True.
    L = config.max_prefill_predict_length
    E = config.base_emb_dim
    return layer.apply(
        params | {"aqt": {}},
        jnp.ones((1, L, E), dtype=jnp.int32),
        None,
        jnp.zeros((1, L), dtype=jnp.int32),
        True,
        model_mode=model_mode,
        rngs={"params": _rng},
        mutable=True,
    )

  num_layers = config.num_decoder_layers
  assert config.first_num_dense_layers == 0, "MiniMax-M2 has no dense layers"

  quantized: dict = {"params": {"decoder": {}}, "aqt": {"decoder": {}}}

  for index in tqdm(range(num_layers), desc="layers"):
    layer_name = f"layers_{index}"
    raw = load_layer_params_from_npy(npy_dir, index, weight_dtype)

    # Layer.apply expects {"params": layer_params, "aqt": {}}
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      _, new_vars = model_apply({"params": raw}, rng_quant)

    if "aqt" not in new_vars or not new_vars["aqt"]:
      max_logging.log(f"[lw-quant] no AQT vars from {layer_name}; keeping bf16")
      quantized["params"]["decoder"][layer_name] = raw
    else:
      aqt_vars = new_vars["aqt"]
      try:
        removed = remove_quantized_params(raw, aqt_vars)
        quantized["params"]["decoder"][layer_name] = removed
        quantized["aqt"]["decoder"][layer_name] = aqt_vars
      except Exception as e:  # pylint: disable=broad-except
        max_logging.log(f"[lw-quant] {layer_name} remove_quantized_params failed: {e}")
        max_logging.log("[lw-quant] raw layer keys:")
        jax.tree_util.tree_map_with_path(
            lambda p, _: max_logging.log(f"  {jax.tree_util.keystr(p)}"), raw)
        max_logging.log("[lw-quant] aqt_vars keys:")
        jax.tree_util.tree_map_with_path(
            lambda p, _: max_logging.log(f"  {jax.tree_util.keystr(p)}"), aqt_vars)
        raise

    # Drop the raw layer arrays from RAM before moving on.
    del raw, new_vars
    gc.collect()

  # Non-layered (embedding, final norm, lm head) stay bf16.
  nonlayered = load_nonlayered_from_npy(npy_dir, weight_dtype)
  quantized["params"]["token_embedder"] = nonlayered["token_embedder"]
  quantized["params"]["decoder"]["decoder_norm"] = nonlayered["decoder"]["decoder_norm"]
  quantized["params"]["decoder"]["logits_dense"] = nonlayered["decoder"]["logits_dense"]

  max_logging.log(f"[lw-quant] saving AQT checkpoint to {config.save_quantized_params_path}")
  maxtext_utils.save_quantized_checkpoint_if_configured(config, quantized)
  max_logging.log("[lw-quant] done")


if __name__ == "__main__":
  main()
