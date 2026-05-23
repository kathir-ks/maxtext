"""Assemble the per-layer pickles produced by `layerwise_quantize_minimax_m2_npy.py`
into a single AQT-format Orbax checkpoint that MaxEngine can serve with
`checkpoint_is_quantized=True`.

The quant step left 62 pickle files at /mnt/dtmpfs/lw_quant_stage/, each
~3.5 GB. Loading all of them at once also OOM-killed the previous run because
tmpfs (file-backed) + Python objects (heap) = ~440 GB on a 400 GB box.

Mitigation: load + delete each pickle in turn. tmpfs shrinks as the heap
grows, total stays bounded.

Usage:
  JAX_PLATFORMS=cpu python -m benchmarks.api_server.assemble_int4_orbax \
      --npy_dir /mnt/disk1/minimax-m2.7-npy \
      --stage_dir /mnt/dtmpfs/lw_quant_stage \
      src/maxtext/configs/base.yml \
      model_name=minimax-m2.7 \
      tokenizer_path=/mnt/dtmpfs/minimax-m2.7-hf tokenizer_type=huggingface \
      ici_tensor_parallelism=1 ici_expert_parallelism=1 \
      ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 \
      quantization=intmp \
      quant_cfg_path=src/maxtext/configs/quantization/int4_weight_only.json \
      max_target_length=128 max_prefill_predict_length=32 \
      per_device_batch_size=1 scan_layers=false attention=dot_product \
      megablox=false sparse_matmul=false weight_dtype=bfloat16 \
      save_quantized_params_path=/mnt/disk1/minimax-m2.7-int4-orbax \
      async_checkpointing=false checkpoint_storage_use_ocdbt=false \
      checkpoint_storage_use_zarr3=false enable_single_controller=true
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib
import pickle
import sys


def main() -> None:
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  os.environ["TPU_NAME"] = ""

  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--npy_dir", required=True,
                      help="Original .npy directory (for non-layered leaves)")
  parser.add_argument("--stage_dir", required=True,
                      help="Per-layer pickle directory")
  parser.add_argument("--keep_stage", action="store_true",
                      help="Don't delete pickles after loading (default: delete)")
  parser.add_argument("--help", "-h", action="store_true")
  args, rest = parser.parse_known_args()
  if args.help:
    print(__doc__)
    return

  npy_dir = pathlib.Path(args.npy_dir).expanduser().resolve()
  stage_dir = pathlib.Path(args.stage_dir).expanduser().resolve()

  from maxtext.configs import pyconfig
  config = pyconfig.initialize(["assemble_int4_orbax"] + rest)
  assert config.save_quantized_params_path, (
      "save_quantized_params_path must be set so the AQT checkpoint is saved"
  )

  import jax
  import jax.numpy as jnp
  from maxtext.utils import max_logging, max_utils, maxtext_utils

  print(f"[assemble] devices: {jax.devices()}", flush=True)
  max_utils.print_system_information()

  num_layers = config.num_decoder_layers

  quantized: dict = {"params": {"decoder": {}}, "aqt": {"decoder": {}}}

  for index in range(num_layers):
    layer_name = f"layers_{index}"
    pickle_path = stage_dir / f"{layer_name}.pkl"
    if not pickle_path.exists():
      raise SystemExit(f"missing {pickle_path}")
    print(f"[assemble] loading {layer_name} ({pickle_path.stat().st_size/1e9:.2f} GB)",
          flush=True)
    with open(pickle_path, "rb") as f:
      stage = pickle.load(f)
    quantized["params"]["decoder"][layer_name] = stage["params"]
    if stage["aqt"] is not None:
      quantized["aqt"]["decoder"][layer_name] = stage["aqt"]
    del stage
    if not args.keep_stage:
      pickle_path.unlink()
    gc.collect()

  # Non-layered weights stay bf16.
  print("[assemble] loading non-layered (embedding / norm / lm_head)", flush=True)
  import numpy as np
  def load_npy(name):
    return jnp.asarray(np.load(npy_dir / name)).astype(jnp.bfloat16)
  quantized["params"]["token_embedder"] = {
      "embedding": load_npy("token_embedder.embedding.npy"),
  }
  quantized["params"]["decoder"]["decoder_norm"] = {
      "scale": load_npy("decoder.decoder_norm.scale.npy"),
  }
  quantized["params"]["decoder"]["logits_dense"] = {
      "kernel": load_npy("decoder.logits_dense.kernel.npy"),
  }

  print(f"[assemble] saving AQT checkpoint to {config.save_quantized_params_path}",
        flush=True)
  maxtext_utils.save_quantized_checkpoint_if_configured(config, quantized)
  print("[assemble] done", flush=True)


if __name__ == "__main__":
  main()
