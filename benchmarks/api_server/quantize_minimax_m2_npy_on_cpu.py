"""One-shot CPU-only AQT quantization driver for MiniMax-M2.7.

Reuses the streaming-converter .npy pile (via `install_load_params_patch`)
and forces `JAX_PLATFORMS=cpu` so the BF16->INT4 transient lives in 400 GB
of host RAM instead of 128 GB of v4-8 HBM. Saves the resulting AQT-format
Orbax checkpoint via `save_quantized_params_path`, then we serve from that
checkpoint with `checkpoint_is_quantized=True`.

Usage:
  JAX_PLATFORMS=cpu python -m benchmarks.api_server.quantize_minimax_m2_npy_on_cpu \
      --npy_dir /mnt/disk1/minimax-m2.7-npy \
      src/maxtext/configs/base.yml \
      model_name=minimax-m2.7 \
      tokenizer_path=/mnt/dtmpfs/minimax-m2.7-hf \
      tokenizer_type=huggingface \
      ici_tensor_parallelism=1 \
      ici_expert_parallelism=1 \
      quantization=intmp \
      quant_cfg_path=src/maxtext/configs/quantization/int4_weight_only.json \
      max_target_length=128 \
      max_prefill_predict_length=32 \
      per_device_batch_size=1 \
      scan_layers=true \
      attention=dot_product \
      megablox=false \
      sparse_matmul=false \
      save_quantized_params_path=/mnt/disk1/minimax-m2.7-int4-orbax \
      async_checkpointing=false \
      checkpoint_storage_use_ocdbt=false \
      checkpoint_storage_use_zarr3=false
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys


def main() -> None:
  # JAX_PLATFORMS must be set BEFORE importing jax.
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  # Disable TPU detection just in case.
  os.environ["TPU_NAME"] = ""

  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--npy_dir", required=True,
                      help="Directory of per-leaf .npy files + manifest.json")
  parser.add_argument("--help", "-h", action="store_true")
  args, rest = parser.parse_known_args()
  if args.help:
    print(__doc__)
    return

  npy_dir = pathlib.Path(args.npy_dir).expanduser().resolve()
  if not (npy_dir / "manifest.json").exists() and not list(npy_dir.glob("manifest.p*.json")):
    raise SystemExit(f"no manifest.json or manifest.p*.json under {npy_dir}")

  # Import config FIRST so it can call jax.distributed.initialize() before any
  # other JAX call materialises the backend. Don't touch jax.devices yet.
  from maxtext.configs import pyconfig

  config = pyconfig.initialize(["quantize_minimax_m2_npy_on_cpu"] + rest)

  # Now safe to import + use JAX.
  import jax
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  from maxtext.inference.decode_minimax_m2_npy import install_load_params_patch
  from maxtext.inference.maxengine import maxengine
  from maxtext.utils import max_utils
  print(f"[quant-cpu] devices: {jax.devices()}", flush=True)

  install_load_params_patch(npy_dir)
  assert config.save_quantized_params_path, (
      "save_quantized_params_path must be set so MaxEngine writes the AQT checkpoint"
  )
  max_utils.print_system_information()

  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)

  # MaxEngine.load_params will:
  #   1. Build abstract state with quantization=intmp wired through (AQT layers).
  #   2. Call our patched .npy-from-disk loader to populate params as BF16.
  #   3. Call self.quantize_params(state, rng3) since checkpoint_is_quantized=False.
  #      That runs a forward pass on the CPU device, producing AQT QTensors.
  #   4. Hits maxtext_utils.save_quantized_checkpoint_if_configured(self.config, params)
  #      which writes Orbax to save_quantized_params_path.
  engine.load_params(rng_load_params)
  print(f"[quant-cpu] saved AQT checkpoint to {config.save_quantized_params_path}",
        flush=True)


if __name__ == "__main__":
  main()
