"""OpenAI-compatible API server entrypoint that loads MiniMax-M2 weights from
the .npy pile produced by ``convert_minimax_m2_streaming.py`` instead of an
Orbax checkpoint.

Mirrors ``maxtext.inference.decode_minimax_m2_npy`` but delegates to
``benchmarks.api_server.maxtext_server`` rather than ``maxtext.inference.decode``,
so the same long-running FastAPI server gets ``/v1/completions`` and
``/v1/chat/completions`` endpoints over the .npy-backed engine.

Usage:
  python -m benchmarks.api_server.serve_minimax_m2_npy \
      --npy_dir /mnt/dtmpfs/minimax-m2.7-npy \
      src/maxtext/configs/base.yml \
      model_name=minimax-m2.7 \
      tokenizer_path=/mnt/dtmpfs/minimax-m2.7-hf \
      tokenizer_type=huggingface \
      ici_tensor_parallelism=4 \
      ici_expert_parallelism=1 \
      quantization=intmp \
      quant_cfg_path=src/maxtext/configs/quantization/int4_weight_only.json \
      max_target_length=65536 \
      max_prefill_predict_length=32768 \
      per_device_batch_size=1 \
      scan_layers=true \
      attention=dot_product \
      megablox=false \
      sparse_matmul=false
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from maxtext.inference.decode_minimax_m2_npy import install_load_params_patch


def main():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--npy_dir", required=True,
                      help="Directory of per-leaf .npy files + manifest.json")
  parser.add_argument("--help", "-h", action="store_true")
  args, rest = parser.parse_known_args()
  if args.help:
    print(__doc__)
    return

  npy_dir = pathlib.Path(args.npy_dir).expanduser().resolve()
  has_replicated = (npy_dir / "manifest.json").exists()
  has_distributed = any(npy_dir.glob("manifest.p*.json"))
  if not (has_replicated or has_distributed):
    raise SystemExit(f"no manifest.json or manifest.p*.json under {npy_dir}")

  install_load_params_patch(npy_dir)

  # maxtext_server.py hardcodes uvicorn host="0.0.0.0" port=8000. We expose
  # both as env vars so the calling shell can pin them.
  port = int(os.environ.get("MAXTEXT_SERVER_PORT", "8000"))
  host = os.environ.get("MAXTEXT_SERVER_HOST", "0.0.0.0")
  import uvicorn
  _orig_Config = uvicorn.Config
  _orig_run = uvicorn.run
  def _patched_Config(app, *a, **kw):
    kw["host"], kw["port"] = host, port
    return _orig_Config(app, *a, **kw)
  def _patched_run(app, *a, **kw):
    kw["host"], kw["port"] = host, port
    return _orig_run(app, *a, **kw)
  uvicorn.Config = _patched_Config
  uvicorn.run = _patched_run

  from benchmarks.api_server import maxtext_server
  sys.argv = ["maxtext_server"] + rest
  maxtext_server.main()


if __name__ == "__main__":
  main()
