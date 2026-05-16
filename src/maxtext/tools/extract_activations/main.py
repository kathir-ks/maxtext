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
"""CLI entry point for activation extraction.

Usage (single host):

  python -m maxtext.tools.extract_activations.main \\
    src/maxtext/configs/base.yml \\
    model_name=qwen3-30b-a3b \\
    load_parameters_path=gs://.../checkpoints/items/0 \\
    activation_extraction_enabled=True \\
    activation_extraction_layers="[12,24,36]" \\
    activation_extraction_hooks="[residual_post]" \\
    activation_extraction_output_path=gs://.../sae/qwen3-30b-a3b/v1 \\
    activation_extraction_shard_size_tokens=2000000 \\
    activation_extraction_dataset=jsonl \\
    activation_extraction_dataset_path=gs://.../pile_subset.jsonl \\
    per_device_batch_size=1 \\
    max_prefill_predict_length=2048
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

from absl import app
import jax

from maxtext.configs import pyconfig
from maxtext.tools.extract_activations.runner import Runner

logger = logging.getLogger(__name__)


def _validate_config(config) -> None:
  if not config.activation_extraction_enabled:
    raise ValueError("activation_extraction_enabled must be True for this tool.")
  if not config.activation_extraction_output_path:
    raise ValueError("activation_extraction_output_path is required.")
  if not config.activation_extraction_dataset_path:
    raise ValueError("activation_extraction_dataset_path is required.")
  if config.using_pipeline_parallelism:
    raise ValueError(
        "pipeline parallelism is not supported by extract_activations (v1)."
    )
  for h in config.activation_extraction_hooks:
    if h not in ("residual_post", "mlp_out", "attn_out"):
      raise ValueError(f"unknown hook {h!r}")


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
  logging.basicConfig(level=logging.INFO)
  config = pyconfig.initialize(argv)
  _validate_config(config)
  Runner(config).run()


if __name__ == "__main__":
  app.run(main)
