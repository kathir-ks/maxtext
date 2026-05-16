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
"""Centralised activation-hook utilities used by all DecoderLayer variants.

`maybe_sow_activations` lets each model's `DecoderLayer` add support for
the extraction tool with a single one-line call instead of duplicating
three `self.sow(...)` statements. The function is a no-op when the
extraction flag is off, and JAX dead-code-eliminates the un-taken
branches so the forward HLO is unchanged in that case.
"""
from __future__ import annotations

from typing import Any

import jax


INTERMEDIATES_COLLECTION: str = "intermediates"

HOOK_RESIDUAL_POST: str = "residual_post"
HOOK_MLP_OUT: str = "mlp_out"
HOOK_ATTN_OUT: str = "attn_out"

ALL_HOOKS: tuple[str, ...] = (HOOK_RESIDUAL_POST, HOOK_MLP_OUT, HOOK_ATTN_OUT)


def maybe_sow_activations(
    module,
    *,
    config: Any,
    layer_output: jax.Array,
    mlp_out: jax.Array | None = None,
    attn_out: jax.Array | None = None,
) -> None:
  """Sow per-hook activations when ``activation_extraction_enabled`` is set.

  Args:
    module: The Flax / NNX module to call ``sow`` on (typically ``self`` inside
      a ``DecoderLayer.__call__``).
    config: The MaxText ``HyperParameters`` config.
    layer_output: The post-block residual stream tensor for this layer
      (shape ``[batch, length, emb_dim]``).
    mlp_out: Optional MLP output **before** the residual add (same shape).
    attn_out: Optional attention output **before** the residual add (same shape).

  The call is conditioned on ``config.activation_extraction_enabled`` so that
  it produces no HLO when extraction is off.

  When wrapped by ``nn.scan``, the runner declares
  ``variable_axes={'intermediates': 0}`` so the collected tensor is
  automatically stacked along the layer axis as
  ``[num_layers, batch, length, emb_dim]``.
  """
  if not getattr(config, "activation_extraction_enabled", False):
    return
  hooks = set(getattr(config, "activation_extraction_hooks", ()) or ())
  if HOOK_RESIDUAL_POST in hooks:
    module.sow(INTERMEDIATES_COLLECTION, HOOK_RESIDUAL_POST, layer_output)
  if mlp_out is not None and HOOK_MLP_OUT in hooks:
    module.sow(INTERMEDIATES_COLLECTION, HOOK_MLP_OUT, mlp_out)
  if attn_out is not None and HOOK_ATTN_OUT in hooks:
    module.sow(INTERMEDIATES_COLLECTION, HOOK_ATTN_OUT, attn_out)
