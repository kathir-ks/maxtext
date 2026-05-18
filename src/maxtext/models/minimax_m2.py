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

"""MiniMax M2 / M2.7 decoder layer for MaxText.

The MiniMax-M2 family (M2, M2.1, M2.5, M2.7) shares one architecture:
sparse MoE (256 experts, 8 active, sigmoid-scored routing with a learnable
routing bias), grouped-query attention with per-layer QK RMSNorm, and partial
RoPE (rotary_dim=64 of head_dim=128). Architecturally this is Qwen3-MoE
attention + DeepSeek-V3-style routing, so we layer on top of the existing
Qwen3 MoE decoder layer: the routing semantics are driven entirely by
`routed_score_func`, `routed_bias`, and `norm_topk_prob` config fields, and
partial RoPE is now driven by `config.partial_rotary_factor` via
AttentionWithNorm.
"""

from maxtext.layers import initializers as max_initializers
from maxtext.layers import nnx_wrappers
from maxtext.models.qwen3 import Qwen3MoeDecoderLayer


class MiniMaxM2DecoderLayer(Qwen3MoeDecoderLayer):
  """MiniMax-M2 family transformer decoder layer (MoE, sigmoid routing)."""


MiniMaxM2DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    MiniMaxM2DecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)
