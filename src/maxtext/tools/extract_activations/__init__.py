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
"""Offline activation extraction for SAE training and interpretability."""

from maxtext.tools.extract_activations.hooks import (
    INTERMEDIATES_COLLECTION,
    HOOK_RESIDUAL_POST,
    HOOK_MLP_OUT,
    HOOK_ATTN_OUT,
    ALL_HOOKS,
    maybe_sow_activations,
)
from maxtext.tools.extract_activations.merge import (
    list_host_dirs,
    iter_shard_paths,
    load_merged,
    load_manifest,
)
