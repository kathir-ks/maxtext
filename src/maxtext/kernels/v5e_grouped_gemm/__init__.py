"""Pallas grouped-GEMM kernel tuned for TPU v5e.

Used by the v5e-allgather MoE forward path (see
``src/maxtext/layers/moe_v5e_allgather.py``) to multiply a batch of token
activations through their per-token-selected expert weights without
relying on v5e's missing ``ragged_all_to_all`` collective.
"""

from maxtext.kernels.v5e_grouped_gemm.v5e_grouped_gemm import v5e_grouped_gemm

__all__ = ["v5e_grouped_gemm"]
