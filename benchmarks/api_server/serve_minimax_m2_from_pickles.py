"""Serve MiniMax-M2.7 on TPU from the per-layer AQT pickles produced by
`layerwise_quantize_minimax_m2_npy.py`, *without* going through Orbax.

Why: Orbax PyTreeCheckpointer.save peaks at ~2x the pytree RAM during
serialization, which OOM-kills on a 400 GB host for a 217 GB AQT pytree.
Loading the same pytree leaf-by-leaf from pickles + device-putting straight
to TPU sidesteps the peak. With `checkpoint_is_quantized=True` MaxEngine
skips the OOM-prone `quantize_params` forward pass and uses the AQT data
directly.

Memory math on v4-8 (4 chips × 32 GB HBM):
  - 62 layer pickles × ~3.5 GB each (int8-stored int4) = 217 GB host-side
  - JAX shards across 4 chips: ~54 GB / chip if stored as int8
  - But AQT QTensor.qvalue.dtype = int4 → XLA packs 2-per-byte in HBM
    → 108 GB total, ~27 GB / chip, leaves ~5 GB headroom for KV+scratch

Usage:
  python -m benchmarks.api_server.serve_minimax_m2_from_pickles \
      --npy_dir /mnt/disk1/minimax-m2.7-npy \
      --stage_dir /mnt/dtmpfs/lw_quant_stage \
      src/maxtext/configs/base.yml \
      model_name=minimax-m2.7 \
      tokenizer_path=/mnt/dtmpfs/minimax-m2.7-hf \
      tokenizer_type=huggingface \
      ici_tensor_parallelism=4 \
      ici_expert_parallelism=1 \
      quantization=intmp \
      quant_cfg_path=src/maxtext/configs/quantization/int4_weight_only.json \
      checkpoint_is_quantized=true \
      max_target_length=65536 \
      max_prefill_predict_length=32768 \
      per_device_batch_size=1 \
      scan_layers=false \
      attention=dot_product \
      megablox=false \
      sparse_matmul=false \
      weight_dtype=bfloat16 \
      quantize_kvcache=true \
      kv_quant_dtype=int8
"""

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import pickle
import sys


def main() -> None:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--npy_dir", required=True,
                      help="Non-layered .npy directory (token_embedder, decoder_norm, logits_dense)")
  parser.add_argument("--stage_dir", required=True,
                      help="Per-layer AQT pickle directory from layerwise_quantize")
  parser.add_argument("--help", "-h", action="store_true")
  args, rest = parser.parse_known_args()
  if args.help:
    print(__doc__)
    return

  npy_dir = pathlib.Path(args.npy_dir).expanduser().resolve()
  stage_dir = pathlib.Path(args.stage_dir).expanduser().resolve()

  if not stage_dir.is_dir():
    raise SystemExit(f"stage_dir not found: {stage_dir}")
  pickles = sorted(stage_dir.glob("layers_*.pkl"))
  if not pickles:
    raise SystemExit(f"no per-layer pickles under {stage_dir}")

  # Init pyconfig FIRST so jax.distributed.initialize runs before any other
  # JAX call materialises the backend.
  from maxtext.configs import pyconfig
  config = pyconfig.initialize(["serve_minimax_m2_from_pickles"] + rest)

  assert config.checkpoint_is_quantized, (
      "must pass checkpoint_is_quantized=true so MaxEngine.load_params skips "
      "the OOM-prone quantize_params step"
  )

  import jax
  import jax.numpy as jnp
  import numpy as np
  from maxtext.inference.maxengine import maxengine
  from maxtext.utils import max_logging, max_utils

  # Patch MaxEngine.load_params to construct params from our pickle-per-layer
  # + non-layered .npy, then route the resulting pytree through the same
  # sharding/init pipeline the original uses.
  from maxtext.utils import maxtext_utils, max_utils as _max_utils
  from flax.linen import partitioning as nn_partitioning

  def _patched_load_params(self, *args, params=None, rng=None, **kwargs):
    if rng is None:
      rng = jax.random.PRNGKey(0)
    if self.model.quant and self.config.checkpoint_is_quantized:
      print("[serve-pickle] loading from pre-quantized pickles "
            f"(checkpoint_is_quantized=true): {len(pickles)} layer files",
            flush=True)
      from maxtext.layers import quantizations
      self.model.quant.quant_mode = quantizations.get_quant_mode("serve")

    rng1, rng2, _rng3 = jax.random.split(rng, 3)
    init_state_fn = functools.partial(
        maxtext_utils.init_initial_state, self.model, None, self.config, False, rng1)
    _, self.state_mesh_annotations, state_mesh_shardings = maxtext_utils.get_abstract_state(
        self.config, self._mesh, init_state_fn, False)

    # Build params pytree by streaming: for each layer, load pickle →
    # device_put that single layer's leaves onto TPU shards → drop host refs
    # → next layer. Host RAM stays at ~10 GB working set instead of 213 GB.
    import gc

    # The state_mesh_shardings pytree mirrors the model pytree shape. Pull
    # per-layer subtrees out of it to use as sharding spec for each pickle.
    shardings_params = state_mesh_shardings.params
    if hasattr(shardings_params, "_dict"):  # FrozenDict
      shardings_params = dict(shardings_params)
    # Some MaxText paths wrap params under a top-level "params" key.
    if "params" in shardings_params and isinstance(shardings_params["params"], dict):
      shardings_params = shardings_params["params"]

    sharded = {"params": {"decoder": {}}, "aqt": {"decoder": {}}}

    def _to_dict(x):
      if hasattr(x, "_dict"):
        return dict(x._dict)
      return x

    def _prune_empty(d):
      """Recursively drop empty-dict children. `remove_quantized_params` replaces
      quantized leaves with {} (rather than deleting them); the abstract state
      has those entries fully absent, so device_put trips on the mismatch."""
      if not isinstance(d, dict):
        return d
      pruned = {}
      for k, v in d.items():
        v2 = _prune_empty(v)
        if isinstance(v2, dict) and not v2:
          continue
        pruned[k] = v2
      return pruned

    weight_dtype = jnp.bfloat16

    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      for idx, pkl in enumerate(pickles):
        layer_name = pkl.stem
        with open(pkl, "rb") as f:
          stage = pickle.load(f)
        # Prune empty-dict leaves left by remove_quantized_params.
        stage["params"] = _prune_empty(stage["params"])

        # Find the layer's sharding sub-tree.
        decoder_shardings = _to_dict(shardings_params.get("decoder", {}))
        layer_shardings = decoder_shardings.get(layer_name)
        aqt_decoder_shardings = _to_dict(shardings_params.get("aqt", {})).get("decoder", {})
        layer_aqt_shardings = _to_dict(aqt_decoder_shardings).get(layer_name)

        if layer_shardings is not None:
          sharded["params"]["decoder"][layer_name] = jax.device_put(
              stage["params"], layer_shardings)
        else:
          # Fallback: device_put with no explicit sharding (replicates).
          sharded["params"]["decoder"][layer_name] = jax.device_put(stage["params"])

        if stage["aqt"] is not None:
          if layer_aqt_shardings is not None:
            sharded["aqt"]["decoder"][layer_name] = jax.device_put(
                stage["aqt"], layer_aqt_shardings)
          else:
            sharded["aqt"]["decoder"][layer_name] = jax.device_put(stage["aqt"])

        del stage
        gc.collect()
        if (idx + 1) % 10 == 0:
          max_logging.log(f"[serve-pickle] placed {idx+1}/{len(pickles)} layers on TPU")

      # Non-layered weights from the original .npy pile.
      def load_npy(name, dtype):
        arr = jnp.asarray(np.load(npy_dir / name)).astype(dtype)
        return arr

      def place(host_pytree, sharding_subtree):
        if sharding_subtree is None:
          return jax.device_put(host_pytree)
        return jax.device_put(host_pytree, sharding_subtree)

      te_shards = _to_dict(shardings_params.get("token_embedder", {}))
      sharded["params"]["token_embedder"] = place(
          {"embedding": load_npy("token_embedder.embedding.npy", weight_dtype)},
          te_shards,
      )

      dec_shards = _to_dict(shardings_params.get("decoder", {}))
      dn_shards = _to_dict(dec_shards.get("decoder_norm", {}))
      sharded["params"]["decoder"]["decoder_norm"] = place(
          {"scale": load_npy("decoder.decoder_norm.scale.npy", weight_dtype)},
          dn_shards if dn_shards else None,
      )
      ld_shards = _to_dict(dec_shards.get("logits_dense", {}))
      sharded["params"]["decoder"]["logits_dense"] = place(
          {"kernel": load_npy("decoder.logits_dense.kernel.npy", weight_dtype)},
          ld_shards if ld_shards else None,
      )

    params_resharded = sharded
    del sharded
    gc.collect()
    state = maxtext_utils.init_decode_state(None, params_resharded)
    state = _max_utils.unbox_logicallypartioned(state)

    self.abstract_params = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)
        if isinstance(x, jax.Array) else None,
        state.params,
    )
    self.prefill_kv_cache_annotations = maxtext_utils.get_prefill_kv_cache_annotations(
        self.model, self.config, rng2, self._mesh, self.page_state)
    self.prefill_kv_cache_shardings = jax.tree_util.tree_map(
        lambda x: jax.sharding.NamedSharding(self._mesh, x),
        self.prefill_kv_cache_annotations,
    )
    if self.config.stack_prefill_result_cache:
      self.prefill_kv_cache_shardings = jax.tree_util.tree_map(
          lambda x: jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec(None, *x.spec)),
          self.prefill_kv_cache_shardings,
      )
      self.prefill_kv_cache_shardings = self.prefill_kv_cache_shardings["decoder"]["layers_0"]
    self.kv_cache_annotations = maxtext_utils.get_kv_cache_annotations(
        self.model, self.config, rng2, self._mesh, self.page_state)
    self.kv_cache_shardings = jax.tree_util.tree_map(
        lambda x: jax.sharding.NamedSharding(self._mesh, x),
        self.kv_cache_annotations,
    )
    self.print_stats("After load_params (pickle path)")
    return state.params

  maxengine.MaxEngine.load_params = _patched_load_params

  # Patch uvicorn port from env (maxtext_server hardcodes 8000)
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

  # maxtext_server.py instantiates MaxTextGenerator(sys.argv) at module top
  # level, which calls pyconfig.initialize(sys.argv). pyconfig rejects unknown
  # flags like --npy_dir/--stage_dir, so strip them out of sys.argv before
  # the import happens.
  sys.argv = ["maxtext_server"] + rest
  from benchmarks.api_server import maxtext_server
  maxtext_server.main()


if __name__ == "__main__":
  main()
