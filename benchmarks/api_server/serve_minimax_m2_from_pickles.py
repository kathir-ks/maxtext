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

  def _hbm_stats(label):
    parts = []
    for d in jax.devices():
      stats = d.memory_stats() if hasattr(d, "memory_stats") else None
      if stats:
        used = stats.get("bytes_in_use", 0) / 1e9
        peak = stats.get("peak_bytes_in_use", 0) / 1e9
        parts.append(f"chip{d.id}: used={used:.2f}G peak={peak:.2f}G")
      else:
        parts.append(f"chip{d.id}: (no stats)")
    print(f"[hbm @{label}] " + " | ".join(parts), flush=True)

  def _patched_load_params(self, *args, params=None, rng=None, **kwargs):
    if rng is None:
      rng = jax.random.PRNGKey(0)
    if self.model.quant and self.config.checkpoint_is_quantized:
      print("[serve-pickle] loading from pre-quantized pickles "
            f"(checkpoint_is_quantized=true): {len(pickles)} layer files",
            flush=True)
      from maxtext.layers import quantizations
      self.model.quant.quant_mode = quantizations.get_quant_mode("serve")

    _hbm_stats("start_of_load_params")

    rng1, rng2, _rng3 = jax.random.split(rng, 3)
    init_state_fn = functools.partial(
        maxtext_utils.init_initial_state, self.model, None, self.config, False, rng1)
    _, self.state_mesh_annotations, state_mesh_shardings = maxtext_utils.get_abstract_state(
        self.config, self._mesh, init_state_fn, False)
    _hbm_stats("after_get_abstract_state")

    # Build params pytree by streaming: for each layer, load pickle →
    # device_put that single layer's leaves onto TPU shards → drop host refs
    # → next layer. Host RAM stays at ~10 GB working set instead of 213 GB.
    import gc

    # The state_mesh_shardings.params is itself a dict like
    # {"params": {...}, "aqt": {...}} when quantization is enabled. Earlier
    # buggy code unwrapped the outer dict and dropped the "aqt" subtree, so
    # every AQT device_put fell through to the no-sharding default and piled
    # the whole quantized model onto chip 0. Keep the outer structure intact.
    raw_shardings = state_mesh_shardings.params
    if hasattr(raw_shardings, "_dict"):  # FrozenDict
      raw_shardings = dict(raw_shardings)

    # If MaxText returned the {"params": ..., "aqt": ...} layout, use those
    # subtrees directly. Otherwise raw_shardings is already the inner dict
    # (no "params" key); in that case AQT shardings don't exist as a sibling
    # subtree — but with checkpoint_is_quantized=true the AQT pytree IS
    # encoded under the "params" subtree, so just iterate it.
    def _to_dict(x):
      if hasattr(x, "_dict"):
        return dict(x._dict)
      return x

    if "aqt" in raw_shardings:
      params_shardings_root = dict(raw_shardings.get("params", {}))
      aqt_shardings_root = dict(raw_shardings.get("aqt", {}))
    else:
      params_shardings_root = raw_shardings
      aqt_shardings_root = {}

    # Diagnostic dump of the actual sharding tree structure on first call.
    if not getattr(self, "_diag_dumped", False):
      print("[diag] raw_shardings top-level keys:", list(raw_shardings.keys()), flush=True)
      print("[diag] params_shardings_root keys:", list(params_shardings_root.keys()), flush=True)
      print("[diag] aqt_shardings_root keys:", list(aqt_shardings_root.keys()), flush=True)
      pdec = _to_dict(params_shardings_root.get("decoder", {}))
      print("[diag] params.decoder keys (first 8):", list(pdec.keys())[:8], flush=True)
      adec = _to_dict(aqt_shardings_root.get("decoder", {}))
      print("[diag] aqt.decoder keys (first 8):", list(adec.keys())[:8], flush=True)
      for name in ("layers_0", "layers"):
        if name in pdec:
          sub = _to_dict(pdec[name])
          print(f"[diag] params.decoder.{name} keys:", list(sub.keys())[:20], flush=True)
          # Recurse one more level for any sub-dict to see leaf sharding
          for k, v in list(sub.items())[:5]:
            v2 = _to_dict(v)
            if isinstance(v2, dict):
              print(f"[diag]   .{k} keys:", list(v2.keys())[:10], flush=True)
              for k2, v3 in list(v2.items())[:3]:
                v4 = _to_dict(v3)
                if not isinstance(v4, dict):
                  print(f"[diag]     .{k}.{k2} sharding:", repr(v4), flush=True)
            else:
              print(f"[diag]   .{k} sharding:", repr(v2), flush=True)
          break
      for name in ("layers_0", "layers"):
        if name in adec:
          sub = _to_dict(adec[name])
          print(f"[diag] aqt.decoder.{name} keys:", list(sub.keys())[:20], flush=True)
          for k, v in list(sub.items())[:5]:
            v2 = _to_dict(v)
            if isinstance(v2, dict):
              print(f"[diag]   .{k} keys:", list(v2.keys())[:10], flush=True)
            else:
              print(f"[diag]   .{k} sharding:", repr(v2), flush=True)
          break
      self._diag_dumped = True

    # Provide a default replicated sharding for any leaf without one (so we
    # never silently fall back to chip 0).
    replicated = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())

    sharded = {"params": {"decoder": {}}, "aqt": {"decoder": {}}}

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

    _hbm_stats("before_layer_loop")
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      for idx, pkl in enumerate(pickles):
        layer_name = pkl.stem
        with open(pkl, "rb") as f:
          stage = pickle.load(f)
        # Re-key MoE expert AQT weights to match MaxText's abstract layout.
        # Pickle stores them as aqt.moe_block.{wi_0, wi_1, wo} (from the manual
        # re-keying in layerwise_quantize), but with non-megablox MoE the
        # einsums are registered at the LAYER level as AqtEinsum_4/5/6 (sibling
        # of moe_block / self_attention). Without this re-key the leaves carry
        # no sharding match → fall back to replicated → 1.82 GB/chip/layer → OOM.
        if stage.get("aqt") is not None:
          moe_aqt = stage["aqt"].get("moe_block", {})
          for src, dst in (("wi_0", "AqtEinsum_4"),
                            ("wi_1", "AqtEinsum_5"),
                            ("wo",   "AqtEinsum_6")):
            if src in moe_aqt:
              stage["aqt"][dst] = moe_aqt.pop(src)
          # Drop moe_block entirely if only stale 'gate' remains and no
          # corresponding sharding tree subkey for it.
        # Prune empty-dict leaves left by remove_quantized_params.
        stage["params"] = _prune_empty(stage["params"])
        if idx < 2 or idx % 10 == 0:
          _hbm_stats(f"pre_layer_{idx}")

        # Find the layer's sharding sub-tree.
        decoder_shardings = _to_dict(params_shardings_root.get("decoder", {}))
        layer_shardings = decoder_shardings.get(layer_name)
        aqt_decoder_shardings = _to_dict(aqt_shardings_root.get("decoder", {}))
        layer_aqt_shardings = aqt_decoder_shardings.get(layer_name)

        def _ensure_sharded(value_tree, sharding_tree):
          """Replace any None sharding with replicated, then device_put."""
          if sharding_tree is None:
            return jax.tree.map(lambda x: jax.device_put(x, replicated), value_tree)
          # Map structure: pad missing leaves with replicated sharding.
          def _walk(v, s):
            if isinstance(v, dict):
              s = s if isinstance(s, dict) else {}
              return {k: _walk(v[k], s.get(k)) for k in v}
            return jax.device_put(v, s if s is not None else replicated)
          return _walk(value_tree, sharding_tree)

        sharded["params"]["decoder"][layer_name] = _ensure_sharded(
            stage["params"], layer_shardings)
        if stage["aqt"] is not None:
          sharded["aqt"]["decoder"][layer_name] = _ensure_sharded(
              stage["aqt"], layer_aqt_shardings)

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

      te_shards = _to_dict(params_shardings_root.get("token_embedder", {}))
      sharded["params"]["token_embedder"] = place(
          {"embedding": load_npy("token_embedder.embedding.npy", weight_dtype)},
          te_shards,
      )

      dec_shards = _to_dict(params_shardings_root.get("decoder", {}))
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
