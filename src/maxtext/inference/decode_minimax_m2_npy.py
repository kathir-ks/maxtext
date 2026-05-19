"""Decode wrapper for MiniMax-M2 that loads weights from streamed per-leaf .npy.

Companion to ``convert_minimax_m2_streaming``. Builds host-distributed JAX
arrays via ``jax.make_array_from_callback`` so no process ever materializes
the full ~460 GB tree in CPU RAM. Each TPU host mmap-loads only the slices
its 4 chips own from its local dtmpfs.

Usage::

    python -m maxtext.inference.decode_minimax_m2_npy \
        --npy_dir /mnt/dtmpfs/minimax-m2.7-npy \
        src/maxtext/configs/base.yml \
        model_name=minimax-m2.7 \
        tokenizer_path=/home/kathirks_gc/minimax-m2.7-tokenizer \
        ...
"""

import argparse
import functools
import json
import os
import pathlib
import sys
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.utils import max_logging


def _nested_to_flat(tree, prefix=""):
  out = {}
  for k, v in tree.items():
    key = f"{prefix}.{k}" if prefix else k
    if isinstance(v, dict):
      out.update(_nested_to_flat(v, key))
    else:
      out[key] = v
  return out


def _flat_to_nested(flat):
  out = {}
  for k, v in flat.items():
    parts = k.split(".")
    d = out
    for part in parts[:-1]:
      d = d.setdefault(part, {})
    d[parts[-1]] = v
  return out


def _detect_distributed(npy_dir: pathlib.Path) -> bool:
  return any(npy_dir.glob("manifest.p*.json"))


def _build_replicated(npy_dir, manifest, flat_shardings) -> dict:
  built = {}
  for leaf_name, meta in manifest["params"].items():
    sharding = flat_shardings.get(leaf_name)
    if sharding is None:
      max_logging.log(f"WARN: leaf {leaf_name!r} has no engine sharding; skipping")
      continue
    shape = tuple(meta["shape"])
    dtype = jnp.dtype(meta["dtype"])
    path = npy_dir / f"{leaf_name}.npy"
    mm = np.load(str(path), mmap_mode="r")
    assert mm.shape == shape, f"{leaf_name}: manifest shape {shape} vs file {mm.shape}"
    def cb(index, _mm=mm, _dtype=dtype):
      return np.asarray(_mm[index]).astype(_dtype)
    built[leaf_name] = jax.make_array_from_callback(shape, sharding, cb)
  return built


def _build_distributed(npy_dir, flat_shardings) -> dict:
  """Build params from per-(process, local_device) shards written by
  ``convert_minimax_m2_distributed``."""
  proc_idx = jax.process_index()
  local_devs = list(jax.local_devices())
  manifest_path = npy_dir / f"manifest.p{proc_idx}.json"
  if not manifest_path.exists():
    raise FileNotFoundError(f"no per-host manifest at {manifest_path}")
  with open(manifest_path, "rt") as f:
    manifest = json.load(f)

  global_shapes = {k: tuple(v) for k, v in manifest["global_shapes"].items()}
  global_dtype = jnp.dtype(manifest["global_dtype"])

  # Group shards by leaf and local device index.
  shards_by_leaf: dict = {}  # leaf -> {local_device_index: shard_meta}
  for shard_meta in manifest["shards"].values():
    shards_by_leaf.setdefault(shard_meta["leaf"], {})[shard_meta["local_device_index"]] = shard_meta

  built = {}
  for leaf_name, sharding in flat_shardings.items():
    if leaf_name not in shards_by_leaf:
      max_logging.log(f"WARN: leaf {leaf_name!r} missing from host manifest; skipping")
      continue
    global_shape = global_shapes[leaf_name]
    # Build a list of per-local-device jax.Arrays, ordered by local_devices().
    local_arrays = []
    for d_idx, dev in enumerate(local_devs):
      meta = shards_by_leaf[leaf_name][d_idx]
      path = npy_dir / f"{leaf_name}.p{proc_idx}.d{d_idx}.npy"
      mm = np.load(str(path), mmap_mode="r")
      arr = jax.device_put(np.asarray(mm).astype(global_dtype), dev)
      local_arrays.append(arr)
    built[leaf_name] = jax.make_array_from_single_device_arrays(
        global_shape, sharding, local_arrays)
  return built


def build_params(npy_dir: pathlib.Path, param_shardings) -> dict:
  """Return a MaxText-shaped pytree where every leaf is a globally-sharded jax.Array."""
  flat_shardings = _nested_to_flat(jax.tree.map(lambda x: x, param_shardings))
  wrap_under_params = all(k.startswith("params.") for k in flat_shardings)
  if wrap_under_params:
    flat_shardings = {k[len("params."):]: v for k, v in flat_shardings.items()}

  if _detect_distributed(npy_dir):
    max_logging.log(f"loading distributed manifest from {npy_dir}")
    built = _build_distributed(npy_dir, flat_shardings)
  else:
    with open(npy_dir / "manifest.json", "rt") as f:
      manifest = json.load(f)
    max_logging.log(f"loading replicated manifest from {npy_dir}")
    built = _build_replicated(npy_dir, manifest, flat_shardings)

  nested = _flat_to_nested(built)
  if wrap_under_params:
    nested = {"params": nested}
  return nested


def install_load_params_patch(npy_dir: pathlib.Path) -> None:
  """Monkey-patch MaxEngine.load_params to return params built from .npy."""
  from maxtext.inference.maxengine import maxengine as _maxengine
  from maxtext.utils import max_utils, maxtext_utils

  _orig = _maxengine.MaxEngine.load_params

  def _patched_load_params(self, *args, params=None, rng=None, **kwargs):
    if rng is None:
      rng = jax.random.PRNGKey(0)
    rng1, rng2, _rng3 = jax.random.split(rng, 3)

    init_state_fn = functools.partial(
        maxtext_utils.init_initial_state, self.model, None, self.config, False, rng1)
    _, self.state_mesh_annotations, state_mesh_shardings = maxtext_utils.get_abstract_state(
        self.config, self._mesh, init_state_fn, False)

    max_logging.log(f"Building params from .npy at {npy_dir} ...")
    npy_params = build_params(npy_dir, state_mesh_shardings.params)

    # Mirror what the original load_params does after sharding the input params:
    params_resharded = jax.device_put(npy_params, state_mesh_shardings.params)
    state = maxtext_utils.init_decode_state(None, params_resharded)
    state = max_utils.unbox_logicallypartioned(state)

    self.abstract_params = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)
        if isinstance(x, jax.Array) else None,
        state.params,
    )
    # KV cache annotations / shardings (matches original load_params).
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

    return state.params

  _maxengine.MaxEngine.load_params = _patched_load_params
  return _orig


def main():
  # Pull --npy_dir off the front; let everything else go to decode.main.
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--npy_dir", required=True)
  parser.add_argument("--help", "-h", action="store_true")
  args, rest = parser.parse_known_args()
  if args.help:
    print(__doc__)
    return

  npy_dir = pathlib.Path(args.npy_dir).expanduser().resolve()
  if not (npy_dir / "manifest.json").exists():
    raise SystemExit(f"no manifest.json found under {npy_dir}")

  install_load_params_patch(npy_dir)

  # Hand off the rest of argv to MaxText's decode.main so we inherit its
  # tokenizer / sampling / batching machinery unchanged.
  from maxtext.inference import decode as _decode
  _decode.main(["decode_minimax_m2_npy"] + rest)


if __name__ == "__main__":
  main()
