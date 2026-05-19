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


def build_params(npy_dir: pathlib.Path, param_shardings) -> dict:
  """Return a MaxText-shaped pytree where every leaf is a globally-sharded jax.Array."""
  with open(npy_dir / "manifest.json", "rt") as f:
    manifest = json.load(f)

  flat_shardings = _nested_to_flat(jax.tree.map(lambda x: x, param_shardings))

  # MaxText flows typically nest params under a top-level "params" subkey
  # (Linen "params" collection). Detect that and remember it so we can
  # mirror it on the way out.
  wrap_under_params = all(k.startswith("params.") for k in flat_shardings)
  if wrap_under_params:
    flat_shardings = {k[len("params."):]: v for k, v in flat_shardings.items()}

  built = {}
  for leaf_name, meta in manifest["params"].items():
    sharding = flat_shardings.get(leaf_name)
    if sharding is None:
      max_logging.log(f"WARNING: manifest leaf {leaf_name!r} has no matching engine sharding; skipping")
      continue
    shape = tuple(meta["shape"])
    dtype = jnp.dtype(meta["dtype"])

    path = npy_dir / f"{leaf_name}.npy"
    mm = np.load(str(path), mmap_mode="r")
    assert mm.shape == shape, f"{leaf_name}: manifest shape {shape} vs file {mm.shape}"

    def cb(index, _mm=mm, _dtype=dtype):
      return np.asarray(_mm[index]).astype(_dtype)

    built[leaf_name] = jax.make_array_from_callback(shape, sharding, cb)

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
