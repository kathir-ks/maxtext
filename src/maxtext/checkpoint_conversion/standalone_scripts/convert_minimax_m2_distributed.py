"""Distributed HF -> per-device .npy converter for MiniMax-M2 / M2.7.

Companion to ``convert_minimax_m2_streaming``. Run as one process per TPU
host on a multi-host pod (16 processes on v6e-64). Each host writes only
the slices of the model its local TPU chips will own at decode time, so
the model is sharded across the pod's dtmpfs instead of replicated 16x.

  - ``jax.distributed.initialize()`` runs first; the engine's mesh and
    shardings then reflect the full 64-chip pod.
  - For each parameter leaf, ``sharding.addressable_devices_indices_map``
    tells us exactly which global-array slice each local chip needs.
  - We download only the HF safetensors files that contain tensors in
    this host's expert range (plus shared attention/embed files).
  - One ``.npy`` file is written per (leaf, local_device); the decode
    loader rebuilds the global jax.Array via
    ``jax.make_array_from_single_device_arrays``.

Per-host dtmpfs after a successful run: ~30 GB (vs. ~460 GB replicated).
"""

import argparse
import functools
import gc
import json
import os
import pathlib
import time
from typing import Dict, List, Set, Tuple

import jax
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from maxtext.utils import max_logging


MODEL_PARAMS_DICT = {
    "minimax-m2": dict(num_hidden_layers=62, num_attention_heads=48, num_key_value_heads=8,
                       hidden_size=3072, head_dim=128, num_experts=256,
                       moe_intermediate_size=1536, vocab_size=200064),
    "minimax-m2.7": dict(num_hidden_layers=62, num_attention_heads=48, num_key_value_heads=8,
                        hidden_size=3072, head_dim=128, num_experts=256,
                        moe_intermediate_size=1536, vocab_size=200064),
}


# Layer axis position in our post-transpose layout. None for non-layered leaves.
_LAYER_AXIS: Dict[str, int] = {
    "decoder.layers.pre_self_attention_layer_norm.scale": 1,
    "decoder.layers.post_self_attention_layer_norm.scale": 1,
    "decoder.layers.self_attention.query.kernel": 1,
    "decoder.layers.self_attention.key.kernel": 1,
    "decoder.layers.self_attention.value.kernel": 1,
    "decoder.layers.self_attention.out.kernel": 1,
    "decoder.layers.self_attention.query_norm.scale": 1,
    "decoder.layers.self_attention.key_norm.scale": 1,
    "decoder.layers.moe_block.gate.kernel": 1,
    "decoder.layers.moe_block.gate.bias": 1,
    "decoder.layers.moe_block.wi_0": 1,
    "decoder.layers.moe_block.wi_1": 1,
    "decoder.layers.moe_block.wo": 1,
}


# Expert axis position. None if leaf has no expert dim.
_EXPERT_AXIS: Dict[str, int] = {
    "decoder.layers.moe_block.wi_0": 0,
    "decoder.layers.moe_block.wi_1": 0,
    "decoder.layers.moe_block.wo": 0,
    "decoder.layers.moe_block.gate.bias": 0,
    "decoder.layers.moe_block.gate.kernel": 2,
}


def leaf_shapes(p: dict) -> Dict[str, Tuple[Tuple[int, ...], str]]:
  L, H = p["num_hidden_layers"], p["hidden_size"]
  Hq, Hkv, D = p["num_attention_heads"], p["num_key_value_heads"], p["head_dim"]
  E, F = p["num_experts"], p["moe_intermediate_size"]
  V = p["vocab_size"]
  return {
      "token_embedder.embedding": ((V, H), "float16"),
      "decoder.decoder_norm.scale": ((H,), "float16"),
      "decoder.logits_dense.kernel": ((H, V), "float16"),
      "decoder.layers.pre_self_attention_layer_norm.scale": ((H, L), "float16"),
      "decoder.layers.post_self_attention_layer_norm.scale": ((H, L), "float16"),
      "decoder.layers.self_attention.query.kernel": ((H, L, Hq, D), "float16"),
      "decoder.layers.self_attention.key.kernel": ((H, L, Hkv, D), "float16"),
      "decoder.layers.self_attention.value.kernel": ((H, L, Hkv, D), "float16"),
      "decoder.layers.self_attention.out.kernel": ((Hq, L, D, H), "float16"),
      "decoder.layers.self_attention.query_norm.scale": ((Hq * D, L), "float16"),
      "decoder.layers.self_attention.key_norm.scale": ((Hkv * D, L), "float16"),
      "decoder.layers.moe_block.gate.kernel": ((H, L, E), "float16"),
      "decoder.layers.moe_block.gate.bias": ((E, L), "float16"),
      "decoder.layers.moe_block.wi_0": ((E, L, H, F), "float16"),
      "decoder.layers.moe_block.wi_1": ((E, L, H, F), "float16"),
      "decoder.layers.moe_block.wo": ((E, L, F, H), "float16"),
  }


# --- HF fetching ----------------------------------------------------------

def _block_dequant_to_bf16(x: torch.Tensor, s: torch.Tensor, block: int = 128) -> torch.Tensor:
  assert x.dim() == 2 and s.dim() == 2
  M, N = x.shape
  x = x.to(torch.float32)
  y = torch.empty_like(x, dtype=torch.bfloat16)
  for i in range(0, M, block):
    for j in range(0, N, block):
      r0, r1 = i, min(i + block, M)
      c0, c1 = j, min(j + block, N)
      scale = s[i // block, j // block]
      y[r0:r1, c0:c1] = (x[r0:r1, c0:c1] * scale).to(torch.bfloat16)
  return y


def _get_tensor_bf16(weight_map, base_path, name, handle_cache):
  fname = weight_map[name]
  if fname not in handle_cache:
    handle_cache[fname] = safe_open(os.path.join(base_path, fname), framework="pt", device="cpu")
  t = handle_cache[fname].get_tensor(name)
  if t.element_size() == 1:
    scale_name = f"{name}_scale_inv"
    if weight_map.get(scale_name) == fname:
      scale = handle_cache[fname].get_tensor(scale_name)
    else:
      scale = _get_tensor_bf16(weight_map, base_path, scale_name, handle_cache)
    return _block_dequant_to_bf16(t, scale)
  return t.to(torch.bfloat16)


def _np_from_torch(t):
  return t.to(torch.float32).numpy().astype(np.float16)


# --- Sharding helpers -----------------------------------------------------

def _nested_to_flat(tree, prefix=""):
  out = {}
  for k, v in tree.items():
    key = f"{prefix}.{k}" if prefix else k
    if isinstance(v, dict):
      out.update(_nested_to_flat(v, key))
    else:
      out[key] = v
  return out


def _normalize_idx(idx: Tuple, global_shape: Tuple[int, ...]) -> Tuple[slice, ...]:
  """Replace any slice(None) entries with concrete bounds from global_shape.

  ``Sharding.addressable_devices_indices_map`` returns ``slice(None, None, None)``
  for axes that are fully replicated across the mesh. Downstream code wants
  concrete ``[start, stop)`` for size arithmetic; this normalizes them.
  """
  out = []
  for s, dim in zip(idx, global_shape):
    if not isinstance(s, slice):
      out.append(slice(int(s), int(s) + 1))
      continue
    start, stop, step = s.indices(dim)
    assert step == 1, f"non-unit slice step not supported: {s}"
    out.append(slice(start, stop))
  return tuple(out)


def _local_shape_from_idx(idx: Tuple) -> Tuple[int, ...]:
  return tuple(s.stop - s.start for s in idx)


def _drop_axis(idx: Tuple, axis: int) -> Tuple:
  return idx[:axis] + idx[axis + 1:]


def _replace_axis(idx: Tuple, axis: int, new) -> Tuple:
  return idx[:axis] + (new,) + idx[axis + 1:]


def maxtext_pyconfig_and_shardings(model_size: str, extra_args: List[str]):
  """Initialize MaxText config + engine, return param sharding tree."""
  from maxtext.configs import pyconfig
  from maxtext.inference.maxengine import maxengine
  from maxtext.utils import maxtext_utils
  config = pyconfig.initialize(
      ["convert_minimax_m2_distributed"]
      + extra_args
      + [
          f"model_name={model_size}",
          "scan_layers=true",
          "weight_dtype=bfloat16",
          "attention=dot_product",
          "quantization=",
      ]
  )
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(0)
  init_state_fn = functools.partial(
      maxtext_utils.init_initial_state, engine.model, None, engine.config, False, rng)
  _, _, state_mesh_shardings = maxtext_utils.get_abstract_state(
      engine.config, engine._mesh, init_state_fn, False)
  return engine._mesh, state_mesh_shardings.params


# --- Writer helper --------------------------------------------------------

def write_layer_slice(arrs: Dict, per_dev_slices: Dict, leaf: str, L: int,
                     source: np.ndarray, local_devs) -> None:
  """Write the L'th layer's contribution to every local device that owns it.

  ``source`` has the same shape as the global tensor but with the layer
  axis squeezed out (e.g. for a global (H, L, Hq, D) leaf, source is
  shape (H, Hq, D)).
  """
  layer_axis = _LAYER_AXIS[leaf]
  for d in range(len(local_devs)):
    idx = per_dev_slices[(leaf, d)]
    l_slc = idx[layer_axis]
    if not (l_slc.start <= L < l_slc.stop):
      continue
    local_L = L - l_slc.start
    src_idx = _drop_axis(idx, layer_axis)
    # Target index in the per-device memmap: shape has the local layer span,
    # we write a single layer at position ``local_L``.
    tgt = arrs[(leaf, d)]
    # Build tgt_idx as full-range on every axis except layer, single index on layer.
    tgt_idx = tuple(local_L if i == layer_axis else slice(None) for i in range(tgt.ndim))
    tgt[tgt_idx] = source[src_idx]


def write_replicated(arrs: Dict, per_dev_slices: Dict, leaf: str,
                    source: np.ndarray, local_devs) -> None:
  """Write a leaf that has no layer axis (token_embedder / decoder_norm / logits_dense)."""
  for d in range(len(local_devs)):
    idx = per_dev_slices[(leaf, d)]
    arrs[(leaf, d)][...] = source[idx]


# --- Main -----------------------------------------------------------------

def main(args):
  jax.distributed.initialize()
  proc_idx = jax.process_index()
  proc_count = jax.process_count()
  local_devs = list(jax.local_devices())

  def log(msg):
    max_logging.log(f"[p{proc_idx}/{proc_count}] {msg}")

  log(f"jax.distributed up: {len(local_devs)} local devices of {len(jax.devices())} global")

  if args.model_size not in MODEL_PARAMS_DICT:
    raise SystemExit(f"unknown model_size {args.model_size!r}")
  p = MODEL_PARAMS_DICT[args.model_size]
  shapes = leaf_shapes(p)

  # 1) Get the engine's per-leaf shardings.
  mesh, param_shardings = maxtext_pyconfig_and_shardings(args.model_size, args.maxtext_args)
  flat = _nested_to_flat(jax.tree.map(lambda x: x, param_shardings))
  if all(k.startswith("params.") for k in flat):
    flat = {k[len("params."):]: v for k, v in flat.items()}

  # 2) For each (leaf, local_device), compute the global-array slice.
  per_dev_slices: Dict[Tuple[str, int], Tuple] = {}
  for leaf, (global_shape, _) in shapes.items():
    if leaf not in flat:
      raise KeyError(f"leaf {leaf!r} not in engine sharding tree; available: {list(flat)[:3]} ...")
    sharding = flat[leaf]
    addr = sharding.addressable_devices_indices_map(global_shape)
    for dev, idx in addr.items():
      per_dev_slices[(leaf, local_devs.index(dev))] = _normalize_idx(idx, global_shape)

  # 3) Compute this host's expert range from the MoE shardings.
  expert_ranges = []
  for leaf, e_ax in _EXPERT_AXIS.items():
    for d in range(len(local_devs)):
      idx = per_dev_slices[(leaf, d)]
      expert_ranges.append((idx[e_ax].start, idx[e_ax].stop))
  host_expert_lo = min(r[0] for r in expert_ranges)
  host_expert_hi = max(r[1] for r in expert_ranges)
  log(f"expert range [{host_expert_lo}, {host_expert_hi}) (host owns {host_expert_hi - host_expert_lo} of {p['num_experts']})")

  # 4) Ensure HF index, then download only needed files.
  hf_dir = pathlib.Path(args.hf_dir)
  hf_dir.mkdir(parents=True, exist_ok=True)
  # Always fetch metadata + tokenizer up front (tiny). The decode wrapper
  # consumes the tokenizer from this same directory, so we co-locate it
  # rather than requiring a separate snapshot_download step.
  metadata_files = [
      "model.safetensors.index.json",
      "config.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "special_tokens_map.json",
      "chat_template.jinja",
      "configuration_minimax_m2.py",
  ]
  missing_metadata = [fn for fn in metadata_files if not (hf_dir / fn).exists()]
  if missing_metadata:
    from huggingface_hub import hf_hub_download
    log(f"downloading {len(missing_metadata)} metadata/tokenizer files")
    for fn in missing_metadata:
      try:
        hf_hub_download(repo_id=args.repo_id, filename=fn, local_dir=str(hf_dir))
      except Exception as exc:  # not every repo has every optional file
        log(f"  skipping optional file {fn}: {exc}")
  with open(hf_dir / "model.safetensors.index.json", "rt") as f:
    weight_map = json.load(f)["weight_map"]

  needed = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
  for L in range(p["num_hidden_layers"]):
    needed.update([
        f"model.layers.{L}.input_layernorm.weight",
        f"model.layers.{L}.post_attention_layernorm.weight",
        f"model.layers.{L}.self_attn.q_proj.weight",
        f"model.layers.{L}.self_attn.k_proj.weight",
        f"model.layers.{L}.self_attn.v_proj.weight",
        f"model.layers.{L}.self_attn.o_proj.weight",
        f"model.layers.{L}.self_attn.q_norm.weight",
        f"model.layers.{L}.self_attn.k_norm.weight",
        f"model.layers.{L}.block_sparse_moe.gate.weight",
        f"model.layers.{L}.block_sparse_moe.e_score_correction_bias",
    ])
    for e in range(host_expert_lo, host_expert_hi):
      needed.update([
          f"model.layers.{L}.block_sparse_moe.experts.{e}.w1.weight",
          f"model.layers.{L}.block_sparse_moe.experts.{e}.w2.weight",
          f"model.layers.{L}.block_sparse_moe.experts.{e}.w3.weight",
      ])
  needed |= {f"{n}_scale_inv" for n in list(needed) if f"{n}_scale_inv" in weight_map}
  needed_files = sorted({weight_map[t] for t in needed if t in weight_map})
  all_files = sorted(set(weight_map.values()))
  log(f"need {len(needed_files)} of {len(all_files)} HF files")

  # Compute file lifecycle: for each needed file, the set of layer indices
  # whose tensors live in it. -1 means "shared / non-layer".
  file_layers: Dict[str, Set[int]] = {}
  for tname in needed:
    if tname not in weight_map:
      continue
    fname = weight_map[tname]
    if tname.startswith("model.layers."):
      L_idx = int(tname.split(".")[2])
      file_layers.setdefault(fname, set()).add(L_idx)
    else:
      file_layers.setdefault(fname, set()).add(-1)
  first_layer = {fn: (min(ls) if min(ls) >= 0 else -1) for fn, ls in file_layers.items()}
  last_layer = {fn: (max(ls) if max(ls) >= 0 else -1) for fn, ls in file_layers.items()}

  from huggingface_hub import hf_hub_download, snapshot_download

  # Parallel upfront fetch of every safetensors file this host will need.
  # ~30 GB per host with `max_workers=8` saturates HF's CDN at ~1 GB/s
  # instead of the sequential ~150 MB/s a single hf_hub_download achieves.
  missing_data = [fn for fn in needed_files if not (hf_dir / fn).exists()]
  if missing_data:
    log(f"snapshot_download: {len(missing_data)} of {len(needed_files)} needed files missing")
    t0 = time.time()
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(hf_dir),
        allow_patterns=missing_data,
        max_workers=args.hf_download_workers,
    )
    log(f"snapshot_download done in {time.time() - t0:.1f}s")
  else:
    log("all needed safetensors already on dtmpfs; skipping snapshot_download")

  def ensure_file(fn: str) -> None:
    # Fallback in case a file got deleted prematurely by drop_file; the
    # main download path is the upfront snapshot_download above.
    if (hf_dir / fn).exists():
      return
    hf_hub_download(repo_id=args.repo_id, filename=fn, local_dir=str(hf_dir))

  def drop_file(fn: str, handle_cache: Dict) -> None:
    if fn in handle_cache:
      del handle_cache[fn]
    p = hf_dir / fn
    try:
      p.unlink()
    except OSError:
      pass

  # 5) Open per-device memmaps for all leaves.
  out_dir = pathlib.Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  arrs: Dict[Tuple[str, int], np.ndarray] = {}
  total_bytes = 0
  for (leaf, d), idx in per_dev_slices.items():
    shp = _local_shape_from_idx(idx)
    path = out_dir / f"{leaf}.p{proc_idx}.d{d}.npy"
    arrs[(leaf, d)] = np.lib.format.open_memmap(
        str(path), mode="w+", shape=shp, dtype=np.dtype(shapes[leaf][1]))
    total_bytes += int(np.prod(shp)) * np.dtype(shapes[leaf][1]).itemsize
  log(f"opened {len(arrs)} memmaps; target = {total_bytes/1e9:.1f} GB")

  # 6) Fill: non-layered leaves first.
  handle_cache: Dict[str, "safe_open"] = {}
  def fetch(name):
    return _get_tensor_bf16(weight_map, str(hf_dir), name, handle_cache)

  log("writing non-layered leaves (embed / final norm / lm_head)")
  emb = _np_from_torch(fetch("model.embed_tokens.weight"))
  write_replicated(arrs, per_dev_slices, "token_embedder.embedding", emb, local_devs)
  del emb

  fnorm = _np_from_torch(fetch("model.norm.weight"))
  write_replicated(arrs, per_dev_slices, "decoder.decoder_norm.scale", fnorm, local_devs)
  del fnorm

  lmh = _np_from_torch(fetch("lm_head.weight")).T.copy()
  write_replicated(arrs, per_dev_slices, "decoder.logits_dense.kernel", lmh, local_devs)
  del lmh

  # Drop any files whose only purpose was shared tensors (last_layer == -1).
  for fn in list(file_layers):
    if max(file_layers[fn]) == -1:
      drop_file(fn, handle_cache)

  H = p["hidden_size"]
  Hq, Hkv, D = p["num_attention_heads"], p["num_key_value_heads"], p["head_dim"]

  log("writing per-layer leaves (all safetensors already on dtmpfs)")
  for L in tqdm(range(p["num_hidden_layers"]), desc="layers"):
    write_layer_slice(arrs, per_dev_slices,
        "decoder.layers.pre_self_attention_layer_norm.scale", L,
        _np_from_torch(fetch(f"model.layers.{L}.input_layernorm.weight")), local_devs)
    write_layer_slice(arrs, per_dev_slices,
        "decoder.layers.post_self_attention_layer_norm.scale", L,
        _np_from_torch(fetch(f"model.layers.{L}.post_attention_layernorm.weight")), local_devs)

    src = _np_from_torch(fetch(f"model.layers.{L}.self_attn.q_proj.weight").T.reshape(H, Hq, D))
    write_layer_slice(arrs, per_dev_slices,
        "decoder.layers.self_attention.query.kernel", L, src, local_devs)
    del src

    src = _np_from_torch(fetch(f"model.layers.{L}.self_attn.k_proj.weight").T.reshape(H, Hkv, D))
    write_layer_slice(arrs, per_dev_slices,
        "decoder.layers.self_attention.key.kernel", L, src, local_devs)
    del src

    src = _np_from_torch(fetch(f"model.layers.{L}.self_attn.v_proj.weight").T.reshape(H, Hkv, D))
    write_layer_slice(arrs, per_dev_slices,
        "decoder.layers.self_attention.value.kernel", L, src, local_devs)
    del src

    src = _np_from_torch(fetch(f"model.layers.{L}.self_attn.o_proj.weight").T.reshape(Hq, D, H))
    write_layer_slice(arrs, per_dev_slices,
        "decoder.layers.self_attention.out.kernel", L, src, local_devs)
    del src

    write_layer_slice(arrs, per_dev_slices,
        "decoder.layers.self_attention.query_norm.scale", L,
        _np_from_torch(fetch(f"model.layers.{L}.self_attn.q_norm.weight")), local_devs)
    write_layer_slice(arrs, per_dev_slices,
        "decoder.layers.self_attention.key_norm.scale", L,
        _np_from_torch(fetch(f"model.layers.{L}.self_attn.k_norm.weight")), local_devs)

    gk = _np_from_torch(fetch(f"model.layers.{L}.block_sparse_moe.gate.weight").T)
    write_layer_slice(arrs, per_dev_slices,
        "decoder.layers.moe_block.gate.kernel", L, gk, local_devs)
    del gk

    gb = _np_from_torch(fetch(f"model.layers.{L}.block_sparse_moe.e_score_correction_bias"))
    write_layer_slice(arrs, per_dev_slices,
        "decoder.layers.moe_block.gate.bias", L, gb, local_devs)
    del gb

    # Expert weights — only fetch experts this host's chips own.
    for e in range(host_expert_lo, host_expert_hi):
      w1 = _np_from_torch(fetch(f"model.layers.{L}.block_sparse_moe.experts.{e}.w1.weight").T)  # (H, F)
      w2 = _np_from_torch(fetch(f"model.layers.{L}.block_sparse_moe.experts.{e}.w2.weight").T)  # (F, H)
      w3 = _np_from_torch(fetch(f"model.layers.{L}.block_sparse_moe.experts.{e}.w3.weight").T)  # (H, F)
      for leaf, src in (("decoder.layers.moe_block.wi_0", w1),
                        ("decoder.layers.moe_block.wi_1", w3),
                        ("decoder.layers.moe_block.wo", w2)):
        for d in range(len(local_devs)):
          e_slc, l_slc, a_slc, b_slc = per_dev_slices[(leaf, d)]
          if not (e_slc.start <= e < e_slc.stop):
            continue
          if not (l_slc.start <= L < l_slc.stop):
            continue
          le, ll = e - e_slc.start, L - l_slc.start
          arrs[(leaf, d)][le, ll, :, :] = src[a_slc, b_slc]
      del w1, w2, w3

    # Drop files whose last_layer == L (no future layer needs them).
    for fn in list(file_layers):
      if last_layer[fn] != -1 and last_layer[fn] <= L:
        if (hf_dir / fn).exists():
          drop_file(fn, handle_cache)
    # Always release handle cache so the unlinked files don't hold dtmpfs pages.
    handle_cache.clear()
    gc.collect()

  for arr in arrs.values():
    arr.flush()

  manifest = {
      "model_size": args.model_size,
      "process_index": proc_idx,
      "process_count": proc_count,
      "local_device_count": len(local_devs),
      "expert_range": [host_expert_lo, host_expert_hi],
      "global_shapes": {leaf: list(s) for leaf, (s, _) in shapes.items()},
      "global_dtype": "float16",
      "shards": {
          f"{leaf}.p{proc_idx}.d{d}": {
              "leaf": leaf,
              "process_index": proc_idx,
              "local_device_index": d,
              "local_shape": list(_local_shape_from_idx(per_dev_slices[(leaf, d)])),
              "global_index": [[s.start, s.stop] for s in per_dev_slices[(leaf, d)]],
          }
          for (leaf, d) in arrs
      },
  }
  with open(out_dir / f"manifest.p{proc_idx}.json", "wt") as f:
    json.dump(manifest, f, indent=2)
  log(f"done; wrote manifest.p{proc_idx}.json")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--hf_dir", required=True)
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--model_size", required=True, choices=list(MODEL_PARAMS_DICT))
  parser.add_argument("--repo_id", default="MiniMaxAI/MiniMax-M2.7")
  parser.add_argument("--skip_download", action="store_true")
  parser.add_argument("--hf_download_workers", type=int, default=8,
                      help="Parallel workers for snapshot_download (default 8). "
                           "Bound by your HF auth tier and host network.")
  parser.add_argument("--maxtext_args", nargs=argparse.REMAINDER, default=[])
  main(parser.parse_args())
