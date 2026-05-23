# Why MiniMax-M2.7 is ~900× slower on v5e-64 than v6e-64

## 1. Measured findings (everything we know)

### Hardware

| | v5e-64 (`v5litepod-64`) | v6e-64 |
|---|---|---|
| Chips | 64 (16 hosts × 4) | 64 (16 hosts × 4) |
| HBM / chip | 16 GB | 32 GB |
| BF16 TFLOPS / chip | ~197 | ~600 |
| HBM bandwidth / chip | ~360 GB/s | ~819 GB/s |
| ICI bandwidth / link | ~70 GB/s | ~140 GB/s |
| **`ragged-all-to-all`** support | ❌ "limited ICI routing" | ✅ |
| Per-host RAM | 188 GB | 708 GB |

The single decisive line is **ragged-all-to-all** on the ICI. It is the
collective MaxText's `megablox` / `sparse_matmul` MoE path uses to ship
each token to the 8 of 256 experts it actually needs. Without it, the
MoE has to compute every token through **every** expert (limited by a
capacity factor), then mask the unused outputs.

### Measured throughput (HEAD `b915df80`, `feature/minimax-m2.7`)

| | v5e-64 (dense MoE fallback) | v6e-64 (sparse MoE) |
|---|---|---|
| Best config | bf16 weights + int4 KV, batch=8, ctx=256 | bf16 weights + int4 KV, batch=96, ctx=128 |
| Step time | 1083 ms | 920 ms |
| **tok/s pod-wide** | **7.4** | **6,678** |
| tok/s / chip | 0.115 | 104.3 |
| Single-stream tok/s (batch=1) | 4.4 | 13.4 |
| Step time at batch=1 | 229 ms | 74 ms |

Raw data: `benchmarks/v5e_minimax_m2_7_2026_05_23/v5e_results.csv` and
`benchmarks/v6e_minimax_m2_7_2026_05_22/*.json`.

### Where the 900× gap comes from (decomposition)

| Factor | v5e | v6e | Contribution to step time |
|---|---|---|---|
| Bytes read / token (full MoE vs sparse) | 460 GB | ~17 GB (8/256 experts) | **27×** less on v6e |
| HBM bandwidth / chip | 360 GB/s | 819 GB/s | **2.3×** faster on v6e |
| TFLOPS / chip | 197 | 600 | **3.0×** faster on v6e |
| Achievable batch (HBM cap) | 8 | 96 | **12×** more amortization on v6e |

Multiplying conservatively: `27 × 2.3 × (batch-amortization)` is in the
right order of magnitude of the 900× observed gap. The single sparse-vs-
dense factor alone explains an order of magnitude; the rest is hardware.

## 2. The problem, stated precisely

> The MiniMax-M2.7 architecture is **256-expert sparse MoE**. At decode
> time only 8 experts are active per token. To exploit that sparsity,
> the runtime must use a **ragged all-to-all collective** on the
> inter-chip interconnect — one chip dispatches a variable-length list
> of tokens to each of its peers, and each peer returns a variable-
> length list of outputs.
>
> **TPU v5e's ICI implementation does not support
> ragged-all-to-all.** XLA on v5e errors with
> `RET_CHECK failure ... !HasLimitedIciRouting() Ragged all-to-all is
> currently not supported in limited ICI routing settings`.
>
> MaxText's only fallback at this point is `megablox=false
> sparse_matmul=false capacity_factor=2.0`, which routes tokens to
> every expert in a *dense* matmul, masks unused outputs, and then
> reduces. This is mathematically equivalent (top-8 routing is still
> applied) but the runtime cost is the cost of processing **all 256**
> experts every token — ~32× the FLOPs and ~32× the HBM bandwidth.
>
> Combined with v5e's lower per-chip HBM bandwidth (2.3×) and compute
> (3.0×), the net effect is a ~900× throughput gap vs v6e — and that
> matches measurement.

Secondary problems uncovered along the way:

- **`qwix` Pallas int8 weight-quantization tile assertion** —
  `v=192 bv=1024 s=192`. The per-chip MLP intermediate dim (`1536/8`)
  doesn't divide qwix's default tile (`1024`). Blocks int8 weight quant
  on both v5e and v6e until the tile size is plumbed through.
- **`bench_steps_minimax_m2_npy.py` silent-kills on v5e** (works on
  v6e). Probably a different jax-distributed lifecycle on v5e. Worked
  around by using `decode_distributed.sh` differentials.
- **HBM cliff at batch=8 on v5e ctx=256** (compared with batch=96 on
  v6e). The dense-MoE compute is what's blocking, not raw HBM.

## 3. Possible solutions

Ordered roughly by effort × payoff.

### A. Take the v6e (easy, biggest payoff)
Just run on v6e-64-europe-west4-a, which already produces 6,678 tok/s.
v6e in europe-west4-a tends to be available; when it isn't, the right
move is to wait + poll, not to fall back to v5e for this model.
**Outcome:** 900× speedup, no code work.

### B. Fix int8 weight quantization
Add `wi_tile_*_mlp_dim=192 wo_tile_*_mlp_dim=192
wo_tile_*_embed_dim=384 wi_tile_*_embed_dim=384` (or other divisors of
the per-chip dims) to the bench scripts so qwix's Pallas kernel
accepts the tile. On **v6e** this should buy 50-100% more throughput
(halves the weight bandwidth from 32 bits/elem to 8 bits/elem). On
**v5e** it still won't beat the sparse-MoE gap but bumps batch=8 from
~7.4 → ~12-15 tok/s pod-wide.
**Outcome:** ~+50% on v6e, modest improvement on v5e. Few hours of work.

### C. Use the AQT int4 pipeline the parallel session built
`benchmarks/api_server/layerwise_quantize_minimax_m2_npy.py` +
`assemble_int4_orbax.py` + `serve_minimax_m2_from_pickles.py`. This
loads pre-quantized int4 weights and lets MaxEngine serve them without
the qwix Pallas issue. On v5e it ~halves the weight bandwidth again
(int8 → int4); on v6e it would further compound.
**Outcome:** ~+50-100% more on v5e and v6e. The pipeline is staged
but un-validated end-to-end.

### D. Re-shard with `tensor=4 expert=16` (or `tensor=2 expert=32`)
The dense-MoE compute is shared across the `expert` axis. More
expert-parallelism means fewer experts per chip → less compute per
step. Requires re-converting the shards (~30 min on v5e). Trade-off:
larger per-chip slice of attention/embedding weights.
**Outcome:** speculative; might be 1.3-2× on v5e if expert compute
truly dominates. Worth one experiment.

### E. CPU-side expert routing + selective weight streaming
For inference, the gating decision (which 8 experts each token goes to)
is cheap to compute on CPU. Pre-compute it per layer, then have the
TPU only load the 8 selected experts' weights per token via a custom
all-gather across the expert axis. Bypasses the missing
ragged-all-to-all entirely. Significant code work — essentially a
custom MoE kernel — but it's the only path that recovers v5e's full
hardware on sparse routing.
**Outcome:** could recover most of the 27× sparse advantage on v5e,
landing v5e somewhere in the 100-500 tok/s range. **Multiple weeks of
work.**

### F. Switch to a model whose architecture fits v5e
If v5e is the only available hardware, run a model that doesn't need
ragged-all-to-all: a dense LLM (Llama, Mistral, Qwen-dense), or a
DenseMoE variant where every expert receives every token. These will
be much faster on v5e than MiniMax-M2 sparse. For the *MiniMax-M2*
family specifically, this isn't an option without retraining.
**Outcome:** strong perf on v5e but different model.

### G. Wait for `v5p` / future TPU with ICI routing
TPU `v5p` already supports ragged-all-to-all on its OPI. If the v5p in
your project has capacity in any zone, it would close most of the gap.
**Outcome:** ~v6e-class throughput; requires moving the workload.

### H. Tighten `capacity_factor` further
Setting `capacity_factor=1.25` or `1.5` (instead of 2.0) reduces the
padding around expert computation. Risks dropping some tokens (output
quality slightly degrades), but gets you ~10-30% more pod-wide tok/s
on v5e at the same batch. Already supported in `decode_distributed.sh`.
**Outcome:** incremental ~+20% on v5e. Worth trying with a quality
check on the output.

### I. AOT-compile a single fixed shape (`per_device_batch=8`,
`ctx=256`) and embed it in a serving binary
Reduces the per-launch ~31 s engine-init + JIT overhead to <5 s. Pure
serving-side win; doesn't change steady-state tok/s.
**Outcome:** lowers cold-start cost dramatically. Useful for
production.

## 4. Recommendation

If the constraint is "**must use v5e-64**" and the model is fixed:
- (H) tighten capacity_factor + (B) int8 weight quant + (C) AQT int4
  ≈ best you'll do on v5e short of writing custom kernels. Expect
  **20-40 tok/s pod-wide**, not 7,000.

If the constraint is "**maximum tok/s, any TPU in our project**":
- Use v6e-64. Already at 6,678 tok/s. Tighten with (B) int8 weight
  quant on v6e for another ~1.5×.

If the constraint is "**run sparse MoE fast on cheap hardware**" long-term:
- Wait for v5p/v6p capacity, or invest in (E) custom routing kernel.

## 5. Open issues this analysis surfaces

- `bench_steps_minimax_m2_npy.py` silent-kill on v5e — root cause not
  identified.
- qwix Pallas `bv=1024` tile vs `v=192` chip dim — tile config knobs
  must be threaded through `bench_steps.sh` / `decode_distributed.sh`.
- AQT int4 pipeline at `benchmarks/api_server/*` is WIP (per parallel
  session) — landing it would close one of the open levers.
- ICI mesh swap (e.g. `tensor=4 expert=16`) requires re-converting
  shards; the converter's mesh is baked into the `.npy` layout.
