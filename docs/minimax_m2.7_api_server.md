# MiniMax-M2.7 inference API server on TPU v6e-64

Long-running HTTP server that exposes the v6e-64 distributed-NPY MiniMax-M2.7
engine via OpenAI-compatible (`/v1/completions`, `/v1/chat/completions`) and
Anthropic-compatible (`/v1/messages`) endpoints. The server is reachable only
through a Google IAP TCP tunnel — no public IP, no firewall rule opened on
the TPU pod. Bearer-token middleware is defense-in-depth on top of IAM.

The same Python process runs on every one of the 16 hosts. Rank 0 binds
uvicorn on 0.0.0.0:8000; ranks 1..15 block in the broadcast-driven batching
loop. Each rank loads only its own weight shard from local dtmpfs.

## Architecture

```
Local laptop / Claude Code
        │
        │  http://localhost:8000  (OpenAI / Anthropic clients)
        ▼
gcloud compute start-iap-tunnel  ──── (the only ingress; no firewall opening)
        │
        ▼
TPU pod v6e-64 (worker 0)
  uvicorn :8000
   ├── /v1/completions, /v1/chat/completions, /v1/models  (OpenAI; bearer-token)
   ├── /v1/messages, /v1/messages/health                  (Anthropic; bearer-token)
   └── /health, /ready, /metrics, /                       (unauthenticated)
       │
       │  request → request_queue → multihost_utils.broadcast_one_to_all
       ▼
all 16 workers
   ├── engine.prefill(...) in lockstep
   ├── engine.generate(...) in lockstep
   └── rank 0 returns response
```

## One-time pod bootstrap

The pod needs the repo, venv, dtmpfs mount, MiniMax-M2.7 weights converted to
distributed `.npy` shards, and the tokenizer. If the pod was re-provisioned,
re-run the bootstrap before starting the server. See
[`scripts/minimax_m2.7/bootstrap_tpu.sh`](../scripts/minimax_m2.7/bootstrap_tpu.sh)
and the v5e64 runbook for the conversion pipeline.

State check (worker 0):

```bash
gcloud compute tpus tpu-vm ssh node-v6e-64-europe-west4-a --zone=europe-west4-a --worker=0 \
  --command='mountpoint -q /mnt/dtmpfs && echo dtmpfs-ok; \
             ls /mnt/dtmpfs/minimax-m2.7-npy-distributed/manifest.p*.json 2>/dev/null | wc -l; \
             ls /mnt/dtmpfs/minimax-m2.7-hf/tokenizer.json 2>/dev/null && echo tok-ok'
```

You need: `dtmpfs-ok`, `1` (at least one manifest), `tok-ok`. Repeat across
all 16 workers (`--worker=all`). If any check fails, run the bootstrap.

## Bring-up (Phase 1: SSH + nohup, recommended for first run)

From the dev VM, NOT from inside the TPU pod:

```bash
cd ~/maxtext/.claude/worktrees/minimax-m2.7

# Optional: protect the server with a bearer token. If unset, the auth
# middleware is a no-op and IAM (via IAP) is the only boundary.
export MAXTEXT_API_KEY="$(openssl rand -hex 32)"
echo "api key: $MAXTEXT_API_KEY  (save this; clients need it)"

# Fan out the server onto every worker, then start an IAP tunnel and
# poll /ready until the server is up. First cold start ~10-15 min.
bash scripts/minimax_m2.7/launch_serve_v6e_64.sh
```

The launcher logs `READY at <timestamp>` once `/ready` returns 200. From
that point the server accepts requests at `http://localhost:8000` on the
dev VM (via the IAP tunnel).

Server logs land at `~/serve_v6e_64.log` on each worker. Tail rank 0:

```bash
gcloud compute tpus tpu-vm ssh node-v6e-64-europe-west4-a --zone=europe-west4-a --worker=0 \
  --command='tail -f ~/serve_v6e_64.log'
```

## Bring-up (Phase 5: systemd, recommended once config is stable)

Install the per-host systemd unit on all 16 workers (one-time):

```bash
bash scripts/minimax_m2.7/install_systemd_v6e64.sh
```

After install, the unit auto-starts on boot, restarts on failure, and is
health-gated on `/ready`. Drop-in for the bearer token:

```bash
gcloud compute tpus tpu-vm ssh node-v6e-64-europe-west4-a --zone=europe-west4-a --worker=all \
  --command='sudo mkdir -p /etc/systemd/system/minimax-m2-serve-v6e64.service.d && \
    echo -e "[Service]\nEnvironment=MAXTEXT_API_KEY=YOUR_KEY_HERE" | \
    sudo tee /etc/systemd/system/minimax-m2-serve-v6e64.service.d/auth.conf && \
    sudo systemctl daemon-reload && sudo systemctl restart minimax-m2-serve-v6e64'
```

To redeploy code (after a git push to the branch):

```bash
bash scripts/minimax_m2.7/reload_serve_v6e64.sh
```

## Client-side: IAP tunnel

The tunnel runs on whatever machine wants to talk to the server (laptop,
dev VM). It binds `localhost:8000` → worker-0:8000 via IAP. Auth = your
GCP IAM identity.

```bash
bash scripts/minimax_m2.7/iap_tunnel.sh
# leaves the foreground; Ctrl-C to stop.
# In another shell: http://localhost:8000 reaches the pod.
```

## Smoke tests

All commands assume the IAP tunnel is up and `MAXTEXT_API_KEY` is set
(or that the server runs without a key).

```bash
# Liveness / readiness / metrics (no auth)
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
curl -fsS http://localhost:8000/metrics | jq

# Models list
curl -fsS http://localhost:8000/v1/models \
  -H "Authorization: Bearer $MAXTEXT_API_KEY" | jq

# OpenAI completion
curl -fsS http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $MAXTEXT_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.7","prompt":"The capital of France is","max_tokens":32}' | jq

# OpenAI chat
curl -fsS http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $MAXTEXT_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.7","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}' | jq

# Anthropic /v1/messages (the Claude Code shape)
curl -fsS http://localhost:8000/v1/messages \
  -H "x-api-key: $MAXTEXT_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.7","max_tokens":32,"messages":[{"role":"user","content":"Hi"}]}' | jq
```

### OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.environ["MAXTEXT_API_KEY"],
)
resp = client.chat.completions.create(
    model="minimax-m2.7",
    messages=[{"role": "user", "content": "Hi"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

### Anthropic Python SDK

```python
from anthropic import Anthropic
client = Anthropic(
    base_url="http://localhost:8000",
    api_key=os.environ["MAXTEXT_API_KEY"],
)
resp = client.messages.create(
    model="minimax-m2.7",
    max_tokens=64,
    messages=[{"role": "user", "content": "Hi"}],
)
print(resp.content[0].text)
```

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=$MAXTEXT_API_KEY
claude -p "What is 2+2?"
```

## Tuning matrix (Phase 4 — to be populated against the live pod)

The throughput benchmark cell (`batch=96 / ctx=128`) maximizes pod-wide
tok/s but starves single-turn latency. For serving, we sweep three
configs and record TTFB p50/p95 + tok/s at concurrencies 1/4/8/16.

| Variant | `per_device_batch` | `max_target_length` | KV quant | TTFB p50 | TTFB p95 | tok/s/stream | tok/s aggregate |
|---|---|---|---|---|---|---|---|
| A: latency | 4 | 4096 | int4 | TBD | TBD | TBD | TBD |
| B: balanced | 8 | 8192 | int4 | TBD | TBD | TBD | TBD |
| C: throughput | 24 | 4096 | int4 | TBD | TBD | TBD | TBD |

The default baked into `serve_v6e_64.sh` is currently variant **B** (the
plan's starting estimate). Adjust after the sweep lands.

## Operations

### Logs
- Per-worker server log: `~/serve_v6e_64.log` on each TPU host.
- systemd: `sudo journalctl -u minimax-m2-serve-v6e64 -f`.
- IAP tunnel client log: `~/iap_tunnel.log` on the dev VM / laptop.

### Restart
- Phase 1 (nohup): re-run `launch_serve_v6e_64.sh`.
- Phase 5 (systemd): `reload_serve_v6e64.sh` (also does a `git pull`),
  or `gcloud ... ssh --worker=all --command='sudo systemctl restart minimax-m2-serve-v6e64'`.

### Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `/ready` never flips to 200 after 30 min | first JIT compile slower than expected, or weight load hung | Tail worker-0 log; if compile is in progress (look for `xla.cpu` / `MaxEngine`), keep waiting (JAX cache will speed restarts). If hung, restart. |
| HBM OOM at first inference | batch × ctx × KV size > 31 GB/chip headroom | Drop `BATCH` or `MAX_TARGET_LENGTH` in `serve_v6e_64.sh` and re-launch. |
| `401 missing bearer token` | client didn't send `Authorization: Bearer ...` | Set `MAXTEXT_API_KEY` on the client or send the header. |
| `403 invalid bearer token` | token mismatch | Check the server's `$MAXTEXT_API_KEY` (or systemd drop-in). |
| `504 Request timed out` | server queue stuck waiting for broadcast | A worker may have died. Check status across all 16; restart the unit. |
| `connection refused` on localhost:8000 | IAP tunnel down | Re-run `iap_tunnel.sh`. |
| `stream=true is not supported` (Anthropic 400) | client requested SSE streaming | Use `stream=false` for now; SSE is a v2 follow-up. |
| Claude Code reports `tool_use` errors | client tried to use tools | v1 adapter rejects tool blocks with 400. Disable tools in the Claude Code session for now. |

### TPU preemption
Per the project recovery policy: if the pod is preempted, wait+poll for
the slice to return; don't switch zones. Systemd will resume once the
pod is back. If the bootstrap state was lost, re-run the bootstrap
before launching the server.

## What's not implemented (yet)

- **Streaming (SSE)** for either OpenAI `stream:true` or Anthropic
  `stream:true`. The current batching loop is non-streaming. v2 follow-up.
- **Tool use** (`tools`, `tool_choice`, `tool_use` content blocks). Rejected
  with 400.
- **Image inputs** (OpenAI vision, Anthropic image blocks). Rejected with 400.
- **Multi-model serving**. Exactly one model is hosted per server process.

## See also

- [`minimax_m2.7.md`](minimax_m2.7.md) — the project overview, weight
  conversion, dtmpfs bootstrap, and per-host throughput numbers.
- [`minimax_m2.7_v5e64_runbook.md`](minimax_m2.7_v5e64_runbook.md) —
  v5e-specific operational notes (analogous pod, different MoE constraints).
