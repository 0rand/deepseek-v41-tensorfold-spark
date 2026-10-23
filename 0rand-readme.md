# Fork notes: native Spark serving, lifecycle and telemetry

This document records what this fork adds on top of
[upstream PR65](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/65), commit
`207273b261196ee491b06e5818031ff54f91fcee`, and the evidence gathered while bringing it up on a
reference pair of DGX Sparks. It is written for operators who want the same setup, not as a new
performance claim. Upstream measurements remain the source for speed and task results; the numbers
here are functional checks unless a table says otherwise.

- Fork: <https://github.com/0rand/deepseek-v41-tensorfold-spark>
- Branch: `pr65-flash17-serving`
- Base: PR65 head `207273b261196ee491b06e5818031ff54f91fcee`
- License: Apache-2.0, with upstream credits and notices preserved

## What this fork adds

| Area | Change | Where |
| --- | --- | --- |
| Two-rank lifecycle | Detached, restartable ranks with ordered start/stop/restart, status and logs | `scripts/zig-stack.sh` |
| Launcher | `--detach`, configurable model alias, `--no-drafts` when drafting is off | `scripts/run-zig.sh` |
| Autostart | Docker `unless-stopped` for managed ranks; intentional stop stays stopped | `scripts/zig-stack.sh`, `docs/ZIG-SERVE.md` |
| EngineMetrics | Native `tensorfold:*` counters, histograms and live gauges on `/metrics`; richer `/health` | `engine/zig/zig/src/server/{metrics,status_routes,spark}.zig` |
| Dashboard adapter | Retains the stream-KV and running/waiting gauges the display reads | `tools/enginemetrics-gauges.patch` |
| SSD sessions | Immediate write-through saves, RAM copy retained, common-prefix recovery at startup | `engine/zig/zig/src/sessions/*`, `families/deepseek_v41/*` |
| Profile | flash17 example with four slots, 1M context, 4K prefill and a 32 GiB SSD tier | `config/prod-zig-flash17.env.example` |
| Documentation | Lifecycle, health/metrics semantics and the generic example | `docs/ZIG-SERVE.md`, `docs/API.md`, `docs/CHANGELOG.md` |

The engine source is still the PR65 tree. The changes are serving, telemetry, session persistence and
launcher work; there is no arithmetic, quantization or kernel rewrite in this fork.

## Deployment profile

The example profile is the one exercised on the reference pair:

| Setting | Value |
| --- | --- |
| API | loopback, port `8100` |
| Model alias | `deepseek-v41-flash` |
| Context / slots | 1,048,576 tokens / four slots |
| Shared KV pool | 2,100,000 configured positions (2,100,224 rounded) |
| Prefill | 4,096-row pieces, 4K workspace, 2K fallback |
| Memory floors | 1 GiB target / 1 GiB hard, measured boot rejection off |
| Sessions | 256 MiB RAM tier, 32 GiB logical SSD tier per node, write-through |
| Transport | TP=2 over RoCE on the CX7 link |

The 1 GiB floors are deliberately less conservative than the upstream 5/4 GiB defaults. They are an
operator choice for this pair, not a universal safety guarantee. Live admission, graph and workspace
checks remain enabled. The pool is shared: four slots do not promise four simultaneous 1M-token
requests.

The generic example uses documentation addresses (`192.0.2.11` / `192.0.2.12`), SSH user `operator`
and `/srv/tensorfold` paths. Replace every one of those with your own values on both nodes. The
example is a template, not a deployment identity.

## Two-rank lifecycle

Run these from the head checkout. The trusted local config supplies `WORKER_SSH`, `WORKER_REPO` and
`WORKER_CONFIG`; the stack script never guesses a remote path.

```bash
scripts/zig-stack.sh start      # worker first, then head; detached
scripts/zig-stack.sh status     # container state, restart policy, /v1/models
scripts/zig-stack.sh logs       # both ranks; logs 0 or logs 1 for one
scripts/zig-stack.sh restart    # coordinated stop, then recreate both
scripts/zig-stack.sh stop       # intentional stop; stays stopped
```

`start` is idempotent when both containers are already running. Otherwise it creates rank 1 over
BatchMode SSH first and rank 0 locally, using `docker run --restart unless-stopped` without `--rm`.
A successful `start` means both containers were created, **not** that the model is usable. Model
loading and first preparation continue in the background; wait for `status` to return a successful
`/v1/models` response before sending traffic.

`stop` disables both restart policies, sends SIGTERM to rank 0 and lets it drain for
`TF_DSV41_DRAIN_S` (default 20 seconds), then stops any remaining follower. It deliberately leaves
restart disabled. A follower can exit naturally while the head drains, and Docker does not record
that as a manual stop; restoring `unless-stopped` immediately would let it come back on a daemon
restart. The next `start` recreates both containers with `unless-stopped`. `ZIG_STOP_TIMEOUT`
defaults to drain plus 60 seconds and may not be lower.

`restart` is the migration path for the original foreground `--rm` ranks: it drains and removes them,
then creates the managed pair. Do not use `docker restart` on one rank; the ranks must be recreated
together because the restore message format and session accounting are pair-wide contracts.

`status` reports container state and restart policy, then queries `/v1/models`. It distinguishes an
unavailable endpoint from a running container. `logs` tails 120 lines by default; set `TAIL=200` for
more.

No systemd unit or watchdog is required for boot autostart, but the Docker daemon must be enabled on
both nodes. A host reboot and a Docker-daemon restart were not exercised on the reference pair.

### Launcher modes

`scripts/run-zig.sh` keeps its original foreground `--rm` mode for direct debugging. `--detach`
creates one managed local rank; use it only through `zig-stack.sh` when you need the ordered pair.
`--dry-run` prints the exact Docker command, and `--check-aot` enumerates the Triton variants the
serving environment needs against the rank's `aot/` directory.

The launcher now passes `--no-drafts` when `TF_DSV41_DRAFTS=0`. Before that change, a plain-decoding
request returned HTTP 500 `NoDrafter` even though the family had drafting disabled. The same request
returned `391` after the fix and a coordinated restart.

## Health and metrics

The Spark serving mode exposes two families on `/metrics`:

- the existing `tensorfold_*` health counters;
- native `tensorfold:*` Prometheus counters, histograms and live gauges.

EngineMetrics reads the native family together with `/health` JSON. The health payload includes
`pool_tokens`, `pool_free_tokens`, `pool_used_tokens`, `pool_kept_tokens`, `kept_prompts`,
`kept_bytes`, `streams`, `context_length` and optional `spill` data.

Semantics worth knowing before building a dashboard:

- Paged-pool occupancy counts allocated page positions. Retained valid positions are deduplicated
  across entries that share pages, so `pool_used_tokens` and `pool_kept_tokens` are not additive.
- `tensorfold:kv_cache_usage_ratio{pool="0"}` is one maximum completed-position/context ratio, not a
  pool ratio. It is zero when idle even if retained pages still occupy the pool.
- KV byte counters cover this rank's compressed KV and indexer buffers only. They exclude weights,
  SWA rings, RoPE tables and workspaces.
- `kept_bytes` counts RAM snapshot buffers outside the pool. Spill bytes are local file sizes; the
  configured disk budget uses the tier's accounting charge.
- Snapshots are published at engine-thread boundaries. HTTP scrapes do not inspect mutable session
  collections or run CUDA operations.

The separate EngineMetrics checkout discarded three gauges its own display read: stream KV ratio and
running/waiting request counts. `tools/enginemetrics-gauges.patch` retains them and renames the
counter list to reflect that it now holds gauges too. Apply it from that checkout:

```bash
git apply --check /path/to/this-repo/tools/enginemetrics-gauges.patch
git apply /path/to/this-repo/tools/enginemetrics-gauges.patch
```

An already-fixed adapter does not need the patch. Run the dashboard against the configured port:

```bash
python3 engine_metrics.py --host 127.0.0.1 --port 8100 --engine tensorfold
```

## Immediate SSD sessions

The native tier previously wrote only on eviction and did not rebuild its prefix index at startup.
This fork adds:

- `TF_DSV41_SESSION_DISK_WRITE_THROUGH=1`: every eligible store save is committed without dropping
  its RAM copy.
- Startup scans and intersects recoverable files across ranks, then assigns deterministic entry IDs
  independently of file timestamps.
- A 32 GiB accounting budget per node in the example. Reported stored bytes are actual local file
  sizes; the budget is logical, not a hard physical ceiling.

Eligibility is unchanged: CED snapshots are on a 16-token grid with a minimum saved prefix of 64
tokens. `TF_DSV41_SESSION_DISK_MIN=0` removes the storage-tier minimum; it does not make arbitrary
tiny turns cacheable. There is no RAM or pool high-water requirement before writing an eligible
state.

Restore is on demand when a prompt matches a saved prefix. It does not remove model-weight loading
time and does not preload the tier. Session directories are private to the flash17 pack; the native
layout/rank checks are not a complete checkpoint identity.

Observed on the reference pair, as functional canaries rather than benchmarks:

| Check | Observation |
| --- | --- |
| Immediate write | An eligible prompt created a 20,762,624-byte rank-0 file while only 7,424 of 2,100,224 pool positions were allocated (about 0.35%); no pressure trigger was needed. |
| RAM reuse | Repeating the request reported 7,360 cached tokens. |
| Full restart | Both ranks stopped and restarted; each logged recovery of one common NVMe entry. Initial RAM-kept count was zero. |
| SSD restore | The request again reported 7,360 cached tokens; one load, 7,360 loaded tokens, 57.5 ms tier restore, 0.5015 s whole request. |

The disk budget can overshoot physically with many small entries; upstream reported roughly 30%.
Reserve headroom and measure actual usage separately from the logical charge.

## 4K prefill and memory

The example enables `TF_DSV41_PF_4K=1`, `TF_DSV41_PF_4K_MEMORY_OK=1`,
`TF_DSV41_PREFILL_CHUNK=4096`, `TF_DSV41_PREFILL_ROWS=4096` and
`TF_DSV41_PF_WS_FLOOR_GIB=1.0`.

On the reference pair both ranks allocated and released their 2,268 MiB 4K workspace, the scheduler
logged prompt pieces of 4,096 rows, and no workspace-fallback warning appeared in the saved runs.
When either rank cannot keep the explicit 1 GiB workspace reserve, the engine falls back to 2K
segments with the same bits. For a conservative 2K profile, set `PF_4K=0`, drop
`PF_4K_MEMORY_OK` and `PF_WS_FLOOR_GIB`, and use 2,048 for chunk and rows.

Changing prefill shapes, slots, rows or context changes the AOT set. Run
`scripts/run-zig.sh N --check-aot` on both ranks and compile what it reports before serving.

The first full-profile boot reached the end of model initialization and then exited with
`PoolDoesNotFit`: the guard measured 5.80 GiB `MemAvailable`, subtracted a 3.00 GiB historical
serving dip, and rejected the predicted 2.80 GiB remainder against a 4 GiB hard floor. That was a
predictive guard decision, not an observed OOM kill. The example profile now uses 1 GiB floors and
disables measured boot rejection; live admission and workspace checks remain.

## Validation performed

### Host and image checks

- Focused host suites: 155 passed, 2 skipped. Real-file regressions cover immediate saves with RAM
  retention, byte-exact reopen, shared-page accounting and stable common recovery despite opposite
  write orders and one-sided files.
- HTTP parity corpus: 312/319 equal, seven pre-existing known differences, zero unexpected
  differences.
- Runtime image library proof passed, including pinned cuBLAS/cuBLASLt and the mlx5 provider.
- The image is a locally patched build on the PR65 base, not an unmodified upstream artifact.

### flash17 bring-up

Ten real inference requests passed on the reference pair:

| Path | Result |
| --- | --- |
| Plain generation with drafting | `17 * 23` returned `391`; draft tokens were generated and accepted |
| Reasoning separation | Final content `156`; reasoning appeared separately in `reasoning_content` and `reasoning` |
| Forced function call | `get_weather` with `{"city":"Paris","units":"celsius"}` and finish reason `tool_calls`; no external tool was executed |
| Native vision | A generated solid-red PNG was identified as `Red` |
| Four concurrent requests | All four returned their own answers: `391`, `156`, `133`, `253` |
| Chunked long prefill | A 6,327-token prompt recovered the exact early marker `SPARK-PR65` |
| SSE streaming | Final text `209`, finish reason `stop`, terminal `[DONE]` |

After those requests, `/health` returned HTTP 200 with 10 requests, zero errors, four supported
streams and a 1,048,576-token configured context. It recorded 119 drafted and 81 accepted tokens,
484 row-graph replays, 27 draft-graph replays and no graph-floor hits. These are functional checks,
not throughput or quality benchmarks.

### Asset and model preparation

- The flash17 `text_config` matches the metadata used to generate the existing full asset set.
- The tokenizer and tokenizer-config SHA-256 values match the published manifest, so tokenizer-derived
  Engram host data and CPU-generated RoPE inputs are unchanged.
- All 150 cubin files were compared by SHA-256 against the verified set; AOT manifest, kernel
  signatures and Engram host data match between ranks.
- RoPE covers 1,050,624 rows and 64 columns.
- The completed download passed a 47-file presence check, 42 indexed shard size checks, F32 `[128]`
  MTP router biases, a BF16 `[1, 5376]` confidence head, and five metadata SHA-256 checks. Weight
  payloads were not reread for full checksums.
- The first boot loaded 95.1 GiB and built/wrote 44 pack-specific prepared files; the weight-load
  stage took 401.6 seconds. A later restart reused the prepared files and became ready in about
  173.5 seconds, including 14.6 seconds of draft-cost calibration.

### Mia compatibility trial

The existing Mia EAQ-v1 2.9 bpw weights and existing 1017 Engram data produced the correct answer on
PR65 Zig with drafting disabled. No replacement weights or Engram data were downloaded.

The full-profile attempt failed before model upload:

```text
mtp.0.ffn.gate.bias: f16 {128}, expected f32 storage of {128}
rank 0 failed in boot: UnexpectedTensor
```

The Mia pack's three MTP router biases and its confidence head are FP16; those tensors are not loaded
in the no-drafts smoke. No loader fix or replacement image was built. Development stopped in favor of
the supported flash17 pack. The compiled asset set remains usable because its context, feature
settings and architecture match the flash17 profile.

## Experiment cleanup

Obsolete Zig/Mia experiment caches, old rig assets, unused build exports and unreferenced Python
display-KV experiment images were removed from the reference pair. No broad Docker image/builder
prune and no forced shared-image removal was used. The cleanup reclaimed roughly 100 GB per node
before concurrent serving writes; the later net gain was about 98 GB per node.

Preserved: model downloads, packed rank-specific Engram shards, the complete working flash17
assets/prepared/state/session/cache directories, the serving image, the latest native builder, the
original Python stack and its cache. Image nominal sizes were not added to filesystem gains or
credited when their layers remained shared.

The old native base tag was reconstructed after its image config was removed. Its ten retained shared
RootFS layers and filesystem size are identical, but the reconstructed image config ID is new. This
keeps the local telemetry Dockerfile's base reference usable without claiming an identical image
config. The active serving image and containers were not changed.

## Publication and privacy

Only reviewed source, script, example and documentation files are committed. The following are
excluded from the public repository:

- local serving configuration and environment files;
- model weights, Engram shards, prepared files, calibration and session data;
- build products, caches, run logs and local benchmark databases;
- the private assessment report and its receipts.

The generic example uses documentation addresses and placeholder paths. The published tree scans clean with
`scripts/check-public.sh`; local configuration, run logs and benchmark databases are untracked and excluded.
The fork is public and its default branch is `pr65-flash17-serving`.

## Limits

Not exercised or not qualified by this fork:

- No A/B benchmark against the Python stack or another quantization.
- No 1M-context request, sustained eviction stress, crash injection or full 32 GiB tier fill.
- No parser transplantation or tool-policy equivalence proof; the Python hardening overlay was not
  ported mechanically.
- No host reboot or Docker-daemon restart; managed stop/start after cutover was deferred on the
  reference pair.
- PR65 does not fix the separate mid-prefill `PoolExhausted` trigger (issue #55) and does not add a
  universal timeout for independent CUDA/transport deadlocks.
- The disk budget is logical, not a hard physical ceiling.
- Engram reads used the native-AIO fallback because `io_uring` returned `PermissionDenied` on the
  reference pair.
- The Mia full profile remains unqualified on this engine.

Treat the functional checks above as evidence that the paths work, not as a production soak. Read
[what is not solved](docs/KNOWN-ISSUES.md) and the upstream limits before relying on the stack.

## Reproduction checklist

1. Build the image from this branch on an ARM64 host, run `scripts/check-zig-image.sh`, and transfer
   the identical artifact to both nodes.
2. Pack the 1017 Engram shards with `scripts/pack_engram.py`; the September shards do not match this
   pack.
3. Generate `rope.bin`, `engram-host.bin` and the complete Triton AOT set for the serving settings.
4. Copy `config/prod-zig-flash17.env.example` to `config/prod-zig.env` on both nodes and replace the
   documentation addresses, SSH user, paths and NIC/HCA names.
5. Run `--check-aot` on both ranks, then `scripts/zig-stack.sh start`.
6. Wait for `scripts/zig-stack.sh status` to return `/v1/models`, then check that `17 * 23` returns
   `391` before sending traffic.
7. Use `scripts/zig-stack.sh restart` for configuration or image changes; never restart one rank
   alone.

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart, and on
[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)'s model, papers and code.
The flash17 pack is by [jayleaton](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw);
EXL3 is by [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3). Upstream credits,
notices and licensing terms are preserved in [CREDITS](docs/CREDITS.md) and
[LICENSING](docs/LICENSING.md).
