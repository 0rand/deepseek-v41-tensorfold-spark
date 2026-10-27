# Operations

## What a start does (`scripts/serve.sh start`)

1. Takes a lock (one start at a time; the watchdog stands down while it is held) and clears the stop marker.
2. Refuses if a CUDA process runs on either node.
3. Preflight (`PREFLIGHT=strict`): ssh to the worker; the image on both nodes with equal content keys (creation time
   + layer digests); `config.json` / tokenizer in both model folders; both Engram shards on each node; a prepared
   folder (warning only); the RoCE ports ACTIVE; nothing on `PORT` / `MASTER_PORT`; the `/cache/roce-failed` marker.
4. Drops the page cache on both nodes and waits until MemFree >= `MEM_GATE_GIB` (104) on both (`MEM_GATE_TIMEOUT`).
   On GB10 the page cache is GPU memory: a load next to a full page cache can start with fewer request slots.
5. Starts rank 1 on the worker, then rank 0 here (`python -m tensorfold serve /model --tp 2 ...`), and waits for
   `/v1/models` (`READY_TIMEOUT`).
6. Checks rank 0's `serving: N slot(s)` line against `PARALLEL` (one retry after a cache drop if short).
7. Drops the caches again, runs `scripts/canary.py` (a greedy chat, a thinking reply with separated reasoning, a forced
   tool call, a JSON-schema reply, `/tokenize`), drops the caches once more. `CANARY=strict` stops both ranks on a
   failed canary and saves their last logs in `STATE_DIR`.

`scripts/serve.sh args 0|1` prints the exact `docker run` arguments without starting anything.

A start does not build the CUDA extensions on purpose: `scripts/serve.sh build` ends with `scripts/serve.sh prebuild`,
which compiles them into `CACHE_VOL` on both nodes with no weights loaded (a build next to ~100 GB of weights once
took the worker to 1.26 GiB MemAvailable). Run `scripts/serve.sh prebuild` yourself after `PREBUILD=0 scripts/serve.sh
build`, after deleting the cache volume, or before turning on a lever whose extension was never built. A start that
finds an extension missing builds it at load time, slowly and beside the weights.

## Knobs

Every `TF_DSV41_*`, `GLM53_TF_*`, `MALLOC_*` and `MIMALLOC_*` line of the config reaches both ranks. A non-empty caller export wins
over the file: `CONTEXT=196608 scripts/serve.sh restart`.

| knob | production | off / fallback |
| --- | --- | --- |
| `TF_DSV41_PREFILL` | `replay` (~2x the kit) | `full`: the exact prefill (with `FULL_CONE=1`: 1.45-2.09x the kit) |
| `TF_DSV41_PREFILL_CHUNK` / `_ROWS` | 2048 / 2048 | 1024 / 1024 (G16-G18 production: replay 32K 1,996 vs 2,310) |
| `TF_DSV41_PREFILL_ADAPT` / `_ADAPT_GIB` | 1 (default) / 4.5: rank 0 steps each round's rows down (2,048 / 1,024 / 512, never below `_MIN` 512) when the tighter rank would keep less than this after the round's priced transient; back up with `_UP_GIB` (0.5) more | `_ADAPT=0`: fixed rows. The engine default `_ADAPT_GIB` (5) flaps on GB10 |
| `TF_DSV41_PF_COPIES`, `TF_DSV41_MHC_PF` | 1, 1 (G17, same bits) | 0 |
| `TF_DSV41_PF_DENSE` / `_TABLE` | `fused` / `/cache/pfdense-table.json` (G17, same bits; `scripts/serve.sh cache` puts `config/pfdense-table.json` there) | `off`. The table file must exist in the cache volume (a missing file fails the prefill); unset `_TABLE` for the engine's built-in tiles (+1.5-3.4% instead of +5.4-6.9%) |
| `TF_DSV41_FULL_CONE` | 1: `full` prefill runs the decoder only over its dependency cone (same bits as plain `full`) | 0 |
| `TF_DSV41_INDEX_BUDGET_MIB` / `_STREAM_MIN` | 64 / 4096 (G16 memory cap, exact) | 256 / 16384 (+1.6 GiB of cached allocator segments in long prefills) |
| `TF_DSV41_POOL_TOKENS` | 614400: the 4 slots share 614K tokens of KV (a request waits for pages, never refused) | unset: 4 x CONTEXT (+0.6 GiB a rank) |
| `MALLOC_ARENA_MAX` | 4 | unset: glibc's default arenas |
| `TF_DSV41_IMAGES` | `native` (+ `TF_DSV41_BIAS_VL`, `_VISION_PREP_MB=64`) | `placeholder` (the engine default: a text notice per image) or `reject` (HTTP 400) |
| `TF_DSV41_PREFILL_KERNELS` | `fast` | `exact` |
| `TF_DSV41_FAST_EXPERTS` | `gm` | `grouped`; `tc` is 3-5x slower |
| `TF_DSV41_EXPERT_TOPP` / `_MIN_K` / `_RENORM` | 0.85 / 3 / kept (lossy, +5%) | delete the three lines: the unpruned model |
| `TF_DSV41_MHC_FN` | `bf16` | unset: fp32 mixing weights (prose -2.4%) |
| `TF_DSV41_L2PF` / `_MB` | 1 / 12 | 0 |
| `TF_DSV41_MHC_CUDA` | 1: an mHC boundary of a decode window as one CUDA launch (G13, same bits) | 0 (the engine's default): the Triton boundary |
| `TF_DSV41_ATTN_CUDA` | 1: CSA2's decode attention core and the indexer's top-k in CUDA (G13, same bits) | 0: Triton |
| `TF_DSV41_DENSE_V3` | 1: dense EXL3 linears over a 16-byte-coalesced repack of the trellis (G13, same bits) | 0: x3seg |
| `TF_DSV41_PLAN_LINK` / `_PLAN_PIN` | `rdma` / `auto` (G14: the round plan through the RoCE host mailbox) | `tcp` (MASTER_PORT + 7) |
| `TF_DSV41_BRANCHES` / `_PRIO` | 1 / `side` (G14, same bits) | 0 |
| `TF_DSV41_L2PF_PACE_GBPS` | 150 (G14, same bits) | 0: unpaced |
| `GLM53_TF_ROCE_FAST` | 1 (G14, the same bytes) | 0 |
| `TF_DSV41_PDL` | 0: must stay 0 with `DENSE_V3=1` (not bit-exact together; refused at boot) | |
| `TF_DSV41_GRAPH_MODE` | `rows` | `mix`: C2 / C4 slower |
| `TF_DSV41_ROUTER` | `gemv` | `fused` / `split` |
| `GLM53_TF_COMM_BACKEND` | `roce` | `nccl` (-11% / -17%). A RoCE failure at run time writes `/cache/roce-failed` in the cache volume and the next start of both ranks uses NCCL; delete it to retry RoCE |
| `CONTEXT` | 300000 (4 x 300K) | 196608 keeps both ranks further from their memory floor |
| `TF_DSV41_THINKING` / `_DEFAULT_EFFORT` | 1 / `high` (75) | requests override |
| `TF_DSV41_GRAMMAR` | 1 | 0: `response_format` and strict tools ignored |
| `MIMALLOC_ALLOW_THP` | 0: no transparent huge pages in torch's CPU allocator (an embedded mimalloc), read at process start | unset: mimalloc's default (on); long prefills then stall in synchronous memory compaction (G12) |

Knobs left at their defaults that you may meet in the code: `TF_DSV41_DEPTH` / `TF_DSV41_DRAFT_DEPTH` (draft depth:
the default policy reads the boot calibration and caps at 5; `static` + 3 is the kit's), `TF_DSV41_DRAFT_HEAD` (full),
`TF_DSV41_VERIFY_BUDGET` (unset; lossy, do not use), `TF_DSV41_TREE_PC` (0), `TF_DSV41_DSPARK_DELTA` (unset).
From G13: `TF_DSV41_MOE_FUSED` (0: the shortened decode MoE chain, exact but +0.7 ms on a 1-row window),
`TF_DSV41_ATTN_CUDA_PARTS` (`attn,topk`; either alone for an A/B) and `TF_DSV41_MHC_CUDA_ROWS` (16, the largest
window the mHC kernel takes; 8 measured the same). From G15-G17, all off: `TF_DSV41_XMAP`, `TF_DSV41_DEPTH_JOINT=2`,
`TF_DSV41_PF_XOVL`, `TF_DSV41_PREFILL_TILES`, `TF_DSV41_PF_4K` (refused at boot without `TF_DSV41_PF_4K_MEMORY_OK=1`),
`TF_DSV41_SPEC_NUCLEUS`, `TF_DSV41_ALLOC_CEIL_GIB` (0), `TF_DSV41_SWITCH_MS` (unset).

Serving safety, on by default (G16-G19):

- `TF_DSV41_FAILFAST` (1): an exception in either rank's round fails every in-flight request with a 500, tells the
  peer over the plan link's TCP connection, and exits both ranks with code 70 after `TF_DSV41_FAILFAST_GRACE_S` (1.0)
  s. `scripts/serve.sh watch` heals an exit 70 on its first tick. 0 restores the old behaviour (the peer waits in its
  exchange: minutes).
- `TF_DSV41_FLOOR_PRICE` (1): admission prices a prompt's prefill transient (window rows, the indexer's blocks, host
  buffers) and admits it only if the tighter rank keeps `TF_DSV41_FLOOR_HARD_GIB` after it; rank 1 reports its usable
  memory every `TF_DSV41_PEER_MEM_S` (0.5) s. A prompt that does not fit waits; with nothing else running it is
  refused with a 503 after `TF_DSV41_FLOOR_REFUSE_S` (30) s (for a streaming request that refusal arrives as an
  in-stream error event after the 200).
- `TF_DSV41_GRAPHS_MAX` (48): the LRU CUDA graph cache; context buckets grow 1.25x past 32K
  (`TF_DSV41_GRAPH_BUCKET_GROW`; 1 restores equal 2,048-token steps); `TF_DSV41_GRAPH_DIP_ROUNDS` (64): captures
  refused within that many rounds of a prefill round evict nothing.
- `TF_DSV41_VISION_MAX_IMAGES` (8): more images in one request are refused (HTTP 400); clients with long agent
  histories should drop old screenshots.

Host memory and stalls (G12; defaults in brackets):

- `TF_DSV41_STALL_S` (300; 0 = off): a round that runs longer prints a stall report on both ranks (rank, round, phase,
  the round's plan, the forward's waits, every thread's stack), again every 4x that. `TF_DSV41_STALL_EXIT_S` (0 =
  never) exits the process after that long in one round.
- `TF_DSV41_ENGRAM_WAIT_S` (300; 0 = none): deadline on an Engram read and on the Engram gate's worker.
- `TF_DSV41_NUMPY_HUGEPAGE` (0): NumPy's huge-page request on its large buffers, off on the ranks.
- `TF_DSV41_HOST_BUFFERS` (6): reused NumPy buffers for a prompt segment's Engram rows.
- `TF_DSV41_HASH_MEMO_ROWS` (16384): rows the Engram hash memo holds. `TF_DSV41_HOST_TRIM_ROWS` (32768; 0 = off):
  `malloc_trim` every that many prompt rows (it reaches glibc only; harmless).
- `TF_DSV41_MEMPROBE=N` (0 = off): one memory line a rank every N rounds (RssAnon / RssShmem, MemAvailable, torch's
  allocator); `TF_DSV41_MEMPROBE_TENSORS=K` and `_SMAPS=1` add live CPU tensors and anonymous RSS by mapping (slow,
  diagnostics only).

## Memory

- **Before a load:** MemFree >= 104 GiB on both nodes.
- **While serving:** `TF_DSV41_FLOOR_GIB=5` is the MemAvailable target the load-time budget plans for;
  `TF_DSV41_FLOOR_HARD_GIB=4` is the level below which nothing new is admitted, checked on both ranks with each
  prompt's prefill transient priced in.
- **Measured (G19, 2,048-row adaptive prefill):** 299K stress minimum head 5.07 / worker 4.81 GiB; 1 h soak minimum
  head 4.63 / worker 3.99 GiB (at `_ADAPT_GIB=4.0`; production runs 4.5), RssAnon falling over the hour. The worker
  bound in G19 with 2 GiB less MemTotal than the head (firmware, equal since 2026-10-24) and 6.2-8.4 GiB at serving
  between identical boots. If either rank runs short, raise `TF_DSV41_PREFILL_ADAPT_GIB` or go back to 1,024-row
  windows. [RESULTS.md](RESULTS.md) section 5.
- **earlyoom:** if the nodes run earlyoom (`-m 2,1` here: SIGTERM below ~2.4 GiB), it is the last backstop under these
  floors and kills a rank; fail-fast then ends the pair in ~1 s and the watchdog restarts it. Never build CUDA
  extensions beside loaded weights (`scripts/serve.sh prebuild` exists for that: G16 lost both ranks to it once).
- Keep other workloads (desktop sessions, other containers, a second model) off both nodes.

## Reported configurations

- **1M context, 1 slot** (reported on this path by @ZackO2o in #5): `KV_DTYPE=fp8`, `CONTEXT=1048576`, `PARALLEL=1`
  booted and served. `/v1/models` reported 1,048,576, retrieval at 60K tokens hit, and prefill ran at 2,048 / 2,104 /
  1,934 tok/s for 16.6K / 135K / 270K-token prompts. Our own runs cover 4 x 300K and 4 x 196K. One slot serves one
  request at a time.
- **The same on the Zig path** ([ZIG-SERVE](ZIG-SERVE.md#split-kv-and-the-pool)): the profile already serves one 1M
  request. `TF_DSV41_CONTEXT=1048576` with `TF_DSV41_KV_SPLIT=1` and the recommended flash17 profile's
  `TF_DSV41_POOL_TOKENS=2100000` holds one slot's whole 1M, two slots at 1M each, or four live slots at about 525K
  each (totals across live slots). The q28-v2 rollback profile's `TF_DSV41_POOL_TOKENS=3000000` holds four live slots
  at about 750K each. For a dedicated
  one-slot server set `TF_DSV41_SLOTS=1` and check the AOT set again (`scripts/run-zig.sh 0 --check-aot`); drafting
  still runs, and `TF_DSV41_SLOT_DRAFTS` then has no effect. `rope.bin` needs `TF_DSV41_CONTEXT` + 2,048 rows. The
  boot check refuses a pool the measured memory cannot hold, and a request whose prompt plus `max_tokens` exceeds the
  context is refused with the context-length error.

## Watchdog and start at boot

- `scripts/serve.sh watch --once` (the `dsv41-tf-watchdog.timer` tick, every minute): a tick is bad when a rank
  exited or `/health` fails. `WATCH_FAILS` (3) bad ticks in a row restart both ranks in the background
  (`WATCH_HEAL=1`), at most once every `WATCH_MIN_HEAL` s (1800). A rank that exited with code 70 (fail-fast) heals on
  the first bad tick, at most once every `WATCH_FF_MIN_HEAL` s (120). It stands down while a start runs, while rank 0 is
  younger than `WATCH_GRACE` (1800 s), after a deliberate `serve.sh stop` during the same boot, and while
  `WATCH_LEASE` (an optional file you touch while benchmarking or maintaining the pair) is fresh. `WATCH_ALERT=<cmd>`
  is called with a message on every alert.
- `scripts/boot-start.sh` (the `dsv41-boot-start.service` oneshot): after a reboot, starts the service if it was
  serving (or crashed) before, once docker, the worker and the RoCE ports are up; `BOOT_DRY_RUN=1` shows the
  decision. The watchdog is the fallback.
- Before benchmarks with `scripts/serve.sh run`, stop the server and the watchdog timer
  (`systemctl --user stop dsv41-tf-watchdog.timer`), or set `WATCH_LEASE` and touch the file.

## Logs and state

- `docker logs dsv41-tf-r0` / `scripts/serve.sh logs 1`; the `[boot]` lines time each start phase.
- `HEAD_STATE` / `WORKER_STATE`: the request log (`requests.jsonl`) and the session NVMe tier (`sessions/`).
- `STATE_DIR` (default `~/.local/state/dsv41-tf`): the start lock, watchdog counters, stop marker, heal and
  boot-start logs, canary-failure logs.
