# Changes since G13

The production engine went `767ad9f` (G13) -> `356a188` (G14) -> `7954c1d` (G15a) -> `da5ae43` (G16) -> `cd4245a`
(G18) -> **`7bd2d67`** (G19). Each window's write-up is in [`docs/campaign/`](campaign/README.md).

## Bugs fixed

- **A duplicate image in one request killed both ranks** (G16). The vision row store counted a reference per
  occurrence of an image, so an agent history that resends the same screenshot released it early; the next request
  with that image raised `KeyError` inside rank 0's prefill forward, rank 1 waited in the exchange, and both ranks
  died. The store now counts each image once per request table. Two tests that fail on the old code.
- **The serving memory leak was the CUDA graph cache** (G16). The verify-graph key carries the context bucket (then
  2,048 tokens), so agent sessions growing to 120K tokens kept capturing new graphs (132 held in one soak, ~15 MiB of
  driver host memory each), until MemAvailable sat at the capture floor. Now an LRU cache capped at 48
  (`TF_DSV41_GRAPHS_MAX`) with context buckets 1.25x apart past 32K (26 buckets to 300K instead of 147; replay ==
  eager as before). Host trims also moved off the round thread (they were not the leak).
- **Graph-eviction cascade** (G17-G19). Under the capture floor the cache evicted a quarter of its graphs at every
  refused capture, so the long-prefill dips of one soak walked the cap from 48 to 8. Now one eviction a memory dip (then the freed pool
  is released and 64 capture attempts are skipped), and captures refused within 64 rounds of a prefill round evict
  nothing (`TF_DSV41_GRAPH_DIP_ROUNDS`).
- **NVMe session-index memory drift** (G18). The NVMe session tier's in-RAM index kept every parked entry's token ids as
  a Python `list[int]` (~36 bytes a token): +0.4 GiB/h of anonymous memory a rank under agent traffic, and 0.5-2.5 GiB
  a rank once the 128 GB tier fills (the slow drift seen since G6). The index now keeps (length, page chain, the < 256
  tokens past the last full page). RssAnon slope after the fix: -0.4 GiB/h.
- **Fail fast across both ranks** (G17-G18, `TF_DSV41_FAILFAST=1`, default). An exception on one rank inside a round
  used to leave the other waiting in an exchange: 120 s (RoCE), forever (NCCL), then the plan link's 300 s, then three
  watchdog ticks. Now both ranks fail the in-flight requests with a 500, tell each other over the plan link's TCP
  connection, and exit with code 70 within ~1-3 s (a CUDA out-of-memory on rank 1: both gone in 3.1 s, the client's
  500 names the error). `scripts/serve.sh watch` heals an exit 70 on its first tick (at most every 2 min).
- **A two-rank, priced memory floor** (G17, `TF_DSV41_FLOOR_PRICE=1`, default). Admission ran on rank 0 only with a
  zero price, so a 299K prompt was admitted at 6.2 GiB on the worker (never consulted) and its own prefill took it to
  3.5. Now rank 1 reports its usable memory every 0.5 s and a prompt is admitted only if its priced prefill transient
  fits on the tighter rank; otherwise it waits.
- **The "empty reply" was a capacity refusal priced too high** (G19). The floor priced an 18K prompt at the full
  2,048-row transient and refused it after 30 s (as an in-stream error: the 200 had gone out). Admission now prices at
  the smallest adaptive step; the soak client records in-stream errors.
- **Adaptive prefill rows** (G19, `TF_DSV41_PREFILL_ADAPT=1`, default). Rank 0 picks each round's prompt rows (2,048 /
  1,024 / 512) from the tighter rank's memory, so 2,048-row windows (+13-16% prefill) no longer push the worker under
  the floor in long prompts. Bit-identical outputs at any row count.
- **Fail-fast exit code race** (G18): rank 1 could exit 1 instead of 70 when rank 0 died first, which the watchdog's
  fast heal did not recognise.
- **Image input on CUDA** (G15): the ViT's attention handed 3-D tensors to the fused SDPA kernels (every image -> HTTP
  500); a typed `<｜deepseek_image｜>` in text is escaped instead of refused.
- **Build and packaging** (this update): the image build failed on `xgrammar.__version__` (xgrammar 0.2.8 has none;
  the Dockerfile now reads the package metadata); an installed engine lacked `csa2/*.cu` / `*.cpp` (ATTN_CUDA could not
  build) and the `pfdense` headers (PF_DENSE=fused could not build): packaging fix `66d0dcd`, and the Dockerfile checks
  the files; `scripts/pack_engram.py` failed with `KeyError: 'engram_layer_ids'` on the EXL3 packs' configs, which
  nest the text model's keys under `text_config`.
- earlyoom note: with earlyoom running on the nodes (`-m 2,1`), building CUDA extensions beside the loaded weights
  once pushed MemAvailable under its line and it killed both ranks (G16). `scripts/serve.sh prebuild` builds every
  extension before any weights are loaded; keep it that way.

Thanks to **WireLLM** for issue #3 (host memory growth on rank 0 until admission stalled: the graph cache and the
session index above), **ZackO2o** for PR #4 (the missing `csa2` kernel sources in an installed engine) and
**flashosophy** for PR #6, whose four-Spark work surfaced the `xgrammar.__version__` and `text_config` bugs. The fixes
here are our own; the PRs themselves are under review.

## New levers

| lever | window | what | measured | in the config |
| --- | --- | --- | --- | --- |
| `TF_DSV41_PLAN_LINK=rdma`, `PLAN_PIN` | G14 | the round plan through a RoCE host mailbox, pinned plan threads | with the three below: 1-row window 23.4 -> 21.8 ms; code +3.7%, prose +3.4%, structured +3.3%, C2 +5.4%, C4 +3.3% | on |
| `TF_DSV41_BRANCHES=1` (`_PRIO=side`) | G14 | CSA2 indexer / compressor on a dedicated high-priority side stream | 1-row -0.53 ms over 3 boots | on |
| `TF_DSV41_L2PF_PACE_GBPS=150` | G14 | the L2 prefetch paced beside the exchanges | 1-row -1.09 ms | on |
| `GLM53_TF_ROCE_FAST=1` | G14 | a shorter critical path in the RoCE all-gather kernel | 1-row -0.16, 16-row -0.51 ms | on |
| calibration VERSION 4 | G15 | every verify-table row measured (no fitted lines) | prose +2.8%, the rest flat | default |
| `TF_DSV41_IMAGES=native` | G15-G16 | DeepSeek's ViT + aligner on rank 0, `bias_vl` routing, Engram shut at image positions | VQA 20 / 20; text replies byte-identical to text-only builds | on |
| `TF_DSV41_PF_COPIES=1` | G16-G17 | one prefill weight copy | replay +1.7-3.3%, full +0.2-6.3% | on |
| `TF_DSV41_MHC_PF=1` | G16-G17 | the mHC prefill kernel | replay +3.8-4.7%, full -0.1 to +11.3% | on |
| `TF_DSV41_PF_DENSE=fused` + table | G16-G17 | the dense EXL3 prefill GEMM with the trellis decoded inside it (mma.sync, no fp16 weight workspace), tile table from a sweep | replay +5.4-6.9% with the swept table (+1.5-3.4% with the built-in one); the three together +12.8-13.6% | on |
| `TF_DSV41_FULL_CONE=1` | G17 | full-mode prefill: the decoder runs only over its dependency cone (the last ~2,541 rows), the same bits as plain full | full 32K 853 -> 1,619 alone, 1,828 with the three above (1,024 rows); 2,025 at 2,048 rows | on |
| adaptive 2,048-row prefill | G19 | `PREFILL_CHUNK` / `_ROWS` 2,048 with rows chosen a round from memory (`_ADAPT_GIB=4.5`) | replay 32K 1,996 -> 2,310 | on |
| `TF_DSV41_PF_XOVL`, `PREFILL_TILES`, 4,096-row windows, `SPEC_NUCLEUS`, `XMAP`, `DEPTH_JOINT=2`, `ALLOC_CEIL_GIB`, `SWITCH_MS` | G15-G17 | (built, exact or opt-in) | slower, no gain, or not needed: [`docs/RESULTS.md`](RESULTS.md) section 4 | off |

The dense-prefill tile table (G17's sweep on GB10, 10 shapes): best 68-78 TFLOP/s at 2,048 rows on the wide layers
(Engram 78, `wo_b` 75, shared gate 70, `wq_a` 68), 54-57 on the narrow ones; the whole table is
[`config/pfdense-table.json`](../config/pfdense-table.json) and `results/campaign/G17-20261004/sweep.txt`.

**Still open:** the items under [What is not solved](KNOWN-ISSUES.md).

## Tools

- **`bench/speed.py`**: one HTTP client for speed and serving checks, with these suites: prefill, decode, multi,
  needle, interleave, exact, reuse and anthropic. Multi-stream cells report the wall-window aggregate
  (Σ tokens − N) / (last token − first token), which is Mia's `tools/bench.py` definition. They also report the
  all-live rate and the first-token spread. Our older `bench_http` aggregate summed the per-stream rates, which
  overstates staggered cells. `--compare` diffs saved runs (Mia's `tools/bench.py` files too) and lists every
  setting that differs. See [BENCHMARKS](BENCHMARKS.md#3-speed-needle-and-serving-checks-benchspeedpy).
- **`scripts/preflight.py`** (opt-in, `PREFLIGHT=1 scripts/run-zig.sh`): checks that the Engram shards are on a
  local filesystem with valid headers, sizes and row ranges, and runs a 4 KiB read-latency probe. It reports
  MemAvailable and `/dev/shm` before a boot. After a boot it reads the engine's boot-check lines and prints the
  largest pool that keeps the floor. See [ZIG-SERVE](ZIG-SERVE.md#preflight-optional). The launcher's default
  behaviour is unchanged.
- Both follow ideas from Mia's recipe (credited in [CREDITS](CREDITS.md) and NOTICE).
