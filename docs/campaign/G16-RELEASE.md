# G16: native image input in production, with a memory cap (2026-10-04)

One held campaign window (`campaign.sh open` at 09:02 with `DEADMAN_MIN=300`), engine **`6e03615`** (`dsv41-060`, the
commit G15b validated: calib VERSION 4, the typed-token escape `8f10167`, the firmer placeholder notice, the 4-D tower
SDPA fix). **No engine change**: the cap is five config words. Raw files: `results/G16-20261004/`. Driver:
`scripts/windows/g16.sh` (new; reuses g15vision's stress and battery, g15depth's speed and gate, G14's quality step)
and `scripts/windows/g16soak.py` (new: the soak and its report).

## Verdict

**SHIPPED 2026-10-04 17:07**: prod runs **da5ae43** (`dsv41-g16`) with native images and the memory cap (config/prod.env's G16
block). Rollback: `TF_COMMIT=7954c1d` (still staged) and drop the G16 words.

## 0. The worker's BERT record (read-only)

`journalctl -b 0` (the boot after the 2026-10-03 21:19:56 reset) has only `BERT: [Hardware Error]: Skipped 1 error
records` / `Total records found: 1`. "Skipped" is the kernel's print limit, not a bad record: `bert.c` prints only
records shorter than 1 KiB, and this one is 1,745 bytes. `rasdaemon` was running (its database: no memory, PCIe AER,
extlog or MCE errors); `ras-mc-ctl` has no EDAC memory controller on GB10; `iasl` is not installed. The table
(`/sys/firmware/acpi/tables/BERT`, OEM `MTKID MTKTABLE`) points at a 1,745-byte region; its data
(`/sys/firmware/acpi/tables/data/BERT`) decoded by hand (`results/G16-20261004/bert-worker-data.hex`):

| field | value | meaning |
| --- | --- | --- |
| Generic Error Status block | block status 0, data length 1,725, **error severity 3** (none) | |
| Generic Error Data Entry | section type **`bf32d4d5-b427-4025-8495-8a9e5d4030e4`** (not a UEFI standard type: no processor, memory or PCIe section), **severity 1 = fatal**, revision 0x300, FRU text **"RAS Registers" / "MMAP_D"** | a vendor (MediaTek) section |
| embedded CPER record header | `CPER` rev 1, 1 section, **severity fatal**, creator `e0a45619-a451-4450-96c2-c7a1e90a95fa` (vendor), **notification type `9a78788a-bbe8-11e4-809e-67611e5d46b0` = SEA** (ARM Synchronous External Abort), no timestamp | |
| section body | a count of 20, then **20 ARM RAS error records** (index + ERR<n>FR / CTLR / STATUS / ADDR / MISC0-3, 72 bytes each) | **every register of every record is 0**: no valid error status, no address, no syndrome |

**Reading:** the firmware logged a **fatal event reported as a synchronous external abort**, and dumped its
memory-mapped RAS nodes ("MMAP_D"), but **none of the 20 nodes had latched an error**. So there is no DRAM ECC error,
no PCIe error and no CPU error syndrome to point at; the record says "something fatal, cause not captured by the RAS
nodes". That fits a firmware-level or power / reset event more than a memory fault. For comparison:

- The worker's earlier boots (−7 to −1, including the 10-02 13:17 reset) have **no BERT lines at all**.
- The head holds a BERT record from its own 10-02 13:17 end (both nodes went down within 10 s that day, so that one
  was a shared event): **severity corrected**, vendor section `3c1e3f4b-1e1a-43df-af28-59820e958e3c` (62 bytes,
  `MTKID`, the same creator GUID), printed by the kernel at boot ("It has been corrected by h/w"). It is a different,
  benign record type.

Nothing was changed on either node.

## 1. Where the memory goes in the worst case

### The stress, measured with attribution

Four 299K stresses (G12's: one 299K prefill, three 64K x 2,048 queued behind it; 2 s wide samplers of `/proc/meminfo`
and the rank's RssAnon, `TF_DSV41_MEMPROBE=4` for torch's counters). `d1` = prod's words on 6e03615 (placeholder):

| node | MemAvailable at serving | min | drop | torch reserved | RssAnon | page cache / shmem / slab | the rest (device memory outside torch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head | 8.94 | 5.39 (+47 s) | 3.55 | **+1.60** | +0.51 | +0.05 / +0.02 / +0.01 | ~1.4 |
| worker | 6.22 | **3.51** (+15 s) | 2.71 | **+1.60** | +0.62 | +0.03 / +0.02 / +0.01 | ~0.5-1.0 |

**Root cause: the dip is one request's prefill transients after admission, not concurrency, sessions or page cache.**

1. **torch's caching allocator keeps the prefill's peak.** Reserved goes 102.24 → 102.87 GiB at the first full
   2,048-row chunk (the chunk buffers), then on to **103.84 by 27K rows** while allocated stays +0.13. The growth
   is the indexer's materialised scores: below `TF_DSV41_INDEX_STREAM_MIN` (16,384 visible keys) every prefill
   window scores in row blocks of up to `TF_DSV41_INDEX_BUDGET_MIB` (256 MiB), and the block size grows with the
   visible keys, so each new size takes a new allocator segment. At 16,384 visible keys (32K positions on the
   ratio-2 layers) the selection streams and the growth stops. The 1.6 GiB stays reserved for the rest of the
   process (unused, cached).
2. **Host:** +0.5-0.6 GiB RssAnon on each rank (the Engram bulk rows and buffers of a 2,048-row segment).
3. **Device memory outside torch:** 0.5-1.4 GiB more leaves MemFree during the first long prefill and is in no
   meminfo category (not anon, cache, shmem, slab or page tables) and not in torch's reserved; it varies by boot.
4. **The worker is the binding node.** the worker's MemTotal is 2.0 GiB below the head's (125.5 vs 127.6 GB; firmware;
   equal since the worker's BIOS and firmware update on 2026-10-24),
   and its serving-start MemAvailable varies 6.2-8.4 GiB between identical boots (torch's counters equal: the
   variation is outside torch). The vision tower is rank 0 only, so the worker breaches with or without native.
5. **The pool is not the problem:** 4 x 300K of FP8 rows is 4,692 pages = 1.2 GiB (1,074 B a token), so `CONTEXT`
   and the pool can buy at most ~0.9 GiB.

### Why the floor gate did not stop it (the bug)

`memory.Floor` (rank 0's admission) has four holes, all visible in `stack.py` / `batch.py` / `memory.py`:

- **rank 0 only:** `floor = memory.Floor.from_env() if rank == 0 else None`. Rank 1 never reports its memory, so
  the tighter node is never checked.
- **at admission only, with `need = 0`:** `_admit` calls `floor.check(0, ...)`. A long prompt is admitted at 6-9
  GiB, and its own prefill then takes 2.7-3.6 GiB. Nothing prices the transient, and nothing can act after admission.
- **the hard floor is 4 GiB, not 5**, and `usable` counts reclaimable page cache, so it would admit at ~4.5 anyway.
- **the existing limits held:** "long prefills (> 32,768 tokens) 1 at a time" held (the three 64K prompts waited
  for the 299K one: `long_waits`), the pool held, `SESSION_RAM_MIB` (bounded state) and `CONTEXT` were never the
  cause. The dip is inside the one admitted prefill.

A proper fix is a two-rank floor (rank 1's MemAvailable in the plan link's ack) plus a reservation that prices a
long prefill's transient. That is an engine change on the latency-critical plan link and is not in this release; the
cap below removes the transient instead.

### Leak check (host memory that grows and is not given back)

**Yes, and it is the bigger problem.** Evidence, in order:

1. **Prod 7954c1d before the window** (read-only, 3.5 h after its 06:17 start; 43 requests in the first 47 min: one
   agent session to 98K tokens, 107K tokens prefilled, 2.3M cached; idle since): rank RssAnon **head 4.54 GiB**
   (boot ~2.0), **worker 3.42 GiB** (boot ~1.8); MemAvailable head 5.58, **worker 4.91 GiB, under the floor at
   idle**. Flat across two samples 15 min apart while idle: it grows with traffic, not time. Where it sits
   (`/proc/<pid>/smaps`): head 2.39 GiB in 64 MiB-aligned **glibc thread arenas** (455 threads), 0.76 [heap], 1.04
   other; worker **1.45 GiB [heap]** (the main thread), 1.06 other, 0.55 arenas.
2. **`malloc_trim(0)` in the live prod ranks (gdb, `trimtest.txt`) gave back almost nothing:** head RssAnon 4.756 →
   4.696 GiB (60 MB), worker 3.583 → 3.573 GiB (10 MB), with `malloc_trim` returning 1 on both. By the time of the
   test, most of the growth was not free memory sitting in glibc's arenas: it was live allocations (or free chunks
   pinned by live neighbours on every page). The first leak hypothesis (glibc arena retention, `64f65cc`'s trims) is
   tested by the A/B soak below, not assumed.
3. **Identical stresses on fresh boots** (d1 / d2): RssAnon +0.58 (head) / +0.55 (worker) GiB each time, then
   +0.1-0.2 GiB more per ~500K further prompt rows (d3 / d4's probes).
4. **The off-arm soak** (fcac1c9 + the cap, native, trims of 6e03615 only; `g16soak.py`: 4 lanes, a growing agent
   session with screenshots, short chats with cancels, 24-200K prompts, image requests), 3-minute means of the 2 s
   samplers (`msamp-r*.log`), MemAvailable / RssAnon GiB:

   | min | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | head | 7.06 / 2.87 | 5.08 / 4.18 | 4.41 / 4.73 | 4.51 / 4.99 | 3.97 / 5.09 | 3.63 / 5.27 | 3.66 / 5.33 | 3.35 / 5.35 | guard |
   | worker | 6.58 / 2.56 | 5.18 / 3.31 | 5.11 / 3.33 | 5.13 / 3.34 | 4.79 / 3.38 | 4.90 / 3.43 | 5.09 / 3.36 | 4.68 / 3.26 | guard |

   The head's RssAnon went **1.86 → 5.35 GiB in 24 min** and was still rising (~+1 GiB/h after the first 12 min);
   torch's reserved stayed flat (102.8 GiB). The worker plateaued at ~3.35 GiB from minute 3. **At +27 min the head
   reached 2.9 GiB and the window's guard (3.0) stopped the ranks** (11:05:13), close to earlyoom's 2.4 GiB kill
   line. The rank logs (and their memprobe breakdown) went with the containers; the wide samplers kept the curve.
   The cap holds the device side; the host growth is what breaches the floor in real traffic.

The agent lane's HTTP 400s in that soak are the image limit (`TF_DSV41_VISION_MAX_IMAGES` = 8: the lane adds a
screenshot every third turn, so turn ~27's history carried 9). That is the server working as built, but **it is a
user-visible limit for agents: a session whose history holds more than 8 images is refused** until the client
drops old images.

## 2. The cap

Each candidate was measured on the same stress with attribution (`results/G16-20261004/stress-summary.txt`,
`memprobe-*.txt`). MemAvailable minimum over the whole run, which includes the TTFT probes after the stress for d3 /
d4; "stress" = the 299K + 3 x 64K phase only:

| run | words (over prod's) | images during the 299K prefill | head min | worker min | reserved growth by 35K rows | 299K TTFT |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| d1 | (prod) placeholder | no | 5.39 | **3.51** | +1.60 | 167.9 s |
| d2 | + `INDEX_BUDGET_MIB=64` | no | 6.13 | 4.10 | +0.79 | 168.7 s |
| d3 | + `INDEX_STREAM_MIN=4096`, `POOL_TOKENS=614400`, native | no (harness: no image dir) | 5.82 (stress 6.37) | 4.78 (stress 5.26) | +0.77 | 169.0 s |
| **d4** | **+ `PREFILL_CHUNK=1024`, `PREFILL_ROWS=1024`**, native | **4 / 4 → 200** | **5.30 (stress ≥ 6.06)** | **5.10 (stress ≥ 6.01)** | +0.70 | 199.2 s |

TTFT probes (fresh random-id prompts, max_tokens 1, best of 2; `ttft-*.txt`), tok/s:

| config | 8K | 16K | 32K | 64K | 128K | 299K (stress) | kit (vLLM) 8K / 32K / 64K / 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prod's words (6e03615, placeholder) | 1,839 | 1,974 | 2,015 | 2,029 | 1,957 | 1,781 | 1,073 / 1,075 / 1,060 / 1,031 |
| d3: index knobs + pool, native | 1,822 | 1,966 | 2,036 | 2,039 | 1,967 | 1,769 | |
| **d4: the cap, native** | **1,652** | **1,723** | **1,753** | **1,740** | **1,690** | **1,501** | |
| d4 vs prod's words | −10% | −13% | −13% | −14% | −14% | −16% | still 1.54-1.65x the kit |

**The cap (shipped):**

| word | prod before | now | what it costs |
| --- | --- | --- | --- |
| `TF_DSV41_INDEX_BUDGET_MIB` | 256 (default) | **64** | nothing measured (TTFT equal); exact (rows are independent of the blocks) |
| `TF_DSV41_INDEX_STREAM_MIN` | 16384 | **4096** | nothing measured; exact (`stream_topk` selects the same positions) |
| `TF_DSV41_POOL_TOKENS` | 1,200,640 (4 x 300K) | **614,400** (2,400 pages, 0.6 GiB saved on each rank) | the 4 slots share 614K tokens of KV (live requests + RAM-tier session pages). A request needing pages the pool lacks **queues** (`pool_waits`) after cold RAM-tier sessions are parked to NVMe; nothing is refused (one request needs at most 300K + max_tokens) |
| `TF_DSV41_PREFILL_CHUNK` / `_ROWS` | 2048 / 2048 | **1024 / 1024** | **prefill 10-16% slower** (above); decode unchanged; outputs unchanged (fast prefill is segmentation-invariant: s2's chunk-128 cold == s1, text probes byte-identical) |
| `CONTEXT` | 300000 | 300000 (unchanged) | |

**User-visible limits:** maximum context stays **300K tokens a request**. Up to **4 streams** at once as before, but
their KV totals **614K tokens**: two 300K conversations at once, or four of up to ~150K, or one 300K plus three
~100K. Past that, a new request waits for pages (it is not refused). Long prefills (> 32K tokens) still run one at a
time. Long prompts take 10-16% longer to first token (a 128K prompt 67 → 78 s; 299K 168 → 199 s).

**What did not make the cut:** the index knobs + pool alone (d3) cost nothing but leave the worker at 4.78 GiB after
the stress; a lower `CONTEXT` buys at most 0.9 GiB (the whole pool is 1.2 GiB); `SESSION_RAM_MIB` is host bounded
state (≤ 0.25 GiB). The chunk is what cuts the transient: the stress phase's minimum moves from ~5.3 to ≥ 6.0 GiB on
the worker.

## 3. Validation

### The first soak found a server-killing native-vision bug (fixed: `fcac1c9`)

The first soak (6e03615 + the cap, native) died 10 min in:

1. `[tensorfold] request failed (500): StopIteration` in `vision.Store.release`;
2. then `KeyError: 5` in `vision.Store.gather`, raised **inside rank 0's prefill forward** (`forward.embed_rows` →
   `vision.fill`), three requests 500;
3. rank 0 left the round, rank 1 waited in the exchange: `RoCE all-gather on rank 0 timed out after 120 s ... The
   runtime is poisoned`, then `plan link: rank 1 consumed no plan in 300 s`. **Both ranks dead** (in prod the watchdog
   would heal after 3 bad ticks, with every in-flight request lost).

Cause: `Store.hold` counted a reference **per occurrence** of a virtual id. An agent's history carries the same
screenshot more than once (each turn resends the earlier tool results), so one request's table holds the same ids
twice. `release` then saw refs > 0 with no other table holding the id: `next()` on an empty generator →
`StopIteration`, and `where` kept pointing at the released table. The next request with that image (the soak's
image lane and agent lane share the screenshot) hit `KeyError` in the forward. G15b's battery never sent one image
twice in one request.

Fix, `fcac1c9` on branch **`dsv41-g16`** (from 6e03615; `dsv41-060`'s checkout on the workstation had other
uncommitted work, so the fix was made in a separate worktree): `hold` counts each id
once per table; `release` walks each id once and forgets an id no live table holds. Rows, ids and outputs unchanged.
Two new tests (same image twice in one request; overlapping requests with duplicates, every release order) **fail on
6e03615 with StopIteration and pass on fcac1c9**. CPU: vision / images / serving app / template 75 passed (workstation,
with the tokenizer); in the image on the head: those + serving batch, slots, replay, long-prefill memory: 81 passed,
38 skipped (no tokenizer in the container). The rest of this section ran on fcac1c9 (gate / quality on 6e03615: the
fix touches only the image-row store, so the text path is the same code).

Not fixed here, and worth a follow-up: **any exception raised on one rank inside a forward desyncs the ranks** and
kills both (rank 1 waits forever in the exchange). Rank 0's image path should validate everything before the job is
queued (it does for missing rows: `gather` raised only because the store was corrupt).

A second incident in the same hour: the first boot on the new sources built CUDA extensions after the weights were
loaded, MemAvailable fell under **earlyoom's 2 % threshold** (`-m 2,1` in `/etc/default/earlyoom` on both nodes:
SIGTERM below ~2.4 GiB, SIGKILL below ~1.2 GiB) and earlyoom killed both ranks (10:23:24). A prebuild with nothing
loaded (16 / 16 both nodes) fixed it. earlyoom is the last backstop under our 5 GiB floor; it kills the rank.

### Battery, gate, quality

| check | build | result |
| --- | --- | --- |
| 299K stress, native, 4 image requests during the long prefill (d4) | 6e03615 + cap | **head min 5.30, worker 5.10 GiB** over the whole run incl. the TTFT probes; the stress phase alone ≥ 6.06 / ≥ 6.01; 4 / 4 images 200; every decode stream 2,048; 0 tracebacks, 0 stalls |
| battery s1 (native + bias_vl + cap) | 6e03615 + cap, and again on **fcac1c9** (s1fix) | smoke 3 / 3, harmony 672 prompt tokens (== G15a), exactness (hit == cold, batched == cold, turn 2, replay tail), agent flow t1 / t2 (1,344 cached) / t3 / thinking, concurrency (4 images beside 2 text streams), limits (9 images, 25 MB, http URL, private https, **typed `<｜deepseek_image｜>` escaped alone / in history / beside a real image**: `/tokenize` 0 image ids for the typed token, 189 beside the corn == the corn alone), VQA 20 / 20, **text probes byte-identical to prod** (4 / 4); 0 tracebacks |
| s2 (`PREFILL_CHUNK=128`) | 6e03615 + cap | cold == s1 cold |
| s3 (no bias_vl) | 6e03615 + cap | VQA 20 / 20 |
| top-1 vs the kit oracle (G5 gate) | 6e03615 + cap | **0.9961** (first copy 0.9502) == prod and G15b |
| MMLU-200 0-shot | 6e03615 + cap, native | **87.5%** (175 / 200) == prod |
| tool chains | same | **11 / 12** PASS (line 10); one tool call `get_weather {"city":"Hanoi"}` |
| CPU suites | fcac1c9 / 2ef0f11 | see above; 64f65cc's own `test_dsv41_host_trim_idle.py` + long-prefill memory + calib knobs pass (97) |
| GPU tests / tower parity | not re-run | passed in G15b on 6e03615 (same tower code); the GPU test's known bf16 threshold issue stands |

### Soak

Four arms of soak.py `--kinds g16` (4 lanes: growing agent sessions to 120K tokens with screenshots, short chats with
cancels, 24-200K prompts, image requests), the release's words (native, the cap, `MALLOC_ARENA_MAX=4`), 2 s / 10 s
samplers on both ranks. Guard at 3.0 GiB MemAvailable.

| arm | engine | extra | traffic | head min | worker min | outcome |
| --- | --- | --- | --- | ---: | ---: | --- |
| off (trims of 6e03615 only) | fcac1c9 | | 27 min | 2.9 | | guard at +27 min |
| trim | 2ef0f11 (64f65cc's idle / busy trims) | | 17 min | 3.5 | **2.9** | **guard at +17 min**: the trims do not fix it |
| g16cap | 2ef0f11 | `TF_DSV41_GRAPHS_MAX=16` | 36 min, 0 errors | 3.82 | 3.06 | ran to the end; below the 4 GiB floor late |
| **fixceil** | **da5ae43** (LRU graph cap 48, 1.25x buckets) | `TF_DSV41_ALLOC_CEIL_GIB=1.5` | **60 min, 0 errors** | **5.02** | **4.54** | ran to the end; never under the floor |

**The trims were not the leak.** In the trim arm the engine's own probe showed glibc *in use* 0.87 -> 2.79 GiB with
torch reserved 102.06 -> 103.42 GiB, while the trims gave back free chunks (2.97 GiB over 27 trims). The rank held
**132 verify graphs** (8 at boot): the graph key carries the context bucket (2,048 tokens), so agent sessions growing to
120K keep creating keys, ~15 MiB of driver host memory each plus their outputs in the shared pool, and the cache only
stopped at the 5 GiB capture floor, parking MemAvailable there. Capping it at 16 (g16cap) kept the run alive.

**The fix (`dsv41-g16` 378971c + da5ae43):** an LRU graph cache (`TF_DSV41_GRAPHS_MAX` 256 -> 48; under the capture
floor it evicts its least recent quarter instead of freezing), context buckets 1.25x apart past 32K (26 to 300K instead
of 147; replay == eager as before), and an opt-in allocator ceiling (`TF_DSV41_ALLOC_CEIL_GIB`: reserved at boot +
N GiB; torch frees cached blocks before growing). Also `TF_DSV41_SWITCH_MS` (GIL switch interval, off). The bucket,
ceiling and switch ideas came from bertholomus/TensorFold `deepseek-v41-tp2`, rewritten here.

fixceil's slopes still fail the strict per-hour gate: torch reserved +0.91 / +0.93 GiB/h (bounded by the ceiling at
+1.5 by construction), RssAnon +0.51 (head) / +0.25 (worker) GiB/h, MemAvailable -1.9 / -1.8 GiB/h; idle returned
RssAnon to 3.70 / 3.29 (max 4.21 / 3.66). Whether RssAnon flattens needs a longer soak (next window: 4 h, fresh
session tiers: `SOAK_FRESH=1`, which the arms above did not have: later arms resumed sessions from the NVMe tier the
earlier ones wrote, so their prefill load differs: trim 877K, g16cap 1.42M prompt rows).

### Speed

2 interleaved boots a config (cut from 3: the soak came first), `g15depth.sh speed prod combo`: `prod` = prod's own
7954c1d sources (+ 6e03615's m2bench), prod's words; `combo` = **2ef0f11** (the release build: the vision fix and
64f65cc's background trims) with the release's words (native, the cap, `MALLOC_ARENA_MAX=4`). `g15-depth-speed.txt`:

| combo vs prod (means of 2 boots) | code T0 | code T.7 | prose T0 | prose T.7 | struct T0 | C1 | C2 | C4 | C2 st | C2-code st | C2-prose st | C4 st | C4-code st | C4-prose st | prose@C2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| delta | −1.1% | −0.4% | +2.9% | +0.2% | +0.1% | −0.6% | −0.1% | −1.5% | −0.7% | −0.3% | +2.0% | −0.4% | −1.4% | +1.9% | +4.1% |

- **Neutral**: the six steady cells average **+0.19%**; the worst cells are C4-code steady −1.4% and the C4 decode
  aggregate −1.5% (G15b measured −1.2% / −0.6% for 6e03615 against the same prod). Prose gains (calib V4's 2-row
  rounds). The report's "FAIL" is its "faster" bar (≥ +0.5%), not a slowdown.
- **Exact** on every boot: drafted == serial, every concurrent stream == serial (`--c-exact`), and combo's replies ==
  prod's.
- m2bench's decode runs on 2ef0f11, so the background trims and `MALLOC_ARENA_MAX=4` cost nothing measurable in
  decode. Prefill: the cap's −10 to −16% (section 2) is the chunk. The trims' own latency effect in serving shows
  in the soak (no stall lines; TTFTs in `soak-trim.json`).

## 4. Production

- **No-ceiling soak (fixnoceil, da5ae43, fresh session tiers, 60 min):** 0 errors, guard never tripped; head min 4.45, worker
  min **3.70 GiB** (under the 4 GiB floor late; the strict gate FAILs on that and on the MemAvailable slope, -1.5 to -1.7
  GiB/h). glibc *in use* flat after warm-up (2.03 -> 2.11 GiB from 10 to 70 min; it rose 0.87 -> 2.79 in 15 min before
  the fix), so the remaining MemAvailable decline is outside the rank's heap (device memory outside torch: section 1,
  point 3). Next memory task.
- **The ceiling stays off:** `TF_DSV41_ALLOC_CEIL_GIB=1.5` held the worker at >= 4.54 GiB (fixceil), but at 1.0 with
  2,048-row chunks (d5ceil) the 299K prefill hit `torch.OutOfMemoryError` on rank 0 and rank 1 waited 300 s on the plan
  link: until a failure on one rank fails both fast, an OOM costs minutes of both ranks. The 1,024-row chunk stays.
- **Speed (g16fix, 3 boots each, prod = 7954c1d):** code T0 82.3 / 85.2 / 84.8 vs 86.2 / 85.9 / 85.1 (boot 1 an outlier:
  2.0 drafted tokens fewer a round; serial decode equal, 43.1-43.5 both), prose T0 +2.1%, C2 +0.2%, C4 -1.4%, C4 steady
  -0.4%; exact (drafted == serial, concurrent == serial, replies == prod) on every boot.
- **Shipped** at 17:07 through `campaign.sh close` (prod-switch.sh restore, canary): sources da5ae43 on both nodes,
  IMAGES=native, POOL_TOKENS=614400, PREFILL_CHUNK / _ROWS=1024, MALLOC_ARENA_MAX=4; watchdog timer on.

Also in this window: upstream PR #300's GPU tests (`tests/cuda/test_cuda_rowgraphs.py`, `test_cuda_kvpool_copies.py`)
on the integration branch 9828a29: 3 passed.

