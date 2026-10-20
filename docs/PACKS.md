# Weight packs

The recipe serves one EXL3 pack of DeepSeek-V4.1-Flash-1017 at a time. **flash17 is the default and the recommended
pack** ([FLASH-1017](FLASH-1017.md)). Any other EXL3 pack with the 1017 layout (the same tensor names and shapes)
can be added as a second profile the same way EAQ is below: check it, widen it if needed, give it its own
profile and folders, and select it with `PACK`.

| Pack id | Repository @ revision | Shards | Tensor bytes (index `total_size`) | Download | vs flash17 a rank (TP=2) | Profile | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `flash17` (default) | [`jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw) | 42 safetensors | 203,755,654,072 | 189.8 GiB | - | [`prod-zig-flash17.env.example`](../config/prod-zig-flash17.env.example) | recommended, measured |
| `eaq` | [`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) @ `6d54be4fca338d79b17c3b03cc9a423bb3ac6718` | 39 | 210,580,703,416 | ~196.2 GiB | **+3.18 GiB** | [`prod-zig-eaq.env.example`](../config/prod-zig-eaq.env.example) | optional, experimental, [measured](#results-on-2x-spark) |

## EAQ: DeepSeek-V4.1-Flash-1017 EXL3 EAQ v1 2.9 bpw

**DeepSeek-V4.1-Flash-1017 EXL3 EAQ v1 2.9 bpw by Mia's AI Lab (MIT), experimental, quality not measured by its
authors.** The model card is at
[Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw);
Mia's own serving recipe for it is
[MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks).
We download the weights and do not redistribute them. Mia's AI Lab found which of the pack's tensors are stored F16
and need widening for this engine; the recipe's tools here are our own implementation of that step.

**Status: measured, optional.** Quality on our suite is close to flash17's (316/356 vs 320/356) and code decode
matches it, but the larger weights leave a ~170K-token KV pool and need eager decode for stability on 2x Spark
([results](#results-on-2x-spark)). flash17 stays the default.

What the headers show (`scripts/check_pack.py`, no tensor data read):

- **Standard EXL3**, `mul1` codebook, every group with a `mul1` marker, no `mcg`; the same tensor names and shapes as
  flash17 except 11 bf16 copies of EXL3 groups flash17 carries and this pack does not (the engine ignores them).
- **2.90 bpw** over the body (flash17: 2.80): routed experts 3 bit, 2 bit on layers 18-22, uniform within each layer
  and with gate width = up width; shared experts 4-5 bit (flash17: 5-6); attention 5 (layer 0: 6); indexer `wk` 8;
  Engram `wkv` 5 / 4; head 6; DSpark (MTP) 4. Every width is on a kernel this engine has, so it runs on the same
  fast paths as flash17.
- **Size:** 6,825,049,344 bytes (6.36 GiB) more tensor data than flash17, about **3.18 GiB more on each rank** at
  TP=2, which comes out of the room for the KV pool, the 4K prefill workspace and the decode graphs
  ([KNOWN-ISSUES](KNOWN-ISSUES.md#low-memory-headroom)).
- **Storage the engine refuses at snapshot `9131e10a`:** `mtp.{0,1,2}.ffn.gate.bias` (F16; the engine takes F32 only)
  and `mtp.2.confidence_head.proj.weight` (F16; F32 or BF16 only), in shards 38 and 39. The other F16 tensors (router
  gates, indexer `weights_proj`, the Markov head) are accepted and rounded to BF16 at load.
- **Vision** is included (the tower and aligner, 263 BF16 tensors, the same names and shapes as flash17's) with the
  43 F32 `gate.bias_vl` image routing biases, so `TF_DSV41_BIAS_VL=/model` works as for flash17.

Shard sizes (GB, `model-000NN-of-00039.safetensors`): 01 6.55, 02 5.32, 03 5.23, 04-08 5.22, 09 5.23, 10-14 5.22,
15 5.30, 16-18 5.22, 19 7.04, 20 7.05, 21 3.52, 22-30 5.22, 31 5.23, 32-34 5.22, 35 5.23, 36-37 5.22, 38 8.12,
39 5.82; 210,662,978,357 bytes in all repository files.

### Disk on each node

About **196.2 GiB** for the download, **13.9 GB** for the widened copies of shards 38 and 39 (every other file is a
hard link), and **~95 GB** for its own prepared folder (`PREPARED_DIR`), plus the 1017 Engram shards and assets you
already have for flash17 (shared). Both nodes need byte-identical model files.

### Download, check, widen, select

On each node (the download and the served folder on the same filesystem):

```bash
hf download Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw \
  --revision 6d54be4fca338d79b17c3b03cc9a423bb3ac6718 --local-dir /srv/models/eaq-download
python3 -I scripts/check_pack.py /srv/models/eaq-download        # refuses: 4 tensors need widening (exit 1)
python3 -I scripts/widen_pack.py /srv/models/eaq-download /srv/models/eaq
python3 -I scripts/check_pack.py /srv/models/eaq                 # pack check: ok
cp config/prod-zig-eaq.env.example config/prod-zig-eaq.env && $EDITOR config/prod-zig-eaq.env
PACK=eaq scripts/run-zig.sh 1 --check-aot                        # rank 1, then rank 0 on its node
PACK=eaq scripts/run-zig.sh 1                                    # rank 1, then PACK=eaq ... 0 on rank 0
```

- `check_pack.py` reads only the index, configs and headers. To check a pack before downloading it, fetch the index
  and configs, then each shard's header (8 bytes of little-endian length, then that many bytes of JSON, saved as
  `<shard>.header.json`), and run `check_pack.py <folder> --headers <header folder>`.
- `widen_pack.py` writes the served folder `/srv/models/eaq` (set `MODEL_DIR` to it): copies of the two shards with
  those four tensors widened exactly to F32, hard links to every other file, an updated index and
  `widen-manifest.json`. Run it again or with `--check` to re-verify (read-only); keep the download unchanged.
- `PACK=eaq` makes `scripts/run-zig.sh` read `config/prod-zig-eaq.env`. Without `PACK` it reads
  `config/prod-zig.env` as before, so flash17 users change nothing. To go back, stop rank 0, then start both ranks
  without `PACK`.
- The first `--check-aot` or start claims the new, empty `STATE_DIR` for the pack (`STATE_DIR/pack-id`). A state
  folder that is already in use but has no `pack-id` (it served flash17 or q28-v2 before packs were marked) is
  refused, not claimed: its draft calibration could be another pack's. Give the pack its own new `STATE_DIR`; only
  if the folder served that pack alone, mark it yourself (`printf '%s\n' <id> > STATE_DIR/pack-id`). A folder marked
  for another pack is refused too, also for a profile without `PACK` (counted as flash17, or the `PACK` selected).

### The profile

[`config/prod-zig-eaq.env.example`](../config/prod-zig-eaq.env.example) is the flash17 profile with:

| Setting | flash17 profile | EAQ profile | Why |
| --- | --- | --- | --- |
| `PACK` | `flash17` | `eaq` | selection; claims a new or empty `STATE_DIR` for the pack |
| `MODEL_DIR` | flash17 pack | the **widened** EAQ folder | the weights |
| `PREPARED_DIR` / `STATE_DIR` | flash17's | **its own** | prepared images are per pack; the draft calibration in `STATE_DIR/calib-zig` is keyed without the pack at `9131e10a` |
| `TF_DSV41_CALIB_FILE` | optional | **unset** until this pack's table is measured on an idle pair | never flash17's table |
| `TF_DSV41_POOL_TOKENS` | 2100000 | **170000** | 3.18 GiB a rank less room (boot check maximum 191,683) |
| `TF_DSV41_PF_WS_FLOOR_GIB` | 3.5 | **4.75** | where 4K prefill falls back to 2K; 3.5 went under the test floor |
| `TF_DSV41_GRAPH_FLOOR_GIB` | unset | **64** (eager) | `hold` went under the test floor in the quality run |

Everything else is unchanged, including expert top-p 0.85 (min 3), which is lossy (top-1 agreement 0.9944 with
unpruned routing, [DECODE](DECODE.md)); remove `TF_DSV41_EXPERT_TOPP` for exact routing.

**The memory values are tested benchmark values, not a qualified long-session profile.** They ran the speed and
quality suites below on 2x Spark; no long agentic session envelope (for example 150K-token resumes) is qualified on
them. Confirm them on your own nodes:

1. **`TF_DSV41_POOL_TOKENS=170000`.** If your boot check refuses it, its log names the most that fits (`kv: ... the
   pool does not fit; the largest that does: TF_DSV41_POOL_TOKENS=<max>`, or `the largest pool that fits: <max>
   tokens` before the pool is made); use 0.9 × that maximum on both nodes. Ours was 191,683 on the tighter rank. A
   smaller pool means less total context across live slots, not a shorter context per slot.
2. **`TF_DSV41_PF_WS_FLOOR_GIB=4.75`.** From the same boot: `kv: rank N: the prefill workspace (... GiB ...) at the
   worst point leaves X GiB, under the 4.0 GiB hard floor: long prompts there run 2,048-row segments`. Lower keeps
   4K pieces with less margin; higher falls back to 2K segments sooner (same bits, slower prefill). Our runs used 2K
   fallback segments; a test arm using 3.5 went under its 4.25 GiB memory floor during an 8K prefill.
3. **`TF_DSV41_GRAPH_FLOOR_GIB=64`** forces every decode window eager (no node has 64 GiB available). With the
   engine's rule (`hold`) a node went under the 4.25 GiB test memory floor during the quality run (4.22 GiB); with 64
   the rest of the run was stable, at about 18% less decode on a long-prompt benchmark. Remove it only after
   measuring your own headroom over long sessions; see [KNOWN-ISSUES](KNOWN-ISSUES.md#low-memory-headroom).

Before load, check both ranks as for flash17: after-boot MemAvailable minus the 3.0 GiB serving dip must leave at
least 4.25 GiB, and watch both ranks during the first long prompts.

### Results on 2x Spark

Measured by us on 2x DGX Spark at engine `9131e10a`, against flash17 on the same profile (same engine settings,
pruning 0.85, 4 slots; only the pack, its folders and the memory values differ). We suggest a 2100-2200 MHz GPU
clock cap to help with temperature control.

**Speed** (single stream, medians of three; the long prompt is one cold request of 37,974 tokens):

| | flash17 | EAQ, graph floor `hold` | EAQ, graph floor 64 (eager decode) |
| --- | ---: | ---: | ---: |
| Code decode, tok/s | 86.6 | 86.5 | 71.9 |
| Prose decode, tok/s | 48.2 | 45.6 | 37.5 |
| 32K native prefill, s | 16.0 | 17.3 | 16.4 |

Code decode is level, prose is 5% slower (fewer accepted draft tokens a round), and prefill is 8% slower. Forcing
eager decode costs about 17-18% decode here and 18% on a separate long-prompt benchmark (46.6 -> 38.1 tok/s), not
prefill; its replies are identical to the graph runs.

**Quality** (fixed 356-item suite, paired against flash17's frozen reference; high effort, thinking on, T=0):
**EAQ 316/356 (88.8%) vs flash17 320/356 (89.9%)**, paired difference **-1.12 pp**, conservative 95% CI
**[-4.47, +2.28]** (4 EAQ-only wins, 8 flash17-only wins). The interval does not establish equivalence; the EAQ run
mixes its first 68 items on graph floor `hold` and the remaining 288 on graph floor 64, and flash17's reference spans
earlier engine stages, so this is not a pure quantization comparison.

| Suite | EAQ | flash17 | Difference, pp | Conservative 95% CI, pp |
| --- | ---: | ---: | ---: | ---: |
| **Total** | **316/356** | **320/356** | **-1.12** | **[-4.47, +2.28]** |
| HumanEval+ | 34/40 | 34/40 | +0.00 | [-18.39, +18.39] |
| MBPP+ | 49/60 | 50/60 | -1.67 | [-12.84, +9.89] |
| LiveCodeBench | 14/20 | 16/20 | -10.00 | [-34.86, +18.82] |
| MMLU-Pro STEM | 93/100 | 93/100 | +0.00 | [-6.20, +6.20] |
| MATH500 | 36/40 | 36/40 | +0.00 | [-10.38, +10.38] |
| SciCode first step | 6/10 | 6/10 | +0.00 | [-35.48, +35.48] |
| MGSM | 40/40 | 40/40 | +0.00 | [-10.38, +10.38] |
| BFCL | 39/40 | 39/40 | +0.00 | [-10.38, +10.38] |
| Tool chains | 5/6 | 6/6 | -16.67 | [-69.13, +51.62] |

Strict success is a score of 1; the intervals are Bonferroni Clopper-Pearson bounds on the paired wins and losses.
**Vision:** 23/23 (20 VQA plus 3 reviewed chart / screenshot / photo probes). **Agentic build:** the OpenCode
SQLite CLI build task passed (visible and held-out tests) in 73 s.

**The trade-off.** EAQ's extra ~3.2 GiB a rank leaves room for a KV pool of only ~170K tokens shared by the slots
(flash17: 2.1M), and on 2x Spark it needs eager decode (graph floor 64) to stay above the memory floor, which costs
~18% decode. flash17 scores within the same interval, keeps 12x the pool and keeps its decode graphs, so it stays the
default; EAQ is for those who want to compare Mia's AI Lab's quantization on their own workload.

Thanks to Mia's AI Lab for the EAQ pack and for finding the F16 tensors this engine needed widened.

### Reused 1017 assets

A 1017-layout pack has the same model dimensions, tokenizer and Engram tables as flash17, so it reuses, unchanged:

- the per-rank **1017 Engram shards** (`ENGRAM_DIR`, packed from `deepseek-ai/DeepSeek-V4.1-Flash-1017`);
- the rank assets in `ASSETS_DIR`: **`rope.bin`**, **`engram-host.bin`** and the **AOT set** (`aot/`, `sigs.json`);
- the dense-prefill table **`config/pfdense-table.json`** (`CACHE_DIR`).

Confirm the AOT set covers the profile with `PACK=eaq scripts/run-zig.sh N --check-aot` on each rank. The first
start writes this pack's prepared folder (several minutes a node).

### With a newer engine

TensorFold branch `perf-1100/eaq-1` adds two loader changes that this recipe's `engine/zig` (snapshot `9131e10a`)
does not have yet: router biases and the DSpark confidence head accepted in F16 / BF16 storage and widened exactly at
load (store class `f32_wide`; `check_pack.py --half-f32-natives` checks against it), and the Zig draft calibration
keyed by the pack's identity. Once `engine/zig` is re-exported from that engine, `widen_pack.py` and the separate
calibration folder become belt-and-braces: keep using them for the current snapshot (both stay valid afterwards;
the first boot after such an update rebuilds every pack's prepared folder, because the prepared key hashes the
loader's source).
