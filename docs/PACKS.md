# Weight packs

The recipe serves one EXL3 pack of DeepSeek-V4.1-Flash-1017 at a time. **flash17 is the default and the recommended
pack** ([FLASH-1017](FLASH-1017.md)). Any other EXL3 pack with the 1017 layout (the same tensor names and shapes)
can be added as a second profile the same way EAQ is below: check it, widen it if needed, give it its own
profile and folders, and select it with `PACK`.

| Pack id | Repository @ revision | Shards | Tensor bytes (index `total_size`) | Download | vs flash17 a rank (TP=2) | Profile | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `flash17` (default) | [`jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw) | 42 safetensors | 203,755,654,072 | 189.8 GiB | - | [`prod-zig-flash17.env.example`](../config/prod-zig-flash17.env.example) | recommended, measured |
| `eaq` | [`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) @ `6d54be4fca338d79b17c3b03cc9a423bb3ac6718` | 39 | 210,580,703,416 | ~196.2 GiB | **+3.18 GiB** | [`prod-zig-eaq.env.example`](../config/prod-zig-eaq.env.example) | optional, experimental, not yet measured here |

## EAQ: DeepSeek-V4.1-Flash-1017 EXL3 EAQ v1 2.9 bpw

**DeepSeek-V4.1-Flash-1017 EXL3 EAQ v1 2.9 bpw by Mia's AI Lab (MIT), experimental, quality not measured by its
authors.** The model card is at
[Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw);
Mia's own serving recipe for it is
[MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks).
We download the weights and do not redistribute them. Mia's AI Lab found which of the pack's tensors are stored F16
and need widening for this engine; the recipe's tools here are our own implementation of that step.

**Status: speed, memory and quality on this recipe are pending Spark measurements.** No numbers are published for it
here yet. flash17 stays the default until this pack has been measured at least level with it on the same task
suite and its memory envelope (pool, 4K prefill pieces, decode graphs) is acceptable for four slots.

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

### The profile

[`config/prod-zig-eaq.env.example`](../config/prod-zig-eaq.env.example) is the flash17 profile with:

| Setting | flash17 profile | EAQ profile | Why |
| --- | --- | --- | --- |
| `PACK` | `flash17` | `eaq` | selection; claims `STATE_DIR` for the pack |
| `MODEL_DIR` | flash17 pack | the **widened** EAQ folder | the weights |
| `PREPARED_DIR` / `STATE_DIR` | flash17's | **its own** | prepared images are per pack; the draft calibration in `STATE_DIR/calib-zig` is keyed without the pack at `9131e10a` |
| `TF_DSV41_CALIB_FILE` | optional | **unset** until this pack's table is measured on an idle pair | never flash17's table |
| `TF_DSV41_POOL_TOKENS` | 2100000 | **`<P_eaq>`**, from the boot check | 3.18 GiB a rank less room |
| `TF_DSV41_PF_WS_FLOOR_GIB` | 3.5 | **`<measured>`** | where 4K prefill falls back to 2K |
| `TF_DSV41_GRAPH_FLOOR_GIB` | unset | unset; optional after measurement | decode graph floor trade-off |

Everything else is unchanged, including expert top-p 0.85 (min 3), which is lossy (top-1 agreement 0.9944 with
unpruned routing, [DECODE](DECODE.md)); remove `TF_DSV41_EXPERT_TOPP` for exact routing.

**Fill the placeholders** (the launcher refuses a value still holding `<...>`):

1. **`P_eaq`.** Boot once with `TF_DSV41_POOL_TOKENS=2100000`. If the boot check refuses it, its log names the most
   that fits (`kv: ... the pool does not fit; the largest that does: TF_DSV41_POOL_TOKENS=<max>`, or `the largest
   pool that fits: <max> tokens` before the pool is made). Set `P_eaq` = 0.9 × that maximum, capped at 2,100,000,
   on both nodes. A smaller pool means less total context across live slots, not a shorter context per slot.
2. **`TF_DSV41_PF_WS_FLOOR_GIB`.** From the same boot: `kv: rank N: the prefill workspace (... GiB ...) at the worst
   point leaves X GiB, under the 4.0 GiB hard floor: long prompts there run 2,048-row segments`. Lower keeps 4K
   pieces with less margin; higher falls back to 2K segments sooner (same bits, slower prefill). It also sets the
   decode graph floor (below).
3. **`TF_DSV41_GRAPH_FLOOR_GIB`** (optional): only if `/health` shows decode graphs running eagerly; see
   [KNOWN-ISSUES](KNOWN-ISSUES.md#low-memory-headroom).

Before load, check both ranks as for flash17: after-boot MemAvailable minus the 3.0 GiB serving dip must leave at
least 4.25 GiB, and watch both ranks during the first long prompts.

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
