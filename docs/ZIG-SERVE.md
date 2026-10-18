# CUDA Zig serving on two DGX Sparks

Recommended weights: **flash17, DeepSeek-V4.1-Flash-1017 EXL3 2.8 bpw**,
[`jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw),
with [`config/prod-zig-flash17.env.example`](../config/prod-zig-flash17.env.example) ([FLASH-1017](FLASH-1017.md)). The
previous pack, q28-v2
([`jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw)), is
the rollback with [`config/prod-zig.env.example`](../config/prod-zig.env.example). The port loads EXL3 directly,
including both packs' per-expert widths. There are two processes: `tensorfold-dsv41 serve` on rank 0 and `tf-dsv41-m1
follow` on rank 1.

## Availability and requirements

The CUDA DeepSeek source is included at [`engine/zig`](../engine/zig), exported from the qualified kit
snapshot on top of [TensorFold 1.0.2](https://github.com/ashhart/TensorFold), under Apache-2.0.
See [source provenance](../engine/zig/SOURCE-EXPORT.md). Generated GPU assets and model weights are separate:
the host build alone embeds empty CUDA blobs and cannot serve a GPU model. The existing Python patches target
0.6.0 and provide the public reference source used by the AOT generators.

Required on both nodes:

- Two DGX Sparks (GB10, 128 GB each), TP=2, cabled/addressed CX7 ports; Docker with NVIDIA Container Toolkit,
  RDMA devices and matching image contents. Set the interface/HCA values to your own link.
- Identical model files on both nodes; rank-specific packed Engram shards on local NVMe, from the pack's own release
  (flash17: `DeepSeek-V4.1-Flash-1017`; q28-v2: the September release). [Engram packing](INSTALL.md)
  writes `engram-l{1,14}-r{rank}of2.bin`. The weights download alone is not sufficient.
- Rank-specific `/assets/rope.bin`, `engram-host.bin`, and `aot/`: RoPE tables covering the configured context plus
  2,048 prefill rows (1,050,624 rows at the 1M context), native host lookup data, and Triton programs for all
  emitted shapes. At four slots, include DSpark,
  row-mode up to 64 rows, CED replay, the device chain variants, 4K prefill pieces and replay runs, and the
  `INDEX_BOUND` twins. `aot-needs` enumerates them from the serving env (see below).
- `/cache/pfdense-table.json`: copy the published `config/pfdense-table.json` into `CACHE_DIR` on both nodes
  (`scripts/run-zig.sh` copies it there when it is missing). Draft costs come from the engine's own
  boot measurement (`CALIB=zig`): the first boot times its windows on both ranks and stores the table under
  `/state/calib-zig`; later boots read it.
- Writable prepared and session directories on local NVMe. Plan roughly 100 GB weights, 95 GB Engram shards,
  another rank's weight-sized prepared cache and up to 128 GiB session files per node, plus image/build space.
  Keep other workloads off the GPUs and preserve the 5 GiB target / 4 GiB hard MemAvailable floor on both ranks.

The source kit's `tools/zig/dsv41_m2c2/build-kit.sh` builds the ARM64 binaries and checks the sm_121 fatbins.
It generates a `knob-names.txt` from the names actually read by Zig. Asset generation uses
`tools/zig/dsv41_rope_tables.py`, `dsv41_engram_host.py`, `triton_fill.py`, the engine's `aot-needs` command and
`dsv41_m4/fill-prod.env` (or the 4K fill). Python reference source, Torch, Triton 3.7.1 and ptxas 13.3.73 are
needed for that fill. Copy assets per rank and validate the fill manifest; a `MissingTritonVariant` is a missing
prerequisite, not a reason to keep running. These generator tools are included in `engine/zig/tools/zig`.
The AOT set depends on the serving settings: see [Changing serving settings](#changing-serving-settings).

## Quick start

The source, build recipe and launcher are included. Generate the assets below before serving. Hardware execution
was not repeated for this publication; the exported host build and host tests are reported in the source guide.

1. Clone this public recipe and download the pack on both nodes:

   ```bash
   git clone --recurse-submodules https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
   cd deepseek-v41-tensorfold-spark
   hf download jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw --local-dir /srv/models/flash17
   cp config/prod-zig-flash17.env.example config/prod-zig.env
   $EDITOR config/prod-zig.env
   ```

   Set the image, rank 0's link address, NIC/HCA settings and each node's own absolute directories. The engine
   settings must match on both ranks. Copy the dense-prefill tuning table into each node's `CACHE_DIR`
   (`cp config/pfdense-table.json "$CACHE_DIR/"`; the launcher does it when the file is missing).
   `config/prod-zig.env` is ignored by git. The template is the recommended flash17
   serving profile, with 4K prefill (it falls back to 2K per piece when memory is short). For q28-v2 copy
   `config/prod-zig.env.example` instead.

2. Build the included source on an ARM64 build host:

   ```bash
   mkdir -p build/zig/src build/zig/fatbins
   git archive HEAD engine/zig | tar -x --strip-components=2 -C build/zig/src/
   # Optionally copy its validated sm_121 *.fatbin files, including tp_mailbox.fatbin, into build/zig/fatbins/.
   touch build/zig/fatbins/.keep
   docker build -f docker/zig/Dockerfile --build-arg ZIG_COMMIT="$(git rev-parse HEAD)" \
     -t tensorfold-dsv41-zig:local build/zig
   scripts/check-zig-image.sh tensorfold-dsv41-zig:local
   docker save tensorfold-dsv41-zig:local -o build/zig-image.tar
   ```

   The runtime stage installs cuBLAS / cuBLASLt 13.6.0.2 (`libcublas-13-3`, held), the release the vision tower's
   exactness was measured on, and its last step checks every library the engine opens at run time (NCCL, verbs and
   its mlx5 provider, cuBLAS by sha256): a missing or different library fails the build. `check-zig-image.sh` runs
   the same kind of check against an image you loaded or pulled.

   The image label records the recipe commit you built (set `ZIG_COMMIT` in the config to the same value); the engine source revision is in `engine/zig/SOURCE-EXPORT.md`. Transfer the tar to rank 1 and run
   `docker load -i build/zig-image.tar` there. Compare the label and root filesystem layers on both nodes with
   `docker image inspect tensorfold-dsv41-zig:local`. Set `IMAGE=tensorfold-dsv41-zig:local` in both configs.

   The multi-stage image uses Zig 0.17.0, CUDA 13, ReleaseSafe ARM64, sm_121 kernels, NCCL and verbs providers.
   Its default nvcc image is CUDA 13.0.1; for prebuilt kernels use the exact CUDA devel/runtime versions validated
   with them (build args `CUDA_DEVEL` / `CUDA_RUNTIME`). Use `ZIG_URL` for the exact Zig compiler tarball if the
   default download URL is unavailable. The first start builds prepared files; do not reuse Python's cache.

3. Check the commands and the AOT set, then run rank 1 on its node followed promptly by rank 0 on its node:

   ```bash
   scripts/run-zig.sh 1 --dry-run    # on rank 1
   scripts/run-zig.sh 0 --dry-run    # on rank 0
   scripts/run-zig.sh 1 --check-aot  # on rank 1: aot/ holds every variant this env launches
   scripts/run-zig.sh 0 --check-aot  # on rank 0
   scripts/run-zig.sh 1             # rank 1: keep running in this terminal
   scripts/run-zig.sh 0             # rank 0: another terminal on its node
   ```

   `--check-aot` runs `tf-dsv41-m1 aot-needs` in the image with this env against the rank's `aot/` (host work only;
   it enumerates every program, about 11 minutes on an x86 desktop CPU) and needs the fill's `sigs.json` in
   `ASSETS_DIR`. The launcher passes the serving settings to both processes, sets TP world/rank/device explicitly,
   and uses Docker `--init` because the follower exits when orphaned. It validates local directories and basic
   assets (including `rope.bin`'s rows against `TF_DSV41_CONTEXT`); it does not replace the engine's memory checks,
   fill validation, or a production watchdog. Keep the same env
   across restarts. A failure on either rank should end both via fail-fast (exit 70).

   To stop, signal rank 0 only and let it drain: on SIGTERM it answers new requests with 503 (`/health` reports
   `draining`), gives requests in progress `TF_DSV41_DRAIN_S` seconds to finish, then halts both ranks at a round
   boundary. Give Docker more time than the drain before it kills, then wait for rank 1 to exit on its node:

   ```bash
   docker stop -t $((DRAIN_S + 60)) tensorfold-zig-r0   # rank 0 node; DRAIN_S = TF_DSV41_DRAIN_S (20)
   docker wait tensorfold-zig-r1                        # rank 1 node
   ```

   The launcher's `--rm` removes both containers once they exit (otherwise `docker rm` them). A second signal
   cuts the drain short.

   Streams that have not sent their first token yet get a keepalive every `TF_DSV41_SSE_KEEPALIVE_S` seconds
   (`: keepalive` comments on OpenAI streams, `ping` events on Anthropic streams), so client or proxy idle
   timeouts do not end long prefills.

4. Wait for the model to load and inspect the boot log: four slots, CED replay, row graphs, successful device-draft
   self-check and RoCE probe. First preparation can take several minutes. Then query rank 0:

   ```bash
   curl -fsS localhost:8000/v1/models
   curl -fsS localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"DeepSeek-V4.1-Flash-TF","messages":[{"role":"user","content":"What is 17 * 23?"}],"temperature":0,"reasoning_effort":"low"}'
   ```

   Check that the answer is 391; do not infer readiness from a live container alone.

## Generate runtime assets

Prepare the already-public Python reference source from the recipe root. Use a GPU-capable ARM64 environment
with Torch, Triton **3.7.1**, and ptxas **13.3.73**, matching the qualified fill toolchain. This stage needs the
model config/tokenizer, but does not require running inference. Set `TRITON_PTXAS_PATH` and
`TRITON_PTXAS_BLACKWELL_PATH` to that compiler executable. Do not use a CPU-only host build for GPU serving.

```bash
mkdir -p build/python-source
git -C vendor/TensorFold archive HEAD | tar -x -C build/python-source
for patch_file in patches/*.patch; do
  patch -d build/python-source -p1 < "$patch_file"
done
export PY_SOURCE="$PWD/build/python-source/src"
export PACK=/srv/models/flash17
export ASSETS=/srv/assets/rank-0
mkdir -p "$ASSETS"
# Run the following in the build image/environment, from engine/zig.
cd engine/zig
python tools/zig/triton_fill.py sigs --py "$PY_SOURCE" --out "$ASSETS/sigs.json"
set -a
source ../../config/prod-zig.env
set +a
# Asset enumeration uses this local destination rather than the serving container mount.
export TF_DSV41_ASSETS="$ASSETS"
export TF_TP_WORLD=2
zig-out/bin/tf-dsv41-m1 aot-needs "$ASSETS/sigs.json" "$ASSETS/needs.json"
python tools/zig/triton_fill.py compile --py "$PY_SOURCE" --needs "$ASSETS/needs.json" \
  --options tools/zig/dsv41_triton_options.json --arch 121 --out "$ASSETS/aot"
# rows default to TF_DSV41_CONTEXT + 2048 from the sourced env (1,050,624 at the 1M context)
PYTHONPATH="$PY_SOURCE" python tools/zig/dsv41_rope_tables.py \
  --config "$PACK/config.json" --out "$ASSETS/rope.bin"
python tools/zig/dsv41_engram_host.py --py-src "$PY_SOURCE" --config "$PACK/config.json" \
  --tokenizer "$PACK/tokenizer.json" --out "$ASSETS/engram-host.bin"
```

For the asset environment, build the builder stage with `docker build --target build -f docker/zig/Dockerfile
-t tensorfold-dsv41-zig:builder build/zig` from the recipe root. Install the matching Torch/Triton toolchain in that
environment. The Docker build's builder stage contains the Zig binaries under `/out/bin`; add them to `PATH` or copy them
into `zig-out/bin` in the asset-generation environment. Mount the public reference, model and asset directories
there. Keep `sigs.json` in the assets directory: `scripts/run-zig.sh N --check-aot` reads it. Repeat/check the
manifest for each rank. Engram host tables do not replace packed
model Engram shards. The existing public reference supports the recommended `R1=1` fill; the optional R1-off
binding against newer Python pruning variants is outside this recipe's qualification.

### Changing serving settings

The AOT set is enumerated from the serving env: prefill rows (`PF_4K`, `PREFILL_CHUNK` / `PREFILL_ROWS`), slots,
`ROWS_CAP`, context, drafting and the knobs `aot-needs` lists. After changing any of them, check the set again and
compile only the missing variants into the existing one, on every rank:

```bash
scripts/run-zig.sh 0 --check-aot   # exit 1 lists the missing variants
# in the asset environment, with the new env sourced as above:
zig-out/bin/tf-dsv41-m1 aot-needs "$ASSETS/sigs.json" "$ASSETS/needs.json"
python tools/zig/triton_fill.py compile --py "$PY_SOURCE" --needs "$ASSETS/needs.json" \
  --options tools/zig/dsv41_triton_options.json --arch 121 --have "$ASSETS/aot" --out "$ASSETS/aot"
```

Variants already in `aot/` are kept and not compiled again. Without this a start stops with `MissingTritonVariant`
and the function and constexprs of the launch it could not find. `rope.bin` also depends on `TF_DSV41_CONTEXT`.

## Serving settings

These are the exact environment names from the serving profile and the port's knob readers, with the measured
all-on additions. **Default means unset engine behavior**, not the template value. The template explicitly opts
into optimizations whose engine defaults remain off. Both ranks need the same settings. `1` means on and `0`
means off unless the row gives another meaning. The recommended column is what the exported template uses;
`verify (unset)` documents an intentional default. This table describes the inspected port snapshot, not generic
upstream TensorFold settings.

| Name | Unset engine default | Recommended profile | What it does |
| --- | --- | --- | --- |
| `TF_DSV41_R1` | 0 | `1` | Complete decode R1 bundle |
| `TF_DSV41_DRAFTS` | 0 | `1` | DSpark drafting |
| `TF_DSV41_OWN_PREFILL` | 0 | `1` | Native prompt prefill rather than decode windows |
| `TF_DSV41_GRAPHS` | 0 | `1` | CUDA graphs for decode windows |
| `TF_DSV41_PREFILL` | full | `replay` | CED encoder prefill and bounded decoder replay, matching Python |
| `TF_DSV41_PROMPT_TAIL` | verify | `verify (unset)` | Last prompt token enters the first decode window |
| `TF_DSV41_PREFILL_PIECES` | 0 | `1` | Interleave prompt pieces with other slots' decode |
| `TF_DSV41_SESSION_SLOTS` | 0 | `1` | Session store with several live slots |
| `TF_DSV41_TAIL_JOIN` | 0 | `1` | Join prompt tail to the first shared verify round; needs PREFILL_PIECES |
| `TF_DSV41_DRAFT_TIMED` | 0 | `1` | Price draft depth from measured round times |
| `TF_DSV41_SLOTS` | 1 | `4` | Concurrent live slots |
| `TF_DSV41_SLOT_DRAFTS` | 0 | `1` | Per-slot drafts with batched verification |
| `TF_DSV41_ROWS_CAP` | 16 | `64` | Maximum rows per decode forward; requires matching AOT |
| `TF_DSV41_POOL_TOKENS` | unset: no paged pool | `2100000` (q28-v2: `3000000`) | Shared paged KV capacity, not capacity per slot ([split KV and the pool](#split-kv-and-the-pool)) |
| `TF_DSV41_KV_SPLIT` | 0 | `1` | Compressed KV rows sharded over the ranks instead of replicated; needs TP >= 2 and `POOL_TOKENS` |
| `TF_DSV41_KV_SPLIT_COMPACT` | 0 | `1` | Split KV's owner-packed exchange where every rank's transport takes device lengths and the shard fits, else dense; `force` packs on any transport |
| `TF_DSV41_KV_SPLIT_UNION_MIB` | 128 | `128 (unset)` | Byte budget of each block of a prefill's union of selected rows under split KV (bounds its exchange and staging) |
| `TF_DSV41_FORMS_INPLACE` | 0 | `1` | Dense weight forms written over their trellis instead of beside it |
| `TF_DSV41_BOOT_TRACE` | 0 | `0 (unset)` | Logs each boot step's memory; for a first boot of a new profile |
| `TF_DSV41_SAMPLING` | 0 | `1` | Keyed sampling for nonzero temperature |
| `TF_DSV41_EXPERT_TOPP` | unset: no pruning | `0.85` | Decode routed-expert top-p pruning |
| `TF_DSV41_INDEX_BUDGET_MIB` | 256 | `64` | Index selection scratch budget |
| `TF_DSV41_PREFILL_CHUNK` | 2048 | `4096` | Segment row limit |
| `TF_DSV41_PREFILL_ROWS` | 2048 | `4096` | Second row limit; smaller of CHUNK and ROWS wins |
| `TF_DSV41_PREFETCH_AHEAD` | 0 | `1` | Read next segment's Engram rows ahead |
| `TF_DSV41_PF_DENSE` | off | `fused` | Fused dense EXL3 prefill GEMM |
| `TF_DSV41_PF_DENSE_TABLE` | unset: heuristic | `/cache/pfdense-table.json` | Shape-specific GEMM tuning table |
| `TF_DSV41_GM_V2` | 0 | `1` | x3gm v2 routed-expert prefill |
| `TF_DSV41_BRANCHES` | 0 | `1` | Indexer, compressor and applicable KV work on a side stream |
| `TF_DSV41_BRANCHES_PRIO` | main | `side` | Side stream priority (eager execution) |
| `TF_DSV41_MHC_DEFER` | 0 | `1` | R1 mHC coefficients on their own stream |
| `TF_DSV41_L2PF` | 0 | `1` | Decode L2 prefetch |
| `TF_DSV41_L2PF_MB` | 12 | `12` | Fixed prefetch budget; other values refused |
| `TF_DSV41_L2PF_PACE_GBPS` | 150 | `150` | Fixed prefetch pacing; other values refused |
| `TF_DSV41_ENGRAM_GATE` | 0 | `1` | GPU waits for decode Engram reads |
| `TF_DSV41_GREEDY_GPU` | 0 | `1` | Device greedy choice and device-started draft round |
| `TF_DSV41_SPEC_DRAFT` | 0 | `1` | Start next speculative DSpark pass after verify |
| `TF_DSV41_DRAFT_GRAPHS` | 0: host path | `1` | Device candidates and graphed drafts; boot self-check |
| `TF_DSV41_ROUTER_GROUP_ROT` | 0 | `1` | Fuse histogram grouping and input rotation |
| `TF_DSV41_MHC_PFDEC` | 0 | `17` | Use prefill mHC kernels for decode windows of at least 17 rows |
| `TF_DSV41_DRAFT_BATCH_PROJ` | 0 | `1` | Project and normalize all slots' main rows together |
| `TF_DSV41_DRAFT_CHAIN` | 0 | `1` | Device draft chain; needs DRAFT_GRAPHS and chain AOT variant |
| `TF_DSV41_DRAFT_OVERLAP` | 0 | `1` | Overlap draft ingests and window tail; needs GREEDY_GPU device start |
| `TF_DSV41_ROUND_GRAPH` | 0 | `1` | Split head/tail graphs and event-based Engram gate |
| `TF_DSV41_KV_NORM_STORE` | 0 | `1` | Fuse KV RMS, SWA RoPE and FP8 store |
| `TF_DSV41_LOOKUP` | 0 | `1` | Copy drafts from repeated token spans |
| `TF_DSV41_ENGRAM_PREFETCH` | 0 | `1` | Prepare decode Engram reads ahead of the forward |
| `TF_DSV41_LOOKUP_PLAN` | 0 | `1` | Cost-priced copy-draft lookup planning |
| `TF_DSV41_PF_4K` | 0 | `1` | 4096-row prefill workspace; may fall back to 2K |
| `TF_DSV41_PF_4K_MEMORY_OK` | unset | `1` | Required acknowledgment for 4K memory cost |
| `TF_DSV41_PF_WS_FLOOR_GIB` | FLOOR_HARD_GIB (4) | `4.75` (q28-v2: `3.5`) | Memory remaining after workspace allocation on every rank |
| `TF_DSV41_PIECE_RUNS` | 0 | `1` | Combine several slots' prompt pieces in one run; needs PREFILL_PIECES and replay |
| `TF_DSV41_REPLAY_RUNS` | 0 | `1` | Several slots' CED decoder replays in one run |
| `TF_DSV41_JOIN_DRAFTS` | 0 | `1` | Prompts finishing in one round draft their first tokens in one pass |
| `TF_DSV41_PF_WS_SHARE` | 0 | `1` | 4K workspace shares the index-stream scratch instead of a second copy |
| `TF_DSV41_LIN2` | 0 | `1` | Dense linears of 17-64-row decode windows in one pass |
| `TF_DSV41_INDEX_BOUND` | 0 | `1` | Row-bounded index scores and decode top-k; needs the twins in the AOT fill |
| `TF_DSV41_MHC_DEFER_AT` | unset | `exchange` | Issue deferred mHC coefficient launches just before their exchange |
| `TF_DSV41_DRAFT_CAPTURE` | 0 | `1` | Capture draft graphs with the round |
| `TF_DSV41_GRAPH_FLOOR` | shed | `hold` | Under the memory floor, run a missing graph eagerly and keep the held ones |
| `TF_DSV41_ENGRAM_AIO` | 0 | `1` | Native async reads for Engram rows |
| `TF_DSV41_CODE_ACCEPT` | 0 | `0` | Expand high-acceptance short-context draft chains; off |

### Data, serving and transport settings

| Name | Unset engine default | Recommended profile | What it does |
| --- | --- | --- | --- |
| `TF_DSV41_CONTEXT` | 4096 | `1048576` | Slot position limit, not a promise that every slot can fill it together |
| `TF_DSV41_ASSETS` | unset | `/assets` | Per-rank rope, host Engram lookup and AOT programs |
| `TF_DSV41_INDEX_KV` | FP8 fixed in this port | `fp8` | Compatibility setting; does not select a different Zig index type |
| `TF_DSV41_ENGRAM_DIR` | unset | `/engram` | Packed per-rank Engram shards |
| `TF_DSV41_CALIB` | cached | `zig` | Engine's own boot measurement, stored and reused; `real` / `cached` read Python's stored tables |
| `TF_DSV41_CALIB_DIR` | user cache directory | `/state/calib-zig` | Directory of stored cost tables |
| `TF_DSV41_CALIB_FILE` | unset | `unset (optional)` | Pin one stored table after the first boot measured it |
| `TF_DSV41_PREPARED` | unset | `/prepared` | Zig prepared layer images (different from Python prepared folders) |
| `TF_DSV41_PREPARED_WRITE` | 0 | `1` | Write missing prepared files |
| `TF_DSV41_PLAN_TIMEOUT` | 300 | `1800` | Seconds allowed for TP rendezvous connection |
| `TF_DSV41_PLAN_LINK` | blocking socket | `rdma` | Spin on the plan socket before blocking; 5000 μs when rdma |
| `TF_DSV41_PLAN_PIN` | off | `1`, a CPU inside `CPUSET_CPUS` (q28-v2: `auto`) | Pin plan threads to that CPU (`auto`: the best available CPU) |
| `TF_CUDA_SCHED` | driver `auto` | `blocking` (q28-v2: unset) | How the host waits for the GPU; `blocking` sleeps instead of spinning (same launches and bits) |
| `TF_DSV41_DEPTH_JOINT` | 1 | `0` | Disable joint draft depth policy, matching the Python reference |
| `TF_DSV41_SESSIONS` | 0 | `1` | Session reuse; SESSION_SLOTS additionally required at four slots |
| `TF_DSV41_SESSION_RAM_MIB` | 256 | `256` | Bounded session state RAM tier |
| `TF_DSV41_SESSION_DISK` | unset: RAM only | `/sessions` | NVMe session tier |
| `TF_DSV41_SESSION_DISK_GIB` | 64 | `128` | NVMe tier file budget |
| `TF_DSV41_SESSION_DISK_MIN` | 1024 | `1024` | Minimum token count for parking a session |
| `TF_DSV41_FLOOR_GIB` | max(5, hard floor) | `5` | Target MemAvailable in GiB per rank |
| `TF_DSV41_FLOOR_HARD_GIB` | 4 | `4` | Hard admission floor in GiB |
| `TF_DSV41_REQUEST_LOG` | unset: off | `/state/requests.jsonl` | Request metadata log |
| `TF_DSV41_DISCONNECT` | 1 | `1` | Cancel generation when client disconnects |
| `TF_DSV41_DRAIN_S` | 20 | `20` | Seconds requests in progress may finish after SIGTERM before both ranks halt |
| `TF_DSV41_SSE_KEEPALIVE_S` | 15 | `15` | Seconds between keepalives on a stream before its first token (0: none) |
| `TF_DSV41_THINKING` | 1 | `1` | Default thinking mode (`run-zig.sh` passes `--no-thinking` for 0); requests can override |
| `TF_DSV41_DEFAULT_EFFORT` | high | `high` | Default effort (high = 75) |
| `TF_DSV41_SCHEMA_PROMPT` | 1 | `1` | Include response-format schema in the prompt |
| `TF_DSV41_IMAGES` | placeholder | `native` | Image input: DeepSeek's vision tower on rank 0 ([IMAGES](IMAGES.md#the-zig-serving-path)); `placeholder` turns each image into a notice, `reject` answers 400 |
| `TF_DSV41_BIAS_VL` | unset: image rows route with the text bias | `/model` (q28-v2: `/cache/dsv41-bias-vl`) | Folder holding the release's `gate.bias_vl` tensors (`bias_vl.safetensors`), loaded on both ranks; boot fails if it names a folder without them |
| `TF_DSV41_GRAMMAR` | 0 | `1` | Grammar-constrained structured output |
| `TF_DSV41_TOOL_GRAMMAR` | 0 | `0` | Automatic tool-schema constraint; remains off |

| Name | Unset engine default | Recommended profile | What it does |
| --- | --- | --- | --- |
| `GLM53_TF_ROCE_MAX_KB` | 256 | 1024 | RoCE message cap; keeps larger multi-slot windows on RoCE |
| `GLM53_TF_ROCE_FAST` | 0 | 1 | Fast RoCE gather path |
| `TF_TP_WORLD` | 1 | 2 (launcher) | Number of TP ranks |
| `TF_TP_RANK` | 0 | 0 or 1 (launcher) | This node's rank |
| `TF_TP_DEVICE` | 0 | 0 (launcher) | Local CUDA device |
| `TF_TP_MASTER` | loopback | your rank 0 link address | TP rendezvous address |
| `TF_TP_PORT` | 29500 | 29571 | Rendezvous port; plan port defaults to this plus 7 |
| `TF_COMM_BACKEND` | nccl | roce | Tensor exchange transport |
| `TF_TP_KERNEL_IMAGE` | unset | image's tp_mailbox.fatbin (launcher) | TP kernel image |
| `NCCL_SOCKET_IFNAME` | library auto-selection | your link interface | NCCL network interface |
| `NCCL_IB_HCA` | library auto-selection | your cabled RDMA devices | NCCL/RoCE HCA selection |

`IMAGE`, `MODEL_DIR`, `ENGRAM_DIR`, `ASSETS_DIR`, `PREPARED_DIR`, `STATE_DIR`, and `CACHE_DIR` are required
launcher inputs, not engine knobs. `HOST=localhost` and `PORT=8000` are launcher defaults for the HTTP listener.
The maximum reply is 32768 tokens and the served model ID is `DeepSeek-V4.1-Flash-TF` in this small launcher.

### 4K prefill and its fallback

The profile sets **all** of `TF_DSV41_PF_4K=1`, `TF_DSV41_PF_4K_MEMORY_OK=1`, `TF_DSV41_PREFILL_CHUNK=4096`, and
`TF_DSV41_PREFILL_ROWS=4096`, with `PF_WS_SHARE=1`. It requires `GM_V2=1`, `PREFILL=replay`, and an AOT fill made from
the same env. The 4K workspace is transient and released after the CED finish. Each prompt piece checks headroom
across both ranks; if either cannot leave `TF_DSV41_PF_WS_FLOOR_GIB` free, both use 2K segments with the same bits.
The boot budget can also refuse a profile that cannot fit.

`TF_DSV41_PF_WS_FLOOR_GIB` defaults to the hard floor (4 GiB). The flash17 profile uses 4.75 GiB, which keeps a
planned 4K workspace above the 4 GiB floor; q28-v2's profile keeps 3.5 GiB, which trades that margin for fewer 2K
fallbacks. The guard ([OPERATIONS](OPERATIONS.md#zig-serving-thermal-and-memory-guard)) refuses a floor under 4. Under `PF_4K=1`, `aot-needs` also lists the
2,048-row segment programs this fallback runs; a fill made from a 4K env with an older `aot-needs` lacks them and
should be completed as in [Changing serving settings](#changing-serving-settings). For a conservative 2K profile set
`PF_4K=0`, drop `PF_4K_MEMORY_OK` and `PF_WS_FLOOR_GIB`, and use 2048 for `PREFILL_CHUNK` / `PREFILL_ROWS`. Do not
enable two-batch prefill overlap (PF_TBO) alongside 4K without a separate memory gate: the tested combination
exhausted memory.

### Split KV and the pool

Without split KV every rank holds every compressed KV row, so the pool is bounded by one node's free memory. With
`TF_DSV41_KV_SPLIT=1` each rank holds its share of the rows and the ranks exchange the selected rows each round, about
1,060 bytes a token on each rank. That is what lets q28-v2's `TF_DSV41_POOL_TOKENS=3000000` boot where the replicated
layout fits about 1.4M (flash17's profile uses 2,100,000 to keep the 4 GiB floor under 4K prefills;
[FLASH-1017](FLASH-1017.md#memory-the-21m-pool)): one slot can fill its whole `TF_DSV41_CONTEXT` (1M), two slots 1M
each, four live slots about 750K each. The pool is shared, so these are totals across live slots, not per-slot
reservations. `TF_DSV41_FORMS_INPLACE=1` frees the dense forms' separate copy for the pool, and
`TF_DSV41_KV_SPLIT_COMPACT=1` sends each rank's packed rows instead of the dense table where the transport allows it.

The boot check prices the pool against each rank's measured memory at the end of boot and refuses a pool that would
leave the worst predicted dip (`TF_DSV41_SERVE_DIP_GIB`, default 3.0 GiB) under the hard floor, so an oversized
`POOL_TOKENS` fails at boot, not under load. When live slots need the pool, admission parks idle sessions from RAM to
the NVMe tier (`TF_DSV41_SESSION_DISK`); a parked chat's next prompt resumes it from disk, and a restore that fails on
any rank is a cache miss: the prompt is prefilled from 0 and serving continues. Split KV costs some decode speed for the
exchange (see [ZIG-RESULTS](ZIG-RESULTS.md#split-kv-and-the-30m-pool)). For a speed-first profile without long contexts,
drop `KV_SPLIT` and `KV_SPLIT_COMPACT` and set `POOL_TOKENS=1400000`.

`PIECE_RUNS`, `REPLAY_RUNS` and `JOIN_DRAFTS` batch several slots' prompt work into one run. They need
`PREFILL_PIECES=1`, replay mode and the multi-segment AOT programs `aot-needs` lists under them.
Do not equate the serving profile with enabling every experimental knob: the engine has more default-off paths
than the profile uses.
