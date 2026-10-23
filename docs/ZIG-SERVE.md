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
- Writable prepared and session directories on local NVMe. Plan roughly 204 GB for the full weights pack on each
  node, 95 GiB packed Engram shards, 95 GiB rank-specific prepared weights, and the template's 32 GiB session cache,
  plus image/build space. The engine's default target/hard MemAvailable floors are 5/4 GiB; this template explicitly
  opts into 1/1 GiB and disables the measured boot guard. Check actual memory headroom on both ranks.

The source kit's `tools/zig/dsv41_m2c2/build-kit.sh` builds the ARM64 binaries and checks the sm_121 fatbins.
It generates a `knob-names.txt` from the names actually read by Zig. Asset generation uses
`tools/zig/dsv41_rope_tables.py`, `dsv41_engram_host.py`, `triton_fill.py`, the engine's `aot-needs` command and
`dsv41_m4/fill-prod.env` (or the 4K fill). Python reference source, Torch, Triton 3.7.1 and ptxas 13.3.73 are
needed for that fill. Copy assets per rank and validate the fill manifest; a `MissingTritonVariant` is a missing
prerequisite, not a reason to keep running. These generator tools are included in `engine/zig/tools/zig`.
The AOT set depends on the serving settings: see [Changing serving settings](#changing-serving-settings).

## Quick start

The source, build recipe and launcher are included. Generate the assets below before serving. This fork's native
flash17 path has been exercised on two DGX Sparks, including 4K prefill and SSD session recovery; the upstream
performance tables are not new benchmark results for this fork.

1. Clone this public recipe and download the pack on both nodes:

   ```bash
   git clone --recurse-submodules https://github.com/0rand/deepseek-v41-tensorfold-spark.git
   cd deepseek-v41-tensorfold-spark
   hf download jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw --local-dir /srv/tensorfold/models/flash17
   cp config/prod-zig-flash17.env.example config/prod-zig.env
   $EDITOR config/prod-zig.env
   ```

   The generic flash17 template uses `/srv/tensorfold`, head link address `192.0.2.11`, worker link address
   `192.0.2.12`, and SSH user `operator`; replace these with the deployment's values. Configure the same serving
   settings on both nodes, with each node's local model, rank-specific Engram/assets, prepared, state and cache data.
   On the head, `WORKER_SSH`, `WORKER_REPO`, and `WORKER_CONFIG` must name the worker's reachable SSH target, matching
   checkout, and its config file. That local trusted config is the only place `scripts/zig-stack.sh` gets remote
   location data. Copy the worker's config to precisely `WORKER_CONFIG` before using the stack script.

   Copy the dense-prefill tuning table into each node's `CACHE_DIR` (`cp config/pfdense-table.json "$CACHE_DIR/"`;
   the launcher does it when missing). `config/prod-zig.env` is ignored by git. The template is the recommended
   flash17 serving profile, with 4K prefill (it falls back to 2K per piece when memory is short). For q28-v2 copy
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

3. Check the commands and AOT set on their respective nodes, then manage both ranks from the head:

   ```bash
   # run on the worker
   CONFIG=/srv/tensorfold/deepseek-v41-tensorfold-spark/config/prod-zig.env scripts/run-zig.sh 1 --dry-run
   CONFIG=/srv/tensorfold/deepseek-v41-tensorfold-spark/config/prod-zig.env scripts/run-zig.sh 1 --check-aot

   # run on the head
   scripts/run-zig.sh 0 --dry-run
   scripts/run-zig.sh 0 --check-aot
   scripts/zig-stack.sh start
   scripts/zig-stack.sh status
   ```

   `zig-stack.sh start` is idempotent when both containers are already running. Otherwise it creates rank 1 over
   BatchMode SSH first and rank 0 locally, using `docker run --restart unless-stopped` without `--rm`; Docker starts
   these containers after a daemon/host reboot unless they were manually stopped. A successful `start` only means both
   containers were created, **not** that the model is usable: model loading and the first preparation continue in the
   background. Wait for `zig-stack.sh status` to return a successful `/v1/models` response before sending traffic;
   it reports an unavailable endpoint separately from container state.

   `run-zig.sh` still defaults to the original foreground `--rm` mode for direct debugging; use
   `scripts/run-zig.sh N --detach` only when deliberately creating one managed local rank. `--check-aot` runs
   `tf-dsv41-m1 aot-needs` in the image with this env against the rank's `aot/` (host work only; it enumerates every
   program, about 11 minutes on an x86 desktop CPU) and needs the fill's `sigs.json` in `ASSETS_DIR`. The launcher
   passes the serving settings to both processes, sets TP world/rank/device explicitly, and uses Docker `--init`
   because the follower exits when orphaned. It validates local directories and basic assets (including `rope.bin`'s
   rows against `TF_DSV41_CONTEXT`); it does not replace the engine's memory checks, fill validation, or a production
   watchdog. Keep the same env across restarts. A failure on either rank should end both via fail-fast (exit 70).

   To apply config or image changes, recreate the pair rather than Docker-restarting one rank:

   ```bash
   scripts/zig-stack.sh restart   # drain rank 0, stop rank 1, then recreate worker first and head second
   scripts/zig-stack.sh logs      # both ranks; use logs 0 or logs 1 for one rank
   scripts/zig-stack.sh stop      # intentional stop: both remain stopped across Docker daemon restarts
   ```

   `stop` disables both restart policies, lets rank 0 receive SIGTERM and drain for
   `TF_DSV41_DRAIN_S`, then stops any remaining follower. It deliberately leaves restart disabled: a follower that
   exits naturally during the head's drain is not recorded as manually stopped by Docker. The next `start` recreates
   both containers with `unless-stopped`. `ZIG_STOP_TIMEOUT` defaults to, and may not be lower than, drain plus
   60 seconds. Do not use `docker restart` on one rank; use `zig-stack.sh restart` for a synchronized recreate.
   The same `restart` migrates prior foreground `--rm` ranks: it drains/removes them, then creates managed containers.
   No systemd unit or watchdog is required for boot autostart; the Docker daemon must be enabled on both nodes.

4. Wait for the model to load and inspect the boot log: four slots, CED replay, row graphs, successful device-draft
   self-check and RoCE probe. First preparation can take several minutes. Then query rank 0:

   ```bash
   curl -fsS localhost:8100/v1/models
   curl -fsS localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"deepseek-v41-flash","messages":[{"role":"user","content":"What is 17 * 23?"}],"temperature":0,"reasoning_effort":"low"}'
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
| `TF_DSV41_PF_WS_FLOOR_GIB` | FLOOR_HARD_GIB (4) | `3.5` | Memory remaining after workspace allocation on every rank |
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
| `TF_DSV41_BOOT_MEASURED` | auto | `0` | Do not apply the measured-boot memory gate in this explicit profile |
| `TF_DSV41_PLAN_LINK` | blocking socket | `rdma` | Spin on the plan socket before blocking; 5000 μs when rdma |
| `TF_DSV41_PLAN_PIN` | off | `auto` | Pin plan threads to the best available CPU |
| `TF_DSV41_DEPTH_JOINT` | 1 | `0` | Disable joint draft depth policy, matching the Python reference |
| `TF_DSV41_SESSIONS` | 0 | `1` | Session reuse; SESSION_SLOTS additionally required at four slots |
| `TF_DSV41_SESSION_RAM_MIB` | 256 | `256` | Bounded session state RAM tier |
| `TF_DSV41_SESSION_DISK` | unset: RAM only | `/sessions` | NVMe session tier |
| `TF_DSV41_SESSION_DISK_GIB` | 64 | `32` | NVMe tier file budget |
| `TF_DSV41_SESSION_DISK_MIN` | 1024 | `0` | No disk-tier length threshold; CED's intrinsic 64-token, grid-16 snapshot rule still applies |
| `TF_DSV41_SESSION_DISK_WRITE_THROUGH` | 0 | `1` | Commit every eligible snapshot to the persistent tier before its store save returns |
| `TF_DSV41_FLOOR_GIB` | max(5, hard floor) | `1` | Explicit less-conservative target MemAvailable reserve in GiB per rank |
| `TF_DSV41_FLOOR_HARD_GIB` | 4 | `1` | Explicit less-conservative hard admission reserve in GiB |
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
launcher inputs, not engine knobs. `HOST=localhost` and `PORT=8000` remain the unset launcher defaults; the flash17
template explicitly uses `127.0.0.1:8100`. `MODEL_NAME` controls the served model ID and defaults to
`deepseek-v41-flash`. `WORKER_SSH`, `WORKER_REPO`, `WORKER_CONFIG`, and `ZIG_STOP_TIMEOUT` belong to
`zig-stack.sh`'s trusted head-side lifecycle configuration, not to the engine.

### 4K prefill and its fallback

The profile sets **all** of `TF_DSV41_PF_4K=1`, `TF_DSV41_PF_4K_MEMORY_OK=1`, `TF_DSV41_PREFILL_CHUNK=4096`, and
`TF_DSV41_PREFILL_ROWS=4096`, with `PF_WS_SHARE=1`. It requires `GM_V2=1`, `PREFILL=replay`, and an AOT fill made from
the same env. The 4K workspace is transient and released after the CED finish. Each prompt piece checks headroom
across both ranks; if either cannot leave `TF_DSV41_PF_WS_FLOOR_GIB` free, both use 2K segments with the same bits.
The boot budget can also refuse a profile that cannot fit.

`TF_DSV41_PF_WS_FLOOR_GIB` defaults to the hard floor (4 GiB). The flash17 profile explicitly chooses `1.0` GiB,
alongside its 1 GiB admission floors, to keep the tested 4K path available. This is less conservative: the workspace
still checks both ranks and falls back to 2K segments under pressure. Under `PF_4K=1`, `aot-needs` also lists the
2,048-row segment programs this fallback runs; a fill made from a 4K env with an older `aot-needs` lacks them and
should be completed as in [Changing serving settings](#changing-serving-settings). For a conservative 2K profile set
`PF_4K=0`, drop `PF_4K_MEMORY_OK` and `PF_WS_FLOOR_GIB`, and use 2048 for `PREFILL_CHUNK` / `PREFILL_ROWS`. Do not
enable two-batch prefill overlap (PF_TBO) alongside 4K without a separate memory gate: the tested combination
exhausted memory.

The workspace floor is separate from `TF_DSV41_FLOOR_GIB` and the measured boot-budget policy. Changing only
`TF_DSV41_PF_WS_FLOOR_GIB` does not change AOT shapes. Confirm actual workspace allocation in both rank logs, not
just the configured `PREFILL_ROWS`.

### Split KV and the pool

Without split KV every rank holds every compressed KV row, so the pool is bounded by one node's free memory. With
`TF_DSV41_KV_SPLIT=1` each rank holds its share of the rows and the ranks exchange the selected rows each round, about
1,060 bytes a token on each rank. That is what lets q28-v2's `TF_DSV41_POOL_TOKENS=3000000` boot where the replicated
layout fits about 1.4M (flash17's profile uses 2,100,000 with its explicit 1 GiB reserves and 4K prefills;
[FLASH-1017](FLASH-1017.md#memory-the-21m-pool)): one slot can fill its whole `TF_DSV41_CONTEXT` (1M), two slots 1M
each, four live slots about 750K each. The pool is shared, so these are totals across live slots, not per-slot
reservations. `TF_DSV41_FORMS_INPLACE=1` frees the dense forms' separate copy for the pool, and
`TF_DSV41_KV_SPLIT_COMPACT=1` sends each rank's packed rows instead of the dense table where the transport allows it.

With `TF_DSV41_BOOT_MEASURED=auto`, the boot check prices the pool against each rank's measured memory and refuses a
pool that would leave the worst predicted dip (`TF_DSV41_SERVE_DIP_GIB`, default 3.0 GiB) under the hard floor. The
flash17 profile explicitly sets `TF_DSV41_BOOT_MEASURED=0`, so it does not add that measured-boot gate; its configured
pool must still fit the actual deployment. When live slots need the pool, admission parks idle sessions from RAM to
the NVMe tier (`TF_DSV41_SESSION_DISK`); a parked chat's next prompt resumes it from disk, and a restore that fails on
any rank is a cache miss: the prompt is prefilled from 0 and serving continues. Split KV costs some decode speed for the
exchange (see [ZIG-RESULTS](ZIG-RESULTS.md#split-kv-and-the-30m-pool)). For a speed-first profile without long contexts,
drop `KV_SPLIT` and `KV_SPLIT_COMPACT` and set `POOL_TOKENS=1400000`.

`PIECE_RUNS`, `REPLAY_RUNS` and `JOIN_DRAFTS` batch several slots' prompt work into one run. They need
`PREFILL_PIECES=1`, replay mode and the multi-segment AOT programs `aot-needs` lists under them.
Do not equate the serving profile with enabling every experimental knob: the engine has more default-off paths
than the profile uses.

### Persistent SSD sessions

For a 32 GiB per-rank tier that writes eligible snapshots immediately rather than waiting for RAM eviction:

```bash
TF_DSV41_SESSIONS=1
TF_DSV41_SESSION_DISK=/sessions
TF_DSV41_SESSION_DISK_GIB=32
TF_DSV41_SESSION_DISK_MIN=0
TF_DSV41_SESSION_DISK_WRITE_THROUGH=1
TF_DSV41_SESSION_RAM_MIB=256
```

Mount `/sessions` on persistent local NVMe on each rank. Write-through waits for the snapshot file's commit
before the store save returns and retains the RAM copy subject to the existing RAM/residency policy.
It does not require pool pressure or a high-water trigger. Turning write-through off restores eviction-driven writes.

`SESSION_DISK_MIN=0` removes the disk tier's minimum length, not the native snapshot eligibility rule:
CED saves prompt prefixes on a 16-token grid from position 64 onward. It does not persist arbitrary tiny
turns or checkpoint every generated token.

Startup indexes compatible files, intersects recovered keys across ranks and assigns stable entry IDs
independently of filesystem timestamps. A later matching prompt restores its saved prefix on demand;
the entire SSD tier is not preloaded at boot. Model weights must still load before serving.
Keep session directories private to the exact weight pack and serving mode; layout/rank checks are not
a substitute for checkpoint isolation.

`/health.spill` and `tensorfold:spill_*` report the budget, file bytes, writes, restores, token counts,
restore latency and failures. Byte counts are per rank. Validate with a real prompt, RAM reuse,
then a full restart and an NVMe prefix hit; a live `/health` alone does not prove cache recovery.

