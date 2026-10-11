# Source export provenance

This is a sanitized source snapshot based on TensorFold 1.0.2, upstream revision
`f8fe17d24629aedabf90bbf78279dd776e6d62e7`, from
[ashhart/TensorFold](https://github.com/ashhart/TensorFold). The DeepSeek-V4.1 source snapshot revision is
`9131e10ad17dde51e94a996da8255227727cd224`, the engine source of the current serving kit. The previous snapshots were
`74ace4be5cc46f2f7444011433c6fad53ea679a6`, `8e96f99cc27743bb80f4827ea78b30481e3b4bcc`,
`55c96c65bf97b8c30455f171e01aec8c59544aeb` and `0d723a8275f5b7879a083e8042a7048242ee624c`.

## Changes since 74ace4be

Every new engine path below is default-off unless noted.

- **TP RoCE:** every RDMA post is bounds-checked against its slot and memory region before the verb; no RDMA write
  goes past 1 MiB, and boot refuses a setting that would ask for one (on); `TF_TP_ROCE_CHUNKED_KB` gathers strides
  past 1 MiB in ops of at most 1 MiB, with an ABI marker the transport fatbin must carry; a failed transport halts its
  queue pairs, drains them and keeps its region registered until exit (on); `TF_TP_ROCE_VAR_KB` routes
  variable-length gathers past the small-op cap over RoCE.
- **Split KV:** `TF_DSV41_KV_SPLIT_DEDUP` sends a slot's row once, `TF_DSV41_KV_SPLIT_EARLY` issues the exchange beside
  the q group, `TF_DSV41_KV_SPLIT_HISTORY` exchanges a prefill union without a plan or a host sync; split KV runs with
  four session slots against its own store and pool.
- **Memory:** `TF_DSV41_ARENA=pack` packs boot's permanent buffers back to back and falls back to plain allocations
  (saying why) where virtual memory management fails; `TF_DSV41_STREAM_LAZY` holds the stream top-k's scratch only
  while prompts stream; `TF_DSV41_POST_BOOT_GIB` prices post-boot one-time cost apart from the serving dip;
  `TF_DSV41_BOOT_TRACE` also logs the kernel memory each boot step grew; `TF_DSV41_EX_STACKS` reads the experts'
  suh / svh stacks where the weights hold them.
- **Images:** served models switch images on before the buffer plan is sized, and boot refuses a plan that lacks an
  image segment's roles (before, the first image request failed with `Unbound`); a window holding an image position
  fails loudly, since a window's embedding fills no image rows; the aligner's first linear reads the unfold's
  transposed view, as torch's addmm does, which makes the vision tower bit-exact with the Python reference.
- **AOT:** a missing Triton variant fails its request instead of both ranks (the window is checked before any launch;
  on, `TF_DSV41_AOT_GUARD=0` turns it off); `TF_DSV41_AOT_TRACE` writes every lookup; `aot-needs` prices the pool as
  the served engine does.
- **Serving:** the boot log names the depth policy in effect.
- **Tools and tests:** a tower-only vision gate and a 30-image shape sweep; a transport fail-fast check on real HCAs;
  host tests for the RoCE bounds and cap, the image plan check, the AOT guard and the post-boot reserve.

## Changes since 8e96f99c

Every new engine path below is default-off unless noted; `config/prod-zig.env.example` opts into the ones the
serving profile uses.

- **Split KV:** a row window exchanges dense at any row count; row mode runs with split KV; a prefill's union of
  selected rows is planned without a sort (a row bitmap and an index map), and its selection and uploads go through
  pinned staging; a prefill workspace past the worst point is a warning, not a refusal.
- **Memory:** under CED replay (`TF_DSV41_PREFILL=replay`) the prefill programs are planned without the head they never
  run, so `w.logits` is no longer sized for 2,048 prefill rows (505 MiB a rank planned); rope.bin's host copy is freed
  once its tables are uploaded; `TF_DSV41_FORMS_INPLACE` writes the dense forms over their trellis instead of beside it;
  `TF_DSV41_BOOT_TRACE` logs each boot step's memory. The KV boot check uses the engine's own figures, measured at the
  end of boot, with a default serving dip of 3.0 GiB (`TF_DSV41_SERVE_DIP_GIB`).
- **Graph caches:** a capture is priced instead of held to a fixed 5 GiB floor; the memory floor's transitions are
  logged and `/health` shows each cache's held, captured, replayed and eager counts.
- **Serving:** `/health` reports the engine's round pace and queue and answers 503 `stalled` when rounds stop; a
  drain's in-stream error carries the 503's code and type; admission waits and long prefills' progress are logged;
  a prompt whose client leaves mid-prefill is kept for its retry.
- **Sessions:** a session resumed from the NVMe tier takes its token history from the prompt instead of the RAM index,
  which no longer holds it; a restore that fails on any rank is an agreed cache miss on every rank (the entry dropped,
  the prompt prefilled from 0) instead of an error that stopped the server. Transport, device and memory errors still
  stop it.
- **Hash maps:** the Engram row cache, the CUDA arena's address maps, the vision id registry and the session page map
  are rehashed before removal markers fill them (always on).
- **Images:** the vision attention's rescaled row sum and its first add are one FMA, as in FlashAttention-2; separate
  multiply and add rounded twice and let the fp32 row sum drift by an ulp after a rescaled block.
- **Kit tools:** `kit-fatbins` refuses a vision fatbin built from another `vision.cu`; `aot-needs` lists the image
  segments' variants under `TF_DSV41_IMAGES=native`.
- New host tests for the union plan, the boot check, the boot trace, in-place forms, the health pace, the rehashes, the
  NVMe resume and the agreed restore miss; the hang-up and drain tests hold their request until their checks are done.

## Changes since 55c96c65

- **Serving:** SIGTERM / SIGINT drains: new requests get 503 with `Retry-After` and `/health` reports `draining`,
  requests in progress get `TF_DSV41_DRAIN_S` seconds (default 20), then both ranks halt at a round boundary; a second
  signal cuts the drain short. A stream that has not sent its first token gets a keepalive every
  `TF_DSV41_SSE_KEEPALIVE_S` seconds (default 15): `: keepalive` comments on OpenAI streams, `ping` events on
  Anthropic streams. The API key store and per-key reply counts are freed when the server stops.
- **CUDA:** a failed kernel launch names its kernel and launch configuration; the server logs the signal that stopped it.
- **Engram:** the row cache's record map is rehashed in place once removal markers reach a quarter of its capacity.
  Before, the markers were never cleaned up, so after a long uncached prefill every lookup scanned the whole map.
- **TP runtime:** `TF_DSV41_KEEP_BATCH`, `TF_DSV41_PT_PINNED` and `TF_DSV41_PIN_ISOLATE` are removed (default off,
  measured no gain); `TF_DSV41_WIN_PROF` times the Engram gate's arm in parts.
- **Bench tools:** per-stream content-chunk arrival times (`bench_http.py --times`), a side-by-side comparison of two
  such cells (`timescmp.py`), and a per-round kernel comparison of two profiler exports (`kcmp.py`).
- New host tests for the drain (including a stop between a request's count and its drain check), the keepalive and
  the rehash.

## Changes since 0d723a82

Every new engine path below is default-off; `config/prod-zig.env.example` opts into the ones the measured serving
profile uses.

- **Prefill:** the 4K workspace shares the index-stream scratch (`TF_DSV41_PF_WS_SHARE`); piece all-gathers write into
  the layer rows, with optional side-stream exchange pieces (`TF_DSV41_PF_OVERLAP_SITE`); row-blocked and prefetching
  stream top-k twins (`TF_DSV41_STREAM_RB`); x3gm gate/up at 4K segments (`TF_DSV41_GM_GU2`) and bounded prefill mHC
  sites (`TF_DSV41_MHC_SITE_ROWS`); CED replay keeps its stash ring in place across piece and replay runs' slot switches.
- **Decode and drafting:** one-pass dense linears for 17-64-row windows (`TF_DSV41_LIN2`); row-bounded index scores and
  decode top-k (`TF_DSV41_INDEX_BOUND`); fused MoE expert epilogues (`TF_DSV41_X3LD_EPI`); mHC coefficient launches
  deferred to just before their exchange (`TF_DSV41_MHC_DEFER_AT=exchange`, `TF_DSV41_COEF_LATE`); draft graph capture
  with the round (`TF_DSV41_DRAFT_CAPTURE`); graphs kept under the memory floor (`TF_DSV41_GRAPH_FLOOR=hold`); the
  engine's own boot calibration of draft costs (`TF_DSV41_CALIB=zig` / `measure`, `TF_DSV41_CALIB_FILE`).
- **TP runtime:** host step profiling (`TF_DSV41_WIN_PROF`). (This snapshot also added `TF_DSV41_KEEP_BATCH`,
  `TF_DSV41_PT_PINNED` and `TF_DSV41_PIN_ISOLATE`; they are removed since 55c96c65.)
- **Serving:** tool-call keys, invoke names and argument values, and queued Anthropic text, are copied instead of
  borrowed from a growing buffer; the JSON writer is bounds-safe on a UTF-8 sequence cut off at a string's end; nested
  completion prompt ids are range-checked; calls to tools the request did not offer are surfaced as `tool_calls`
  (`Tools.unknown_calls`, default true); a reply with DSML markup but no call is logged once.
- **Bench tools:** a mixed-window cell (`bench_http.py --mixwin`) whose long-stream rate counts tokens
  (`long_live_tok_s`), not decode rounds.
- New host tests for each of the above, GPU micro-benches for the new kernels, and their AOT fill inputs.

The source derives from the public upstream tree and is added through logical source commits. Development
history is not imported. Source changes include removal of infrastructure references, conversion of
machine-specific benchmark configuration to environment inputs, and default-off controls for the new
family's drafting, native prefill, lookup, session retention, and Engram prefetch.

Apache-2.0 license headers, attribution, `LICENSE`, `NOTICE`, third-party notices, and applicable
third-party license texts are retained. Weight and dataset licenses remain separate.

Excluded material includes performance diaries and infrastructure notes; private benchmark results
and logs; profiler captures; databases; packaged kits; caches; scratch files; generated build outputs;
and large generated binary goldens. Small checked-in source fixtures remain; no huge binary fixtures are included. The source tools generate
checkpoint-dependent fixtures and AOT assets locally; the public serving recipe documents asset
generation. No checkpoint weights or generated AOT bundle are shipped in this source export.

The export is intended to make the source inspectable and buildable. Historical device receipts are
identified separately from checks on the exported tree. Host compilation does not qualify hardware,
models, arithmetic, or precision choices. See [the CUDA guide](docs/DEEPSEEK-V41-CUDA.md) and the pull
request for build/test outcomes, unrun gates, and the outstanding upstream lean-rule cleanup.
