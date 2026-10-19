# What is not solved

- **4 streams on the mixed cell are below the 120 target.** C4 mixed decode aggregate is ~101 tok/s (136 while all
  four streams are live). The cell's wall is mostly its prose stream finishing alone; 4 distinct code streams are
  estimated at ~125-135 but that cell has not been measured. The built-but-off `TF_DSV41_SPEC_NUCLEUS` measured
  +1.3% on C2 steady and -0.6% on C4 steady: not adopted.
- **Prose drafting.** DSpark keeps ~1.7 tokens a round on prose (3.7 on code), so prose (1.44x) and 2 streams (1.59x)
  are not at 2x. Self-distilling the drafter on our own logs helped prose and hurt code; not shipped.
- **The graph cache still shrinks under repeated memory dips.** Long prefills at 2,048 rows dip the worker's memory;
  G19 keeps the graphs through dips inside 64 decode-only rounds of a prefill, but in the 1 h soak five longer dips
  still took the LRU cap 48 -> 11 (G18: 48 -> 8). Fewer cached graphs cost speed until the next restart.
- **Capacity refusals of streaming requests arrive as in-stream errors.** When the priced admission floor refuses a
  streaming request after its `200` headers went out (it waited up to 30 s for memory first), the client gets an SSE
  `{"error": ...}` event, not an HTTP 503. Sending the headers lazily would fix it. G19's pricing at the smallest
  adaptive step removed the case the soak hit (0 refusals since).
- **The adaptive-row threshold.** Production runs `TF_DSV41_PREFILL_ADAPT_GIB=4.5`; the 1 h soak ran 4.0 (worker
  minimum 3.99 GiB, 9 MiB under our own 4.0 line, no errors). 4.5 has more margin but was not soaked, and not measured
  at 64K / 128K. The engine default (5) makes rows flap on GB10 (replay 32K 1,912 tok/s).
- **The worker is the binding node.** Its MemTotal is 2 GiB below the head's (firmware), and its MemAvailable at
  serving varies 6.2-8.4 GiB between identical boots, outside torch. earlyoom on the nodes (`-m 2,1` here) is the last
  backstop under our 4-5 GiB floors: it kills a rank below ~2.4 GiB, and fail-fast then takes the pair down in ~1 s
  for the watchdog to restart.
- **Two G13 rewrites ship off.** The shortened decode MoE chain (`TF_DSV41_MOE_FUSED`) is exact but measured +0.7 ms
  on a 1-row window in the graph. (`TF_DSV41_BRANCHES`, off in G13, is on since G14 on dedicated priority streams.)
- **Strict mode on the current engine** (every precision trade off, `full` prefill with the cone) has not been
  measured; the strict numbers above are G13's.
- **`/health`** reports `drafted_total` / `accepted_total` as 0 for this family (the counters are not wired).
- **No trimmed draft-head vocabulary is shipped** (`TF_DSV41_DRAFT_HEAD=trim`, off by default and not adopted). The
  development ranking was counted from private chat transcripts and is excluded, so `trim` needs
  `TF_DSV41_DRAFT_VOCAB=<file>` and `tests/test_dsv41_draft_head.py::test_shipped_ranking` fails. A ranking of your
  own traffic (`scripts/campaign/draftvocab.py`) or of public text (the GLM recipe's `bench/draftvocab_public.py`
  method on this tokenizer) can be used.

## Low memory headroom

A larger pack, a larger pool or other software on a node takes the engine into its low-memory fallbacks. None of
them changes the output; they cost speed, and the decode one is easy to miss. The EAQ pack
([PACKS](PACKS.md)) is 3.18 GiB a rank larger than flash17 and meets all three on 2x Spark; its profile sets a
170K pool and forces eager decode (`TF_DSV41_GRAPH_FLOOR_GIB=64`).

- **A smaller pool.** The boot check prices the pool against each rank's measured memory and refuses one that would
  leave the worst predicted dip under the 4 GiB hard floor; its log names the largest pool that fits
  (`kv: ... the largest that does: TF_DSV41_POOL_TOKENS=<max>`). A larger pack lowers that maximum. Size the pool
  from it with some margin (the EAQ profile uses 0.9 × the maximum, capped at 2,100,000).
- **2K prefill segments instead of 4K pieces.** Each prompt piece needs the 4K workspace (about 2.27 GB) to leave
  `TF_DSV41_PF_WS_FLOOR_GIB` free on every rank; otherwise that piece runs 2,048-row segments (same bits, slower
  prefill). Boot warns when it already expects this (`kv: rank N: the prefill workspace ... leaves X GiB, under the
  4.0 GiB hard floor: long prompts there run 2,048-row segments`), and each fallback logs `prefill workspace: ...
  this prompt piece runs 2,048-row segments`.
- **Decode windows without CUDA graphs.** A new decode graph is captured only while MemAvailable minus its ~64 MiB
  price stays at or above max(4.0 GiB hard floor, `TF_DSV41_PF_WS_FLOOR_GIB` + 1/16 GiB) on every rank. Under that,
  with the profile's `TF_DSV41_GRAPH_FLOOR=hold`, the graphs already held keep replaying but every new (rows,
  context bucket) key and every new draft-pass key runs eagerly, slower, for as long as the node stays under the
  floor; with a pool sized near the limit that can be the rest of the process's life. Drafted decode keeps making
  new keys (1-64 rows, a context bucket every 2,048 tokens), so the cost grows with use. It shows in `/health` under
  `graphs.<cache>.eager`, `floor_hits` and `under_floor` (per cache: decode windows, draft passes), and in the log
  as `graph cache <name>: under the memory floor on this/another rank ...: new keys eager` (then `still under the
  memory floor: N eager steps`, and `room again on every rank` when it recovers).

`TF_DSV41_GRAPH_FLOOR_GIB=<GiB>` replaces that rule with a fixed floor: graphs capture while MemAvailable is at least
that many GiB. A lower value keeps decode on graphs, but each capture takes the node nearer the 4 GiB hard floor
(and the nodes' out-of-memory backstop); a high value (for example 64) forces every miss eager, which is a safe way to
measure the eager penalty. Prefer freeing memory (a smaller pool, fewer other processes) before lowering it.
