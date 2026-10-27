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
- **Either rank can bind.** Both nodes have the same MemTotal (127.6 GB) since the worker's BIOS and firmware were
  brought to the head's versions (2026-10-24); before that the worker's was 2 GiB lower, which made it the binding
  node in G16-G19. The worker's MemAvailable at serving varied 6.2-8.4 GiB between identical boots, outside torch.
  Every floor here is checked on both ranks, so nothing assumes which one is tighter. earlyoom on the nodes (`-m 2,1`
  here) is the last backstop under our 4-5 GiB floors: it kills a rank below ~2.4 GiB on either node, and fail-fast
  then takes the pair down in ~1 s for the watchdog to restart.
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
