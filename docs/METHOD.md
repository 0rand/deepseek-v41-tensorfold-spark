# How the results were measured

The headline tables of the Python recipe are in
[PREVIOUS-RESULTS](PREVIOUS-RESULTS.md#historical-29-bpw-results-python-vs-vllm); every table and the lever-by-lever
history are in [RESULTS.md](RESULTS.md).

The decode cells are G19's check of the shipped build (`7bd2d67` with the production words, mean of 2 boots,
[`docs/campaign/G19-RESULTS.md`](campaign/G19-RESULTS.md)); at temperature 0.7: code 78.2, prose 48.0, structured
122.9. The prefill cells are G19's 2,048-row windows: 8K and 32K on the shipped words (`TF_DSV41_PREFILL_ADAPT_GIB=4.5`),
64K and 128K on the soaked 4.0 (the same 32K within 1 tok/s; 4.5 was not run at 64K / 128K). Over HTTP (the stress
server's first-token probe, best of 2) replay reads 2,052 / 2,224 / 2,305 / 2,290 / 2,202 tok/s at 8K / 16K / 32K /
64K / 128K. Start to ready was measured on the G10 / G11 test servers and not re-measured since.

Where it moved since the last update (G13, `767ad9f`): code 82.8 -> 84.7, prose 44.25 -> 46.8, structured 117.8 ->
121.3, C1 / C2 / C4 85.1 / 70.3 / 96.6 -> 87.6 / 74.1 / 101.1 (G14's window levers and calibration VERSION 4); replay
prefill at 32K 2,043 -> 2,310 and full prefill 1,004 -> 2,025 (G17's prefill kernels, the full-mode cone, G19's
2,048-row windows with adaptive rows). Between G16 and G18 production ran 1,024-row windows for memory (replay 32K
1,753 -> 1,990); G19 brought 2,048 back.

## Quality and robustness
Exactness, checked in every run: **drafted == serial** (speculative decoding never changes a reply, at T = 0 and
T > 0), **batched == alone** (every concurrent stream equals the same request run alone, serial: `--c-exact`), and the
prefill levers give **bit-identical outputs across prefill row sizes** (2,048 / 1,024 / 512 rows, adaptive or fixed:
first token and a 32-token reply digest equal at every size, mode and boot in G17-G19; the CPU suites compare whole
slot states). Top-1 ran on G17's build (`0bdd276`, which G19 changes only in the prefill row choice), MMLU-200 and the
tool chains (thinking off) on G16's (`da5ae43` with native images), the soak and the stress on `7bd2d67`; structured output, the
needles and the 20-question MMLU are older runs on the same paths (G10 and G7).

## What measures what

- **Decode cells, ours:** `m2bench` (inside the engine, both ranks, no HTTP; `scripts/serve.sh run`), tok/s from the
  first to the last token of a 384-token reply, the median of the repetitions (2 by default), request slots sized
  for 16K tokens. Prompts: an LRU-cache
  class with tests (code), a 400-word essay (prose), counting 1 to 200 (structured). The cells are the mean of G19's
  two boots of the shipped build (`results/campaign/G19-20261005/m2-g19s1-combo-b{1,2}.json`, `g15-depth-speed.txt`).
- **Decode cells, kit:** HTTP clients (`glmbench`, `multiturn` of the GLM recipe): code = a 64-token code reply
  (41.9) and a 512-token one (45.0), prose = a 200-token essay, structured = count 1-200, thinking off.
- **2 / 4 streams:** both sides report decode aggregate = all tokens / (last token - first token). Ours mixes code,
  prose, a JSON task and a copy-heavy edit, every other stream at T = 0.7; the kit's mix is its chat / code prompts,
  256 tokens each. Same metric, different prompts. "Steady" is the aggregate while all four streams are live: the mixed
  cell's wall is mostly the prose stream finishing alone ([`docs/campaign/G17-LEVERS.md`](campaign/G17-LEVERS.md)).
- **Prefill:** one request, cold, after a 2K warm-up. Ours: fresh random text through `m2bench --prefill
  --prefill-reply 32` (the median of 2 boots; the first token and a 32-token reply digest are compared with the
  1,024-row reference at every size). The kit: a repeated filler document over HTTP.
  Repeated fillers can make Engram reads look cheaper, so the kit's cells are not pessimistic.
- **Start to ready:** ours = `docker run` to `/v1/models` answering, with prepared folders and compiled kernels
  cached, page cache dropped first. The kit's = its start script to `/health` on freshly rebooted nodes. The first
  start of a new image is slower (kernel builds, ~80 s) and the very first writes the prepared folders (~95 GB a node).
- **Top-1:** the kit's `prompt_logprobs` (top 5) over 8 built-in prompts repeated to 2,048 tokens
  ([`results/kit-baseline/oracle-kit.json`](../results/kit-baseline/oracle-kit.json)), our engine teacher-forced on the
  same token ids. "First copy" counts only each prompt's first pass, before the model can copy itself.
- **Not RigMark.** No RigMark receipt exists for this model yet; every cell comes from the clients above.

[`docs/RESULTS.md`](RESULTS.md) has every table, the lever-by-lever history and the negatives;
[`docs/BENCHMARKS.md`](BENCHMARKS.md) how to rerun each cell.

## What "exact" means here, and what is approximate

- **Exact:** speculative decoding never changes a reply. Every verify row is the serial step at its position (row-
  invariant kernels) and its token is the request's keyed choice at that absolute position, so drafted == serial at
  T = 0 and at T > 0, and batched == alone. Every benchmark run checks it (`exact_all`).
- **Approximate by design, on in the measured config:**
  - CED decoder bounded replay for prompts (`TF_DSV41_PREFILL=replay`, DeepSeek's own technique: the decoder half
    runs only over a prompt's last 128 tokens). `full` is the exact prefill: with `TF_DSV41_FULL_CONE=1` (on in the
    config, the same bits as plain `full`) the decoder runs only over the ~2,541 rows whose state survives the prompt,
    so `full` is now 1.45-2.09x the kit too.
  - Routed-expert pruning in decode (`TF_DSV41_EXPERT_TOPP=0.85`, at least 3 experts, renormalized): +5% decode,
    MMLU-200 88.5% alone. Delete three lines of the config to serve the unpruned model.
  - mHC mixing weights in bf16 (`TF_DSV41_MHC_FN=bf16`): +2-5% prose, top-1 vs the kit 0.9963.
- **Not bit-identical to the kit.** Different kernels and summation orders; the agreement is the top-1 row above.
- **Every lever added since G13 is exact:** the G14 window levers, the G17 prefill kernels, the full-mode cone and the
  adaptive prefill rows change no bits (on == off, compared on the GPU and in the CPU suites).

## Strict mode: every precision trade off

Measured in G13 and not re-run since. The same engine and the G13 rewrites (which change no bits) with every knob
that trades precision turned off:
`TF_DSV41_EXPERT_TOPP=0` (no expert pruning), `EXPERT_RENORM=orig`, `MHC_FN=fp32`, `KIT_ROUNDING=0`, `LOGITS=fp32`,
`INDEX_KV=bf16`, `PREFILL=full` (no bounded replay). The fast prefill GEMMs and the fused prefill attention stay on.
Measured in G13, same build, same session ([`docs/campaign/G13-RESULTS.md`](campaign/G13-RESULTS.md)):

| | strict | production | kit | strict / kit |
| --- | ---: | ---: | ---: | ---: |
| Code, 1 stream, greedy | **76.9** | 82.8 | 41.9-45.0 | 1.71-1.83x |
| Prose, 1 stream, greedy | **44.2** | 44.25 | 32.5 | 1.36x |
| Structured, 1 stream, greedy | **111.9** | 117.8 | 38.0-50.2 | 2.24-2.95x |
| 1 / 2 / 4 streams, decode aggregate | **79.0 / 64.0 / 89.5** | 85.1 / 70.3 / 96.6 | 32.2 / 46.7 / 37.6 | 2.45x / 1.37x / 2.38x |
| Cold prefill 8K / 32K / 64K / 128K | **915 / 965 / 959 / 923** | (replay) 1,833-2,068 | 1,073 / 1,075 / 1,060 / 1,031 | **0.85-0.90x** |
| Teacher-forced top-1 vs the kit | 0.9963 | 0.9961 | 1 | |

Decode costs 5-9% against production (prose at T = 0 is level only because the strict reply drafts a little better on
that prompt; at T = 0.7 it is -6.9%), and is still 1.4-2.9x the kit. Prefill without replay was below the kit then;
G17's full-mode cone and prefill kernels (all exact) have since taken `full` to 1,561-2,197 tok/s on the production
words, but a strict-mode receipt on the current engine has not been run. Strict MMLU and tool chains were not run.

## Where we are not at 2x

Prose (1.44x), 2 streams (1.59x) and full-mode prefill of short prompts (8K: 1.45x). Prose drafts poorly: DSpark
keeps ~1.7 tokens a round on prose against ~3.7 on code, so prose speed is the verify window's cost. A 1-row window is
~21.8 ms since G14 (23.4 after G13, 26.9 before) against a
bandwidth floor of ~17 ms a rank (the 2.9 bpw weights read once), and the second row costs ~6 ms more because a
second token brings ~5 new experts a layer. That is why G13's rewrites helped prose (+7.2%) far more than code (+1.3%:
code verifies ~4.6 rows a round, where the savings are smaller). 2 streams pair a code stream with a T = 0.7 prose
stream, so the prose stream sets the pace; the same holds for the 4-stream mixed cell (101 aggregate, 136 while all
four are live). Full prefill under ~2.6K tokens has no cone to skip.
[`docs/DECODE.md`](DECODE.md) has the roofline and what was tried.
