# README measurement provenance

This is a selected, sanitized transcription of gated receipts behind the README update. No benchmark was rerun for
this documentation change. Eligible source changes are merged recipe PRs [#18](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/18),
[#23](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/23),
[#24](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/24) and
[#26](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/26). No figures from unmerged work are included.

## Exported engine rerun, 2026-10-10

Hardware: two NVIDIA DGX Sparks, GB10, 128 GB each, TP=2 over CX7/RoCE. Pack: q28-v2 EXL3 2.8 bpw.
Engine: `8e96f99c`, the source exported by #18. Time: 2026-10-10 07:44–08:01 UTC. Replicated 1.4M pool,
placeholder images, 4K prefill, Zig boot calibration. Fresh boot, cold repetition, three warm repetitions per short
cell, three long/mixed passes and two salted long passes. Best and median are the receipt's reported precision.

Source ledger: `NIGHT-20261009.md`, section “2026-10-10 07:44-08:01 UTC: full published bench on prod kit 8e96f99c”,
lines 479–513. Its run identifier is `final-8e96-20261010-074450`. Original published benchmark identifier:
`final-zig-20261010-000701`, measured kit `12b1b2c3`; the Python reference is engine `8474f31`, measured earlier
that night. The raw final benchmark records are not exported. The existing [ZIG-RESULTS](ZIG-RESULTS.md) describes
the original comparison, not this later rerun.

| README cell | Old Zig best | Latest gated `8e96f99c` best / median | Ledger line |
| --- | ---: | ---: | ---: |
| code T0 ×1 | 90.3 | 90.68 / 90.39 | 489 |
| code T0 ×4 | 156.4 | 155.55 / 154.73 | 490 |
| code T0.7 ×1 | 92.2 | 92.48 / 92.16 | 491 |
| code T0.7 ×4 | 153.0 | 152.74 / 152.27 | 492 |
| prose T0 ×1 | 54.7 | 54.87 / 54.66 | 493 |
| prose T0 ×4 | 104.7 | 104.90 / 103.00 | 494 |
| prose T0.7 ×1 | 50.6 | 50.67 / 50.57 | 495 |
| prose T0.7 ×4 | 103.3 | 102.85 / 101.62 | 496 |
| 32K prompt | 2,686 | 2,689.6 / 2,687.1 | 497 |
| 131K prompt | 2,496 | 2,497.0 / 2,495.6 | 498 |
| 32K decode | 63.6 | 64.75 / 63.41 | 500 |
| 131K decode | 69.3 | 68.42 / 68.34 | 501 |
| mixed code ×4 | 155.9 | 156.21 / 155.55 | 503 |
| mixed windows aggregate | 155.1 | 155.70 / 155.69 | 504 |

All throughput figures above are tok/s. Three cold prompt passes produced best TTFT 14.12 / 61.75 s at the
32K / 131K workload sizes (line 499); these encode to 37,974 / 154,183 prompt tokens. The rerun matched 16/16 short
replies and every base/salted long-reply hash. Both ranks had zero AOT misses, graph-eager lines and panics; four
4K workspace allocations and zero fallbacks. No cell was more than 2% below the original published Zig reference.

Python values are retained, not remeasured. The README restores reported precision for these old rounded cells:

| Python cell | Old README | Recorded value | Source |
| --- | ---: | ---: | --- |
| 32K prompt | 2,440 | 2,440.2 | Ledger line 497; ZIG-RESULTS long table |
| 131K prompt | 2,264 | 2,263.5 | Ledger line 498; ZIG-RESULTS long table |
| 32K decode | 63.5 | 63.47 | Ledger line 500; ZIG-RESULTS long table |
| 131K decode | 67.0 | 67.04 | Ledger line 501; ZIG-RESULTS long table |
| mixed code ×4 | 145.1 | 145.10 | Ledger line 503 |
| mixed windows aggregate | 143.4 | 143.39 | Ledger line 504 |

The old Python-comparison percentages are removed, rather than recalculated from rounded reference values or
represented as a same-window comparison with the split-KV kit:

| Cell | Old percentage removed |
| --- | ---: |
| code T0 ×1 / ×4 | +2.6% / +8.6% |
| code T0.7 ×1 / ×4 | +1.0% / +5.6% |
| prose T0 ×1 / ×4 | +1.7% / +6.6% |
| prose T0.7 ×1 / ×4 | +2.1% / +5.6% |
| 32K / 131K prompt | +10.1% / +10.3% |
| 32K / 131K decode | +0.2% / +3.3% |
| mixed code ×4 / mixed windows | +7.5% / +8.2% |

The prose ranges +5.6–8.6%, +1.0–2.6%, +10% and +3.3% likewise stop describing the current kit.

## Split KV and images, 2026-10-11

Same two GB10 Sparks, 128 GB each, TP=2 over CX7/RoCE, same q28-v2 pack. Engine
`9131e10ad17dde51e94a996da8255227727cd224`, exported in #24. Baseline kit `17add462`, same window.
Source: `prodimg-window-20261011-pi2/run.log`, lines 5, 9, 11–16, 20–35, and `NIGHT-20261010.md:180–183`.

| Text gate, one warm repetition | Prior kit | `9131e10a` | Reported change |
| --- | ---: | ---: | ---: |
| code ×1 | 87.11 tok/s | 86.96 tok/s | −0.2% |
| code ×4, all-four-live aggregate | 146.91 tok/s | 148.03 tok/s | +0.8% |

Text gate completed at 09:28 UTC **with images still off**: 16/16 replies matched, long hashes matched, both ranks
reported 3,000,320 logical pool token positions. That is one shared capacity with rank-local shards, not 6M tokens.
The profile requests 3,000,000 pool tokens, 1,048,576 context tokens per slot and four slots; allocation granularity
explains the reported pool. Configuration is not proof of four simultaneous full-context requests.

Native images were subsequently enabled and gated: 25 requests with three repetitions, 75/75 exact image outputs,
20/20 VQA. One text near-tie remains documented in #24; it is not hidden by the image result. The soak receipt
`soak-20261011-0935/soak.log:4,6–8` records 120 minutes, 24 clean five-minute checks, and passing text/image probes
at start and end. This is not the older 906-request soak workload.

Earlier merged ancestor `74ace4be`, 2026-10-10, images off, passed the long-context/session gate on the same pair:
two 996,011-token requests and four approximately 740K-token requests completed, with serial prefill. An NVMe
restore cached 996,000 of 996,011 tokens in 1.3 s and reproduced the reply. Source: `NIGHT-20261010.md:35–40`.
The context limit itself was not filled exactly; this receipt does not assert four 1,048,576-token live requests.

#23 supplies setup/AOT host fixes. #26 supplies image run-planning host regression proof and explicitly supplies no
new Spark measurement. No unmerged correction, TBO prototype or review-fixes performance number is used here.

## Historical values retained

The older 2.9 bpw Python/vLLM table keeps its figures because no newer matched vLLM gate exists. Baseline date:
2026-10-01. Python speed/prefill and soak/stress: `7bd2d67`, 2026-10-05; startup 34–44 s: `38f6500`, 2026-10-03.
Hardware for both tables is the same two GB10 Sparks, 128 GB each, CX7/RoCE. Quality and robustness receipts retain
their per-engine dates in the README and [METHOD](METHOD.md), including [G7](campaign/G7-RESULTS.md),
[G13](campaign/G13-RESULTS.md), [G15b](campaign/G15b-RESULTS.md), [G16](campaign/G16-RELEASE.md),
[G17](campaign/G17-PREFILL.md) and [G19](campaign/G19-RESULTS.md). Teacher-forced agreement is not draft acceptance.
No newer comparable merged draft-acceptance fraction, startup time or full prefill campaign was found.

The repository description's unqualified “1.8–1.9x code, 2.5x multi-stream, 1.7–1.95x prefill vs the vLLM kit”
should be replaced with a description of the current q28-v2 serving path. Those ratios must not be promoted to a
matched current Zig/vLLM claim. The README's separately dated historical 2.9 bpw ratios remain historical.
