# Earlier results: q28-v2 and the historical 2.9 bpw Python vs vLLM comparison

These sections were in the main README before flash17 became the recommended pack. They are kept unchanged here as the
record for the previous pack (q28-v2, now the [rollback](FLASH-1017.md#rollback-to-q28-v2)) and the original Python
recipe. flash17's results are in [FLASH-1017](FLASH-1017.md).

## q28-v2: latest merged kit, split KV and native images

Two DGX Sparks (GB10, 128 GB each), TP=2 over CX7/RoCE, q28-v2 EXL3 2.8 bpw. Engine `9131e10a`,
2026-10-11 09:22–09:28 UTC; same-window baseline: prior Zig kit `17add462`. **One warm repetition**, code workload;
×4 is the all-four-live decode aggregate. This text gate ran **before images were enabled**.
Sources: [merged #24](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/24) and
[the selected receipts](README-STATS.md). These are not comparisons with Python or vLLM.

| Text gate | Prior kit `17add462` | Kit `9131e10a` | Reported Δ |
| --- | ---: | ---: | ---: |
| code T0 ×1 (tok/s) | 87.11 | 86.96 | −0.2 % |
| code T0 ×4 (tok/s) | 146.91 | 148.03 | +0.8 % |

On the same hardware and engine, 2026-10-11: the text gate matched 16/16 replies and the long-context hashes.
The subsequent native-image gate passed 75/75 outputs (25 requests, three repetitions), VQA 20/20, and a
120-minute soak had 24/24 clean five-minute checks, with text and image probes passing at start and end.
One text prompt has a known T0 near-tie (`shorter` / `short`); the 75 image outputs were exact.

| Capacity / profile | Value | Evidence scope |
| --- | ---: | --- |
| Shared split-KV pool observed at boot | 3,000,320 token positions | `9131e10a`, 2026-10-11; both ranks report the same logical capacity, not twice the pool |
| Requested pool in the q28-v2 profile | 3,000,000 tokens | [Serving profile](../config/prod-zig.env.example); boot capacity includes allocation granularity |
| Configured context per slot / slots | 1,048,576 tokens / 4 | Configuration, not a receipt for four full-context requests |
| Long requests completed (serial prefill) | 996,011 tokens ×2; about 740K tokens ×4 | Earlier merged engine `74ace4be`, 2026-10-10, same two Sparks; images off |
| NVMe resume in that long-context gate | 1.3 s; 996,000 of 996,011 prompt tokens cached | `74ace4be`, 2026-10-10; restored reply identical |

[Merged #26](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/26) fixes host-side image run planning;
it supplies no new Spark speed measurement. No newer complete prefill, draft-acceptance-rate or startup-latency
measurement is claimed for `9131e10a`.

## Earlier q28-v2 benchmark: exported Zig engine and Python reference

Two DGX Sparks (GB10, 128 GB each), TP=2 over CX7/RoCE, q28-v2 EXL3 2.8 bpw, replicated 1.4M pool,
placeholder images, 4K prefill and Zig boot calibration. The latest complete gated rerun of the engine exported in
[merged #18](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/18) is `8e96f99c`,
2026-10-10 07:44–08:01 UTC. Zig figures are **best / median of three** warm repetitions; prompt rows use three cold
prefills. Python `8474f31` figures retain the earlier night's reference measurement. This is not a fresh paired
comparison with the current split-KV kit. [Receipt provenance and the old→new table](README-STATS.md).

| Cell | Python best (earlier run) | Zig `8e96f99c` best / median |
| --- | ---: | ---: |
| code T0 ×1 | 88.0 | 90.68 / 90.39 |
| code T0 ×4 | 144.1 | 155.55 / 154.73 |
| code T0.7 ×1 | 91.3 | 92.48 / 92.16 |
| code T0.7 ×4 | 145.0 | 152.74 / 152.27 |
| prose T0 ×1 | 53.8 | 54.87 / 54.66 |
| prose T0 ×4 | 98.2 | 104.90 / 103.00 |
| prose T0.7 ×1 | 49.6 | 50.67 / 50.57 |
| prose T0.7 ×4 | 97.9 | 102.85 / 101.62 |
| 32K prompt (tok/s) | 2,440.2 | 2,689.6 / 2,687.1 |
| 131K prompt (tok/s) | 2,263.5 | 2,497.0 / 2,495.6 |
| 32K / 131K time to first token (s, best) | 15.56 / 68.12 | 14.12 / 61.75 |
| 32K decode | 63.47 | 64.75 / 63.41 |
| 131K decode | 67.04 | 68.42 / 68.34 |
| mix: code ×4 aggregate beside a 131K stream | 145.10 | 156.21 / 155.55 |
| mixed windows: aggregate | 143.39 | 155.70 / 155.69 |

The 32K / 131K workload labels encode to 37,974 / 154,183 prompt tokens. The rerun matched all 16 short replies and
the long reply hashes, including salted passes. Its historical Python reference was not rerun with it, so the old
percentage column and the claim that those percentages describe the current kit have been removed.
[ZIG-RESULTS](ZIG-RESULTS.md) retains the original earlier comparison and its cold/first-repetition detail.

## Historical 2.9 bpw results: Python vs vLLM

The following tables retain the original measurements and pack. They are **not a q28-v2 vLLM comparison**;
there are no new vLLM measurements for the Zig path here.

Two DGX Sparks (GB10, 128 GB each, one QSFP cable, RoCE), weights
[`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw).
Baseline: [MiaAI-Lab's 2x DGX Spark vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) on
the same pair and weights, measured 2026-10-01. Ours: [`config/prod.env.example`](../config/prod.env.example) on
engine `7bd2d67` (G19), 2026-10-05. No newer matched vLLM gate is available; these historical values are retained.

### Speed (tok/s)

| | **TensorFold (this recipe)** | MiaAI-Lab vLLM kit | ratio |
| --- | ---: | ---: | ---: |
| Code, 1 stream, greedy | **84.7** | 41.9-45.0 | **1.9-2.0x** |
| Prose (essay), 1 stream, greedy | **46.8** | 32.5 | 1.44x |
| Structured (count 1-200), 1 stream, greedy | **121.3** | 38.0 (50.2 on a JSON task) | **2.4-3.2x** |
| 1 stream, decode aggregate | **87.6** | 32.2 | **2.7x** |
| 2 streams, decode aggregate | **74.1** | 46.7 | 1.59x |
| 4 streams, decode aggregate (steady, all four live) | **101.1** (136.0) | 37.6 | **2.7x** |
| Cold prefill 8K / 32K / 64K / 128K (CED replay, the default) | **2,028 / 2,310 / 2,294 / 2,118** | 1,073 / 1,075 / 1,060 / 1,031 | **1.89-2.16x** |
| Cold prefill, `TF_DSV41_PREFILL=full` (exact: no replay) | **1,561 / 2,025 / 2,197 / 2,152** | same | **1.45-2.09x** |
| Start to ready | **34-44 s** | 378 s | ~9x |

Decode: mean of 2 boots of the shipped build. Prefill: 8K / 32K on the shipped `TF_DSV41_PREFILL_ADAPT_GIB=4.5`,
64K / 128K on the soaked 4.0. Start to ready retains engine `38f6500` (G10/G11), 2026-10-03; it was not re-measured.

### Quality and robustness

Historical Python gates on the same two GB10 Sparks, 128 GB each, TP=2 over CX7/RoCE, 2026-10-02 to 10-05.
These are not fresh measurements on the Zig kit. Engine/date provenance: `767ad9f` (G13), 2026-10-03 for the original
top-1, MMLU and tool-chain gates; `0bdd276` (G17), 2026-10-04 for the top-1 recheck; `6e03615` plus the native-image cap (G16), 2026-10-04 for
MMLU and tool-chain rechecks; `e87e813` (G7), 2026-10-02 for needles and the preamble MMLU; `38f6500` (G10),
2026-10-03 for structured output; `6e03615` (G15b), 2026-10-04 for images; and `7bd2d67` (G19), 2026-10-05 for soak
and stress. Details and per-gate limits: [METHOD](METHOD.md) and [campaign receipts](campaign/README.md).
Top-1 agreement is not a draft-acceptance fraction. The historical figures below have no newer comparable merged gate.

| | **TensorFold** | kit |
| --- | --- | --- |
| Teacher-forced top-1 agreement with the kit (8 prompts x 2,048 positions) | **0.9961** (first copy of each prompt: 0.9502); the same with every G13-G17 lever on or off | 1 by definition |
| MMLU-200, 0-shot, greedy, thinking off | **87.5%** | 87.5% |
| MMLU-200 with a 20-question preamble (~2.1K-token prompts), replay vs full prefill | 81.1% / 81.1% (178 of 180 answers equal) | - |
| Needle at 32K / 128K / 299K (replay prefill) | found (19.9 s / 74.0 s / 195 s) | - |
| Multi-step tool chains (`bench/tooleval/chains.py`, 6 scenarios, 12 points), thinking off / high | 11 / 12, 11 / 12 | - |
| tool-eval-bench category C (multi-step), thinking off | 8 / 8 (score 100) | - |
| Structured output: 12 JSON-schema cases + 10 tool-choice cases (drafted == serial == batched, valid, no markup) | 22 / 22 | - |
| Image input (native: DeepSeek's ViT on rank 0): 20 visual questions | 20 / 20 | - |
| 1 h soak (agent sessions to 120K tokens with screenshots, short chats with cancels, 24-200K prompts, image requests) | **906 requests, 0 errors**, 0 capacity refusals; RssAnon falling (-0.4 GiB/h a rank); MemAvailable min head 4.63 / worker 3.99 GiB | - |
| Stress: one 299K prefill + three 64K prompts decoding 2,048 tokens each + 4 image requests | every stream completes; 299K first token **157 s**; MemAvailable min head 5.07 / worker 4.81 GiB | 4.40 GiB at <= 256K (lighter load) |

Every run checks **drafted == serial**, **batched == alone**, and bit-identical outputs across prefill row sizes. How
each cell was measured, what is exact and what is approximate, and strict mode: [`docs/METHOD.md`](METHOD.md).
