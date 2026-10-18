Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# DeepSeek-V4.1-Flash on TensorFold, 2x NVIDIA DGX Spark

The recommended pack is **flash17, DeepSeek-V4.1-Flash-1017 EXL3 2.8 bpw**:
[`jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw),
with [its serving profile](docs/FLASH-1017.md). Serve across two NVIDIA DGX Sparks, TP=2 over the 200 Gb/s CX7 link,
behind an OpenAI-compatible API. The previous pack, **q28-v2**
([`jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw)),
is the documented rollback.

The [CUDA Zig serving path](docs/ZIG-SERVE.md) exports engine `9131e10a`, based on TensorFold 1.0.2 plus the DeepSeek
CUDA port, with split KV, a shared paged KV pool (2.1M tokens for flash17, 3.0M for q28-v2) and native images. The
latest measured gates and earlier benchmarks are separated below. Its source is included in [`engine/zig`](engine/zig),
with the Apache-2.0 upstream license and provenance. The serving profile (4K prefill, the engine's own boot calibration)
and generated AOT asset preparation are documented there. The existing Python recipe remains TensorFold 0.6.0 plus the
three published patches, with exact DSpark speculative decoding, CED replay, four request slots, NVMe sessions, native
image input, structured output and tool calls. The shipped Python patches predate the q28-v2 ragged-expert loader; keep
their original 2.9 bpw pack until that engine update is exported.

> **Work in progress.** Measured on one pair of Sparks, against one baseline. Read
> [what is not solved](docs/KNOWN-ISSUES.md) before relying on it.

## Recommended pack: flash17 (DeepSeek-V4.1-Flash-1017, 2.80 bpw)

A 2.80 bpw EXL3 pack of DeepSeek's 2026-10-17 update, `deepseek-ai/DeepSeek-V4.1-Flash-1017`, 203.8 GB, vision
included: [`jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw).
Its profile, [`config/prod-zig-flash17.env.example`](config/prod-zig-flash17.env.example), runs the same Zig `9131e10a`
engine with the 1017 Engram shards and a **2.1M-token pool**: with 3.0M, q28-v2's pool, the 4K prefill workspace took
the head under the 4 GiB floor. We suggest a 2100-2200 MHz GPU clock cap (`nvidia-smi -lgc 300,2100`) to help with
temperature control. Details: [FLASH-1017](docs/FLASH-1017.md).

On 353 of the 356 matched task items, flash17 scores at least as high as Mia's 2.9 bpw pack on every suite and at
least as high as q28-v2; **every paired 95 % interval includes zero**, so no statistical superiority is claimed.
Short decode at a 3.0M pool and 2250 MHz measured +3.6 % (1 stream) and +7.3 % (4 streams) over q28-v2; speed at the
recommended profile is **[FINAL LAUNCH: PENDING]**. Rollback is the unchanged q28-v2 profile.

## Previous pack q28-v2: latest merged kit, split KV and native images

Two DGX Sparks (GB10, 128 GB each), TP=2 over CX7/RoCE, q28-v2 EXL3 2.8 bpw. Engine `9131e10a`,
2026-10-11 09:22–09:28 UTC; same-window baseline: prior Zig kit `17add462`. **One warm repetition**, code workload;
×4 is the all-four-live decode aggregate. This text gate ran **before images were enabled**.
Sources: [merged #24](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/24) and
[the selected receipts](docs/README-STATS.md). These are not comparisons with Python or vLLM.

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
| Requested pool in the q28-v2 profile | 3,000,000 tokens | [Serving profile](config/prod-zig.env.example); boot capacity includes allocation granularity |
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
comparison with the current split-KV kit. [Receipt provenance and the old→new table](docs/README-STATS.md).

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
[ZIG-RESULTS](docs/ZIG-RESULTS.md) retains the original earlier comparison and its cold/first-repetition detail.

## Historical 2.9 bpw results: Python vs vLLM

The following tables retain the original measurements and pack. They are **not a q28-v2 vLLM comparison**;
there are no new vLLM measurements for the Zig path here.

Two DGX Sparks (GB10, 128 GB each, one QSFP cable, RoCE), weights
[`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw).
Baseline: [MiaAI-Lab's 2x DGX Spark vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) on
the same pair and weights, measured 2026-10-01. Ours: [`config/prod.env.example`](config/prod.env.example) on
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
and stress. Details and per-gate limits: [METHOD](docs/METHOD.md) and [campaign receipts](docs/campaign/README.md).
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
each cell was measured, what is exact and what is approximate, and strict mode: [`docs/METHOD.md`](docs/METHOD.md).

## Quick start: Zig

See [the Zig quick start](docs/ZIG-SERVE.md#quick-start), including the bundled source, build context,
per-rank assets, and `scripts/run-zig.sh`. Image input on the Zig path: [IMAGES](docs/IMAGES.md#the-zig-serving-path).

## Quick start: published Python recipe

```bash
git clone --recurse-submodules https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
cp config/prod.env.example config/prod.env && $EDITOR config/prod.env
scripts/serve.sh build && scripts/serve.sh preflight && scripts/serve.sh start
```

Weights, Engram shards, requirements and unattended operation: [`docs/INSTALL.md`](docs/INSTALL.md).

## Documentation

| | |
| --- | --- |
| [FLASH-1017](docs/FLASH-1017.md) | the recommended flash17 pack: profile, memory, task results, rollback to q28-v2 |
| [ZIG-SERVE](docs/ZIG-SERVE.md) / [ZIG-RESULTS](docs/ZIG-RESULTS.md) | Zig build, TP=2 launch, exact settings, q28-v2 comparisons |
| [INSTALL](docs/INSTALL.md) | requirements, weights, Engram shards, build, start, watchdog |
| [API](docs/API.md) / [IMAGES](docs/IMAGES.md) | endpoints, thinking and effort, image input |
| [OPERATIONS](docs/OPERATIONS.md) | settings, memory gates, turning each lever off |
| [METHOD](docs/METHOD.md) / [RESULTS](docs/RESULTS.md) / [BENCHMARKS](docs/BENCHMARKS.md) | how it was measured, every table, how to rerun |
| [CHANGELOG](docs/CHANGELOG.md) | bug fixes and new levers since G13 |
| [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) | what is not solved |
| [PROJECT](docs/PROJECT.md) / [ENGINE](docs/ENGINE.md) / [ARCHITECTURE](docs/ARCHITECTURE.md) / [DECODE](docs/DECODE.md) | patches, layout, model split, decode roofline |
| [campaign/](docs/campaign/README.md) | the development log, windows G1-G19 |

## Credits

- [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): DeepSeek-V4.1-Flash, its prompt encoding and its
  image path (the engine's image input re-implements `inference/vision.py` and `image_processor.py`, MIT; the
  unmodified files are test oracles in `patches/0002`);
  the ideas this engine implements come from DeepSeek's papers and code: the
  [V4.1-Flash technical report](https://arxiv.org/abs/2609.19969) (CED and bounded replay, CSA2, mHC),
  [DSpark](https://arxiv.org/abs/2607.05147) with [DeepSpec](https://github.com/deepseek-ai/DeepSpec), and
  [Engram](https://arxiv.org/abs/2601.07372) with [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram).
- [Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold): the engine, the EXL3 kernels, the drafting
  and the server this recipe builds on.
- Mia / MiaAI-Lab: the [DeepSeek-V4.1-Flash 2x DGX Spark vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
  this recipe is measured against (whose packed Engram shards the measured runs used), and the
  [2.9 bpw EXL3 pack](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) on Hugging Face
  ([Mia-AiLab](https://huggingface.co/Mia-AiLab)); their GLM-5.3 kit's fat-expert MoE design also lives on in
  `patches/0001`.
- [dealignai](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw): the uncensored 2.9 bpw
  variant measured here.
- [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3): the EXL3 format.
- [Cruz (vcruz305)](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe): the finding that DSpark
  verify and plain decode take different EXL3 paths in the vLLM kits (the case for exact speculative decoding), the
  expert-union counts per verify row behind our round model, and the host-side Engram hashing fix
  ([`docs/campaign/LANDSCAPE.md`](docs/campaign/LANDSCAPE.md), [`docs/campaign/TARGETS.md`](docs/campaign/TARGETS.md));
  his [SAGE 1.59 bpw pack](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw) was evaluated
  ([`docs/campaign/PARKED-SAGE-1.59.md`](docs/campaign/PARKED-SAGE-1.59.md)).
- PCTree, [arXiv 2608.02123](https://arxiv.org/abs/2608.02123): the parent-conditioned draft trees implemented as
  `TF_DSV41_TREE_PC` (measured, not adopted: [`docs/DECODE.md`](docs/DECODE.md)).
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (Luke Alonso and contributors): the
  "RoCEnante" one-shot RoCE all-gather that `patches/0001`'s RoCE path (used by this family) adapts, and the prefill
  kernel designs it re-implements.
- [Reederey87](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark): the Apache-2.0 kernel code the GLM
  stack's fat-expert kernels adapt.
- [The vLLM project](https://github.com/vllm-project/vllm): the DeepSeek-V4 / V4.1 implementation our kernels' math
  follows (cited per file; no code copied).
- [bertholomus / BertholomusAI (Albert Lee)](https://github.com/bertholomus/TensorFold), branch `deepseek-v41-tp2`
  (Apache-2.0): ideas we re-implemented (no code copied): geometric context buckets for the CUDA graphs (ours: 1.25x
  past 32K), an allocator ceiling (its `_memory_ceiling`; ours `TF_DSV41_ALLOC_CEIL_GIB`, shipped off), the GIL
  switch-interval knob (`TF_DSV41_SWITCH_MS`) and failing fast across both ranks.
- Issue and pull-request authors on this repository: **WireLLM** (#3, the host memory growth), **ZackO2o** (#4, the
  missing `csa2` kernel sources) and **flashosophy** (#6, the four-Spark work that surfaced the `xgrammar` and
  `text_config` build bugs); see [Changes since G13](docs/CHANGELOG.md).
- [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar): the grammar engine behind structured output.
- [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) and
  [Weschera/spark-bench](https://github.com/Weschera/spark-bench): the tool-calling benchmarks.
- [MMLU](https://github.com/hendrycks/test) (Hendrycks et al.): the 200 questions in `bench/data/`.
- NVIDIA: the DGX Spark and the PyTorch container.

## Licensing

This project's code, patches, scripts, benchmarks and docs: **Apache License 2.0** ([`LICENSE`](LICENSE),
[`NOTICE`](NOTICE)). TensorFold, the added files, the adapted kernels, the base image and the model weights each keep
their own terms: [`docs/LICENSING.md`](docs/LICENSING.md).
