Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# DeepSeek-V4.1-Flash on TensorFold, 2x NVIDIA DGX Spark

The recommended pack is **q28-v2 EXL3 2.8 bpw**:
[`jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw)
(publication pending; repository name provisional). Serve across two NVIDIA DGX Sparks, TP=2 over the 200 Gb/s CX7
link, behind an OpenAI-compatible API.

The new [CUDA Zig serving path](docs/ZIG-SERVE.md) is based on TensorFold 1.0.2 plus the DeepSeek CUDA port.
Its source is included in [`engine/zig`](engine/zig), with the Apache-2.0 upstream license and provenance.
The serving profile (4K prefill, the engine's own boot calibration) and generated AOT asset preparation are documented there. The existing Python
recipe remains TensorFold 0.6.0 plus the two published patches, with exact DSpark speculative decoding, CED replay,
four request slots, NVMe sessions, native image input, structured output and tool calls. The shipped Python patches
predate the q28-v2 ragged-expert loader; keep their original 2.9 bpw pack until that engine update is exported.

> **Work in progress.** Measured on one pair of Sparks, against one baseline. Read
> [what is not solved](docs/KNOWN-ISSUES.md) before relying on it.

## Current q28-v2 results: Zig vs Python

Two DGX Spark nodes, TP=2 over RoCE, q28-v2. Zig with the [serving profile](config/prod-zig.env.example) against a
fresh same-night run of the Python serving path; **best of 3** warm repetitions for both engines (medians, first
repetitions and cold runs in [the full results](docs/ZIG-RESULTS.md)). Decode tok/s; ×4 is the four-stream aggregate.

| Cell | Python | Zig | Δ |
| --- | ---: | ---: | ---: |
| code T0 ×1 | 88.0 | 90.3 | +2.6 % |
| code T0 ×4 | 144.1 | 156.4 | +8.6 % |
| code T0.7 ×1 | 91.3 | 92.2 | +1.0 % |
| code T0.7 ×4 | 145.0 | 153.0 | +5.6 % |
| prose T0 ×1 | 53.8 | 54.7 | +1.7 % |
| prose T0 ×4 | 98.2 | 104.7 | +6.6 % |
| prose T0.7 ×1 | 49.6 | 50.6 | +2.1 % |
| prose T0.7 ×4 | 97.9 | 103.3 | +5.6 % |
| 32K prompt (tok/s, 3 cold prefills) | 2,440 | 2,686 | +10.1 % |
| 131K prompt (tok/s, 3 cold prefills) | 2,264 | 2,496 | +10.3 % |
| 32K decode | 63.5 | 63.6 | +0.2 % |
| 131K decode | 67.0 | 69.3 | +3.3 % |
| mix: code ×4 aggregate beside a 131K stream | 145.1 | 155.9 | +7.5 % |
| mixed windows: aggregate | 143.4 | 155.1 | +8.2 % |

Zig is ahead in every cell: +5.6 to +8.6 % on four-stream code, +1.0 to +2.6 % on one-stream code, +10 % on long
prompts, +3.3 % on 131K decode. Replies are identical (16/16 short replies, and the long reply hashes). One pair of
nodes, one night: treat differences of a few percent as indicative.

## Historical 2.9 bpw results: Python vs vLLM

The following tables retain the original measurements and pack. They are **not a q28-v2 vLLM comparison**;
there are no new vLLM measurements for the Zig path here.

Two DGX Sparks (GB10, 128 GB each, one QSFP cable, RoCE), weights
[`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw).
Baseline: [MiaAI-Lab's 2x DGX Spark vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) on
the same pair and weights. Ours: [`config/prod.env.example`](config/prod.env.example) on engine `7bd2d67` (G19).

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
64K / 128K on the soaked 4.0. Start to ready not re-measured since G11.

### Quality and robustness

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
