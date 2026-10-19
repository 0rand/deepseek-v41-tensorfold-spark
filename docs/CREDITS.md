# Credits

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
- Mia's AI Lab ([Mia-AiLab](https://huggingface.co/Mia-AiLab)): the
  [DeepSeek-V4.1-Flash-1017 EXL3 EAQ v1 2.9 bpw pack](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
  (MIT), the optional second pack ([PACKS](PACKS.md)), and the finding, in their
  [1017 serving recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks), of which of its
  tensors are stored F16 and must be widened for this engine (our tools re-implement that step; no code copied).
- [dealignai](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw): the uncensored 2.9 bpw
  variant measured here.
- [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3): the EXL3 format.
- [Cruz (vcruz305)](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe): the finding that DSpark
  verify and plain decode take different EXL3 paths in the vLLM kits (the case for exact speculative decoding), the
  expert-union counts per verify row behind our round model, and the host-side Engram hashing fix
  ([`docs/campaign/LANDSCAPE.md`](campaign/LANDSCAPE.md), [`docs/campaign/TARGETS.md`](campaign/TARGETS.md));
  his [SAGE 1.59 bpw pack](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw) was evaluated
  ([`docs/campaign/PARKED-SAGE-1.59.md`](campaign/PARKED-SAGE-1.59.md)).
- PCTree, [arXiv 2608.02123](https://arxiv.org/abs/2608.02123): the parent-conditioned draft trees implemented as
  `TF_DSV41_TREE_PC` (measured, not adopted: [`docs/DECODE.md`](DECODE.md)).
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
  `text_config` build bugs); see [Changes since G13](CHANGELOG.md).
- [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar): the grammar engine behind structured output.
- [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) and
  [Weschera/spark-bench](https://github.com/Weschera/spark-bench): the tool-calling benchmarks.
- [MMLU](https://github.com/hendrycks/test) (Hendrycks et al.): the 200 questions in `bench/data/`.
- NVIDIA: the DGX Spark and the PyTorch container.
