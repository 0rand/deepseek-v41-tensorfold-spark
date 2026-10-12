# Licensing

| Part | License |
| --- | --- |
| This project's code, patches, scripts, benchmarks and docs | **Apache License 2.0** ([`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE)). Redistributions, modified or not, must keep the copyright line and the NOTICE attributions and state their changes. |
| TensorFold (`vendor/TensorFold`) | Apache License 2.0 from 0.6.0 (code written before 0.6.0 keeps its MIT notice), Copyright 2026 TensorFold contributors; unmodified submodule, the patches are applied at build time. The TensorFold code the patches modify stays under its license. Its third-party notices: `vendor/TensorFold/THIRD_PARTY_NOTICES.md` (the patches extend it). |
| TensorFold DeepSeek-V4.1 tools (`tools/tensorfold-dsv41/`) | this project's Apache-2.0, except files carrying an MIT SPDX notice (MIT, Copyright (c) 2026 Jay Leaton) and `triton_aot_manifest.py` (TensorFold's, Apache-2.0, TensorFold contributors); see its [README](../tools/tensorfold-dsv41/README.md). |
| Files the patches add | keep the SPDX notice written in them: the DeepSeek-V4.1-Flash family (`families/deepseek_v41/`, its tests) is MIT, Copyright (c) 2026 Jay Leaton; the GLM Spark engine (`families/glm5_next/spark/`) is MIT ([`NOTICE`](../NOTICE)). |
| RoCE all-gather and fast-prefill kernels in `patches/0001` | adapted from / re-implementing [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0, Luke Alonso and the b12x contributors); details in [`NOTICE`](../NOTICE). |
| Fat-expert MoE kernel structure in `patches/0001` | adapted from the Apache-2.0 [Reederey87 kit](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) (code MiaAI-Lab contributed under MIT before 2026-09-07); its NOTICE is reproduced in [`NOTICE`](../NOTICE). |
| Ported upstream code in `patches/0001` | from later TensorFold releases (0.3.6.2, 0.5.0), MIT, Copyright (c) 2026 TensorFold contributors; each ported piece names its source commit. |
| xgrammar (structured output) | Apache-2.0 ([mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)), installed into the image by pip, not vendored. |
| Docker base image | NVIDIA Deep Learning Container License (`nvcr.io/nvidia/pytorch:26.07-py3`) |
| Model weights (not included) | `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (measured here), its base `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`, and `deepseek-ai/DeepSeek-V4.1-Flash` (for the Engram tables): each under its model card's terms. The uncensored weights have refusals removed; you are responsible for how you use them. |

Nothing from the MiaAI-Lab DeepSeek kit's AGPL-3.0 code is included: the engine reads the pack and the Engram shard
format (file-format facts), and implements DeepSeek's prompt encoding from DeepSeek's own MIT `encoding.py`.
