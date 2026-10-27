# The engine and the repository layout

# The engine

`vendor/TensorFold` is upstream TensorFold v0.6.0, unmodified. `patches/` holds three patches, applied in order by the
Dockerfile:

| patch | what | licence |
| --- | --- | --- |
| [`0001-spark-stack-060.patch`](../patches/0001-spark-stack-060.patch) | the GLM-5.3-Flash two-Spark engine (`families/glm5_next/spark/`) rebased onto 0.6.0, the CUDA communicator interface (`cuda/comm.py`), the family `CUDA_SERVE` hook (`cli.py`, `families/glm5_next/__init__.py`), the server's descriptor fix (`server/cancellation.py`), packaging (`pyproject.toml`), recipes and tests |
| [`0002-deepseek-v41-family.patch`](../patches/0002-deepseek-v41-family.patch) | `families/deepseek_v41/` and its tests, the EXL3 linear's device-side skip (`cuda/exl3/linear.*`), fp64 in `cuda/comm.py`, `--kv-dtype fp8` (`cli_args.py`), model aliases in the GLM server, G14's RoCE changes to the GLM stack (the faster all-gather kernel, the host mailbox the round plan uses, the exchange benchmark and split tools, `families/glm5_next/spark/roce*`), packaging (the CUDA sources and headers as package data), NOTICE entries |
| [`0003-dsv41-literal-added-tokens.patch`](../patches/0003-dsv41-literal-added-tokens.patch) | `families/deepseek_v41/cuda/literal_tokens.py` and its tests: a DeepSeek added token quoted inside a chat request (`</think>`, `<｜User｜>`, `<｜end▁of▁sentence｜>`, `<｜DSML｜>`, ...) stays text instead of becoming that token; `app.py` hides the literals before the template renders and encodes them as plain pieces |

Together they are every engine change production runs (development commit `7bd2d67`, plus `66d0dcd`, a packaging-only
fix: 455 files over v0.6.0): applying them to v0.6.0 reproduces that tree except for reworded comments, two
documentation strings, one benchmark argument and the excluded draft-vocabulary files ([`docs/ENGINE.md`](ENGINE.md)
lists each difference).

[`docs/ENGINE.md`](ENGINE.md) explains how the patches were produced, how to get the same tree as a git branch,
and how to run the engine's test suites. [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) summarises the model and the
TP=2 split.

# Layout

| path | what |
| --- | --- |
| `vendor/TensorFold` | upstream TensorFold v0.6.0 (submodule) |
| `patches/` | the engine changes ([`docs/ENGINE.md`](ENGINE.md)) |
| `docker/Dockerfile` | the image: NVIDIA PyTorch 26.07 + xgrammar + TensorFold with the patches |
| `config/prod.env.example` | the measured configuration, with placeholders for your hosts and paths |
| `config/pfdense-table.json` | the measured tile table of the dense prefill GEMM (`TF_DSV41_PF_DENSE_TABLE`; `scripts/serve.sh prebuild` copies it into the cache volume) |
| `scripts/serve.sh` | build / prebuild / cache / preflight / start / stop / status / watchdog / `run` (engine benchmarks on both ranks) |
| `scripts/prebuild_ext.py` | builds every CUDA extension a rank loads (`scripts/serve.sh prebuild` runs it in the image on both nodes) |
| `scripts/pack_engram.py` | the per-rank Engram shards from DeepSeek's checkpoint |
| `scripts/canary.py`, `scripts/boot-start.sh`, `scripts/systemd/` | post-start canary, start at boot, watchdog units |
| `scripts/check-public.sh`, `scripts/check_secrets.py` | the sanitizer this repository was checked with, and its credential scan (allowlist: `scripts/check-secrets-allow.txt`) |
| `scripts/redact.py` | masks API keys and tokens in command lines, environments, logs and results before a tool saves or prints them |
| `bench/` | HTTP clients: quality (MMLU, needles), structured output, soak, stress, tool calling |
| `docs/` | results, benchmark method, architecture, decode roofline and lessons, operations, the engine |
| `results/` | the raw files behind the tables ([`results/README.md`](../results/README.md)); `results/campaign/` the tracked results of the development windows G1-G19 |
| `docs/campaign/` | the development log: plans, targets, the landscape study, the results of every window G1-G19, the vision design ([`docs/campaign/README.md`](campaign/README.md)) |
| `engine/`, `tests/` | the development staging tree: the PyTorch reference model (the correctness oracle), the kernels and serving layer before they were ported into the TensorFold family, and their tests |
| `scripts/campaign/` | analysis tools of the windows (nsys window / idle / skew breakdowns, summaries), the draft-vocabulary study tool, the porting script |
