# The engine: TensorFold v0.6.0 + three patches

## What ships

| | |
| --- | --- |
| `vendor/TensorFold` | upstream [TensorFold](https://github.com/ashhart/TensorFold) at tag `v0.6.0` (commit `c464617`), unmodified submodule |
| `patches/0001-spark-stack-060.patch` | `families/glm5_next/spark/`: the GLM-5.3-Flash two-Spark engine (TensorFold 0.3.4's GLM CUDA engine with the [glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) patch series 0001-0620) rebased onto 0.6.0; the CUDA communicator interface (`cuda/comm.py`); a family `CUDA_SERVE` hook; a server fix for sockets past descriptor 1023. The DeepSeek family reuses its RoCE all-gather, communicator and HTTP server. |
| `patches/0002-deepseek-v41-family.patch` | `families/deepseek_v41/` (MIT) and its tests; a device-side skip in the EXL3 linear (graph-safe), fp64 in the communicator's dtype table, `--kv-dtype fp8`, more model ids in `/v1/models`; G14's changes to the GLM stack's RoCE path (`GLM53_TF_ROCE_FAST`, the host mailbox `roce_msg.h` the round plan uses, `roce_bench` / `roce_split`); package data for the family's CUDA sources and headers; NOTICE and THIRD_PARTY_NOTICES entries |
| `patches/0003-dsv41-literal-added-tokens.patch` | `families/deepseek_v41/cuda/literal_tokens.py` and its tests: a DeepSeek added token quoted inside a chat request (`</think>`, `<｜User｜>`, `<｜end▁of▁sentence｜>`, `<｜DSML｜>`, ...) stays text instead of becoming that token; `app.py` hides the literals before the template renders and encodes them as plain pieces |

`docker/Dockerfile` applies all three in order with `git apply` on a copy of the submodule and installs the result
with pip.

## How the patches were made, and what differs from production

The engine was developed on a private branch: 219 commits on `v0.6.0` up to the first publication (85 for the GLM
stack, 134 for the DeepSeek family), then 9 more (G11's drafter self-distillation tooling and G12's host-memory, stall
and RoPE fixes), then 8 more (G13: a test-order fix, the five decode rewrites behind levers that default to 0, and
test fixes made on the hardware), then 39 more (G14-G19: the window levers, image input, the serving-memory fixes,
fail-fast and the priced floor, the prefill levers, adaptive rows), then one packaging fix. Production runs commit
`7bd2d67` of that branch (it ran `38f6500` until G12, `a6f5792` until G13, `767ad9f` until G14, then `356a188`,
`7954c1d`, `da5ae43` and `cd4245a`); `66d0dcd` on top of it only adds the family's CUDA sources and headers to the
package data (production mounts the source tree, so it never needed them; an installed engine does). For publication
the branch was squashed into the two commits above and re-authored; each update regenerated `0002` the same way
(`0001` is unchanged since the first publication, so G14's RoCE changes to the GLM stack's files are in `0002`). Every
engine change production runs is in them (455 files over v0.6.0).

Applying `patches/` to v0.6.0 and diffing the result against `66d0dcd` leaves exactly these differences:

| file | difference | why |
| --- | --- | --- |
| 32 files in `families/deepseek_v41/` (17), `families/glm5_next/spark/` (`decode_stream.py`, `l2pf.py`, `roce.py`, `roce.cu`, `roce_bench.py`) and `tests/` (10) | comments and docstrings only, except the three rows below | machine names, development paths, link addresses and agent / development-process wording reworded |
| `families/deepseek_v41/cuda/pf4k.py` | `DOCS`, the string a refused 4,096-row boot prints, names this repository's `docs/campaign/` instead of the private development repository | the same reason |
| `families/glm5_next/spark/roce_bench.py` | `--master` is required (it defaulted to the development link's address) | the same reason |
| `families/deepseek_v41/cuda/draft_vocab.txt` | absent | counted from private chat transcripts. Only `TF_DSV41_DRAFT_HEAD=trim` reads it (off, not adopted); `test_dsv41_draft_head.py::test_shipped_ranking` fails without it |
| `families/glm5_next/spark/draft_vocab.txt` | replaced | same origin; replaced by the public-text ranking the GLM recipe publishes (its `patches/0420`, `bench/draftvocab_public.py`). Read only with `GLM53_TF_DRAFT_VOCAB` (GLM engine) |
| `NOTICE`, `THIRD_PARTY_NOTICES.md` | an entry for the DeepSeek family; the deployment names | licensing record |

No other code differs: the two rows above are an error message and a benchmark's command line.
`THIRD_PARTY_NOTICES.md` also gains two lines on the image path (DeepSeek's reference, MIT) and on the ideas
re-implemented from bertholomus/TensorFold (no code copied). Checked for this update: the patches apply to v0.6.0 with
`git apply --check`, the result's tree equals the squashed commit, an offline wheel build of it contains every CUDA
source and header of the family, and the family's CPU suites pass (see Tests).

## The CUDA extensions

The family compiles its CUDA extensions with `torch.utils.cpp_extension` at first use, into `TORCH_EXTENSIONS_DIR`
(`/cache/torch_extensions` in the `CACHE_VOL` volume). Building them beside ~100 GB of weights once took the worker
to 1.26 GiB MemAvailable (G6), so `scripts/serve.sh build` ends with `scripts/serve.sh prebuild`: on both nodes, with
no weights loaded, it removes stale build locks and runs [`scripts/prebuild_ext.py`](../scripts/prebuild_ext.py),
which builds every extension a rank can load (18: the G13 ones, `mhc_cuda`, `csa2.attn_cuda` with the top-k kernels,
`moe_fused`, `dense3`; `l2pf` and G14's paced prefetch; the G16-G17 prefill kernels `mhc_pf` and `pfdense`). A build
that fails is reported and makes the command fail. Then it puts the cache volume's data files in place
(`scripts/serve.sh cache`): `config/pfdense-table.json` at `TF_DSV41_PF_DENSE_TABLE`, and for native image input the
66 KB image routing bias at `TF_DSV41_BIAS_VL` (`python -m tensorfold.families.deepseek_v41.cuda.bias_vl_fetch DIR`,
HTTP range requests against `deepseek-ai/DeepSeek-V4.1-Flash`). Run `scripts/serve.sh prebuild` again after changing
the image or clearing the volume.

## The same tree as a git branch

```bash
git clone https://github.com/ashhart/TensorFold.git && cd TensorFold
git checkout -b deepseek-v41-tensorfold-spark v0.6.0
git am /path/to/deepseek-v41-tensorfold-spark/patches/*.patch
```

A public fork carrying this branch (suggested: `<you>/TensorFold`, branch `deepseek-v41-tensorfold-spark`) would let the
Dockerfile build from it directly (`TF_SRC=<export of the branch> PATCHES=none`); none is published yet.

## Tests

The family's tests are in the patched tree (`tests/test_dsv41_*.py`, `tests/cuda/test_dsv41_*.py`). CPU suites (fake
forward, Triton interpreter, emulators) run anywhere with torch, numpy, safetensors and tokenizers; the CUDA suites
need a GB10 and some the real pack (`TF_DSV41_TEST_MODEL=/model`). In the image, against an export of the patched tree:

```bash
mkdir -p build/tf && git -C vendor/TensorFold archive HEAD | tar -x -C build/tf
(cd build/tf && for p in ../../patches/*.patch; do git apply "$p"; done)
docker run --rm --gpus all -v "$PWD/build/tf:/tf" -w /tf -e PYTHONPATH=/tf/src --entrypoint python \
    dsv41-tensorfold:060 -m pytest -q tests/test_dsv41_serving_app.py tests/test_dsv41_serving_batch.py
```

## Upstream

The communicator interface and the descriptor fix in `0001` were offered upstream separately. The DeepSeek family is
deployment code for two Sparks and this pack; it is not proposed for upstream as is.
