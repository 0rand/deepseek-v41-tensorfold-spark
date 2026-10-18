# TensorFold serving DeepSeek-V4.1-Flash on two DGX Sparks (GB10, sm_121, aarch64).
#
# Base: NVIDIA's PyTorch container (CUDA toolkit, PyTorch, Triton). TensorFold comes from the pinned submodule
# (vendor/TensorFold = v0.6.0) with the patches in patches/ applied in order, so the image is reproducible from this repo.
# scripts/serve.sh build builds it on the head and ships it to the worker (docker save | ssh worker docker load).
#
# The production measurements in README.md ran an image of this content: the same base, xgrammar 0.2.8,
# transformers 5.17.0, and the engine tree these patches produce (engine commit 7bd2d67 in the development history,
# plus a packaging-only fix; G19's rows ran 7bd2d67; older rows name their commit).
FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TENSORFOLD_NO_UPDATE_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton \
    CUDA_CACHE_PATH=/cache/nv/ComputeCache \
    CUDA_CACHE_MAXSIZE=4294967296
# Every kernel cache lives in the /cache volume (scripts/serve.sh: CACHE_VOL), so a new container reuses them:
# torch C++ / CUDA extensions, Triton's compiled kernels, the driver's PTX JIT cache, the native Engram reader and
# the measured verify-cost tables. scripts/serve.sh build prebuilds the CUDA extensions on both nodes with no weights
# loaded (scripts/serve.sh prebuild, scripts/prebuild_ext.py); the first start compiles the Triton kernels.

# The RoCE all-gather (GLM53_TF_COMM_BACKEND=roce) builds its RDMA proxy against libibverbs at first use. The 26.07
# base ships libibverbs-dev + ibverbs-providers; install them only if a base ever drops them.
RUN test -f /usr/include/infiniband/verbs.h || (apt-get update \
 && apt-get install -y --no-install-recommends libibverbs-dev ibverbs-providers && rm -rf /var/lib/apt/lists/*)

# Structured output (TF_DSV41_GRAMMAR=1): xgrammar 0.2.8 (Apache-2.0). The base image's torch is a pre-release, so
# pip cannot resolve xgrammar's `torch>=1.10` against it: xgrammar goes in with --no-deps, then transformers (which
# does not require torch). The build fails if torch's version changed.
RUN python -c "import torch; print(torch.__version__)" > /tmp/torch-before.txt \
 && pip install --no-deps "xgrammar==0.2.8" \
 && pip install "transformers==5.17.0" \
 && python -c "import importlib.metadata as md, torch, xgrammar, transformers; \
v = open('/tmp/torch-before.txt').read().strip(); assert torch.__version__ == v, (torch.__version__, v); \
print('xgrammar', md.version('xgrammar'), 'transformers', transformers.__version__, 'torch', torch.__version__)" \
 && rm /tmp/torch-before.txt

# TensorFold v0.6.0 (the submodule) + patches/*.patch in order. TF_SRC: another source tree in the build context
# instead (for example an export of the engine branch: TF_SRC=build/tf PATCHES=none).
ARG TF_SRC=vendor/TensorFold
ARG PATCHES=""
COPY ${TF_SRC} /src/TensorFold
COPY patches /src/patches
RUN cd /src/TensorFold \
 && { test -f pyproject.toml || { echo "vendor/TensorFold is empty: git submodule update --init" >&2; exit 1; }; } \
 && rm -rf .git \
 && if [ "$PATCHES" != none ]; then for p in $(ls /src/patches/*.patch | sort); do \
      echo "applying $p"; git apply --whitespace=nowarn "$p" || patch -p1 --forward < "$p" || exit 1; done; fi \
 && pip install --no-deps /src/TensorFold \
 && pip install "huggingface-hub>=0.24" "tokenizers>=0.20" "safetensors>=0.4" "jinja2>=3.1" \
 && python -c "import tensorfold.families.deepseek_v41 as f, torch, triton, pathlib; \
d = pathlib.Path(f.__file__).parent / 'cuda'; \
miss = [n for n in ('csa2/attn_cuda.cpp', 'csa2/attn_cuda.cu', 'csa2/topk_cuda.cu', 'pfdense.cuh', 'pfdense_args.h') if not (d / n).is_file()]; \
assert not miss, ('kernel sources missing from the installed package', miss); \
print('tensorfold', f.__file__, 'torch', torch.__version__, 'triton', triton.__version__)"

ARG TF_SRC_COMMIT=""
LABEL dsv41.tf_src=${TF_SRC} dsv41.tf_src_commit=${TF_SRC_COMMIT}

ENTRYPOINT ["python", "-m", "tensorfold"]
