# TensorFold DeepSeek-V4.1 tools

The development tools of TensorFold's DeepSeek-V4.1 Zig family, kept here instead of in the engine: references,
fixture generators, Triton AOT capture and fill, kernel and SASS diagnostics, benchmarks and profile analysis.
They were written beside the family's upstream pull requests and moved out of
[ashhart/TensorFold](https://github.com/ashhart/TensorFold) unchanged apart from the paths noted below.

- Source: TensorFold DeepSeek-V4.1 stack `5aefae17` (the commit that last changed any of them). `MAP.tsv` lists every
  file's TensorFold path, its path here and the commit it was taken from.
- `triton_aot_manifest.py` is a copy of TensorFold's own `tools/zig/triton_aot_manifest.py` at `e7c523eb` (upstream
  main), which the capture, fill and reference tools import. TensorFold keeps its original.
- Generator labels inside written files (`"generator": "tools/zig/..."`) still name the TensorFold path, as the
  fixtures committed in TensorFold do.

The engine itself does not need any of them: its build and tests read the stored fixtures these tools once wrote.

## Layout

| Here | Was in TensorFold | What it is |
| --- | --- | --- |
| `dsv41_*.py`, `triton_*.py`, `dsv41_triton_options.json` | `tools/zig/` | references (decode, forward digests, KV, sampling, slots), Triton capture / AOT fill, rope tables, Engram host tables, expert widths |
| `dsv41_{bench,perf,diverge,gm2pf,sched,session_slots,slots,ops,triton}/` | `tools/zig/` | benchmarks, profile and acceptance analysis, divergence, scheduler / session goldens, Triton kernel checks |
| `kernels/` | `zig/kernels/cuda/deepseek_v41/` | the offline SASS identity check (`sass_check.sh`, `sass_check.py`, `py_sources.py`), `proto/nsys_classes.py` |
| `draft/` | `zig/src/families/deepseek_v41/draft/tools/` | the five drafting golden generators |
| `cuda_tests/` | `zig/tests/cuda/deepseek_v41/` | `oracle.py` / `oracle_glue.py` (device-test fixtures), `dsv41_capture.py` (replay capture plugin) |
| `server_tests/` | `zig/tests/server/dsv41/` | DSML, sampling, template and tokenizer golden generators |

`dsv41_ops/http_golden.py` and `wire_probe.py` produced the reference-server wire golden that TensorFold no longer
carries; they are kept for the record.

## Dependencies

Only `check.py` and the tests run on the standard library alone (Python 3.10 or later). The rest, by tool:

- the TensorFold Python engine (`tensorfold` package, passed as `--py-src`, `--src` or on `PYTHONPATH`), PyTorch,
  Triton, NumPy: references, capture, fill, oracles and goldens; a CUDA GPU where a tool says so
- `tokenizers`: `dsv41_diverge/diverge.py`, `server_tests/gen_tokenizer.py`, `dsv41_ops/http_golden.py`
- `jinja2`: `server_tests/gen_template.py`
- Nsight Systems SQLite exports: `dsv41_bench/*timeline*.py`, `dsv41_perf/{nsys_draft,roundgap,kcmp}.py`
- Docker and the NGC image `nvcr.io/nvidia/pytorch:26.07-py3`: `kernels/sass_check.sh`
- `dsv41_decode_ref.py --grammar` loads `grammar_ref.py` from `TF_DSV41_GRAMMAR`, else from this repository's
  engine snapshot, `engine/zig/tools/zig/dsv41_grammar/`

## Paths that changed

- `kernels/sass_check.sh` used to compile the kernel copies in its own directory. Point it at a TensorFold checkout:

  ```
  OUT=/scratch/sass SM=121 TF_DSV41_KERNELS=$TF/zig/kernels/cuda/deepseek_v41 TF_DSV41_REF_SRC=$REF \
    tools/tensorfold-dsv41/kernels/sass_check.sh
  ```

  `py_sources.py` no longer guesses a `tensorfold` clone beside the checkout: `TF_DSV41_REF_SRC` is required, and the
  `_X3GM2` / `_R1` keys default to it. A missing setting stops the check before anything is built.
- `dsv41_decode_ref.py --grammar`: see Dependencies.

## Checks

```
python3 -I tools/tensorfold-dsv41/check.py
```

It compiles every file, checks every import resolves inside this directory or names a dependency above, runs each
argparse tool's `--help`, checks the scripts' executable bits, and runs the unit tests (`tests/`, the comparator and
source map of the SASS check, the AOT merge, the closure check itself; `dsv41_perf/acclog_replay_test.py`;
`dsv41_bench/kv_glue_gpu_test.py`, a GPU-free dry run) and `dsv41_perf/gap.py selftest`.

These checks qualify the tools, not the engine. Reference runs, captures, AOT compiles, SASS identity and every
GPU measurement stay separate receipts, made with these tools on the hardware they name.

## License

Files with an `SPDX-License-Identifier: MIT` line keep it (Copyright (c) 2026 Jay Leaton; their
`THIRD_PARTY_NOTICES.md` is TensorFold's). `triton_aot_manifest.py` is TensorFold's (Apache-2.0, TensorFold
contributors). The rest is under this repository's [license](../../LICENSE).
