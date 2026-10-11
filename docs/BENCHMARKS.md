# Benchmark method, and how to rerun each cell

Two kinds of measurement: **engine benchmarks** (`m2bench`, `gate`) run inside the engine on both ranks with no HTTP
server, and **HTTP clients** (`bench/`) run against a started server. The kit was measured with HTTP clients only.
Each table in [RESULTS.md](RESULTS.md) says which one it is.

Before any run: nothing else on the GPUs, the page cache dropped (`scripts/serve.sh` does it), and MemFree >= 104 GiB
on both nodes (`MEM_GATE_GIB`). Single-run differences under ~2% are noise on this pair; the tables give ranges over
runs where there were several.

## 1. Engine benchmarks: `scripts/serve.sh run`

`scripts/serve.sh run MODULE [ARGS]` stops nothing: stop the server first. It starts rank 1 on the worker and rank 0
here with the production config's knobs (a caller export overrides one), adds `--model /model --rank R --master
HEAD_IP --port MASTER_PORT`, and mounts `OUT` (default `results/run-<time>/`) at `/out` on rank 0.

**Decode speed** (the README's code / prose / structured / C1 / C2 / C4 cells):

```bash
scripts/serve.sh stop
OUT=$PWD/results/run-speed scripts/serve.sh run tensorfold.families.deepseek_v41.cuda.m2bench \
    --out /out/m2.json --only code,prose,structured --streams 1,2,4 --c-sampling mixed --reps 2 \
    --c-same code,prose --c-reps 2 --c-exact
```

The G15-G19 speed checks ran this set two or three times on fresh boots, interleaving the configurations compared.

- Single stream: each workload at T = 0 and T = 0.7 (keyed), tok/s from the first to the last token of a 384-token
  reply, the median of `--reps`. Every reply is compared with the serial reply (`draft=False`): `exact_all` must be
  true (exit code 3 otherwise).
- `--streams 1,2,4`: 1, 2 and 4 concurrent requests cycling code, prose, a JSON task and a copy-heavy edit, every
  other one at T = 0.7 (`--c-sampling mixed`; `greedy` for all T = 0). `decode_aggregate_tok_s` = all tokens / (last
  token - first token): the cells the README reports, the same definition as the kit's `multiturn` aggregate.
- `--c-same code,prose`: same-workload cells (C2-code, C4-prose, ...); `--c-distinct` gives each stream its own
  prompt of the workload (otherwise the same prompt runs on every stream). Every multi-stream cell also reports the
  **steady** aggregate (all streams live). `--c-exact` replays every concurrent stream alone and serial and compares
  the replies (`concurrent_exact_all`).
- `m2bench` sizes the request slots for `--context` (16,384 by default), not the production 300K pool.
- `TF_DSV41_PHASES=1` adds per-phase host timings; `--nsys code|prose|c2|...` runs one workload under
  `cudaProfilerStart/Stop` for an nsys capture (add the nsys entrypoint yourself).

**Cold prefill:**

```bash
OUT=$PWD/results/run-prefill scripts/serve.sh run tensorfold.families.deepseek_v41.cuda.m2bench \
    --out /out/prefill.json --prefill 8192,32768,65536,131072 --prefill-reply 32 --context 139264 --parallel 1
TF_DSV41_PREFILL=full OUT=... scripts/serve.sh run ... --prefill ...     # the exact prefill
```

One slot, fresh random text, after a 2K warm-up; tok/s and TTFT from the batcher. `--prefill-reply 32` also records
the first token and a SHA-256 of a 32-token greedy reply at each size: the G17-G19 tables compare them across
configurations and boots (prefill levers and row counts must not change them).

**Prefix cache.** A prompt that repeats a filler document hits the prefix cache: the repeated part is not prefilled
again, so the prefill rate comes out impossibly fast (one report: an 87K-token prompt back in about 0.4 s). Check
`usage.prompt_tokens_details.cached_tokens` in the reply: a value close to `prompt_tokens` means the cache served the
prompt. Use unique text per prompt for prefill numbers, as `--prefill` does. The Zig server also reports it per reply
(`tensorfold.cached`), in `/health` (`cached_tokens_total`) and in `/metrics` (`tensorfold_cached_tokens_total`).

**Top-1 agreement with the kit** (needs the kit's prompt-logprobs capture, included):

```bash
cp results/kit-baseline/oracle-kit.json "$OUT/"
OUT=$PWD/results/run-gate scripts/serve.sh run tensorfold.families.deepseek_v41.cuda.gate \
    --capture /out/oracle-kit.json --out /out/gate.json
```

The capture is the kit's `prompt_logprobs` (top 5) for 8 prompts repeated to 2,048 tokens. The gate feeds the same
token ids through our engine teacher-forced and reports whole-sequence and first-copy top-1 agreement (the first
copy is the part where the model cannot copy itself). Run it with `TF_DSV41_PREFILL_KERNELS=exact` to gate decode
levers (the fast kernels only cover prompt rows).

## 2. HTTP clients: `bench/` (standard library only)

Against a running server (`scripts/serve.sh start`; a second test server: `PORT=8001 NAME=dsv41-tf-test
MASTER_PORT=29561 HEAD_STATE=... WORKER_STATE=... scripts/serve.sh start`).

| client | what | gate used here |
| --- | --- | --- |
| `bench/quality.py mmlu --data bench/data/mmlu200.jsonl` | MMLU-200, 0-shot, thinking off, greedy, max_tokens 8, first A-D letter | within 1 point of the kit (87.5%) |
| `bench/quality.py mmlu ... --shots 20` | the same after a 20-question preamble: ~2.1K-token prompts, so CED replay really runs | replay vs full within 1 point |
| `bench/quality.py needle --sizes 32768,131072,299000` | a passkey at 1/3 depth of a filler document | found |
| `bench/quality.py compare --replay R.json --full F.json --kit 0.875` | the two runs side by side | |
| `bench/structured.py` | 6 JSON schemas x thinking off / on, each drafted, serial (`"draft": false`) and 4 at a time; tool_choice required / named / strict / none / streamed | drafted == serial == batched, valid, no markup |
| `bench/tooleval/chains.py --mode off\|low\|high` | 6 multi-step tool scenarios with simulated tools, history sent back as agent clients do; 12 points | >= 10 |
| `bench/tooleval/run.sh setup`, then `teb\|sb\|both\|chains\|own off\|low\|high` | tool-eval-bench (69 scenarios; category C = multi-step) and spark-bench TrueScore at pinned commits | |
| `bench/soak.py --minutes 30` | 1-4 streams of mixed kinds (code, prose, structured, tools, thinking, 8K-48K documents), ~15% streamed cancels, ~5% client disconnects; `/health` every 10 s | 0 errors, inflight back to 0, 17*23 = 391 |
| `bench/soak.py --minutes 60 --kinds g16` | the same plus growing agent sessions (40 tools, to 120K tokens, screenshots), image requests and idle gaps: the G16-G19 serving-memory soaks. In-stream error events (a refusal after the 200) count as errors | 0 errors; sample RssAnon and MemAvailable on both nodes (G19: RssAnon slope < 0) |
| `bench/stress.py stress --long 299000 --dctx 65536 --decode 2048` | one 299K prefill while three 64K prompts decode 2,048 tokens (ignore_eos) | every stream complete; watch MemAvailable on both nodes |
| `bench/stress.py admit` | seconds to the first token of one request after a start (admission) | |

Common flags: `--base http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-TF --out R.json`. Sample MemAvailable on
both nodes while they run (`awk '/MemAvailable/' /proc/meminfo` every second; avoid polling `nvidia-smi` every second,
see the README).

## 3. The kit baseline

The kit's cells come from the GLM recipe's HTTP clients against the kit's server: `bench/glmbench.py` (suites tf,
tweet, kit, edit), `bench/multiturn.py concurrent --streams 1,2,4`, `bench/quality.py` (MMLU-200) and a long-prefill
client (an exact token-id prompt of a repeated filler document, 8K-256K). Raw files: `results/kit-baseline/`.
