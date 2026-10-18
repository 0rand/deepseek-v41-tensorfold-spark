# DeepSeek-V4.1-Flash-1017 EXL3 2.8 bpw (flash17): the recommended pack

flash17 is the **recommended pack** for this recipe, served by the Zig engine with
[`config/prod-zig-flash17.env.example`](../config/prod-zig-flash17.env.example). The previous pack, q28-v2, remains
documented as the [rollback](#rollback-to-q28-v2) with its own profile,
[`config/prod-zig.env.example`](../config/prod-zig.env.example).

**Status:** flash17 is serving as the production default on our two Sparks with this profile, under the model ID
`DeepSeek-V4.1-Flash-TF`.

## The pack

| | |
| --- | --- |
| Repository | [`jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw) |
| Base | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170` (MIT) |
| Plan | `q28-v3a` (flash17); the `v3b` alternative was encoded but not evaluated |
| Bits per weight | **2.80** (2.8003 on the 2.9 bpw pack's definition); routed experts 2.77 (2/3/4/5 bits per expert), attention 5.0, shared experts 5.38, CSA2 5.0, Engram projection 5.0, head 6, MTP / draft 4 |
| Size | **203.8 GB** in 47 pack files (39 MB smaller than q28-v2), at most 102.09 GB a rank |
| Vision | included: the 1017 vision tower and aligner (263 bf16 tensors, 0.971 GB) and the 43 learned image routing biases, in `native.safetensors` |
| Manifest | `SHA256SUMS` in the repository; first 16 hex digits of its own sha256: `5d4b0c1b650da70a` |

Calibration was weighted toward this recipe's use: 24 % code, 24 % math + science reasoning, 30 % tool calls, with
chat, prose, web text and multilingual text (4 %) as guards. The method and the full card are in the Hugging Face
repository.

What it needs beyond the general Zig setup ([ZIG-SERVE](ZIG-SERVE.md)):

- **The 1017 Engram shards.** Every Engram weight and scale changed in 1017; the September shards do not match this
  pack. Pack them from the 1017 release as in [INSTALL step 3](INSTALL.md), with `deepseek-ai/DeepSeek-V4.1-Flash-1017`
  as the source, and run `--check` on the result.
- **`TF_DSV41_BIAS_VL=/model`**: the pack's own 1017 image routing biases, read from the model folder on both ranks.
  No separate `bias_vl.safetensors` fetch is needed for this pack.
- **Its own prepared and state folders.** Do not reuse another pack's prepared files, calibration tables or sessions.
- **Chat template note.** 1017 ships no `chat_template.jinja`. The pack carries September's file (the same file
  q28-v2 has) only because some older TensorFold builds look for it at boot. Zig `9131e10a` renders prompts with its
  own encoder and does not read it. 1017's `encoding/encoding.py` is byte-identical to the September repository's
  current one: DeepSeek's 2026-10-16 encoder fix (a user turn holding only tool results does not start a new turn) is
  in both releases, so it is not a 1017 change. Zig `9131e10a`'s encoder predates that fix, for both packs alike; it
  only affects continuations that carry tool results without declaring `tools`.

## Recommended serving profile

[`config/prod-zig-flash17.env.example`](../config/prod-zig-flash17.env.example) is the q28-v2 serving profile with
only these changes, served behind the thermal and memory guard. Engine Zig `9131e10a` (the exported [`engine/zig`](../engine/zig)), TP=2, four slots, 1M context,
split + compact KV, expert top-p 0.85 / min 3, native images, 4K prefill, floors 5 / 4 GiB and the 3.0 GiB serving dip
are all unchanged.

| Setting | q28-v2 profile | flash17 profile | Why |
| --- | --- | --- | --- |
| `MODEL_DIR` | q28-v2 pack | flash17 pack | the weights |
| `ENGRAM_DIR` | September shards | **1017 shards** | the tables changed in 1017 |
| `PREPARED_DIR` / `STATE_DIR` | q28-v2's | **its own** | prepared files and calibration belong to one pack |
| `TF_DSV41_BIAS_VL` | `/cache/dsv41-bias-vl` | **`/model`** | the 1017 learned image biases |
| `TF_DSV41_POOL_TOKENS` | 3000000 | **2100000** | memory headroom ([below](#memory-the-21m-pool)) |
| `TF_DSV41_PF_WS_FLOOR_GIB` | 3.5 | **4.75** | a planned 4K workspace stays above the 4 GiB floor |
| `TF_CUDA_SCHED` | unset | **`blocking`** | the host sleeps instead of spinning while it waits for the GPU |
| `TF_DSV41_PLAN_PIN` | `auto` | **`1`**, inside `CPUSET_CPUS` | plan thread on a chosen host CPU |
| `CPUSET_CPUS` | unset | **host CPUs from your topology** | [host CPU affinity](OPERATIONS.md#zig-serving-thermal-and-memory-guard) |
| `PORT` | 8000 | **8001** | the guard serves 8000 and forwards to the engine |

We suggest a 2100-2200 MHz GPU clock cap (`nvidia-smi -lgc 300,2100`) to help with temperature control.

The full profile, with the guard, was qualified over a 3-hour soak of long agentic sessions up to ~156K tokens with
RAM resume, 322 requests, no errors. Setup, the guard and the watchdog are in
[OPERATIONS](OPERATIONS.md#zig-serving-thermal-and-memory-guard); qualify your own machines before you rely on it
([Qualifying your own setup](OPERATIONS.md#qualifying-your-own-setup)).

Each path key is set per node, so beyond the settings above the profile changes model, prepared, Engram and state
folders on each of the two nodes, plus `TF_DSV41_BIAS_VL`.

With a 2.1M-token pool, one slot can still fill its 1M context and two slots 1M each; four live slots share about
525K each. The pool is shared across live slots, not reserved per slot.

```bash
cp config/prod-zig-flash17.env.example config/prod-zig.env   # on each node; set the paths and link
scripts/run-zig.sh 1 --check-aot                             # rank 1, then rank 0
scripts/run-zig.sh 1                                         # rank 1, then rank 0 on its node
```

Before any load, check both ranks: after-boot MemAvailable minus the 3.0 GiB serving dip must leave at least
4.25 GiB (the 4 GiB floor plus 0.25 GiB margin). Then watch both ranks' MemAvailable during the first long prompts,
not just at idle.

## Memory: the 2.1M pool

All on the same two Sparks, Zig `9131e10a`, unchanged knobs except the pool, 2026-10-17/18 UTC. Lows are sampled
about every 1.4-2 s, not continuous.

| Pool (tokens) | Load | Head / worker before load (GiB) | Engine guard worst point (GiB) | Observed low head / worker (GiB) | Against the 4 GiB floor |
| ---: | --- | --- | --- | --- | --- |
| 3,000,000 | agent + task load, first boot | 6.93 / 6.95 (after canaries) | not retained | **3.977** / 5.106 | under |
| 3,000,000 | same, reboot with populated state | 7.52 / 7.64 | 4.42 / 4.63 | **3.558** / 4.13 (engine log) | under, first 4K prefill |
| 2,400,000 | same | 8.14 / 7.91 | 5.09 / 4.91 | **3.960** / 4.361 | under, first 4K prefill |
| 2,200,000 | same | 8.23 / 8.46 | 5.27 / 5.53 | 4.153 / 4.974 | above by 0.15 GiB |
| **2,100,000** | 4K-prefill agent replay + tasks | 8.41 / 8.74 | 5.47 / 5.43 | **4.226** / 5.132 | above |
| **2,100,000** | task suites, three boots | 8.23-8.39 / 8.11-8.49 | 5.37-5.46 / 5.19-5.58 (two boots) | **5.072** / 5.604 (lowest of the three) | above |

**The 4K prefill workspace, not the pack, sets the low point.** Each long prompt allocates a transient 2,268 MiB
(~2.27 GB) 4K prefill workspace on both ranks and releases it after the prompt. The engine admits it when current
MemAvailable minus the workspace stays above `TF_DSV41_PF_WS_FLOOR_GIB`, which the profile sets to 3.5 GiB, below
the 4 GiB hard floor; it does not reserve the 3.0 GiB serving dip on top. `TF_DSV41_PF_4K_MEMORY_OK=1` acknowledges
that cost; it does not bypass the check. So a boot that passes the guard can still go under 4 GiB on the head during
the first long prompt: at 3.0M a reboot passed at 4.42 GiB modelled and fell to 3.558 GiB on a 9,891-token prompt;
its logs show the head recover from 3.41 to 5.64 GiB when the workspace was released. The engine's boot guard warned
that the workspace would leave 2.15 / 2.37 GiB at the modelled worst point; that warning does not block requests.
This mechanism is shared with q28-v2's profile, which has not had the same mixed load as a control.

flash17 also starts about 0.62 GiB a rank lower after boot than q28-v2 on the same engine. Only 21 MiB of that is
weights (the prepared rank folders); the rest is symmetric across ranks and not attributed. The head is the binding
rank because it also holds the HTTP server, the request log, the image cache and the vision tower. **Use 2,100,000
tokens (or less) for this pack**; 3.0M, q28-v2's pool, does not hold the floor.

## Fidelity to the release (retention, as served)

Next-token KL / top-1 agreement % / confident mistakes per 10,000 positions (top-1 differs where the release is at
least 90 % sure), each pack against **its own** release, with the serving expert pruning (top-p 0.85, min 3), on a
held-out per-domain set of about 24.5K tokens a domain. Measured on a RunPod RTX PRO 6000, not on the Sparks.

| Domain | flash17 vs 1017 | q28-v2 vs September | Mia 2.9 bpw vs September | flash17 KL / Mia KL |
| --- | --- | --- | --- | ---: |
| code | 0.0328 / 95.17 / 7 | 0.0269 / 95.35 / 4 | 0.0330 / 94.77 / 7 | x0.99 |
| science | 0.0242 / 94.93 / 2 | 0.0243 / 94.70 / 6 | 0.0310 / 93.88 / 0 | x0.78 |
| math | 0.0197 / 96.58 / 3 | 0.0187 / 96.33 / 3 | 0.0264 / 95.67 / 5 | x0.75 |
| tools | 0.2013 / 87.64 / 109 | 0.1394 / 89.66 / 105 | 0.1889 / 87.94 / 187 | x1.07 |
| multilingual | 0.0982 / 90.68 / 55 | 0.0902 / 91.03 / 31 | 0.1122 / 90.22 / 57 | x0.88 |
| chat | 0.0688 / 91.96 / 27 | 0.0626 / 92.28 / 20 | 0.0731 / 91.45 / 26 | x0.94 |
| all | 0.0739 / 92.84 / 34 | 0.0601 / 93.24 / 28 | 0.0771 / 92.34 / 47 | x0.96 |

The bases differ (1017 vs September), so the columns are not paired. flash17 keeps more of its release than Mia's
pack keeps of September on science, math, chat and multilingual, ties on code, and is slightly worse on tool calls
(KL x1.07, top-1 -0.30 points, with fewer confident mistakes). q28-v2 retains more of September than flash17 retains
of 1017 on code and tools. Teacher-forced short-context fidelity did not predict task results for q28-v2, so the
task suite decides usefulness.

## Task results (356 items)

Same frozen items and scorers as the matched baseline: greedy (T0, top_p 1, seed 0), thinking on, effort high,
max_tokens 32,768, pass@1 (strict score == 1). Conservative paired 95 % intervals: Bonferroni Clopper-Pearson 97.5 %
bounds on the two discordant counts of each comparison, per suite, not simultaneous across suites. Half-width is half
the interval's width.

| Suite (completed / planned) | flash17 | Mia 2.9 bpw | q28-v2 | Δ vs Mia, points [95 % CI]; half-width | Δ vs q28-v2, points [95 % CI]; half-width |
| --- | ---: | ---: | ---: | --- | --- |
| HumanEval+ (40 / 40) | 34 / 40 | 34 / 40 | 33 / 40 | +0.0 [-14.9, +14.9]; 14.9 | +2.5 [-10.3, +14.9]; 12.6 |
| MBPP+ (60 / 60) | 50 / 60 | 49 / 60 | 50 / 60 | +1.7 [-9.9, +12.8]; 11.4 | +0.0 [-12.6, +12.6]; 12.6 |
| MMLU-Pro STEM (100 / 100) | 93 / 100 | 91 / 100 | 90 / 100 | +2.0 [-4.1, +7.9]; 6.0 | +3.0 [-5.3, +10.9]; 8.1 |
| MATH-500 (40 / 40) | 36 / 40 | 36 / 40 | 36 / 40 | +0.0 [-10.4, +10.4]; 10.4 | +0.0 [-10.4, +10.4]; 10.4 |
| SciCode first step (10 / 10) | 6 / 10 | 6 / 10 | 5 / 10 | +0.0 [-35.5, +35.5]; 35.5 | +10.0 [-35.4, +49.1]; 42.2 |
| LiveCodeBench (20 / 20) | 16 / 20 | 13 / 20 | 13 / 20 | +15.0 [-17.2, +41.1]; 29.2 | +15.0 [-17.2, +41.1]; 29.2 |
| BFCL (40 / 40) | 39 / 40 | 39 / 40 | 39 / 40 | +0.0 [-10.4, +10.4]; 10.4 | +0.0 [-10.4, +10.4]; 10.4 |
| Tool chains (6 / 6) | 6 / 6 | 6 / 6 | 4 / 6 | +0.0 [-51.8, +51.8]; 51.8 | +33.3 [-48.8, +81.6]; 65.2 |
| MGSM (40 / 40) | 40 / 40 | 40 / 40 | 40 / 40 | +0.0 [-10.4, +10.4]; 10.4 | +0.0 [-10.4, +10.4]; 10.4 |

**Every confidence interval includes zero.** No suite shows a statistically significant difference against either pack,
and no statistical superiority is claimed. On these 356 items flash17 scores at least as high as Mia's 2.9 bpw pack on
every suite (equal on six, higher on MBPP+, MMLU-Pro STEM and LiveCodeBench), and at least as high as q28-v2 on every
suite.

The Mia and q28-v2 columns are their results on the same items. Generations that hit the 32,768-token limit
without an answer count as failures and were not retried.

Limits:

- **Sample.** A fixed set of 356 items; the intervals describe these paired items only.
- **Mixed runtime.** The flash17 rows ran across several boots of the same engine and pack, at pools of 3.0M, 2.2M and
  2.1M and clock caps from 2000 to 2250 MHz; generation settings were the same throughout.
- **Different engines and bases.** The Mia and q28-v2 references ran on the Python engine `8474f31` with pools of
  524,288 and 1,400,000 tokens; flash17 ran on Zig `9131e10a`. 1017 is also a newer base model than September, so a
  task difference would not by itself be a quantization difference.
- GPQA-Diamond is not in the suite (gated access). SciCode is first subproblems only; tool chains are simulated.

## Images, tools and agent sessions

| Check | Result | Config |
| --- | --- | --- |
| VQA (20 questions) | 20 / 20 | pools 3.0M, 2.4M, 2.2M (2250 MHz) and 2.1M (2200 MHz) |
| `image_url` chart, screenshot and photo probes | 3 / 3, answers reviewed | the same four configs |
| BFCL subset / tool chains | 39 / 40, 6 / 6 | pool 3.0M, 2250 MHz |
| OpenCode sessions (multi-file build, bug fix, read / edit / bash) | 3 / 3, protected tests unchanged | pool 3.0M, 2250 MHz |
| OpenCode multi-file build | completed, visible and held-out tests passed; strict harness false (one sandbox permission denial, recovered) | pools 2.2M (2250 MHz) and 2.1M (2200 MHz) |
| **VQA + `image_url` probes, final launch** | **20 / 20 + 3 / 3**, answers reviewed | **recommended profile: pool 2.1M, 2100 MHz** |
| **OpenCode build session, final launch** | **passed**: 7 valid tool calls, protected held-out checks passed, normal stop | **recommended profile: pool 2.1M, 2100 MHz** |

## Speed

**At the recommended profile** (pool 2.1M, 2100 MHz cap), single stream, greedy, thinking off, 384 output tokens on
the short cell (mean of three runs) and 128 on the long ones (one run each). The q28-v2 column is q28-v2's **earlier
measurement**, on the same engine and probe but a different clock cap and its 3.0M pool: this is not a matched A/B.

| Metric | q28-v2 (earlier) | flash17 |
| --- | ---: | ---: |
| Short decode (tok/s) | 85.75 | **87.9** (+2.5 %) |
| 32K prompt: prompt tokens / TTFT (tok/s) | 2,630 | 2,349 |
| Decode after the 32K prompt (tok/s) | 66.7 | 64.1 |
| 131K prompt: prompt tokens / TTFT (tok/s) | 2,460 | 2,198 |
| Decode after the 131K prompt (tok/s) | 64.5 | 64.9 |

The 32K / 131K prompts are 37,979 / 154,188 tokens for flash17. TTFT includes HTTP and admission time, so the prompt
rows are not isolated prefill speed. No speed interval is inferred from three runs.

**Earlier flash17 measurement, pool 3.0M, 2250 MHz caps** (not the recommended profile), Zig `9131e10a`, one boot each,
same probe: greedy, thinking off, 384 output tokens on short cells and 128 on long ones. Short cells are the mean of
three repetitions; long cells are one run. No statistical speed interval. The q28-v2 column is the same engine and pool.

| Metric | q28-v2 | flash17 | Change |
| --- | ---: | ---: | ---: |
| Short decode, 1 stream (tok/s) | 85.75 | 88.84 | +3.60 % |
| Short decode, 4 streams aggregate (tok/s) | 143.24 | 153.76 | +7.34 % |
| 32K-label prompt throughput (tok/s) | 2,630.2 | 2,636.2 | +0.23 % |
| 131K-label prompt throughput (tok/s) | 2,459.8 | 2,463.4 | +0.15 % |
| Decode after the 32K prompt (tok/s) | 66.69 | 61.14 | -8.32 % |
| Decode after the 131K prompt (tok/s) | 64.45 | 62.70 | -2.72 % |

The prompts encode to 37,985 and 154,194 tokens for flash17 (37,977 and 154,186 for q28-v2); prompt throughput is
prompt tokens over time to first token (flash17: 14.409 s and 62.593 s), including HTTP overhead. The recommended
profile uses a smaller pool and a lower clock cap, so these numbers do not describe it.

## Rollback to q28-v2

q28-v2 ([`jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw))
is the previous recommended pack; its profile, [`config/prod-zig.env.example`](../config/prod-zig.env.example), is
unchanged. Stop rank 0 first and let it drain, then start both ranks with the q28-v2 config (its own model, September
Engram, prepared and state folders, `TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl`, pool 3,000,000):

```bash
cp config/prod-zig.env.example config/prod-zig-q28v2.env              # once, on each node; set the paths and link
docker stop -t $((DRAIN_S + 60)) tensorfold-zig-r0                    # rank 0 node
docker wait tensorfold-zig-r1                                         # rank 1 node
CONFIG=config/prod-zig-q28v2.env scripts/run-zig.sh 1                 # rank 1, then rank 0 on its node
```

Check the answer to the quick-start canary (391) and, if you kept them, your saved replies. Keep the q28-v2 pack,
Engram shards and prepared folders on disk if you want a quick rollback; rebuilding the prepared folders takes several
minutes. On the test nodes every flash17 evaluation ended by restoring q28-v2 on the same engine, with the saved
replies, long-context hashes, image probe and health checks passing.
