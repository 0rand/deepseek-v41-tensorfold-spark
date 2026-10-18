# Flash-1017 pack (flash17) on two DGX Sparks: candidate profile

**Status: candidate, not the recommended pack.** The recommended pack and profile remain q28-v2 with
[`config/prod-zig.env.example`](../config/prod-zig.env.example). The final validation run (2.1M pool, clock ladder
2200 → 2100 → 2000 MHz) **passed memory but stopped on temperature**: the worker reached the 92 °C stop at 2000 MHz
and one request at a time, in a hot room without air conditioning. q28-v2 also reached 92 °C on the same nodes during
a restore, so this is treated as a limit of the test environment, not a property of the pack. The task results are
**PARTIAL (300 of 356 items)** and all within noise. The Hugging Face repository is still **private** while its upload
completes.

## The pack

| | |
| --- | --- |
| Repository | [`jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw) (**private**, upload in progress) |
| Base | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170` (MIT) |
| Plan | `q28-v3a` (flash17); the `v3b` alternative was encoded but not evaluated |
| Bits per weight | **2.80** (2.8003 on the 2.9 bpw pack's definition); routed experts 2.77 (2/3/4/5 bits per expert), attention 5.0, shared experts 5.38, CSA2 5.0, Engram projection 5.0, head 6, MTP / draft 4 |
| Size | **203.8 GB** in 47 pack files (39 MB smaller than q28-v2), at most 102.09 GB a rank |
| Vision | included: the 1017 vision tower and aligner (263 bf16 tensors, 0.971 GB) and the 43 learned image routing biases, in `native.safetensors` |
| Manifest | `SHA256SUMS` in the repository; first 16 hex digits of its own sha256: `5d4b0c1b650da70a` |

Calibration was weighted toward this recipe's use: 24 % code, 24 % math + science reasoning, 30 % tool calls, with
chat, prose, web text and multilingual text (4 %) as guards. The method and the full card are in the Hugging Face
repository.

What it needs beyond q28-v2's setup ([ZIG-SERVE](ZIG-SERVE.md)):

- **The 1017 Engram shards.** Every Engram weight and scale changed in 1017; the September shards do not match this
  pack. Pack them from the 1017 release as in [INSTALL step 3](INSTALL.md), with `deepseek-ai/DeepSeek-V4.1-Flash-1017`
  as the source, and run `--check` on the result.
- **`TF_DSV41_BIAS_VL=/model`**: the pack's own 1017 image routing biases, read from the model folder on both ranks,
  instead of the September copy in `/cache`.
- **Fresh prepared and state folders.** Do not reuse q28-v2's prepared files, calibration tables or sessions.
- **Chat template note.** 1017 ships no `chat_template.jinja`. The pack carries September's file (the same file
  q28-v2 has) only because some older TensorFold builds look for it at boot. Zig `9131e10a` renders prompts with its
  own encoder and does not read it. That encoder does **not** include 1017's prompt change (a user turn holding only
  tool results does not start a new turn); it only affects continuations that carry tool results without declaring
  `tools`.

## Candidate serving profile

[`config/prod-zig-flash17.env.example`](../config/prod-zig-flash17.env.example) is the q28-v2 serving profile with
only these changes. Engine Zig `9131e10a` (the exported [`engine/zig`](../engine/zig)), TP=2, four slots, 1M context,
split + compact KV, expert top-p 0.85 / min 3, native images, 4K prefill, floors 5 / 4 GiB and the 3.0 GiB serving dip
are all unchanged.

| Setting | q28-v2 profile | flash17 candidate | Why |
| --- | --- | --- | --- |
| `MODEL_DIR` | q28-v2 pack | flash17 pack | the new weights |
| `ENGRAM_DIR` | September shards | **1017 shards** | the tables changed in 1017 |
| `PREPARED_DIR` / `STATE_DIR` | q28-v2's | **fresh** | prepared files and calibration belong to one pack |
| `TF_DSV41_BIAS_VL` | `/cache/dsv41-bias-vl` | **`/model`** | the 1017 learned image biases |
| `TF_DSV41_POOL_TOKENS` | 3000000 | **2100000** | memory headroom ([below](#memory-and-thermal-findings)) |
| GPU clock cap (host, both nodes) | 2250 MHz on the test nodes | **2000 MHz** | thermal headroom ([below](#clock-ladder)) |

The clock cap is a host setting, not an engine setting. The test nodes apply it through their existing clock-cap
service with `nvidia-smi -lgc 300,2000`, and it persists across reboots on both nodes, for q28-v2 as well. Check the
applied clock on both nodes before load.

Each path key is set per node, so the measured candidate changed nine entries: model, prepared, Engram and state
folders on each of the two nodes, plus `TF_DSV41_BIAS_VL`. The pool and the clock cap were changed afterwards, for
memory and then for temperature.

With a 2.1M-token pool, one slot can still fill its 1M context and two slots 1M each; four live slots share about
525K each. The pool is shared across live slots, not reserved per slot.

```bash
cp config/prod-zig-flash17.env.example config/prod-zig-flash17.env   # on each node; set the paths and link
CONFIG=config/prod-zig-flash17.env scripts/run-zig.sh 1 --check-aot   # rank 1, then rank 0
CONFIG=config/prod-zig-flash17.env scripts/run-zig.sh 1               # rank 1, then rank 0 on its node
```

Before any load, check both ranks: after-boot MemAvailable minus the 3.0 GiB serving dip must leave at least
4.25 GiB (the 4 GiB floor plus 0.25 GiB margin). Then watch both ranks' MemAvailable and temperature under load, not
just at idle; on the test nodes the worker ran hotter and set the thermal limit.

## Memory and thermal findings

All on the same two Sparks, Zig `9131e10a`, unchanged knobs except the pool and the clock cap, 2026-10-17/18 UTC.
Lows and temperatures are sampled about every 1.4-2 s, not continuous. "Conc." is the number of generation
requests admitted at once (the server keeps four slots).

| Run | Pool (tokens) | Clock cap | Conc. | Head / worker before load (GiB) | Engine guard worst point (GiB) | Observed low head / worker (GiB) | Outcome |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| q28-v2 restores (reference) | 3,000,000 | 2250 MHz | - | 7.54-7.58 / 7.57 | 4.54-4.58 / 4.57 | no matched load run | prod |
| flash17, first boot | 3,000,000 | 2250 MHz | 4 | 6.93 / 6.95 (after canaries) | not retained | **3.977** / 5.106 | stop: memory |
| flash17, reboot (populated state) | 3,000,000 | 2250 MHz | 4 | 7.52 / 7.64 | 4.42 / 4.63 | **3.558** / 4.13 (engine log) | stop: memory, first 4K prefill |
| flash17 | 2,400,000 | 2250 MHz | 4 | 8.14 / 7.91 | 5.09 / 4.91 | **3.960** / 4.361 | stop: memory, first 4K prefill |
| flash17 | 2,200,000 | 2250 MHz | 4 | 8.23 / 8.46 | 5.27 / 5.53 | 4.153 / 4.974 | stop: head 94.1 °C |
| flash17, with the 4K prefill replay | 2,100,000 | 2200 MHz | 4 | 8.41 / 8.74 | 5.47 / 5.43 | 4.226 / 5.132 | stop: worker 92.1 °C |
| flash17 | 2,100,000 | 2200 MHz | 1 | 8.28 / 8.49 | 5.46 / 5.58 | 5.735 / 6.415 | stop: fourth thermal pause in 30 min |
| **flash17, final ladder** | **2,100,000** | **2200 → 2100 → 2000 MHz** | **1** | **8.229 / 8.105** | **5.37 / 5.19** | **5.095 / 5.819** | **stop: worker 92 °C at 2000 MHz** |

**Memory passes at 2.1M.** In the final ladder the sampled minimum was about 5.1 GiB (head 5.095 GiB), and no run at
2.1M went under the 4 GiB floor: the lowest was 4.226 GiB on the head, during the same 4K-prefill agent replay that
took 3.0M and 2.4M under it. All three 2.1M runs stopped on temperature, none on memory.

**The 4K prefill workspace, not the pack, sets the low point.** Each long prompt allocates a transient 2,268 MiB
(~2.27 GB) 4K prefill workspace on both ranks and releases it after the prompt. The engine admits it when current
MemAvailable minus the workspace stays above `TF_DSV41_PF_WS_FLOOR_GIB`, which the profile sets to 3.5 GiB, below
the 4 GiB hard floor; it does not reserve the 3.0 GiB serving dip on top. `TF_DSV41_PF_4K_MEMORY_OK=1` acknowledges
that cost; it does not bypass the check. So a boot that passes the guard can still go under 4 GiB on the head during
the first long prompt: the reboot above passed at 4.42 GiB modelled and fell to 3.558 GiB on a 9,891-token prompt;
its logs show the head recover from 3.41 to 5.64 GiB when the workspace was released. The engine's boot guard warned
that the workspace would leave 2.15 / 2.37 GiB at the modelled worst point; that warning does not block requests.
This mechanism is shared with q28-v2's profile, which has not had the same mixed load as a control.

flash17 also starts about 0.62 GiB a rank lower after boot than q28-v2 on the same engine. Only 21 MiB of that is
weights (the prepared rank folders); the rest is symmetric across ranks and not attributed. The head is the binding
rank because it also holds the HTTP server, the request log, the image cache and the vision tower.

### Thermal limit

At the 2.1M runs an admission gate in front of the server enforced the thermal policy. It admits no new request at
90 °C or above and resumes only when both nodes are at 85 °C or below. It stops the run at 92 °C, or at a fourth
pause within 30 minutes. A separate guard stops the containers at 94 °C or under 4 GiB.

| Run | Clock cap | Conc. | Head / worker max (°C) | What stopped it |
| --- | --- | ---: | --- | --- |
| flash17, 2.2M | 2250 MHz | 4 | 94.1 / 92.2 | 94 °C guard (before the 92 °C rule existed) |
| flash17, 2.1M | 2200 MHz | 4 | 89.5 / 92.1 (MMLU-Pro STEM) | 92 °C stop on the worker |
| q28-v2 restore after that run | 2200 MHz | restore checks | 90.9 / 92.0 | (restore; 32K / 131K long-context checks) |
| flash17, 2.1M | 2200 MHz | 1 | 90.5 / 91.6 | fourth pause in 30 min |
| flash17, 2.1M, final ladder | 2100 → 2000 MHz | 1 | 91.5 / 92.1 | 92 °C stop on the worker, at 2000 MHz |
| q28-v2 restore after the ladder | 2000 MHz | 1 | 89.0 / 88.8 | none (restore passed) |

The final ladder's maxima by suite and cap:

| Suite / cap | Head max (°C) | Worker max (°C) | Head min available (GiB) | Worker min available (GiB) |
| --- | ---: | ---: | ---: | ---: |
| MATH-500 / 2100 MHz | 91.5 | 88.9 | 5.954 | 6.345 |
| MMLU-Pro STEM / 2100 MHz | 89.0 | 89.5 | 6.190 | 6.466 |
| SciCode / 2100 MHz | 89.5 | 90.1 | 5.650 | 6.268 |
| SciCode / 2000 MHz | 89.3 | 89.5 | 5.845 | 6.295 |
| LiveCodeBench / 2000 MHz | 90.3 | 92.1 | 5.095 | 5.819 |

**This is an environment limit, not a property of the pack.** The runs were in a hot room without air conditioning.
q28-v2, the recommended pack, also reached 92.0 °C on the worker on the same nodes at 2200 MHz during its restore
checks. No matched thermal run of q28-v2 under the same task load exists, so these numbers do not show that flash17
runs hotter or cooler than q28-v2. Even at 2000 MHz and one request at a time, this room did not keep the worker under
92 °C for a sustained benchmark. 2000 MHz was the lowest approved cap, and the stop thresholds were not relaxed; the
next attempt needs a cooler room or better node cooling, and resumes the remaining 56 items.

### Clock ladder

| Step | Cap (both nodes) | When (UTC, 2026-10-18) | Why |
| --- | --- | --- | --- |
| prior production | 2250 MHz | - | the 2.2M run reached 94.1 °C here |
| approved | 2200 MHz | from 02:26 | 2.1M runs at concurrency 4, then 1 |
| ladder step 1 | 2100 MHz | 05:32:55 / 05:32:59 | applied while idle |
| ladder step 2 | 2000 MHz | 05:57:21 | after five minutes at or above 88 °C on the worker; applied idle at 81.4 / 81.0 °C |
| now | **2000 MHz, persistent** | - | kept after rollback, for q28-v2 too |

Speed at 2000 MHz was not measured, for flash17 or for q28-v2.

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

## Task results: PARTIAL (300 of 356 items)

**PARTIAL: MGSM (40 items) and 15 of the 20 LiveCodeBench items were not run, nor one MBPP+ item; the thermal stop
ended the run.** All differences are within noise: every interval includes zero. No claim is made that flash17 beats
Mia's pack or q28-v2 on tasks.

Same frozen items and scorers as the matched baseline: greedy (T0, top_p 1, seed 0), thinking on, effort high,
max_tokens 32,768, pass@1 (strict score == 1). Conservative paired 95 % intervals: Bonferroni Clopper-Pearson 97.5 %
bounds on the two discordant counts of each comparison, per suite, not simultaneous across suites. Half-width is half
the interval's width.

| Suite (completed / planned) | flash17 | Mia 2.9 bpw | q28-v2 | Δ vs Mia, points [95 % CI]; half-width | Δ vs q28-v2, points [95 % CI]; half-width |
| --- | ---: | ---: | ---: | --- | --- |
| HumanEval+ (40 / 40) | 34 / 40 | 34 / 40 | 33 / 40 | +0.0 [-14.9, +14.9]; 14.9 | +2.5 [-10.3, +14.9]; 12.6 |
| MBPP+ (59 / 60) | 50 / 59 | 49 / 59 | 50 / 59 | +1.7 [-10.0, +13.0]; 11.5 | +0.0 [-12.8, +12.8]; 12.8 |
| LiveCodeBench (5 / 20) | 5 / 5 | 4 / 5 | 4 / 5 | +20.0 [-58.1, +76.4]; 67.3 | +20.0 [-58.1, +76.4]; 67.3 |
| MMLU-Pro STEM (100 / 100) | 93 / 100 | 91 / 100 | 90 / 100 | +2.0 [-4.1, +7.9]; 6.0 | +3.0 [-5.3, +10.9]; 8.1 |
| MATH-500 (40 / 40) | 36 / 40 | 36 / 40 | 36 / 40 | +0.0 [-10.4, +10.4]; 10.4 | +0.0 [-10.4, +10.4]; 10.4 |
| SciCode first step (10 / 10) | 6 / 10 | 6 / 10 | 5 / 10 | +0.0 [-35.5, +35.5]; 35.5 | +10.0 [-35.4, +49.1]; 42.2 |
| BFCL (40 / 40) | 39 / 40 | 39 / 40 | 39 / 40 | +0.0 [-10.4, +10.4]; 10.4 | +0.0 [-10.4, +10.4]; 10.4 |
| Tool chains (6 / 6) | 6 / 6 | 6 / 6 | 4 / 6 | +0.0 [-51.8, +51.8]; 51.8 | +33.3 [-48.8, +81.6]; 65.2 |
| MGSM (0 / 40) | not run | - | - | unavailable | unavailable |

The Mia and q28-v2 columns are their results on the same completed items. All science items (MMLU-Pro STEM, MATH-500,
SciCode) are complete. One saved MMLU-Pro STEM row exhausted the 32,768-token limit without an answer; it counts as a
failure and was not retried. Requests cut off by a stop are excluded as infrastructure errors, not counted as wrong.

Limits of these partial results:

- **Selection.** Suites ran in a fixed order and stopped early, so the completed subset is completion-selected. The
  intervals describe the observed paired items only.
- **Mixed runtime.** The 300 rows ran across the attempts above: 145 at 3.0M / 2250 MHz, 90 at 2.2M / 2250 MHz, 7 at
  2.1M / 2200 MHz, 50 at 2.1M / 2100 MHz and 8 at 2.1M / 2000 MHz. Generation settings were the same throughout;
  admission concurrency was 4 in the earlier attempts and 1 in the last two.
- **Different engines and bases.** The Mia and q28-v2 references ran on the Python engine `8474f31` with pools of
  524,288 and 1,400,000 tokens; flash17 ran on Zig `9131e10a`. 1017 is also a newer base model than September, so a
  task difference would not by itself be a quantization difference.
- GPQA-Diamond is not in the suite (gated access). SciCode is first subproblems only; tool chains are simulated.

## Images, tools and agent sessions

| Check | Result | Config |
| --- | --- | --- |
| VQA (20 questions) | 20 / 20 | pools 3.0M, 2.4M, 2.2M (2250 MHz) and 2.1M (2200 MHz), concurrency 4 |
| `image_url` chart, screenshot and photo probes | 3 / 3, answers reviewed | the same four configs |
| BFCL subset / tool chains | 39 / 40, 6 / 6 | pool 3.0M, 2250 MHz |
| OpenCode sessions (multi-file build, bug fix, read / edit / bash) | 3 / 3, protected tests unchanged | pool 3.0M, 2250 MHz |
| OpenCode multi-file build | completed, visible and held-out tests passed; strict harness false (one sandbox permission denial, recovered) | pools 2.2M (2250 MHz) and 2.1M (2200 MHz) |
| Final ladder (2.1M, 2100 / 2000 MHz) | not run: the thermal stop came first | - |

## Speed

**Config: pool 3.0M, 2250 MHz caps**, Zig `9131e10a`, one boot each, same probe: greedy, thinking off, 384 output
tokens on short cells and 128 on long ones. Short cells are the mean of three repetitions; long cells are one run.
No statistical speed interval. The q28-v2 column is the same engine and pool.

| Metric | q28-v2 | flash17 | Change |
| --- | ---: | ---: | ---: |
| Short decode, 1 stream (tok/s) | 85.75 | 88.84 | +3.60 % |
| Short decode, 4 streams aggregate (tok/s) | 143.24 | 153.76 | +7.34 % |
| 32K-label prompt throughput (tok/s) | 2,630.2 | 2,636.2 | +0.23 % |
| 131K-label prompt throughput (tok/s) | 2,459.8 | 2,463.4 | +0.15 % |
| Decode after the 32K prompt (tok/s) | 66.69 | 61.14 | -8.32 % |
| Decode after the 131K prompt (tok/s) | 64.45 | 62.70 | -2.72 % |

The prompts encode to 37,985 and 154,194 tokens for flash17 (37,977 and 154,186 for q28-v2); prompt throughput is
prompt tokens over time to first token (flash17: 14.409 s and 62.593 s), including HTTP overhead. The pool and the
clock cap both changed for the candidate profile, so these numbers do not describe it.

**Speed at the final clock (2000 MHz) was not measured**, for flash17 or q28-v2: the thermal stop came before the speed
gate. Expect lower decode than the table above at a 2000 MHz cap.

## Rollback to q28-v2

The q28-v2 profile is unchanged by this candidate. Stop rank 0 first and let it drain, then start both ranks with the
q28-v2 config (its own model, Engram, prepared and state folders, `TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl`, pool
3,000,000). Keep the clock cap your room needs; the test nodes keep 2000 MHz for q28-v2 too:

```bash
docker stop -t $((DRAIN_S + 60)) tensorfold-zig-r0   # rank 0 node
docker wait tensorfold-zig-r1                        # rank 1 node
CONFIG=config/prod-zig.env scripts/run-zig.sh 1      # rank 1, then rank 0 on its node
```

Check the answer to the quick-start canary (391) and, if you kept them, your saved replies. Keep the q28-v2 pack,
Engram shards and prepared folders on disk while evaluating flash17; the rollback needs them, and rebuilding the
prepared folders takes several minutes. Every measured flash17 attempt above ended by restoring q28-v2 on the same
engine. The final restore at 2000 MHz passed the 16 saved replies, both long-context hashes, the image probe and the
health checks. One earlier restore at 2200 MHz could not finish its long-context check: it was paused for temperature
(see the thermal table).
