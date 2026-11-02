# Upstream upgrade considerations

Assessment date: 2026-11-02

## Recommendation

**Keep the existing `main-0rand` stack as the production baseline. Do not undertake a full upstream migration without a specific requirement or a demonstrated, material benefit on the actual workload.**

The operator reports that the current stack is highly stable and performant. The upstream evidence inspected here does not establish that replacing it with the current upstream engine would improve either property. The migration would also require reimplementing and requalifying session persistence, memory handling, lifecycle, and telemetry behavior that the fork already provides.

This is not a claim that the new engine is necessarily slower or less stable. There is no controlled comparison of the exact deployed fork against the current upstream engine in the evidence examined. Upstream has useful fixes and features; assess those individually rather than assuming that a full upgrade is the only way to benefit.

## Scope and branch comparison

| Item | Assessed revision |
| --- | --- |
| Upstream recipe `main` | `ed96ef610693ff68ac5b1bec5881de72854554bd` |
| Customized recipe `main-0rand` | `6ff9b2d2d4482dbbc2ea89d87f4a5738726601cd` |
| Common ancestor | `60b66888a1a379fb38bf85c232b27ec60b007373` |
| Upstream TensorFold runtime pin | `58bece1d3184269059fd3f7f755c52f4f295113d`, release candidate G |

The recipe branches have 12 upstream-only and 13 fork-only commits. At the time of the assessment, the local `main-0rand` tip was one commit ahead of `origin/main-0rand`. The direct branch diff spans 332 files, much of it historical Python patches, removed tooling, and benchmark artifacts rather than serving-logic changes.

A merge simulation reported conflicts in `README.md`, `config/prod-zig-flash17.env.example`, `docs/ZIG-SERVE.md`, and `scripts/run-zig.sh`. The larger integration problem is not textual conflicts: upstream now builds a separate pinned TensorFold source tree, while the fork changes `engine/zig`. A merge that takes upstream's build pipeline would leave those local Zig changes in the retained export, **not in the serving image**.

See the [fork notes](../0rand-readme.md) for the fork's own implementation, historical validation, and limitations. The recorded operator experience should not be confused with a new benchmark or soak performed for this assessment.

## What changed upstream

### Runtime and build ownership

Upstream commit `44b3dcd` switched the image to build the `dsv41-serve` target from an exact TensorFold revision. The assessed recipe fetches candidate G from `jayleaton/TensorFold`; the recipe documentation says that pin was not yet on `ashhart/TensorFold` main. The previous local Zig export remains for generator and reference dependencies, while the older Python serving path and its historical patches were removed.

The native image builds the pinned source, installs `tensorfold-dsv41` and `tf-dsv41`, and includes grammar, JPEG, and WebP runtime libraries with dependency checks. The launcher compares the image revision label and hashes image runtime configuration, layers, common assets, and common profile settings into a two-rank identity.

A supplied `TENSORFOLD_SOURCE` checkout or bundle still builds the exact committed revision in `config/tensorfold.rev`; uncommitted checkout changes are not used. Fork engine changes would therefore need a committed TensorFold-derived revision and a regenerated settings manifest.

Sources: [upstream engine documentation at the assessed recipe revision](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/ed96ef610693ff68ac5b1bec5881de72854554bd/docs/ENGINE.md), [upstream native builder](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/ed96ef610693ff68ac5b1bec5881de72854554bd/scripts/build-native.sh), and [upstream Dockerfile](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/ed96ef610693ff68ac5b1bec5881de72854554bd/docker/zig/Dockerfile).

### Session persistence: a fork capability G does not preserve

The fork supports eligible-snapshot write-through while retaining the RAM copy, reconstruction of common entries after restart, and rank-independent logical SSD budget accounting. Its historical record includes a TP2 restart and SSD restore; that is evidence for the fork build, not candidate G.

In the assessed G source:

- DeepSeek `Sessions.init` opens the disk tier with `fresh_start = true`.
- Startup removes regular `.tfs` and `.tmp` files **within the matching compatibility/rank namespace**. It does not imply deletion of every older namespace under the session root.
- `Disk.scan` returns `SplitReconcileUnsupported` when `world > 1`. Turning off `fresh_start` alone would not provide safe TP2 restart recovery.
- The new session identity binds checkpoint, configuration, layout, rank, and bounded-state shape. No migration path from the fork's existing snapshots was established.

Use a separate state directory for any experiment and preserve rollback copies. Restoring the fork's behavior would require a real TP2 recovery protocol, including common-entry reconciliation, corrupt or missing peer files, deterministic IDs, and consistent logical charges—not just copying a setting or disabling the cleanup.

Sources: [G DeepSeek sessions](https://github.com/jayleaton/TensorFold/blob/58bece1d3184269059fd3f7f755c52f4f295113d/zig/src/families/deepseek_v41/sessions_gpu.zig), [G namespace cleanup](https://github.com/jayleaton/TensorFold/blob/58bece1d3184269059fd3f7f755c52f4f295113d/zig/src/sessions/disk_namespace.zig), [G disk scan](https://github.com/jayleaton/TensorFold/blob/58bece1d3184269059fd3f7f755c52f4f295113d/zig/src/sessions/disk.zig), and the [fork's session notes](../0rand-readme.md).

### Prefill correctness and memory behavior

Fork commit `006b9df` fixes a held-4K-workspace issue: it rechecks available memory on both ranks between prompt pieces and releases the held workspace and falls back to 2K if either rank is below the configured floor. G's [`wsAgreed`](https://github.com/jayleaton/TensorFold/blob/58bece1d3184269059fd3f7f755c52f4f295113d/zig/src/families/deepseek_v41/forward_prefill_k4.zig) still returns immediately when the workspace buffer is already present. The two implementations are not equivalent.

Upstream commit `33cb5e8` / #78 separately fixes missing or undersized index scratch for intermediate positions and ragged prefill shapes. A refactored version is present in G's [`index_scratch.zig`](https://github.com/jayleaton/TensorFold/blob/58bece1d3184269059fd3f7f755c52f4f295113d/zig/src/families/deepseek_v41/index_scratch.zig). This is a legitimate candidate for a targeted backport to the existing fork if the workload can exercise the affected shapes. Its regression focuses on 4K, but the change also touches shared planning paths; do not assume a 2K profile makes every aspect irrelevant without checking.

The fork documents a historical local 2K rollout as well as a tracked 4K example. Neither document establishes the effective configuration of a currently running container.

### Operations, guard, and observability

The fork's `scripts/zig-stack.sh` synchronizes the head configuration to the worker, manages ordered rank startup and drain, and uses Docker `unless-stopped`. Upstream uses `scripts/serve.sh` and `native_ops.py` for rank ownership, pair identity, readiness, durable STOP, and optional watchdog healing. These are different lifecycle contracts; do not enable two independent recovery authorities for the same pair.

Upstream also adds an optional thermal/memory guard with bounded proxy admission, sampling, and STOP/ownership controls. The assessed example is disabled, has no qualified profiles, and does not automatically install or enable services. The guard's profile validation rejects a 4K workspace floor below 4 GiB. Its native port and unattended recovery require qualification on the actual pair.

G already exposes standard `tensorfold:` request gauges, counters, histograms, and prompt-cache metrics. It does not provide all fork-specific shared-pool, retained-entry, KV-byte, and SSD-spill fields. The fork's EngineMetrics integration and dashboards require a field-by-field compatibility review; replacing G's existing metrics wholesale would also risk changing their semantics. G hides model details on authenticated `/health`.

The upstream launcher hard-codes the served model name in the assessed revision and does not carry the fork's conditional `--no-drafts` behavior. Those are consumer-facing compatibility requirements to preserve or explicitly change.

### Tracked profile differences

These values compare the committed **example profiles**, not the actual effective configuration on either node:

| Setting | `main-0rand` example | Upstream `main` example |
| --- | ---: | ---: |
| Session disk budget | 32 GiB | 128 GiB |
| Session disk minimum | 0 tokens, subject to snapshot eligibility | 1,024 tokens |
| Memory target / hard floor | 1 / 1 GiB | 5 / 4 GiB |
| 4K workspace floor | 1 GiB | 4.75 GiB |
| API port | 8100 | 8000 |

The fork's session-disk budget is a logical accounting budget, not a hard physical filesystem quota. Its `MIN=0` does not override the intrinsic snapshot eligibility and alignment rules. The fork's historical notes also describe a local 2K profile with lower admission floors; those notes should not be treated as a live-state inspection.

The assessed upstream flash17 example enables `TF_DSV41_COMPUTE_LANE=1`, although the engine default is off. A request offering `code_interpreter` can cause model-written Python to run in a Landlock/seccomp sandbox inside the serving container. If testing G, leave it off initially unless the feature and its security boundary are explicitly wanted. Upstream's file-based authentication and credential sanitization are useful improvements, but do not by themselves establish that the current deployment is insecure. Assess actual exposure and credential handling separately.

## What the performance evidence shows

| Upstream comparison | Reported result | Scope and limitation |
| --- | --- | --- |
| Engine H vs. old upstream engine A | 8K prefill **+6.3%**; four-stream decode **-2.7%** | Two-pair spot comparison; neither arm is the customized fork or current G pin |
| Candidate G vs. F | **+0.5%** geometric mean across the reported coding cells | Different engine comparison using the unpublished K6 MTP pack variant |
| `STREAM_RB=4`, older engine | Long cold-prefill time **6.5–14.6% lower** at 131K–500K; 32K **5.2% higher** | One baseline/candidate pair, not the exact fork; the full strict concurrency gate did not pass |

The RB4 receipt localized its observed short-input concurrent/solo mismatch outside the long-context RB4 active range, but did not establish the root cause or claim the full serving gate passed. Its kernel generator and AOT artifacts require separate arithmetic and device qualification. Candidate G's performance measurements used an unpublished K6 MTP pack variant, and the upstream documentation explicitly says the current pin still requires pack/device/API, session, sampling, image, and structured-output qualification.

These observations do not prove that the current fork is faster in every workload, nor that G cannot improve it. They do mean there is **no demonstrated net gain for this deployment** to set against the migration risk. Long cold-prefill latency may justify an isolated RB4 experiment if it matters to the workload; it is not evidence for a wholesale engine migration.

Sources: [upstream H/A comparisons](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/ed96ef610693ff68ac5b1bec5881de72854554bd/docs/ENGINE.md), [G/F benchmark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/ed96ef610693ff68ac5b1bec5881de72854554bd/docs/BENCHMARKS.md), [K6 pack provenance](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/ed96ef610693ff68ac5b1bec5881de72854554bd/docs/FLASH-1017.md), and [RB4 serving receipt](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/ed96ef610693ff68ac5b1bec5881de72854554bd/results/prefill/stream-rb4-e2e-20261030/README.md).

## Risks worth monitoring without upgrading

1. **Index-scratch fix #78:** assess whether actual prompt lengths, segmentation, and prefill settings can exercise it. Backport and stage the specific fix only if its risk reduction justifies a new image and validation.
2. **Session identity:** the fork notes state that its layout/rank checks are not a complete checkpoint identity. Keep sessions, prepared files, and calibration state separated by pack and relevant configuration; do not assume old snapshots can be reused across model or engine changes.
3. **Resource and liveness limits:** the fork's historical record identifies mid-prefill `PoolExhausted` and independent CUDA/transport deadlocks as unresolved limits. G's optional stalled-engine exit is off by default and had not been triggered on the documented pair; a full upgrade is not proven to eliminate every such failure.
4. **Credential and network exposure:** review the actual serving and operator boundary independently of performance. Upstream's file-based authentication and subprocess sanitization may be worth adapting if they address a real exposure; they are not proof of a current vulnerability.
5. **Recovery evidence:** the fork's historical record identifies some unexercised scenarios, including full-tier fill, crash injection, and host/daemon reboot. That historical record is not a claim about subsequent operator testing.

## If a full migration becomes justified

1. **Preserve rollback:** retain both ranks' exact image, effective configs, pack and asset manifests, and separate backups of session and prepared state. Keep the old stack deployable.
2. **Port the engine contract onto the chosen upstream TensorFold revision:** held-workspace rechecking; eligible write-through; TP2 restart reconciliation and logical accounting; and only the missing cache/spill telemetry. Preserve the newer engine's checkpoint/namespace safety checks. Do not merely change `fresh_start`.
3. **Commit and pin the resulting engine revision:** regenerate `config/tensorfold-settings.txt`, build one image for both ranks, and compare image/profile identities. Port or replace the fork's config sync, model name, draft-off, lifecycle, and dashboard contracts without creating competing supervisors.
4. **Use isolated state and qualify actual assets:** verify native token map, RoPE, Engram shards, AOT variants, prepared files, and calibration against the selected pack and final settings. Do not assume fork snapshots or AOT assets are compatible.
5. **Exercise the real workload and failure paths:** compare end-to-end throughput and latency, output correctness, memory minima, long/ragged prefill, held-workspace fallback, session restore after restart, asymmetric/corrupt peer state, SSD-budget eviction, metrics, authentication, rank failure, drain, STOP, and recovery. Qualify four and eight slots separately if both matter.
6. **Cut over both ranks together only after the agreed gates pass.** Keep compute lane, guard healing, and RB4 as separate opt-ins until each is qualified. Retain the old image and state for rollback.

## Assessment verification

- `git merge-tree` was used to simulate the branch merge without changing the original working tree; it reported the four conflicts listed above.
- Upstream's actual settings validator was exercised with the tracked fork example. It rejected ten legacy names. After removing those names for the diagnostic, it warned that `TF_DSV41_SESSION_DISK_WRITE_THROUGH` is not read or forwarded by G.
- The pinned G source was inspected for session cleanup and TP2 scan restrictions, index scratch, held-workspace handling, server CLI, and metrics.
- No build, test suite, benchmark, GPU-serving run, or deployment action was performed as part of the assessment. Performance figures above are attributed to the cited upstream records, not newly measured results.
