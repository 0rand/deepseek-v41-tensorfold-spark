#!/usr/bin/env bash
# The vision gate on the two DGX Sparks (port doc 0a, gate, "FOR THE NIGHT RUN"): job.sh's vision_gate
# with one rank a host. Prod must be stopped; start each two-host phase on both hosts within about a minute.
#   both:   ./tools/zig/dsv41_vision/spark.sh R OUT ref        Python at TP=2 over the hosts (prod's tree 8474f31, layers
#                                                              0-24, 6 image prompts, p4..p6 with prod's bias_vl), ~15 min
#   rank 0 host: ./tools/zig/dsv41_vision/spark.sh 0 OUT vit        the Zig tower == Python, stage by stage (GPU 0), ~2 min
#   both:   ./tools/zig/dsv41_vision/spark.sh R OUT prefill    the Zig prefill gate on p1..p6 (p4..p6 with
#                                                              TF_DSV41_BIAS_VL), every state role, ~2-3 min a prompt
#   rank 0 host: ./tools/zig/dsv41_vision/spark.sh 0 OUT generate [bias]   the served engine replays the trace,
#   rank 1 host: ./tools/zig/dsv41_vision/spark.sh 1 OUT follow             replies == Python's, ~4 min each
# Every phase prints PASS / FAIL lines into OUT/vision-steps.txt. Knobs: PROMPTS (the prefill list, "1 2 3 4 5 6"),
# STEPS (32), MASTER, IMAGE, CACHE_VOL (prod's: /cache/dsv41-bias-vl holds the release's gate.bias_vl).
set -u
RANK=${1:?rank}; OUT=${2:?outdir}; PHASE=${3:?ref|vit|tower|prefill|generate|follow}; VARIANT=${4:-}
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LEASE=${LEASE_FILE:-$KIT/.tensorfold-lease}
if [[ "$RANK" == 0 && -f "$LEASE" ]]; then echo "the test-window lease is held ($LEASE): another window owns the Sparks"; exit 3; fi
MASTER=${MASTER:?set MASTER to rank 0 rendezvous address}
IMAGE=${IMAGE:?set IMAGE to the CUDA Python reference image}
CACHE_VOL=${CACHE_VOL:?set CACHE_VOL to the Python reference cache volume}
C=8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47
TF=${TF_TREE:?set TF_TREE to the Python reference checkout}; MODEL=${PACK:?set PACK to the model pack directory}
BIAS=/cache/dsv41-bias-vl
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
res() { echo "$*" | tee -a "$OUT/vision-steps.txt"; }
# the containers write OUT as root: removals and copies under it run as root in the image too
asroot() { docker run --rm -v "$OUT:/out" --entrypoint bash "$IMAGE" -c "$1"; }
NET=(--rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
     -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$OUT:/out"
     -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}
     -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1})
# the Zig ranks: one a host over the hybrid collective (M2c-2's settings), index Engram rows (as the reference),
# expert top-p 0.85 (the reference's), images native with the rows held from the reference's images on rank 0
ZIG=("${NET[@]}" -v "$CACHE_VOL:/cache:ro" -e TF_TP_WORLD=2 -e TF_TP_RANK="$RANK" -e TF_TP_DEVICE=0 -e TF_TP_MASTER="$MASTER"
     -e TF_COMM_BACKEND="${BACKEND:-roce}" -e TF_TP_ROCE_FALLBACK=error -e TF_TP_KERNEL_IMAGE=/kit/tp_mailbox.fatbin
     -e TF_DSV41_PLAN_TIMEOUT=1800 -e TF_DSV41_FAILFAST=0 -e TF_DSV41_IMAGES=native -e TF_DSV41_VISION_HOLD=/out/img)
# GATE_KNOBS=prod (default): prod-zig.env's same-bits prefill / decode knobs; replay and 4K need Python's replay ref
[[ "${GATE_KNOBS:-prod}" == prod ]] && ZIG+=(-e TF_DSV41_PF_DENSE=fused -e TF_DSV41_PF_DENSE_TABLE=/cache/pfdense-table.json
     -e TF_DSV41_GM_V2=1 -e TF_DSV41_INDEX_BOUND=1 -e TF_DSV41_INDEX_BUDGET_MIB=64 -e TF_DSV41_PREFETCH_AHEAD=1)
# prod's assets/aot is a capture plus the kit's fill: the reference's sets get the fill too (by hash, idempotent)
fill() {
    [[ -d "$KIT/triton-fill/aot" ]] || { res "FAIL vision $PHASE: the kit has no triton-fill/aot"; exit 1; }
    local d; for d in "$@"; do docker run --rm -v "$OUT:/out" -v "$KIT:/kit:ro" --entrypoint python "$IMAGE" -B \
        /kit/tools/zig/dsv41_aot_merge.py "/out/$d" /kit/triton-fill/aot || { res "FAIL vision $PHASE: fill into $d"; exit 1; }; done
}
NCCL='export TF_NCCL_LIB=$(python -c "import nvidia.nccl, os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), \"lib\", \"libnccl.so.2\"))" 2>/dev/null || echo libnccl.so.2)'
case "$PHASE" in
    ref)
        if docker ps --format '{{.Names}}' | grep -q 'dsv41'; then echo "a dsv41 container is running: stop prod first"; exit 2; fi
        asroot 'rm -rf /out/ref /out/img /out/zig-p*'
        docker run "${NET[@]}" -v "$TF:/dsv41-tf:ro" -v "$CACHE_VOL:/cache" -e TF_DSV41_EXPERT_TOPP=0.85 \
            -e PYTHONPATH=/dsv41-tf/src:/cache/pylib -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
            --name "dsv41-zig-vref-r$RANK" --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_vision/vision_ref.py \
            --vdir /out/img --bias-vl "$BIAS" -- --pack /model --out /out/ref --layers 0-24 --steps "${STEPS:-32}" \
            --prompts 1,2,3,4,5,6 --rank "$RANK" --master "$MASTER" --port 29861 > "$OUT/vision-ref.log" 2>&1 \
        && docker run "${NET[@]}" -v "$TF:/dsv41-tf:ro" --name "dsv41-zig-vengram-r$RANK" --entrypoint bash "$IMAGE" -c \
            'python -B /kit/tools/zig/dsv41_engram_host.py --py-src /dsv41-tf/src --config /model/config.json \
             --tokenizer /model/tokenizer.json --out /out/ref/engram-host.bin && for d in /out/ref/p*; do cp /out/ref/engram-host.bin "$d/"; done' \
            >> "$OUT/vision-ref.log" 2>&1
        rc=$?
        tail -4 "$OUT/vision-ref.log"
        if (( rc != 0 )); then res "FAIL vision ref rank $RANK (vision-ref.log)"; exit 1; fi
        n=$(ls "$OUT"/ref/p[1-6]/ref.json 2>/dev/null | wc -l)
        if (( n == 6 )) && [[ -s "$OUT/ref/trace.jsonl" && -s "$OUT/ref/trace-bias.jsonl" ]]; then res "PASS vision ref rank $RANK: 6 prompts, both traces"
        else res "FAIL vision ref rank $RANK: $n / 6 prompts' ref.json (vision-ref.log)"; exit 1; fi ;;
    vit)
        [[ "$RANK" == 0 ]] || { echo "vit runs on rank 0 (the tower's host)"; exit 2; }
        docker run "${NET[@]}" --name dsv41-zig-vvit --entrypoint /kit/tf-dsv41-m1 "$IMAGE" vision /model /out/img > "$OUT/vision-vit.log" 2>&1
        rc=$?; tail -5 "$OUT/vision-vit.log"
        if (( rc == 0 )) && grep -q '^PASS vision ViT' "$OUT/vision-vit.log"; then res "PASS vision vit"; exit 0; fi
        res "FAIL vision vit (first differing stage in vision-vit.log)"
        # each stage from Python's bits of the stage before (TF_DSV41_VISION_RESYNC=1): every op still off, apart
        docker run "${NET[@]}" -e TF_DSV41_VISION_RESYNC=1 --name dsv41-zig-vvit-ops --entrypoint /kit/tf-dsv41-m1 "$IMAGE" \
            vision /model /out/img > "$OUT/vision-vit-ops.log" 2>&1
        grep -E "differ|bytes, Python" "$OUT/vision-vit-ops.log" | head -60
        exit 1 ;;
    # tower [sweep]: one host, prod may serve (GPU 0, no network): the Python tower then the Zig one; dumps removed
    tower)
        d=tower${VARIANT:+-$VARIANT}; sw=(); [[ "$VARIANT" == sweep ]] && sw=(--sweep --light)
        asroot "rm -rf /out/$d"
        V=(--rm --gpus all -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$OUT:/out")
        docker run "${V[@]}" -v "$TF:/dsv41-tf:ro" -v "$CACHE_VOL:/cache:ro" -e PYTHONPATH=/dsv41-tf/src:/cache/pylib \
            --name "zig-vtower-ref-r$RANK" --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_vision/vision_ref.py \
            --tower-only --vdir "/out/$d" --pack /model "${sw[@]}" > "$OUT/vision-$d-ref.log" 2>&1 \
            || { res "FAIL vision $d: the Python tower (vision-$d-ref.log)"; tail -5 "$OUT/vision-$d-ref.log"; exit 1; }
        docker run "${V[@]}" --name "zig-vtower-r$RANK" --entrypoint /kit/tf-dsv41-m1 "$IMAGE" vision /model "/out/$d" \
            > "$OUT/vision-$d.log" 2>&1
        rc=$?; asroot "rm -rf /out/$d"; grep -E "^vision |differ|bytes, Python|^(PASS|FAIL)" "$OUT/vision-$d.log" | tail -40
        if (( rc == 0 )) && grep -q '^PASS vision ViT' "$OUT/vision-$d.log"; then res "PASS vision $d: $(tail -1 "$OUT/vision-$d.log")"; exit 0; fi
        res "FAIL vision $d (rc $rc): $(tail -1 "$OUT/vision-$d.log")"; exit 1 ;;
    prefill)
        st=0; for p in ${PROMPTS:-1 2 3 4 5 6}; do fill "ref/p$p/aot"; done
        for p in ${PROMPTS:-1 2 3 4 5 6}; do
            bias=(); (( p > 3 )) && bias=(-e TF_DSV41_BIAS_VL="$BIAS")
            mkdir -p "$OUT/zig-p$p"
            timeout --kill-after=30 1200 docker run "${ZIG[@]}" "${bias[@]}" -e TF_TP_PORT=$(( 29870 + p )) -e M2B_TOPP=0.85 \
                -e M2B_PREFILL=1 --name "dsv41-zig-vpf-r$RANK" --entrypoint bash "$IMAGE" \
                -c "$NCCL; exec /kit/tf-dsv41-m1 m2b /model /out/ref/p$p /out/zig-p$p" > "$OUT/vision-prefill-p$p.log" 2>&1
            rc=$?
            docker rm -f "dsv41-zig-vpf-r$RANK" > /dev/null 2>&1
            line="$(grep -h '^PASS Zig\|^FAIL Zig' "$OUT/vision-prefill-p$p.log" | tail -1) / $(grep -h '^PASS M2b\|^FAIL M2b' "$OUT/vision-prefill-p$p.log" | tail -1)"
            if (( rc == 0 )) && grep -q '^PASS M2b' "$OUT/vision-prefill-p$p.log"; then res "PASS vision prefill p$p${bias:+ bias_vl} rank $RANK: $line"
            else res "FAIL vision prefill p$p${bias:+ bias_vl} rank $RANK (rc $rc): $line"; st=1; fi
            asroot "rm -rf /out/zig-p$p/rank*/state"
        done
        exit $st ;;
    generate|follow)
        bias=(); trace=trace.jsonl; port=29880
        if [[ "$VARIANT" == bias ]]; then bias=(-e TF_DSV41_BIAS_VL="$BIAS"); trace=trace-bias.jsonl; port=29881; fi
        [[ "$PHASE" == generate && "$RANK" != 0 ]] && { echo "generate runs on rank 0; rank 1 runs follow"; exit 2; }
        fill ref/p1/aot
        # the served engine on the reference's layers, drafts off; the assets (rope.bin, aot/, engram-host.bin): p1's
        GEN=("${ZIG[@]}" "${bias[@]}" -v "$OUT/ref/p1:/assets:ro" -e TF_DSV41_ASSETS=/assets -e TF_TP_PORT=$port
             -e TF_DSV41_LAYERS=0-24 -e TF_DSV41_DRAFTS=0 -e TF_DSV41_EXPERT_TOPP=0.85
             -e TF_DSV41_PROMPT_TAIL=prefill)   # vision_ref.py's replies follow Forward.prompt
        if [[ "$PHASE" == follow ]]; then
            timeout --kill-after=30 900 docker run "${GEN[@]}" --name dsv41-zig-vgen-r1 --entrypoint bash "$IMAGE" \
                -c "$NCCL; exec /kit/tf-dsv41-m1 follow /model /assets" > "$OUT/vision-follow${VARIANT:+-$VARIANT}.log" 2>&1
            rc=$?; docker rm -f dsv41-zig-vgen-r1 > /dev/null 2>&1; tail -3 "$OUT/vision-follow${VARIANT:+-$VARIANT}.log"; exit $rc
        fi
        log=$OUT/vision-generate${VARIANT:+-$VARIANT}.log
        timeout --kill-after=30 900 docker run "${GEN[@]}" --name dsv41-zig-vgen-r0 --entrypoint bash "$IMAGE" \
            -c "$NCCL; exec /kit/tf-dsv41-m1 generate /model /assets /out/ref/$trace" > "$log" 2>&1
        rc=$?; docker rm -f dsv41-zig-vgen-r0 > /dev/null 2>&1
        grep -E "^(PASS|FAIL) M4|\"equal\"" "$log" | tail -5
        if (( rc == 0 )) && grep -q '^PASS M4' "$log"; then res "PASS vision generate${VARIANT:+-$VARIANT} (replies == Python's)"
        else res "FAIL vision generate${VARIANT:+-$VARIANT} (rc $rc): $(tail -1 "$log")"; tail -25 "$log"; exit 1; fi ;;
    *) echo "phase: ref | vit | tower [sweep] | prefill | generate [bias] | follow [bias]"; exit 2 ;;
esac
