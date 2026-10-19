#!/usr/bin/env bash
# Run one local rank. Invoke separately on each node with the same serving env.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rank=${1:?usage: run-zig.sh 0|1 [--dry-run|--check-aot]}
[[ "$rank" == 0 || "$rank" == 1 ]] || { echo 'rank must be 0 or 1' >&2; exit 2; }
# PACK picks the served weight pack's config (docs/PACKS.md): unset or flash17 reads config/prod-zig.env (the
# recommended flash17 profile copied there), any other id reads config/prod-zig-<id>.env (eaq:
# config/prod-zig-eaq.env from config/prod-zig-eaq.env.example). CONFIG names a file directly and wins.
selected=${PACK:-}
[[ -z "$selected" || "$selected" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || { echo "PACK=$selected: use a pack id such as eaq" >&2; exit 2; }
if [[ -z "$selected" || "$selected" == flash17 ]]; then default_config=config/prod-zig.env
else default_config=config/prod-zig-$selected.env; fi
config=${CONFIG:-$default_config}
[[ -f "$config" ]] || { echo "missing $config; copy its config/*.env.example first (docs/PACKS.md)" >&2; exit 2; }
unset PACK
set -a
source "$config"
set +a
# A config may name its pack (PACK=<id>); a selected PACK must agree with it.
declared=${PACK:-}
if [[ -n "$selected" && -n "$declared" && "$selected" != "$declared" ]]; then
    echo "PACK=$selected selected, but $config is the profile of pack $declared" >&2; exit 2
fi
for key in IMAGE TF_TP_MASTER NCCL_SOCKET_IFNAME NCCL_IB_HCA MODEL_DIR ENGRAM_DIR ASSETS_DIR PREPARED_DIR STATE_DIR CACHE_DIR; do
    [[ -n "${!key:-}" && "${!key}" != *'<'* ]] || { echo "set $key in $config" >&2; exit 2; }
done
# Pass values by environment name, preserving spaces in values. A value still holding a <placeholder> stops here:
# some engine knobs fall back to their default on a value they cannot parse.
envs=()
while IFS='=' read -r key value; do
    case "$key" in TF_DSV41_*|GLM53_TF_*|NCCL_*)
        [[ "$value" != *'<'* ]] || { echo "set $key in $config (still the placeholder $value)" >&2; exit 2; }
        envs+=(-e "$key");;
    esac
done < <(env)
# --check-aot: every Triton variant this env launches, checked against this rank's aot/ (host only, no GPU work).
# Run it before the first start and after any serving-setting change; it needs the fill's sigs.json in ASSETS_DIR.
if [[ "${2:-}" == --check-aot ]]; then
    [[ -f "$ASSETS_DIR/sigs.json" && -f "$ASSETS_DIR/aot/aot.json" ]] || {
        echo "--check-aot needs $ASSETS_DIR/sigs.json (triton_fill.py sigs) and $ASSETS_DIR/aot/aot.json" >&2; exit 2; }
    mkdir -p "$STATE_DIR"
    docker run --rm --init -v "$ASSETS_DIR:/assets:ro" -v "$STATE_DIR:/state" "${envs[@]}" -e TF_TP_WORLD=2 "$IMAGE" \
        tf-dsv41-m1 aot-needs /assets/sigs.json "/state/aot-needs-r$rank.json" /assets/aot || {
        echo "rank $rank: aot/ does not cover this env (missing variants above; every needed one in $STATE_DIR/aot-needs-r$rank.json)" >&2
        echo "compile them: triton_fill.py compile --needs <that file> --have <aot> --out <aot> (docs/ZIG-SERVE.md)" >&2
        exit 1; }
    exit
fi
args=(docker run --rm --init --name "tensorfold-zig-r$rank" --gpus all --ipc=host --network host
      --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
      -v "$MODEL_DIR:/model:ro" -v "$ENGRAM_DIR:/engram:ro" -v "$ASSETS_DIR:/assets:ro"
      -v "$PREPARED_DIR:/prepared" -v "$STATE_DIR:/state" -v "$STATE_DIR/sessions:/sessions"
      -v "$CACHE_DIR:/cache" "${envs[@]}")
args+=(-e TF_TP_WORLD=2 -e "TF_TP_RANK=$rank" -e TF_TP_DEVICE=0
       -e "TF_TP_MASTER=$TF_TP_MASTER" -e "TF_TP_PORT=${TF_TP_PORT:-29571}"
       -e "TF_COMM_BACKEND=${TF_COMM_BACKEND:-roce}"
       -e TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin "$IMAGE")
if [[ "$rank" == 0 ]]; then
    args+=(tensorfold-dsv41 serve /model --host "${HOST:-localhost}" --port "${PORT:-8000}"
           --parallel "$TF_DSV41_SLOTS" --context "$TF_DSV41_CONTEXT"
           --name DeepSeek-V4.1-Flash-TF --max-tokens 32768)
    if [[ "${TF_DSV41_THINKING:-1}" == 0 ]]; then args+=(--no-thinking); fi  # the default; requests still override
else
    args+=(tf-dsv41-m1 follow /model /assets)
fi
if [[ "${2:-}" == --dry-run ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit; fi
for dir in "$MODEL_DIR" "$ENGRAM_DIR" "$ASSETS_DIR" "$CACHE_DIR"; do
    [[ -d "$dir" ]] || { echo "missing directory $dir" >&2; exit 2; }
done
for asset in rope.bin engram-host.bin aot; do
    [[ -e "$ASSETS_DIR/$asset" ]] || { echo "missing asset $asset" >&2; exit 2; }
done
# rope.bin ("DSV41RP1", u32 rows, ...) covers the context plus the 2,048 prefill rows the engine reads past it
rope_rows=$(od -An -t u4 -j 8 -N 4 "$ASSETS_DIR/rope.bin" | tr -d ' ')
need_rows=$(( ${TF_DSV41_CONTEXT:-4096} + 2048 ))
if (( rope_rows < need_rows )); then
    echo "rope.bin holds $rope_rows rows; TF_DSV41_CONTEXT ${TF_DSV41_CONTEXT:-4096} needs $need_rows. Regenerate it:" >&2
    echo "  PYTHONPATH=\$PY_SOURCE python engine/zig/tools/zig/dsv41_rope_tables.py --config $MODEL_DIR/config.json \\" >&2
    echo "    --rows $need_rows --out $ASSETS_DIR/rope.bin   (docs/ZIG-SERVE.md, Generate runtime assets)" >&2
    exit 2
fi
# image input with the routing bias: the folder TF_DSV41_BIAS_VL names inside /cache must hold the fetched tensors
bias=${TF_DSV41_BIAS_VL:-}
if [[ "${TF_DSV41_IMAGES:-}" == native && "$bias" == /cache/* ]] &&
    ! compgen -G "$CACHE_DIR/${bias#/cache/}/*.safetensors" > /dev/null; then
    echo "TF_DSV41_BIAS_VL=$bias: no *.safetensors in $CACHE_DIR/${bias#/cache/}; the engine would refuse to boot." >&2
    echo "  fetch bias_vl.safetensors there (docs/IMAGES.md, The Zig serving path), or unset TF_DSV41_BIAS_VL" >&2
    exit 2
fi
# the published pfdense tuning table, placed where the env points inside /cache when it is not there yet
table=${TF_DSV41_PF_DENSE_TABLE:-}
if [[ "$table" == /cache/* && ! -e "$CACHE_DIR/${table#/cache/}" ]]; then
    mkdir -p "$(dirname "$CACHE_DIR/${table#/cache/}")"
    cp config/pfdense-table.json "$CACHE_DIR/${table#/cache/}"
    echo "copied config/pfdense-table.json to $CACHE_DIR/${table#/cache/} (TF_DSV41_PF_DENSE_TABLE=$table)" >&2
fi
mkdir -p "$PREPARED_DIR" "$STATE_DIR/sessions"
# A config that names its pack owns its state folder: the draft calibration under it is not keyed by pack at
# engine 9131e10a, so a state folder that served another pack is refused rather than silently reused.
if [[ -n "$declared" ]]; then
    owner_file="$STATE_DIR/pack-id"
    if [[ -f "$owner_file" ]]; then
        owner=$(< "$owner_file")
        [[ "$owner" == "$declared" ]] || {
            echo "STATE_DIR $STATE_DIR belongs to pack $owner ($owner_file), not $declared: give each pack its own" >&2
            exit 2; }
    else
        printf '%s\n' "$declared" > "$owner_file"
    fi
fi
exec "${args[@]}"
