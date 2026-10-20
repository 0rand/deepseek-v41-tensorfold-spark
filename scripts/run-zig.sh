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
# STATE_DIR ownership, settled before anything is written into STATE_DIR (--check-aot writes aot-needs-r*.json, a
# start makes sessions/). The draft calibration under it (calib-zig) is not keyed by pack at engine 9131e10a, so one
# pack must never start on another pack's state. A config that names its pack claims the folder (STATE_DIR/pack-id)
# only while it is new or empty; a used folder without pack-id (it served flash17 or q28-v2 before packs were
# marked) is refused, never adopted. A config that names no pack (the default config/prod-zig.env) claims nothing but
# is refused a folder marked for any pack other than the selected one (flash17 when PACK is unset).
claim_state() {
    local pack=${declared:-${selected:-flash17}} owner_file="$STATE_DIR/pack-id" owner tmp
    [[ "$pack" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || { echo "PACK=$pack in $config: use a pack id such as eaq" >&2; exit 2; }
    mkdir -p "$STATE_DIR"
    if [[ -n "$declared" && ! -e "$owner_file" && ! -L "$owner_file" ]]; then
        # Empty but for another launcher's claim in flight (.pack-id.*): claim with a complete temp file hard-linked
        # into place. ln never replaces an existing pack-id, so of racing launchers exactly one claims.
        if [[ -z "$(find "$STATE_DIR" -mindepth 1 -maxdepth 1 ! -name '.pack-id.*' -print -quit)" ]]; then
            tmp=$(mktemp "$STATE_DIR/.pack-id.XXXXXX")
            printf '%s\n' "$declared" > "$tmp"
            ln "$tmp" "$owner_file" 2> /dev/null || true
            rm -f "$tmp"
        elif [[ ! -e "$owner_file" ]]; then
            echo "STATE_DIR $STATE_DIR is in use but marked for no pack; refusing to claim it for pack $declared." >&2
            echo "  Its draft calibration may be another pack's (docs/PACKS.md). Give $declared its own new STATE_DIR, or," >&2
            echo "  only if this folder served $declared alone: printf '%s\\n' $declared > $owner_file" >&2
            exit 2
        fi
        [[ -e "$owner_file" ]] || { echo "could not create $owner_file" >&2; exit 2; }
    fi
    if [[ -e "$owner_file" || -L "$owner_file" ]]; then
        [[ -f "$owner_file" && ! -L "$owner_file" ]] || { echo "$owner_file is not a regular file" >&2; exit 2; }
        owner=$(< "$owner_file")
        [[ "$owner" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
            echo "$owner_file holds no pack id; fix or remove it only once you know which pack used $STATE_DIR" >&2
            exit 2; }
        [[ "$owner" == "$pack" ]] || {
            echo "STATE_DIR $STATE_DIR belongs to pack $owner ($owner_file), not $pack: give each pack its own" >&2
            exit 2; }
    fi
}
if [[ "${2:-}" != --dry-run ]]; then claim_state; fi
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
exec "${args[@]}"
