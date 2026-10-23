#!/usr/bin/env bash
# Manage the two detached Zig ranks from the head node.
#
# usage: zig-stack.sh start|stop|restart|status|logs [0|1|both]
#
# The trusted, local CONFIG file supplies WORKER_SSH, WORKER_REPO, and
# WORKER_CONFIG. It is sourced here and by each local-rank launcher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG=${CONFIG:-config/prod-zig.env}
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG; copy config/prod-zig-flash17.env.example first" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

for key in WORKER_SSH WORKER_REPO WORKER_CONFIG; do
    [[ -n "${!key:-}" && "${!key}" != *'<'* ]] || { echo "set $key in $CONFIG" >&2; exit 2; }
done

DRAIN_S=${TF_DSV41_DRAIN_S:-20}
[[ "$DRAIN_S" =~ ^[0-9]+$ ]] || { echo "TF_DSV41_DRAIN_S must be an integer" >&2; exit 2; }
MIN_STOP_TIMEOUT=$(( DRAIN_S + 60 ))
STOP_TIMEOUT=${ZIG_STOP_TIMEOUT:-$MIN_STOP_TIMEOUT}
[[ "$STOP_TIMEOUT" =~ ^[0-9]+$ && "$STOP_TIMEOUT" -ge "$MIN_STOP_TIMEOUT" ]] || {
    echo "ZIG_STOP_TIMEOUT must be an integer of at least $MIN_STOP_TIMEOUT (TF_DSV41_DRAIN_S + 60)" >&2; exit 2; }
TAIL=${TAIL:-120}
RANK0=tensorfold-zig-r0
RANK1=tensorfold-zig-r1

log() { printf '[zig-stack] %s\n' "$*"; }
shell_quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
remote() {
    local arg command=""
    for arg in "$@"; do command+="$(shell_quote "$arg") "; done
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$command"
}

head_state() {
    docker info >/dev/null || return
    docker inspect -f '{{if .State.Running}}running{{else}}stopped{{end}}' "$RANK0" 2>/dev/null || printf 'absent\n'
}
worker_state() {
    remote bash -c 'docker info >/dev/null && { docker inspect -f "{{if .State.Running}}running{{else}}stopped{{end}}" "$1" 2>/dev/null || printf "absent\\n"; }' bash "$RANK1"
}
worker_launch() {
    remote bash -c 'cd -- "$1" && CONFIG="$2" ./scripts/run-zig.sh 1 --detach' bash "$WORKER_REPO" "$WORKER_CONFIG"
}
head_launch() { CONFIG="$CONFIG" ./scripts/run-zig.sh 0 --detach; }

stop_pair() {
    local r0 r1 rc=0
    r0=$(head_state)
    r1=$(worker_state)

    # Keep restart disabled after an intentional stop: a follower can exit
    # naturally during the head's drain without Docker recording a manual stop.
    # start_pair recreates both containers with unless-stopped.
    if [[ "$r0" != absent ]]; then docker update --restart=no "$RANK0" >/dev/null; fi
    if [[ "$r1" != absent ]]; then remote docker update --restart=no "$RANK1" >/dev/null; fi
    if [[ "$r0" == running ]]; then docker stop --time "$STOP_TIMEOUT" "$RANK0" >/dev/null || rc=$?; fi
    if [[ "$r1" == running && "$(worker_state)" == running ]]; then
        remote docker stop --time "$STOP_TIMEOUT" "$RANK1" >/dev/null || rc=$?
    fi
    (( rc == 0 )) || return 1
}

start_pair() {
    local r0 r1
    r0=$(head_state)
    r1=$(worker_state)
    if [[ "$r0" == running && "$r1" == running ]]; then
        log 'both containers are already running; this does not establish API readiness'
        return 0
    fi
    if [[ "$r0" == running || "$r1" == running ]]; then
        echo "ranks are inconsistent (head=$r0 worker=$r1); use scripts/zig-stack.sh restart" >&2
        return 1
    fi
    worker_launch
    if ! head_launch; then
        log 'rank 0 did not launch; rank 1 was started and remains inspectable with logs'
        return 1
    fi
    log 'both containers were created with restart=unless-stopped; model loading continues in the background'
    log "wait for status to show a successful /v1/models response before sending traffic (port ${PORT:-8000})"
}

status() {
    local r0 r1
    r0=$(head_state)
    printf 'head (%s): %s\n' "$RANK0" "$r0"
    if [[ "$r0" != absent ]]; then
        docker inspect -f '{{.Name}} {{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}}' "$RANK0"
    fi
    r1=$(worker_state)
    printf 'worker (%s): %s\n' "$RANK1" "$r1"
    if [[ "$r1" != absent ]]; then
        remote docker inspect -f '{{.Name}} {{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}}' "$RANK1"
    fi
    if curl -fsS --max-time 5 "http://127.0.0.1:${PORT:-8000}/v1/models"; then
        printf '\n'
    else
        printf 'API not ready or unreachable; running containers are not a readiness signal\n' >&2
    fi
}

logs() {
    case "${1:-both}" in
        0) docker logs --tail "$TAIL" "$RANK0" ;;
        1) remote docker logs --tail "$TAIL" "$RANK1" ;;
        both) printf '%s\n' "==> $RANK0 <=="; docker logs --tail "$TAIL" "$RANK0"
              printf '%s\n' "==> $RANK1 <=="; remote docker logs --tail "$TAIL" "$RANK1" ;;
        *) echo 'logs accepts 0, 1, or both' >&2; return 2 ;;
    esac
}

case "${1:-}" in
    start) start_pair ;;
    stop) stop_pair; log 'both ranks are stopped with restart disabled until the next start' ;;
    restart) stop_pair; start_pair ;;
    status) status ;;
    logs) logs "${2:-both}" ;;
    *) sed -n '2,6p' "$0" >&2; exit 2 ;;
esac
