#!/usr/bin/env bash
# Watchdog for the Zig TP=2 pair (scripts/run-zig.sh on each node). Run on the rank 0 node, by tf-zig-watchdog.timer.
#
#   scripts/watch-zig.sh --once    one tick (exit 0 healthy or standing down, 1 unhealthy)
#   scripts/watch-zig.sh           a tick every WATCH_INTERVAL s
#
# Stands down while WATCH_LEASE (a file you touch while you benchmark or maintain the pair) is younger than
# WATCH_LEASE_MIN minutes, while WATCH_STATE_DIR/stopped exists (touch it when you stop the pair on purpose), while a
# heal holds the lock, and while the guard's lifecycle check (GUARD_CONFIG, scripts/guard) holds the pair.
# A tick is bad when a rank is not running or /health fails; WATCH_FAILS bad ticks in a row heal (WATCH_HEAL=1 runs
# WATCH_RESTART_CMD in the background) at most once every WATCH_MIN_HEAL s. Fail-fast (exit 70), the engine's own
# "stalled" verdict and WATCH_PROBE_FAILS failed probes in a row heal on that tick, at most every WATCH_FF_MIN_HEAL s.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config=${CONFIG:-config/prod-zig.env}
if [[ -f "$config" ]]; then set -a; source "$config"; set +a; fi
NAME="${NAME:-tensorfold-zig}"
WORKER_SSH="${WORKER_SSH:-}"
BASE="${WATCH_BASE:-http://127.0.0.1:${WATCH_PORT:-8000}}"
WATCH_MODEL="${WATCH_MODEL:-DeepSeek-V4.1-Flash-TF}"
WATCH_STATE_DIR="${WATCH_STATE_DIR:-${XDG_STATE_HOME:-${HOME:-/tmp}/.local/state}/tf-zig-watch}"
GUARD_CONFIG="${GUARD_CONFIG:-}"
WATCH_GRACE="${WATCH_GRACE:-1800}"; WATCH_FAILS="${WATCH_FAILS:-3}"; WATCH_HEAL="${WATCH_HEAL:-0}"
WATCH_MIN_HEAL="${WATCH_MIN_HEAL:-1800}"; WATCH_FF_MIN_HEAL="${WATCH_FF_MIN_HEAL:-120}"; WATCH_ALERT="${WATCH_ALERT:-}"
WATCH_LEASE="${WATCH_LEASE:-}"; WATCH_LEASE_MIN="${WATCH_LEASE_MIN:-20}"; WATCH_RESTART_CMD="${WATCH_RESTART_CMD:-}"
# The probe: /health stays 200 while a wedged engine answers nothing. Every WATCH_PROBE_EVERY s one tiny completion
# (max_tokens 1, no reasoning, header X-TF-Probe: watchdog) must answer within WATCH_PROBE_TIMEOUT s. It runs only when
# the pair is idle: skipped while every slot is busy, and the guard defers it (429 watchdog_deferred) behind user work,
# queued requests and admission holds, so load is never counted as an engine failure.
WATCH_PROBE="${WATCH_PROBE:-1}"; WATCH_PROBE_EVERY="${WATCH_PROBE_EVERY:-180}"; WATCH_PROBE_TIMEOUT="${WATCH_PROBE_TIMEOUT:-30}"
WATCH_PROBE_FAILS="${WATCH_PROBE_FAILS:-3}"

log() { echo "[tf-zig-watch $(date -u +%H:%M:%S)] $*" >&2; }
alert() { log "ALERT: $*"; [[ -z "$WATCH_ALERT" ]] || "$WATCH_ALERT" "$*" || true; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_SSH" "$@"; }
lease_fresh() { [[ -n "$WATCH_LEASE" && -f "$WATCH_LEASE" ]] && (( $(date +%s) - $(stat -c %Y "$WATCH_LEASE") < WATCH_LEASE_MIN * 60 )); }
guard_permit() { # the guard's lifecycle decision: 0 permit, 75 hold, 78 durable STOP (never heal through a STOP)
    [[ -n "$GUARD_CONFIG" ]] || return 0
    python3 scripts/guard/lifecycle.py --config "$GUARD_CONFIG" check --operation "$1" > /dev/null
}
running() { # $1 = rank: true / false / absent (/ unreachable for rank 1)
    local out
    if [[ "$1" == 0 ]]; then out=$(docker inspect -f '{{.State.Running}}' "$NAME-r0" 2>/dev/null) || { echo absent; return; }
    else
        [[ -n "$WORKER_SSH" ]] || { echo unreachable; return; }
        out=$(wssh "docker inspect -f '{{.State.Running}}' $NAME-r1 2>/dev/null || echo absent") || { echo unreachable; return; }
    fi
    echo "$out"
}

probe_tick() { # $1 = now, $2 = the /health body. 1: WATCH_PROBE_FAILS tiny completions in a row got nothing; else 0
    [[ "$WATCH_PROBE" == 1 ]] || return 0
    local last_f="$WATCH_STATE_DIR/probe_last" pf_f="$WATCH_STATE_DIR/probe_fails" last pf out code busy
    last=$(cat "$last_f" 2>/dev/null || echo 0); (( $1 - last >= WATCH_PROBE_EVERY )) || return 0
    busy=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(d.get("requests_running",0) >= (d.get("streams") or {}).get("max",1e9)))' "$2" 2>/dev/null || echo 0)
    if [[ "$busy" == 1 ]]; then log "watch: every request slot busy: probe skipped"; return 0; fi
    echo "$1" > "$last_f"
    out=$(curl -s -m "$WATCH_PROBE_TIMEOUT" -H 'Content-Type: application/json' -H 'X-TF-Probe: watchdog' -w '\n%{http_code}' \
        -d '{"model":"'"$WATCH_MODEL"'","messages":[{"role":"user","content":"ok"}],"max_tokens":1,"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}' \
        "$BASE/v1/chat/completions" || true)
    code=${out##*$'\n'}
    if [[ "$code" == 429 && "$out" == *watchdog_deferred* ]]; then
        log "watch: guard deferred the idle-only probe (user work or admission hold)"; return 0
    fi
    if [[ "$code" == 200 && "$out" == *'"choices"'* ]]; then echo 0 > "$pf_f"; return 0; fi
    pf=$(( $(cat "$pf_f" 2>/dev/null || echo 0) + 1 )); echo "$pf" > "$pf_f"
    log "watch: probe $pf of $WATCH_PROBE_FAILS failed (http ${code:-none} within ${WATCH_PROBE_TIMEOUT} s)"
    (( pf >= WATCH_PROBE_FAILS )) || return 0
    echo 0 > "$pf_f"; return 1
}

# The native idle round clock can be old when newly active work starts prefill (docs/KNOWN-ISSUES.md). Only this
# exact young-request verdict gets a bounded recheck; all other failures keep their policy.
startup_stale_idle() {
    [[ "$1" == 503 ]] || return 1
    python3 -c '
import json,math,re,sys
def number(v): return type(v) in (int,float) and math.isfinite(v)
try:
 d=json.loads(sys.argv[1]); age=(d.get("rounds") or {}).get("last_age_s")
 oldest=d.get("oldest_s"); active=d.get("inflight")
 valid=(d.get("ok") is False and type(d.get("errors")) is int and d["errors"] == 0
        and not d.get("fatal") and d.get("status") == "stalled"
        and isinstance(d.get("stall_reason"),str)
        and re.fullmatch(r"no round for [0-9]+ s with work in flight",d["stall_reason"]) is not None
        and number(oldest) and 0 <= oldest <= 3 and number(age) and age > 120
        and type(active) is int and active > 0)
except (ValueError,TypeError,AttributeError): valid=False
sys.exit(0 if valid else 1)
' "$2"
}

startup_recheck_healthy() {
    [[ "$1" == 200 ]] || return 1
    python3 -c '
import json,sys
try:
 d=json.loads(sys.argv[1])
 valid=d.get("ok") is True and type(d.get("errors")) is int and d["errors"] == 0 and not d.get("fatal")
except (ValueError,TypeError,AttributeError): valid=False
sys.exit(0 if valid else 1)
' "$2"
}

startup_retry() { # $1 = code, $2 = body; prints "code<newline>body" after at most 3 s of rechecks
    local code=$1 body=$2 retry_deadline retry_left retry_timeout
    retry_deadline=$(( ${EPOCHREALTIME/./} + 3000000 ))
    log "watch: young prefill with stale idle round clock; rechecking health for at most 3 s"
    while startup_stale_idle "$code" "$body"; do
        retry_left=$(( retry_deadline - ${EPOCHREALTIME/./} ))
        (( retry_left > 0 )) || break
        # Curl itself is bounded by the remaining window, including a hung health endpoint.
        printf -v retry_timeout '%d.%06d' "$((retry_left / 1000000))" "$((retry_left % 1000000))"
        body=$(curl -s -m "$retry_timeout" -w '\n%{http_code}' "$BASE/health" || true)
        code=${body##*$'\n'}; body=${body%$'\n'*}
        startup_stale_idle "$code" "$body" || break
        retry_left=$(( retry_deadline - ${EPOCHREALTIME/./} ))
        if (( retry_left > 200000 )); then sleep 0.2; fi
    done
    if startup_recheck_healthy "$code" "$body" && (( ${EPOCHREALTIME/./} <= retry_deadline )); then
        log "watch: startup false-stall 503 resolved by actual healthy 200 within 3 s"; return 0
    fi
    printf '%s\n%s' "$code" "$body"; return 1
}

evidence_dump() { # $1 = why. Before a heal: WATCH_STATE_DIR/heal-evidence/<UTC stamp>/ gets health, logs and state
    local d; d="$WATCH_STATE_DIR/heal-evidence/$(date -u +%Y%m%d-%H%M%S)"; mkdir -p "$d" || return 0
    echo "$1" > "$d/why.txt" || true
    curl -s -m 10 "$BASE/health" > "$d/health.json" 2>&1 || true
    docker logs --timestamps "$NAME-r0" > "$d/r0.log" 2>&1 || true
    docker inspect -f '{{json .State}}' "$NAME-r0" > "$d/r0-state.json" 2>&1 || true
    if [[ -n "$WORKER_SSH" ]]; then
        wssh "docker logs --timestamps $NAME-r1 2>&1" > "$d/r1.log" 2>/dev/null || true
        wssh "docker inspect -f '{{json .State}}' $NAME-r1" > "$d/r1-state.json" 2>&1 || true
    fi
    log "watch: heal evidence in $d"
}

watch_tick() {
    mkdir -p "$WATCH_STATE_DIR"
    local fails_f="$WATCH_STATE_DIR/fails" heal_f="$WATCH_STATE_DIR/last_heal" fails r0 r1 age code body now bad="" last
    local probe_bad=0 retried
    guard_permit watch || { log "watch: guard lifecycle holds; standing down"; return 0; }
    fails=$(cat "$fails_f" 2>/dev/null || echo 0); now=$(date +%s)
    if lease_fresh; then echo 0 > "$fails_f"; log "watch: $WATCH_LEASE is fresh; standing down"; return 0; fi
    if [[ -e "$WATCH_STATE_DIR/stopped" ]]; then echo 0 > "$fails_f"; log "watch: stopped on purpose; standing down"; return 0; fi
    if ! flock -n "$WATCH_STATE_DIR/lock" true; then log "watch: a heal is in progress; standing down"; return 0; fi
    r0=$(running 0); r1=$(running 1)
    if [[ "$r0" != true ]]; then bad="rank 0 $r0"
    elif [[ "$r1" == unreachable ]]; then log "watch: worker unreachable over ssh (not counted)"
    elif [[ "$r1" != true ]]; then bad="rank 1 $r1"
    else
        age=$(( now - $(date -d "$(docker inspect -f '{{.State.StartedAt}}' "$NAME-r0")" +%s) ))
        body=$(curl -s -m 10 -w '\n%{http_code}' "$BASE/health" || true)
        code=${body##*$'\n'}; body=${body%$'\n'*}
        if startup_stale_idle "$code" "$body"; then
            if retried=$(startup_retry "$code" "$body"); then code=200; body='{"ok":true}'
            else
                code=${retried%%$'\n'*}; body=${retried#*$'\n'}
                bad="stalled: startup health recheck unresolved: /health $code ${body:0:300}"; probe_bad=1
            fi
        fi
        if [[ -n "$bad" ]]; then :
        elif [[ "$code" == 200 ]]; then
            probe_tick "$now" "$body" || { bad="wedged: $WATCH_PROBE_FAILS tiny completions in a row got nothing within ${WATCH_PROBE_TIMEOUT} s (/health 200)"; probe_bad=1; }
        elif [[ "$code" == 000 && $age -lt $WATCH_GRACE ]]; then log "watch: loading (${age}s)"; echo 0 > "$fails_f"; return 0
        elif [[ "$code" == 503 && "$body" =~ \"status\":\ *\"stalled\" ]]; then
            # the engine's own stall verdict: heal now, like a fail-fast exit
            bad="stalled: ${body:0:300}"; probe_bad=1
        else bad="/health $code ${body:0:300}"; fi
    fi
    if [[ -z "$bad" ]]; then echo 0 > "$fails_f"; return 0; fi
    local need=$WATCH_FAILS min=$WATCH_MIN_HEAL e0 e1=""
    e0=$(docker inspect -f '{{.State.ExitCode}}' "$NAME-r0" 2>/dev/null || true)
    [[ -z "$WORKER_SSH" ]] || e1=$(wssh "docker inspect -f '{{.State.ExitCode}}' $NAME-r1" 2>/dev/null || true)
    if [[ "$r0" == false && "$e0" == 70 ]] || [[ "$r1" == false && "$e1" == 70 ]]; then
        need=1; min=$WATCH_FF_MIN_HEAL; bad="$bad (fail-fast exit 70)"
    fi
    # the probe's failures were counted by probe_tick already
    if [[ "$probe_bad" == 1 ]]; then need=1; min=$WATCH_FF_MIN_HEAL; fi
    fails=$((fails + 1)); echo "$fails" > "$fails_f"
    log "watch: bad tick $fails of $need: $bad"
    (( fails >= need )) || return 0
    last=$(cat "$heal_f" 2>/dev/null || echo 0)
    if [[ "$WATCH_HEAL" != 1 || -z "$WATCH_RESTART_CMD" ]]; then alert "unhealthy ($bad); not restarting (WATCH_HEAL=1 and WATCH_RESTART_CMD heal)"; return 1; fi
    if (( now - last < min )); then alert "unhealthy ($bad); healed $((now - last))s ago, waiting"; return 1; fi
    guard_permit start || { log "watch: guard lifecycle holds the heal; standing down"; return 0; }
    echo "$now" > "$heal_f"; echo 0 > "$fails_f"
    evidence_dump "$bad"
    alert "unhealthy ($bad); restarting both ranks"
    setsid flock "$WATCH_STATE_DIR/lock" bash -c "$WATCH_RESTART_CMD" >> "$WATCH_STATE_DIR/heal.log" 2>&1 < /dev/null &
    return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ "${1:-}" == --once ]]; then watch_tick; exit $?; fi
    while :; do watch_tick || true; sleep "${WATCH_INTERVAL:-60}"; done
fi
