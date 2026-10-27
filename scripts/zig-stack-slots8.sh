#!/usr/bin/env bash
# Run the staged 8-slot stack: config/prod-zig-slots8.env + tensorfold-dsv41-zig:main-0rand-5ff0d5c.
# Thin wrapper over scripts/zig-stack.sh; every argument is passed through.
#
#   scripts/zig-stack-slots8.sh start|stop|restart|sync-config|status|logs [0|1|both]
#
# The staged config keeps the live memory guards and the 2.1M pool; only TF_DSV41_SLOTS=8.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG=config/prod-zig-slots8.env
IMAGE=tensorfold-dsv41-zig:main-0rand-5ff0d5c
ZIG_COMMIT=5ff0d5c56f97f34770dc597a8537393a693fcc84
export CONFIG IMAGE ZIG_COMMIT

[[ -f "$CONFIG" ]] || { echo "missing staged config: $CONFIG" >&2; exit 2; }

configured_image=$(grep -m1 '^IMAGE=' "$CONFIG" | cut -d= -f2- || true)
[[ "$configured_image" == "$IMAGE" ]] || {
    echo "staged config $CONFIG points at '$configured_image', expected '$IMAGE'" >&2
    exit 2
}
configured_commit=$(grep -m1 '^ZIG_COMMIT=' "$CONFIG" | cut -d= -f2- || true)
[[ "$configured_commit" == "$ZIG_COMMIT" ]] || {
    echo "staged config $CONFIG records '$configured_commit', expected '$ZIG_COMMIT'" >&2
    exit 2
}

exec ./scripts/zig-stack.sh "$@"
