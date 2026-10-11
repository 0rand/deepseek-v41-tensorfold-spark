#!/usr/bin/env bash
# Check that a Zig serving image holds every shared library the engine opens at run time.
#   scripts/check-zig-image.sh IMAGE    from the build host (docker; another architecture runs under emulation)
#   check-zig-image.sh --inside         inside the image, e.g. as the Dockerfile's last RUN step
# The engine links only libc; NCCL, verbs and cuBLAS / cuBLASLt are opened by name, so a missing one shows up only
# when its path first runs (cuBLAS: the first image request). libcuda.so.1 comes from the host driver at run time.
# CUBLAS_VERSION: the cuBLAS / cuBLASLt release the vision tower's bit-exactness was measured on; any other fails.
set -uo pipefail
CUBLAS_VERSION=${CUBLAS_VERSION:-13.6.0.2}
BIN=${BIN:-/opt/tensorfold/bin}
if [[ "${1:-}" != --inside ]]; then
    image=${1:?usage: check-zig-image.sh IMAGE | --inside}
    platform=$(docker image inspect -f '{{.Os}}/{{.Architecture}}' "$image") || exit 2
    exec docker run --rm -i --platform "$platform" --entrypoint bash -e CUBLAS_VERSION="$CUBLAS_VERSION" "$image" \
        -s -- --inside < "${BASH_SOURCE[0]}"
fi
TEXT="libnccl.so.2 libibverbs.so.1"     # TP over NCCL and RoCE, every boot
IMAGES="libcublas.so.13 libcublasLt.so.13"  # TF_DSV41_IMAGES=native: the vision tower's GEMMs
fail=0
names=$(grep -aohE 'lib[A-Za-z0-9_]+\.so(\.[0-9]+)*' "$BIN"/* | sort -u | tr '\n' ' ')
echo "names the engine can open: $names"
for n in $names; do
    case " libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 libcuda.so.1 libcublas.so libcublasLt.so $TEXT $IMAGES " in
        *" $n "*) ;;
        *) echo "  $n: not covered by this check"; fail=1 ;;
    esac
done
for s in $TEXT $IMAGES; do
    f=$(ldconfig -p | awk -v s="$s" '$1 == s {print $NF; exit}')
    if [[ -z "$f" ]]; then echo "  $s: MISSING"; fail=1; continue; fi
    real=$(readlink -f "$f")
    unresolved=$(ldd "$real" 2>&1 | grep -c 'not found')
    note=""
    (( unresolved == 0 )) || { note=" ($unresolved dependencies not found)"; fail=1; }
    if [[ $s == libcublas* && $real != *".so.$CUBLAS_VERSION" ]]; then
        note+=" (not $CUBLAS_VERSION)"; fail=1
    fi
    echo "  $s -> $real$note"
done
provider=$(ls /usr/lib/*-linux-gnu/libibverbs/ 2>/dev/null | grep -m1 mlx5)
echo "  verbs mlx5 provider: ${provider:-MISSING}"
[[ -n "$provider" ]] || fail=1
(( fail == 0 )) && echo "check-zig-image: OK" || echo "check-zig-image: FAILED"
exit $fail
