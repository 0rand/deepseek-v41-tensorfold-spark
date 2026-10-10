#!/usr/bin/env bash
# The kit's / prod image's ONE fatbin dir (build host, no GPU): the gated sm_121 dsv41_*.fatbin of FATBINS plus
# tp_mailbox.fatbin built from this tree's zig/src/fabric/tp/mailbox.cu (or TP_FATBIN, checked the same way), checked:
#   - nvcc is the image's 13.3 (TF_ZIG_IMAGE, default nvcr.io/nvidia/pytorch:26.07-py3: nvcc V13.3.73, the e1e2 prod
#     image's; zig/kernels/cuda/deepseek_v41/sass_check.sh uses the same);
#   - tp_mailbox.fatbin: zig/build/tp.zig mailboxFatbin's flags (sm_120 + sm_121 SASS, compute_90 PTX), and its tp_roce
#     holds GLM53_TF_ROCE_FAST's code (0d4ff22: wait_flag_shared, NANOSLEEP) on both SASS targets: a fatbin from before
#     10f61d13 / 0d4ff22 (RoceArgs without opts / lens) is refused;
#   - dsv41_glue.fatbin: every glue.cu kernel present, and its sm_121 SASS (instructions and encodings) equal to a fresh
#     compile of this tree's glue.cu with zig/build/cuda.zig's flags (torch's, -O3, --fmad=false): a glue fatbin from
#     another glue.cu is refused;
#   - every kernel zig/build/cuda.zig lists (the `install` step embeds them all: Nemotron's too) is in OUT: one FATBINS
#     lacks is built here with cuda.zig's flags for sm_121 when it is not a DeepSeek kernel (TF 1.0.2's lane_gemv and
#     sample: DeepSeek never loads them); a missing dsv41_* is refused (those come gated).
# Writes OUT/FATBINS.txt (nvcc, tree commit, sha256 of every file). OUT is what build-kit.sh embeds and ships as
# <kit>/fatbins and what the prod image takes (zig-serve: ZIG_FATBINS=<kit>/fatbins scripts/zig-image.sh stage).
#   FATBINS=<dir of sm_121 dsv41_*.fatbin> [TP_FATBIN=<prebuilt tp_mailbox.fatbin>] [KIT_COMMIT=<hash>] bash kit-fatbins.sh OUT
set -euo pipefail
H=${HEAVY:-}
[[ -n "${HEAVY_HELD:-}" || ! -x "$H" ]] || exec env HEAVY_HELD=1 "$H" bash "$0" "$@"
OUT=${1:?out dir}; FB=$(cd "${FATBINS:?sm_121 fatbin dir}" && pwd)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); TREE=$(cd "$HERE/../../.." && pwd)
IMAGE=${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}
TP_FLAGS="-fatbin -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121 -gencode=arch=compute_90,code=compute_90"
# the flags must still be tp.zig's (the Dockerfile's nvcc fallback carries the same line)
for f in $TP_FLAGS; do grep -qF "\"$f\"" "$TREE/zig/build/tp.zig" || { echo "[kit-fatbins] $f is not in zig/build/tp.zig: update TP_FLAGS" >&2; exit 2; }; done
compgen -G "$FB/dsv41_*.fatbin" >/dev/null || { echo "[kit-fatbins] no dsv41_*.fatbin in $FB" >&2; exit 2; }
[[ -f "$FB/dsv41_glue.fatbin" ]] || { echo "[kit-fatbins] no dsv41_glue.fatbin in $FB" >&2; exit 2; }
rm -rf "$OUT" && mkdir -p "$OUT/.chk" && OUT=$(cd "$OUT" && pwd)
cp "$FB"/*.fatbin "$OUT/"
# cuda.zig's kernels missing from FATBINS: "name|source|flags|arch suffix" lines for the container
FILL=$(python3 - "$TREE/zig/build/cuda.zig" "$OUT" <<'PY'
import os, re, sys
src = open(sys.argv[1]).read()
torch_ops = re.search(r'const torch_ops = &\[_\]\[\]const u8\{([^}]*)\}', src).group(1)
torch_flags = re.search(r'const torch_flags = \[_\]\[\]const u8\{([^}]*)\}', src).group(1)
q = lambda t: " ".join(re.findall(r'"([^"]*)"', t))
for table in re.findall(r'const \w+ = \[_\]Kernel\{(.*?)\n\};', src, re.S):
    for m in re.finditer(r'\.\{ \.name = "(\w+)"(?:, \.src = "([^"]+)")?, \.flags = (torch_ops|&\.\{[^}]*\})(, \.arch_specific = true)? \}', table):
        name, sub, flags, arch = m.groups()
        if os.path.exists(os.path.join(sys.argv[2], name + ".fatbin")):
            continue
        if name.startswith("dsv41_"):
            sys.exit(f"[kit-fatbins] FATBINS has no {name}.fatbin (a DeepSeek kernel: bring the gated one)")
        print(f"{name}|zig/kernels/cuda/{sub or name}.cu|{q(torch_flags)} {q(torch_ops) if flags == 'torch_ops' else q(flags)}|{'a' if arch else ''}")
PY
)
TPM=()
if [[ -n "${TP_FATBIN:-}" ]]; then cp "$TP_FATBIN" "$OUT/tp_mailbox.fatbin"; TPM=(-e TP_PREBUILT=1); fi
commit=$(git -C "$TREE" rev-parse HEAD 2>/dev/null || echo "${KIT_COMMIT:?KIT_COMMIT: the commit of an archived tree (no .git)}")
docker run --rm -i --network none --memory 8g -u "$(id -u):$(id -g)" -v "$TREE:/tree:ro" -v "$OUT:/out" "${TPM[@]}" \
    -e TP_FLAGS="$TP_FLAGS" -e FILL="$FILL" --entrypoint bash "$IMAGE" -s <<'EOF'
set -euo pipefail
cd /out
v=$(nvcc --version | tail -2 | head -1); echo "[kit-fatbins] $v"
[[ "$v" == *V13.3.* ]] || { echo "[kit-fatbins] want nvcc 13.3 (the prod image's), got: $v" >&2; exit 3; }
echo "$v" > .chk/nvcc
: > .chk/fill
while IFS='|' read -r name src flags arch; do
    [[ -n "$name" ]] || continue
    # shellcheck disable=SC2086
    nvcc -fatbin $flags "-gencode=arch=compute_121$arch,code=sm_121$arch" -o "$name.fatbin" "/tree/$src"
    echo "$name (nvcc -fatbin $flags sm_121$arch $src)" >> .chk/fill
done <<< "$FILL"
# tp_mailbox.fatbin
if [[ -z "${TP_PREBUILT:-}" ]]; then
    # shellcheck disable=SC2086
    nvcc $TP_FLAGS -o tp_mailbox.fatbin /tree/zig/src/fabric/tp/mailbox.cu
    echo "built: nvcc $TP_FLAGS -o tp_mailbox.fatbin zig/src/fabric/tp/mailbox.cu" > .chk/tp
else
    echo "prebuilt (TP_FATBIN), checked" > .chk/tp
fi
elf=$(cuobjdump --list-elf tp_mailbox.fatbin); ptx=$(cuobjdump --list-ptx tp_mailbox.fatbin)
for want in sm_120.cubin sm_121.cubin; do [[ "$elf" == *"$want"* ]] || { echo "[kit-fatbins] tp_mailbox.fatbin has no $want" >&2; exit 4; }; done
[[ "$ptx" == *sm_90.ptx* ]] || { echo "[kit-fatbins] tp_mailbox.fatbin has no compute_90 PTX" >&2; exit 4; }
for sm in 120 121; do
    s=$(cuobjdump -sass -arch "sm_$sm" tp_mailbox.fatbin; cuobjdump -elf -arch "sm_$sm" tp_mailbox.fatbin)
    # ROCE_FAST (0d4ff22): the shared-flag pollers (a device function of tp_roce: its ELF symbol) and their start stagger
    for want in 'Function : tp_roce' 'Function : tp_mailbox' '$tp_roce$_ZN4tftp16wait_flag_shared' 'NANOSLEEP'; do
        grep -qF "$want" <<< "$s" || { echo "[kit-fatbins] tp_mailbox.fatbin sm_$sm lacks '$want': built before mailbox.cu 0d4ff22 (ROCE_FAST)" >&2; exit 4; }
    done
    echo "sm_$sm: tp_roce + tp_mailbox, wait_flag_shared, $(grep -c NANOSLEEP <<< "$s") NANOSLEEP" >> .chk/tp
done
# dsv41_glue.fatbin: every kernel of this tree's glue.cu, SASS equal to a fresh compile (zig/build/cuda.zig's dsv41_glue)
G=/tree/zig/kernels/cuda/deepseek_v41/glue.cu
nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 --fmad=false \
    -gencode=arch=compute_121,code=sm_121 -o .chk/glue.fatbin "$G"
cuobjdump -sass -arch sm_121 dsv41_glue.fatbin > .chk/glue-kit.sass
cuobjdump -sass -arch sm_121 .chk/glue.fatbin > .chk/glue-tree.sass
n=0
for k in $(grep '__global__' "$G" | grep -oE '\b[A-Za-z_][A-Za-z0-9_]*_kernel\(' | tr -d '('); do
    grep -q "Function : .*$k" .chk/glue-kit.sass || { echo "[kit-fatbins] dsv41_glue.fatbin lacks $k (glue.cu has it): an older glue fatbin" >&2; exit 5; }
    n=$((n + 1))
done
cmp -s .chk/glue-kit.sass .chk/glue-tree.sass \
    || { echo "[kit-fatbins] dsv41_glue.fatbin's sm_121 SASS differs from this tree's glue.cu (diff .chk/glue-*.sass)" >&2; exit 5; }
echo "dsv41_glue.fatbin: $n kernels of glue.cu, sm_121 SASS == a fresh compile of the tree's glue.cu" > .chk/glue
# dsv41_vision.fatbin: SASS equal to this tree's vision.cu (cuda.zig's flags, fmad on as torch's)
V=/tree/zig/kernels/cuda/deepseek_v41_vision/vision.cu
nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 \
    -gencode=arch=compute_121,code=sm_121 -o .chk/vision.fatbin "$V"
cuobjdump -sass -arch sm_121 dsv41_vision.fatbin > .chk/vision-kit.sass
cuobjdump -sass -arch sm_121 .chk/vision.fatbin > .chk/vision-tree.sass
cmp -s .chk/vision-kit.sass .chk/vision-tree.sass \
    || { echo "[kit-fatbins] dsv41_vision.fatbin's sm_121 SASS differs from this tree's vision.cu (diff .chk/vision-*.sass)" >&2; exit 5; }
echo "dsv41_vision.fatbin: sm_121 SASS == a fresh compile of the tree's vision.cu" > .chk/vision
EOF
{
    echo "# tensorfold-zig-dsv41 kit fatbins (tools/zig/dsv41_m2c2/kit-fatbins.sh)"
    echo "tree $commit"
    echo "from FATBINS=$FB"
    echo "nvcc $(cat "$OUT/.chk/nvcc") ($IMAGE)"
    echo "tp_mailbox.fatbin: $(cat "$OUT/.chk/tp" | tr '\n' ';' | sed 's/;$//')"
    cat "$OUT/.chk/glue"
    cat "$OUT/.chk/vision"
    [[ -s "$OUT/.chk/fill" ]] && sed 's/^/built here, not in FATBINS: /' "$OUT/.chk/fill"
    (cd "$OUT" && sha256sum -- *.fatbin)
} > "$OUT/FATBINS.txt"
rm -rf "$OUT/.chk"
cat "$OUT/FATBINS.txt" | sed -n '1,7p'
echo "[kit-fatbins] $OUT: $(ls "$OUT"/*.fatbin | wc -l) fatbins"
