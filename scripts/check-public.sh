#!/usr/bin/env bash
# Scan this repo for private infrastructure details and credentials before publishing.
# Credentials (API keys, tokens, private keys, credential assignments) are scripts/check_secrets.py's patterns, run
# from here over the same files; its value-masked findings fail this check too.
#
#   scripts/check-public.sh           precise patterns; exit 1 on any finding
#   scripts/check-public.sh --broad   also list every secret / password / api-key / bearer / authorization mention
#                                     for a manual read (informational, does not fail)
#   scripts/check-public.sh --history the same patterns over every blob of every commit (git grep), plus every
#                                     commit's author / committer address (only ALLOWED_AUTHOR passes)
#
# Scans every tracked and untracked (not ignored) file, the engine patches included, except the vendor/TensorFold
# submodule (upstream TensorFold, unmodified) and this script.
#
# Your own private names (hostnames, user names, tailnet name, home paths, internal repos) belong in
# .check-public.local (git-ignored, one extended regex per line, # comments allowed) or in CHECK_PUBLIC_EXTRA='a|b'.
# They are matched case-insensitively. Lines of the form  regex<TAB>path-glob  skip that path for that regex
# (for a known false positive such as a benchmark question).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SELF="scripts/check-public.sh"
LOCAL_LIST=".check-public.local"
ALLOWED_AUTHOR="${ALLOWED_AUTHOR:-contact@buildthings.co}"

# The GLM engine's image-URL hardening (in patches/0001) refuses private / link-local / CGNAT / metadata addresses;
# its docstring lists the cloud metadata address 100.100.100.200, not ours.
SSRF_FIXTURES="patches/0001-spark-stack-060.patch"

# Recorded Flash Next kernel role keys are identifiers, not addresses; keep the path and tokens exact.
FLASHNEXT_ROLE_EMAILS='^engine/zig/zig/(src/families/flashnext/(replay|roles_gen)|tests/flashnext_bench)\.zig:[0-9]+:'
FLASHNEXT_ROLE_EMAILS+='(lane_qmm_bytes_grouped@(gdn\.(in|out)|att\.(proj|o)|ple\.kv|mtp\.(att(\.proj)?|draft|fc[eh]))'
FLASHNEXT_ROLE_EMAILS+='|q4_attn_prep@mtp\.att|q4_rms_rows@mtp\.(enorm|hnorm)|q4_router_float@mtp\.moe'
FLASHNEXT_ROLE_EMAILS+='|qa_expert_down_y@(mtp\.)?moe\.down|qa_expert_gateup@(mtp\.)?moe\.gate|qa_hc_(down|up)@mtp\.(ahc|mhc|mix))$'
# Homebrew's version substitution marker inside the fixed archive name is not an address.
TEMPLATE_EMAILS='^engine/zig/packaging/homebrew/tensorfold\.rb\.in:[0-9]+:VERSION@-macos-arm64\.tar\.gz$'

# name|regex (grep -E, case-insensitive)|path globs excluded for this pattern (space-separated, may be empty)
PATTERNS=(
    "private IPv4 192.168/16|\\b192\\.168\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|$SSRF_FIXTURES"
    "private IPv4 10/8 (the launcher tests use a fake 10.0.0.1)|\\b10\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|tests/test_serve_ops.py $SSRF_FIXTURES"
    "private IPv4 172.16/12|\\b172\\.(1[6-9]|2[0-9]|3[01])\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|$SSRF_FIXTURES"
    "CGNAT / Tailscale IPv4|\\b100\\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\\.[0-9]{1,3}\\.[0-9]{1,3}\\b|$SSRF_FIXTURES"
    'Tailscale IPv6|fd7a:|'
    'Tailscale names|\.ts\.net\b|tailnet|tailscale|'
    'gmail address|[a-z0-9._%+-]+@gmail\.com|'
    'MAC address|\b([0-9a-f]{2}:){5}[0-9a-f]{2}\b|'
    'serial number|serial[ _-]?(number|no\.?)[ :=]|'
    'SSH public key|ssh-(ed25519|rsa|dss) AAAA|'
    'home directories|/home/[a-z0-9_-]+/|'
    'root home paths|(^|[^a-z0-9_.-])/root/|'
    'machine hostnames (dgx-01 style)|\bdgx-?[0-9]{1,2}\b|'
    'per-user cache / work paths|\.cache/tf[0-9]+-port|~/ai/|\.prod-stack|test-window-lease|'
    'host OS tuning|polkit|sparkdash|irqbalance|smp_affinity|unattended-upgrade|auto-?reboot|\bwi-?fi\b|networkmanager|results/*greedy-*.json'
    'agent tooling / chatter|\bt3[-_ ]?(code|gateway|agent)|agent-storage|claude[ -]code|\bcoordinator\b|\b(gpu|perf|dense|exchange|research|window) agent\b|'
    'private chat-history corpus output|local opencode database|opencode\.db|scripts/campaign/draftvocab.py'
    'GPU / board identifiers|\bGPU-[0-9a-f]{8}-[0-9a-f]{4}|\bMIG-[0-9a-f]{8}|'
)
if [[ -f "$LOCAL_LIST" ]]; then
    while IFS= read -r line; do
        [[ -z "${line//[[:space:]]/}" || "$line" == \#* ]] && continue
        re="${line%%$'\t'*}"; skip=""
        [[ "$line" == *$'\t'* ]] && skip="${line#*$'\t'}"
        PATTERNS+=("local list ($LOCAL_LIST)|$re|$skip")
    done < "$LOCAL_LIST"
fi
[[ -n "${CHECK_PUBLIC_EXTRA:-}" ]] && PATTERNS+=("CHECK_PUBLIC_EXTRA|$CHECK_PUBLIC_EXTRA|")

mapfile -t FILES < <(git ls-files -co --exclude-standard 2>/dev/null |
                     grep -vE '^vendor/TensorFold(/|$)' | grep -vxF "$SELF" | grep -vxF "$LOCAL_LIST" |
                     while IFS= read -r f; do [[ -f "$f" ]] && printf '%s\n' "$f"; done)
if [[ ${#FILES[@]} -eq 0 ]]; then echo "no files to scan (not a git checkout?)" >&2; exit 2; fi

# /home/<name>/ is allowed for these generic placeholders
HOME_OK='/home/(user|you|<you>|<user>|ubuntu)/'

fail=0
for entry in "${PATTERNS[@]}"; do
    name="${entry%%|*}"; rest="${entry#*|}"
    skip="${rest##*|}"; re="${rest%|*}"
    hits=$(for f in "${FILES[@]}"; do
               # shellcheck disable=SC2053
               skipped=0
               read -ra globs <<< "$skip"
               for g in "${globs[@]}"; do [[ "$f" == $g ]] && { skipped=1; break; }; done
               [[ $skipped -eq 1 ]] && continue
               grep -HnIiE -- "$re" "$f" 2>/dev/null
           done)
    [[ "$name" == "home directories" && -n "$hits" ]] && hits=$(grep -viE "$HOME_OK" <<< "$hits")
    if [[ -n "$hits" ]]; then
        fail=1
        echo "== $name"
        cut -c1-220 <<< "$hits"
    fi
done

# credentials: one pattern set, shared with scripts/redact.py; allowlist in scripts/check-secrets-allow.txt
secrets=$(python3 scripts/check_secrets.py "${FILES[@]}" 2>&1)
if [[ $? -ne 0 ]]; then fail=1; echo "== credentials (scripts/check_secrets.py)"; echo "$secrets"; fi

# e-mail addresses other than placeholder / example domains (model outputs in results/ invent a few)
emails=$(for f in "${FILES[@]}"; do grep -HnoIiE '[a-z0-9][a-z0-9._%+-]*@[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}' "$f" 2>/dev/null; done |
         grep -viE '@(([a-z0-9-]+\.)*example(\.(com|org|net))?|company\.com|novatech\.io)$|:noreply@anthropic\.com$' |
         grep -vF ":$ALLOWED_AUTHOR" | grep -vE "$FLASHNEXT_ROLE_EMAILS" | grep -vE "$TEMPLATE_EMAILS" | cut -c1-200)
if [[ -n "$emails" ]]; then fail=1; echo "== e-mail addresses (not on the example-domain allowlist)"; echo "$emails"; fi

# files that should never be published
# run logs are published only under results/campaign/ (sanitized engine / client output of the development windows)
bad=$(printf '%s\n' "${FILES[@]}" | grep -E '\.(log|out|pem|key)$|(^|/)\.env$|^config/[^/]+\.env$|id_(rsa|ed25519)' |
      grep -vE '^results/campaign/.*\.(log|out)$' || true)
if [[ -n "$bad" ]]; then fail=1; echo "== files that should not be published"; echo "$bad"; fi

if [[ "${1:-}" == --history ]]; then
    commits=$(git rev-list --all 2>/dev/null)
    echo "== history: $(wc -w <<< "$commits") commit(s)"
    bad_auth=$(git log --all --format='%h %ae %ce' | awk -v ok="$ALLOWED_AUTHOR" '$2 != ok || $3 != ok')
    if [[ -n "$bad_auth" ]]; then fail=1; echo "== commits not authored / committed as $ALLOWED_AUTHOR"; echo "$bad_auth"; fi
    hsecrets=$(python3 scripts/check_secrets.py --history 2>&1)
    if [[ $? -ne 0 ]]; then fail=1; echo "== history: credentials (scripts/check_secrets.py)"; echo "$hsecrets"; fi
    msgs=$(git log --all --format='%h %B')
    for entry in "${PATTERNS[@]}"; do
        name="${entry%%|*}"; rest="${entry#*|}"; skip="${rest##*|}"; re="${rest%|*}"
        pathspec=(-- . ':!scripts/check-public.sh' ':!vendor/TensorFold')
        for g in $skip; do pathspec+=(":!$g"); done
        hits=""
        for c in $commits; do
            h=$(git grep -I -n -i -E -e "$re" "$c" "${pathspec[@]}" 2>/dev/null)
            [[ "$name" == "home directories" && -n "$h" ]] && h=$(grep -viE "$HOME_OK" <<< "$h")
            [[ -n "$h" ]] && hits+="$h"$'\n'
        done
        m=$(grep -niE -e "$re" <<< "$msgs" || true)
        [[ -n "$m" ]] && hits+="commit message: $m"$'\n'
        if [[ -n "${hits//[[:space:]]/}" ]]; then fail=1; echo "== history: $name"; cut -c1-220 <<< "$hits" | sed '/^$/d'; fi
    done
fi

if [[ "${1:-}" == --broad ]]; then
    echo "== broad keyword list (manual review; informational)"
    for f in "${FILES[@]}"; do
        grep -HnIiE 'secret|password|api[_-]?key|bearer|authorization' "$f" 2>/dev/null
    done | cut -c1-200
fi

if [[ $fail -ne 0 ]]; then
    echo; echo "check-public: FINDINGS above (${#FILES[@]} files scanned)"; exit 1
fi
echo "check-public: clean (${#FILES[@]} files scanned$([[ -f "$LOCAL_LIST" ]] && echo ", plus $LOCAL_LIST"))"
