#!/bin/bash
# Wrapper for ld.lld that adds system library search paths
SYSROOT="/home/aiuser/.rustup/toolchains/stable-x86_64-unknown-linux-gnu"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if --dynamic-linker is already in the arguments; if not, add it.
# ld.lld has no built-in default dynamic linker path (unlike GNU ld), so we
# must supply it explicitly to produce a working PT_INTERP segment.
has_dynamic_linker=false
for arg in "$@"; do
    if [[ "$arg" == "--dynamic-linker" || "$arg" == "-dynamic-linker" ]]; then
        has_dynamic_linker=true
        break
    fi
done

extra_args=()
if ! $has_dynamic_linker; then
    extra_args+=(--dynamic-linker /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2)
fi

exec "${SYSROOT}/lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld/ld.lld" \
    -L"${SCRIPT_DIR}/syslibs" \
    -L/lib/x86_64-linux-gnu \
    -L/usr/lib/x86_64-linux-gnu \
    "${extra_args[@]}" \
    "$@"
