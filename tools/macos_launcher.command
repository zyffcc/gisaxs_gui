#!/bin/bash
# The portable runtime lives outside the application, so read-only / moved app folders work.
set -euo pipefail
package_dir="$(cd "$(dirname "$0")" && pwd)"
archive="$package_dir/runtime.tar.gz"
if [ "$(uname -m)" != "@ARCH@" ]; then
    echo "This package needs @ARCH@. Download the package for this Mac's processor."
    exit 1
fi
runtime_root="$HOME/Library/Application Support/GIMaP/runtimes"
runtime_dir="$runtime_root/@RUNTIME_SHA@"
mkdir -p "$runtime_root"
if [ ! -f "$runtime_dir/.ready" ]; then
    echo "Preparing GIMaP for the first launch. Please keep this window open."
    if ! mkdir "$runtime_dir.lock" 2>/dev/null; then
        echo "Another launch is preparing GIMaP. Wait for it to finish, then try again."
        exit 1
    fi
    stage_dir="$(mktemp -d "$runtime_root/.prepare.XXXXXX")"
    trap 'rm -rf "$stage_dir"; rmdir "$runtime_dir.lock" 2>/dev/null || true' EXIT
    actual_sha="$(shasum -a 256 "$archive" | cut -d ' ' -f 1)"
    if [ "$actual_sha" != "@RUNTIME_SHA@" ]; then
        echo "The runtime download is incomplete or damaged. Download the package again."
        exit 1
    fi
    tar -xzf "$archive" -C "$stage_dir"
    # Relocate only after the runtime has reached its permanent, writable location.
    if [ -e "$runtime_dir" ]; then rm -rf "$runtime_dir"; fi
    mv "$stage_dir" "$runtime_dir"
    "$runtime_dir/bin/python" "$runtime_dir/bin/conda-unpack"
    touch "$runtime_dir/.ready"
    rmdir "$runtime_dir.lock"
    trap - EXIT
fi
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
cd "$HOME"
echo "Starting GIMaP…"
exec "$runtime_dir/bin/python" "$package_dir/app/main.py" "$@"
