#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# This workstation currently needs Mesa's GLX implementation for the native window.
export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export MENAGERIE_CACHE_DIR=outputs/menagerie_cache

exec uv run --locked mimicvla sim-view
