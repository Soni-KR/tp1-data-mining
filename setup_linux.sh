#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export UV_UNMANAGED_INSTALL="$PWD/.runtime/bin"
export UV_CACHE_DIR="$PWD/.runtime/cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python"
mkdir -p .runtime
if [ ! -x .runtime/bin/uv ]; then
    curl -LsSf https://astral.sh/uv/install.sh -o .runtime/install-uv.sh
    sh .runtime/install-uv.sh
fi
.runtime/bin/uv venv --clear --python 3.12 .linux-venv
.runtime/bin/uv pip install --python .linux-venv/bin/python -r requirements.txt
