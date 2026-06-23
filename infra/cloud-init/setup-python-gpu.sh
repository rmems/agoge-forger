#!/bin/bash
# Stub script to setup python and uv on a new GPU node
set -euo pipefail

UV_VERSION="0.6.14"
python3 -m pip install --no-cache-dir "uv==${UV_VERSION}"
