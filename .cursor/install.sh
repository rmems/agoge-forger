#!/usr/bin/env bash
# Cursor Cloud Agent install script for agoge-forger (`install` in .cursor/environment.json).
#
# Cursor runs this from the repository root during every Build, on its default
# Ubuntu base image (CPU only: cloud agents have no GPU), then snapshots the disk.
# It must be idempotent. Shell exports don't survive into agent runs, so the tools
# it installs are exposed through /etc/profile.d and /usr/local/bin.
# See https://cursor.com/docs/cloud-agent/setup
#
# Installs only what this repo's CI and manifests need:
#   - apt: curl, ca-certificates
#   - uv 0.12.7 (GitHub release digest checked) + Python 3.12
#   - uv venv + uv pip install -e ".[dev]" (as in every CI job)
#   - tool directories exposed to later shells (/etc/profile.d + /usr/local/bin links)
#
# It ends with a dependency fetch/prebuild, not a test run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Install apt packages that are not already present.
apt_install() {
  local missing=() pkg
  for pkg in "$@"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$pkg")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    $SUDO apt-get -o Acquire::Retries=5 update -qq
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 install -y --no-install-recommends "${missing[@]}"
  fi
}

# --- System packages (curl + CA certs for the installers below) ---
apt_install curl ca-certificates

# --- Python (uv) ---
# Same uv release as .github/workflows/qodana_code_quality.yml. Digests are the
# GitHub release checksums for that tag, checked before the binaries are installed.
export PATH="$HOME/.local/bin:$PATH"
UV_VERSION="0.12.7"
install_pinned_uv() {
  local target archive digest tmp
  case "$(uname -m)" in
    x86_64) target="x86_64-unknown-linux-gnu" ;;
    aarch64) target="aarch64-unknown-linux-gnu" ;;
    *)
      echo "unsupported architecture for uv: $(uname -m)" >&2
      exit 1
      ;;
  esac
  archive="uv-${target}.tar.gz"
  case "$target" in
    x86_64-unknown-linux-gnu)
      digest="788f18abea7c5f55d6216e4f5613fd89d4d59b631efeec117b2b07fe72f1da21"
      ;;
    aarch64-unknown-linux-gnu)
      digest="66393193038dd7eb108abd7a218d9cec04ac70ab98242b0720fa94de19223b7c"
      ;;
  esac
  tmp="$(mktemp -d)"
  curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${archive}" -o "${tmp}/${archive}"
  echo "${digest}  ${tmp}/${archive}" | sha256sum -c -
  tar -xzf "${tmp}/${archive}" -C "$tmp"
  mkdir -p "$HOME/.local/bin"
  install -m 0755 "${tmp}/uv-${target}/uv" "$HOME/.local/bin/uv"
  install -m 0755 "${tmp}/uv-${target}/uvx" "$HOME/.local/bin/uvx"
  rm -rf "$tmp"
}
if ! command -v uv >/dev/null 2>&1 || [[ "$(uv --version 2>/dev/null || true)" != "uv ${UV_VERSION} "* ]]; then
  install_pinned_uv
fi
# Drop a hashed path to any older uv found before the pinned binary was installed.
hash -r
# python_smoke_test.yml uses 3.12 (other jobs use the ubuntu runner default, also 3.12).
uv python install 3.12
# CI forces CPU; keep torch/bitsandbytes off any GPU probing during install.
export CUDA_VISIBLE_DEVICES=-1
# A reused snapshot can keep a .venv from another Python. Recreate it unless it is 3.12.
if ! { [ -x .venv/bin/python ] && .venv/bin/python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; }; then
  uv venv --python 3.12 --clear .venv
fi
# Match CI's editable install. `uv sync --frozen` aborts while uv.lock disagrees
# with pyproject.toml (ruff 0.16.7 vs ==0.16.8); refreshing the lock is out of scope.
uv pip install --python .venv/bin/python -e ".[dev]"

# --- Expose the tools to later shells ---
# The PATH exports above last only for this script; Cursor starts the agent's shells
# separately. Login shells get these directories from /etc/profile.d, and every other
# shell finds the entry points through symlinks in /usr/local/bin (on the default PATH).
# In login shells the venv's bin comes first, so python3 is the venv's Python.
# The links skip python and pip so the system python3 stays the default elsewhere.
repo_root="$(pwd)"
tool_dirs=("$repo_root/.venv/bin" "$HOME/.local/bin")
quoted_dirs=()
for dir in "${tool_dirs[@]}"; do
  # Quote each directory so a special character stays inside one PATH entry.
  quoted_dirs+=("$(printf '%q' "$dir")")
done
# shellcheck disable=SC2016 # $PATH must expand when the profile is sourced, not now.
printf 'export PATH=%s:$PATH\n' "$(IFS=:; echo "${quoted_dirs[*]}")" |
  $SUDO tee /etc/profile.d/cursor-env-agoge-forger.sh >/dev/null
for dir in "${tool_dirs[@]}"; do
  [ -d "$dir" ] || continue
  for tool in "$dir"/*; do
    name="${tool##*/}"
    case "$name" in
      # uv's installer also drops env/env.fish (sourced, not run) in ~/.local/bin.
      python* | pip* | activate* | deactivate | Activate.ps1 | env | env.fish) continue ;;
    esac
    if [ -f "$tool" ] && [ -x "$tool" ]; then
      $SUDO ln -sfn "$tool" "/usr/local/bin/$name"
    fi
  done
done

echo "Cursor install for agoge-forger finished."
