#!/usr/bin/env bash
set -euo pipefail

# Install only the Python environment and, optionally, build the external
# OneLOopBridge bindings.  The bridge checkout is never vendored into the repo.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
CLONE_ONELOOPBRIDGE=0
CACHE_TARBALL="${FSD_CACHE_TARBALL:-}"
CACHE_URL="${FSD_CACHE_URL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clone-oneloopbridge)
      CLONE_ONELOOPBRIDGE=1
      shift
      ;;
    --cache-tar)
      if [[ $# -lt 2 ]]; then
        echo "--cache-tar requires a path" >&2
        exit 2
      fi
      CACHE_TARBALL="$2"
      shift 2
      ;;
    --cache-url)
      if [[ $# -lt 2 ]]; then
        echo "--cache-url requires a URL" >&2
        exit 2
      fi
      CACHE_URL="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: ./install.sh [--clone-oneloopbridge] [--cache-tar FSD_cache.tar.gz] [--cache-url URL]" >&2
      exit 2
      ;;
  esac
done

install_cache_archive() {
  local archive="$1"
  if [[ ! -f "${archive}" ]]; then
    echo "FSD cache archive does not exist: ${archive}" >&2
    exit 1
  fi
  local tmpdir
  tmpdir="$(mktemp -d)"
  tar -xzf "${archive}" -C "${tmpdir}"
  mkdir -p "${ROOT_DIR}/cache"
  if [[ -d "${tmpdir}/cache" ]]; then
    cp -R "${tmpdir}/cache/." "${ROOT_DIR}/cache/"
  elif [[ -d "${tmpdir}/subtraction_formulae" ]]; then
    mkdir -p "${ROOT_DIR}/cache/subtraction_formulae"
    cp -R "${tmpdir}/subtraction_formulae/." "${ROOT_DIR}/cache/subtraction_formulae/"
  else
    echo "FSD cache archive must contain cache/ or subtraction_formulae/" >&2
    exit 1
  fi
  rm -rf "${tmpdir}"
  echo "FSD formula cache installed under ${ROOT_DIR}/cache"
}

if [[ -n "${CACHE_URL}" && -n "${CACHE_TARBALL}" ]]; then
  echo "Use only one of --cache-url/FSD_CACHE_URL or --cache-tar/FSD_CACHE_TARBALL" >&2
  exit 2
fi

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
# pySecDec source builds compile bundled GiNaC documentation on some systems.
# A conservative C locale avoids macOS failures when LC_ALL=C.UTF-8 is not
# supported by the local Perl/makeinfo toolchain.
LC_ALL=C LANG=C "${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"

if [[ -n "${CACHE_URL}" ]]; then
  mkdir -p "${ROOT_DIR}/.deps"
  CACHE_TARBALL="${ROOT_DIR}/.deps/FSD_cache.tar.gz"
  if command -v curl >/dev/null 2>&1; then
    curl -L "${CACHE_URL}" -o "${CACHE_TARBALL}"
  else
    "${VENV_DIR}/bin/python" - <<PY
from urllib.request import urlretrieve
urlretrieve(${CACHE_URL@Q}, ${CACHE_TARBALL@Q})
PY
  fi
fi

if [[ -n "${CACHE_TARBALL}" ]]; then
  install_cache_archive "${CACHE_TARBALL}"
fi

# Prefer a user-supplied checkout.  The clone option is only a convenience and
# puts the external repository under ignored .deps/.
if [[ -n "${ONELOOPBRIDGE_SRC:-}" ]]; then
  BRIDGE_SRC="${ONELOOPBRIDGE_SRC}"
elif [[ "${CLONE_ONELOOPBRIDGE}" == "1" ]]; then
  mkdir -p "${ROOT_DIR}/.deps"
  BRIDGE_SRC="${ROOT_DIR}/.deps/OneLOopBridge"
  if [[ ! -d "${BRIDGE_SRC}/.git" ]]; then
    git clone https://github.com/SecretGmG/OneLOopBridge.git "${BRIDGE_SRC}"
  fi
else
  BRIDGE_SRC=""
fi

if [[ -n "${BRIDGE_SRC}" ]]; then
  if [[ ! -d "${BRIDGE_SRC}" ]]; then
    echo "OneLOopBridge source path does not exist: ${BRIDGE_SRC}" >&2
    exit 1
  fi
  (
    cd "${BRIDGE_SRC}"
    # maturin develop installs the Python extension directly into this venv.
    "${VENV_DIR}/bin/python" -m pip install 'maturin[patchelf]'
    VIRTUAL_ENV="${VENV_DIR}" PATH="${VENV_DIR}/bin:${PATH}" \
      "${VENV_DIR}/bin/maturin" develop --release --features python
  )
fi

# The CLI has no no-benchmark mode, so fail setup if the bridge cannot import.
if ! "${VENV_DIR}/bin/python" - <<'PY'
import oneloop_bridge
print("OneLOopBridge import OK")
PY
then
  cat >&2 <<'EOF'
OneLOopBridge is required but could not be imported.

Use one of:
  ONELOOPBRIDGE_SRC=/path/to/OneLOopBridge ./install.sh
  ./install.sh --clone-oneloopbridge

The cloned checkout is placed under ignored .deps/ and is not vendored.
EOF
  exit 1
fi

echo "FSD environment ready: ${VENV_DIR}"
