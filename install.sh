#!/usr/bin/env bash
set -euo pipefail

# Install the uv-managed Python environment and, optionally, build the external
# OneLOopBridge bindings.  The bridge checkout is never vendored into the repo.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
CLONE_ONELOOPBRIDGE=0
CACHE_TARBALL="${FSD_CACHE_TARBALL:-}"
CACHE_URL="${FSD_CACHE_URL:-}"
BUILD_SYMBOLICA_COMMUNITY_DEV="${FSD_BUILD_SYMBOLICA_COMMUNITY_DEV:-1}"
SYMBOLICA_COMMUNITY_SRC="${FSD_SYMBOLICA_COMMUNITY_SRC:-}"
SYMBOLICA_DEV_REV="${FSD_SYMBOLICA_DEV_REV:-6262adc56784f19c6bb370b4121cf155f7090624}"
SYMJIT_VERSION="${FSD_SYMJIT_VERSION:-2.19.2}"

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
    --skip-symbolica-community-dev)
      BUILD_SYMBOLICA_COMMUNITY_DEV=0
      shift
      ;;
    --symbolica-community-src)
      if [[ $# -lt 2 ]]; then
        echo "--symbolica-community-src requires a path" >&2
        exit 2
      fi
      SYMBOLICA_COMMUNITY_SRC="$2"
      shift 2
      ;;
    --symbolica-dev-rev)
      if [[ $# -lt 2 ]]; then
        echo "--symbolica-dev-rev requires a git revision" >&2
        exit 2
      fi
      SYMBOLICA_DEV_REV="$2"
      shift 2
      ;;
    --symjit-version)
      if [[ $# -lt 2 ]]; then
        echo "--symjit-version requires a version" >&2
        exit 2
      fi
      SYMJIT_VERSION="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: ./install.sh [--clone-oneloopbridge] [--cache-tar FSD_cache.tar.gz] [--cache-url URL] [--skip-symbolica-community-dev] [--symbolica-community-src PATH] [--symbolica-dev-rev REV] [--symjit-version VERSION]" >&2
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

install_patched_symbolica_community() {
  # Temporary local workaround: Symbolica dev currently pins symjit = "=2.18",
  # while FSD needs the real-valued jit_compile fix released in symjit 2.19.2.
  # Build symbolica-community against a local Symbolica checkout whose only
  # patch is the symjit dependency version.  Remove this once upstream
  # Symbolica itself depends on a fixed symjit release.
  mkdir -p "${ROOT_DIR}/.deps"

  local community_src="${SYMBOLICA_COMMUNITY_SRC}"
  if [[ -z "${community_src}" ]]; then
    community_src="${ROOT_DIR}/.deps/symbolica-community-dev"
    if [[ ! -d "${community_src}/.git" ]]; then
      git clone https://github.com/symbolica-dev/symbolica-community.git "${community_src}"
    fi
  fi
  if [[ ! -f "${community_src}/Cargo.toml" ]]; then
    echo "symbolica-community source path does not contain Cargo.toml: ${community_src}" >&2
    exit 1
  fi

  local patched_symbolica="${ROOT_DIR}/.deps/symbolica-dev-symjit-${SYMJIT_VERSION}"
  if [[ ! -d "${patched_symbolica}/.git" ]]; then
    git clone https://github.com/symbolica-dev/symbolica.git "${patched_symbolica}"
  fi
  git -C "${patched_symbolica}" fetch origin dev
  git -C "${patched_symbolica}" checkout "${SYMBOLICA_DEV_REV}"

  PATCHED_SYMBOLICA="${patched_symbolica}" \
  COMMUNITY_SRC="${community_src}" \
  SYMJIT_VERSION="${SYMJIT_VERSION}" \
  "${VENV_DIR}/bin/python" - <<'PY'
import os
import re
from pathlib import Path

patched_symbolica = Path(os.environ["PATCHED_SYMBOLICA"]).resolve()
community_src = Path(os.environ["COMMUNITY_SRC"]).resolve()
symjit_version = os.environ["SYMJIT_VERSION"]

symbolica_toml = patched_symbolica / "Cargo.toml"
text = symbolica_toml.read_text()
text, count = re.subn(
    r'symjit\s*=\s*"=[^"]+"',
    f'symjit = "={symjit_version}"',
    text,
    count=1,
)
if count != 1:
    raise SystemExit(f"could not patch symjit dependency in {symbolica_toml}")
symbolica_toml.write_text(text)

community_toml = community_src / "Cargo.toml"
text = community_toml.read_text()
symbolica_rel = os.path.relpath(patched_symbolica, community_src)
graphica_rel = os.path.relpath(patched_symbolica / "lib" / "graphica", community_src)
numerica_rel = os.path.relpath(patched_symbolica / "lib" / "numerica", community_src)

replacements = {
    "symbolica": f'symbolica = {{ path = "{symbolica_rel}" }}',
    "graphica": f'graphica = {{ path = "{graphica_rel}", package = "graphica" }}',
    "numerica": f'numerica = {{ path = "{numerica_rel}", package = "numerica" }}',
}
lines = text.splitlines()
out = []
in_patch = False
seen = set()
inserted = False

def emit_missing():
    global inserted
    if inserted:
        return
    for crate, replacement in replacements.items():
        if crate not in seen:
            out.append(replacement)
    inserted = True

for line in lines:
    stripped = line.strip()
    if stripped == "[patch.crates-io]":
        in_patch = True
        out.append(line)
        continue
    if in_patch and stripped.startswith("[") and stripped.endswith("]"):
        emit_missing()
        in_patch = False
    if in_patch:
        matched = False
        for crate, replacement in replacements.items():
            if re.match(rf"^{crate}\s*=", stripped):
                out.append(replacement)
                seen.add(crate)
                matched = True
                break
        if matched:
            continue
    out.append(line)

if in_patch:
    emit_missing()
elif "[patch.crates-io]" not in text:
    out.extend(["", "[patch.crates-io]"])
    emit_missing()

community_toml.write_text("\n".join(out) + "\n")
PY

  (
    cd "${community_src}"
    cargo update -p symbolica -p graphica -p numerica -p symjit
    if ! grep -A2 'name = "symjit"' Cargo.lock | grep -q "version = \"${SYMJIT_VERSION}\""; then
      echo "failed to resolve symjit ${SYMJIT_VERSION} in ${community_src}/Cargo.lock" >&2
      exit 1
    fi
    "${VENV_DIR}/bin/maturin" build --release
    local wheel
    wheel="$(ls -t target/wheels/symbolica-*.whl | head -n 1)"
    "${VENV_DIR}/bin/python" -m pip install --force-reinstall "${wheel}"
  )

  "${VENV_DIR}/bin/python" - <<PY
import symbolica
from importlib.metadata import version
print("Patched symbolica-community installed")
print("  symbolica.__version__ =", getattr(symbolica, "__version__", "n/a"))
print("  distribution version =", version("symbolica"))
print("  symjit override =", ${SYMJIT_VERSION@Q})
PY
}

if [[ -n "${CACHE_URL}" && -n "${CACHE_TARBALL}" ]]; then
  echo "Use only one of --cache-url/FSD_CACHE_URL or --cache-tar/FSD_CACHE_TARBALL" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first, then rerun ./install.sh." >&2
  exit 1
fi

# pySecDec source builds compile bundled GiNaC documentation on some systems.
# A conservative C locale avoids macOS failures when LC_ALL=C.UTF-8 is not
# supported by the local Perl/makeinfo toolchain.
(
  cd "${ROOT_DIR}"
  UV_PROJECT_ENVIRONMENT="${VENV_DIR}" LC_ALL=C LANG=C uv sync --group dev
)

if [[ "${BUILD_SYMBOLICA_COMMUNITY_DEV}" == "1" ]]; then
  install_patched_symbolica_community
else
  echo "Skipping patched symbolica-community build; real-valued jit_compile may still be affected by the upstream symjit pin."
fi

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
