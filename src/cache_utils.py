"""Shared paths for FSD-generated formula caches.

The cache is intentionally outside the tracked ``assets`` tree.  It is a
runtime/distribution artifact: installations may populate it from an external
``FSD_cache.tar.gz`` archive, and cold generation fills in missing entries.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback.
    fcntl = None


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FORMULA_CACHE_ENV = "FSD_SUBTRACTION_FORMULA_CACHE_DIR"


class FormulaCacheLockUnavailable(RuntimeError):
    """Raised when a non-blocking formula-cache lock is already held."""


def default_formula_cache_dir() -> Path:
    """Return the top-level generated/distributed formula-cache directory."""
    return PACKAGE_ROOT / "cache" / "subtraction_formulae"


def legacy_formula_cache_dir() -> Path:
    """Return the older in-tree cache directory kept as a read fallback."""
    return PACKAGE_ROOT / "assets" / "subtraction_formulae"


def formula_cache_dir() -> Path:
    """Return the primary cache directory used for new writes."""
    configured = os.environ.get(FORMULA_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return default_formula_cache_dir()


def formula_cache_read_roots() -> list[Path]:
    """Return cache roots to search, newest first.

    The environment override is treated as the primary write location, but the
    default top-level cache and legacy assets cache remain readable so older
    local warm-up runs can seed new bundles.
    """
    roots: list[Path] = []
    for root in (formula_cache_dir(), default_formula_cache_dir(), legacy_formula_cache_dir()):
        expanded = root.expanduser()
        if expanded not in roots:
            roots.append(expanded)
    return roots


def mirror_cache_entry_to_primary(
    metadata_path: Path,
    data: dict[str, Any],
    *,
    sidecar_fields: tuple[str, ...] = (),
) -> None:
    """Copy a cache hit from any readable root into the primary cache root.

    Older development runs wrote warm caches under ``assets/subtraction_formulae``.
    The distribution layout now uses the top-level ``cache/subtraction_formulae``
    directory.  Cache loading is therefore self-migrating: when a valid entry is
    found in a fallback root, copy its metadata JSON and any referenced evaluator
    or expression sidecars to the primary root.  The operation is best-effort
    because cache mirroring is an optimization, not a correctness condition.
    """
    try:
        source = metadata_path.expanduser().resolve()
        destination_root = formula_cache_dir().expanduser()
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / metadata_path.name
        if source != destination.resolve():
            _copy_if_missing(source, destination)

        for field in sidecar_fields:
            raw_names = data.get(field)
            if raw_names is None:
                continue
            names = raw_names if isinstance(raw_names, list) else [raw_names]
            for raw_name in names:
                if not raw_name:
                    continue
                name = str(raw_name)
                sidecar = metadata_path.parent / name
                if not sidecar.is_file() and sidecar.suffix != ".gz":
                    compressed = sidecar.with_name(sidecar.name + ".gz")
                    if compressed.is_file():
                        sidecar = compressed
                if not sidecar.is_file():
                    continue
                target = destination_root / sidecar.name
                if sidecar.expanduser().resolve() != target.resolve():
                    _copy_if_missing(sidecar, target)
    except Exception:
        return


@contextmanager
def formula_cache_lock(metadata_path: Path, *, blocking: bool = True):
    """Serialize cold generation for one formula-cache metadata file."""

    lock_root = metadata_path.expanduser().parent / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{metadata_path.name}.lock"
    handle = lock_path.open("a+")
    locked = False
    try:
        if fcntl is not None:
            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
            try:
                fcntl.flock(handle.fileno(), flags)
            except BlockingIOError as exc:
                raise FormulaCacheLockUnavailable(str(lock_path)) from exc
            locked = True
        yield
    finally:
        if fcntl is not None and locked:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _copy_if_missing(source: Path, destination: Path) -> None:
    """Copy one cache artifact unless the destination already exists."""
    if destination.exists():
        return
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, tmp)
    tmp.replace(destination)
