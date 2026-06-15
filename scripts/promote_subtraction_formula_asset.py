#!/usr/bin/env python3
"""Promote generated universal subtraction formula JSON files to curated assets.

Generated cache files in ``assets/subtraction_formulae`` are intentionally
ignored by git.  Once a formula signature has been validated and benchmarked,
copy it into ``assets/subtraction_formulae/curated`` with this script.  Curated
assets are source files: the FSD loader prefers them over generated cache files,
and regular-Taylor curated assets bypass the cold-build guard by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


VALID_PREFIXES = ("endpoint_projector_", "regular_taylor_")


def _cache_root(path: str | None) -> Path:
    """Return the formula cache root used by FSD."""
    if path is not None:
        return Path(path).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "assets" / "subtraction_formulae"


def _resolve_formula_path(cache_root: Path, value: str) -> Path:
    """Resolve either a filename under the cache root or an explicit path."""
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    candidate = cache_root / value
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"formula cache file not found: {value}")


def _validate_formula_file(path: Path) -> dict[str, Any]:
    """Load a formula cache file and reject files that are not FSD formula JSON."""
    if not any(path.name.startswith(prefix) for prefix in VALID_PREFIXES):
        raise ValueError(
            f"{path.name}: expected filename starting with one of {VALID_PREFIXES}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data.get("signature_payload")
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}: missing signature_payload")
    kind = payload.get("kind")
    if path.name.startswith("endpoint_projector_") and kind != "endpoint-projector":
        raise ValueError(f"{path.name}: endpoint filename has kind {kind!r}")
    if path.name.startswith("regular_taylor_") and kind != "regular-taylor":
        raise ValueError(f"{path.name}: regular filename has kind {kind!r}")
    if "input_names" not in data:
        raise ValueError(f"{path.name}: missing input_names")
    if kind == "endpoint-projector" and "output_expressions" not in data:
        raise ValueError(f"{path.name}: endpoint projector has no output_expressions")
    if kind == "regular-taylor" and not (
        "output_expressions" in data or "scalar_expression" in data
    ):
        raise ValueError(
            f"{path.name}: regular-Taylor formula has neither output_expressions "
            "nor scalar_expression"
        )
    return data


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` atomically within the curated directory."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name, suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    try:
        shutil.copy2(src, tmp_name)
        os.replace(tmp_name, dst)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "formula",
        nargs="+",
        help="Generated formula JSON filename or path to promote.",
    )
    parser.add_argument("--cache-dir", default=None, help="Override formula cache root.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing curated asset with the same filename.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print what would be copied without writing.",
    )
    args = parser.parse_args()

    cache_root = _cache_root(args.cache_dir)
    curated_root = cache_root / "curated"
    for value in args.formula:
        src = _resolve_formula_path(cache_root, value)
        data = _validate_formula_file(src)
        dst = curated_root / src.name
        if dst.exists() and not args.replace:
            raise FileExistsError(
                f"curated asset already exists: {dst}; pass --replace to overwrite"
            )
        size_mib = src.stat().st_size / 1024 / 1024
        kind = data.get("signature_payload", {}).get("kind", "unknown")
        if args.dry_run:
            print(f"would promote {src.name} ({kind}, {size_mib:.3g} MiB) -> {dst}")
            continue
        _atomic_copy(src, dst)
        print(f"promoted {src.name} ({kind}, {size_mib:.3g} MiB) -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
