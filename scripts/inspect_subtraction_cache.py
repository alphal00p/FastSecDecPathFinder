#!/usr/bin/env python3
"""Summarize generated and curated subtraction formula cache files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def _cache_root(path: str | None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    return Path(__file__).resolve().parents[1] / "assets" / "subtraction_formulae"


def _formula_kind(path: Path) -> str:
    if path.name.startswith("endpoint_projector_"):
        return "endpoint-projector"
    if path.name.startswith("regular_taylor_"):
        return "regular-taylor"
    return "unknown"


def _signature_version(signature: Any) -> Any:
    if isinstance(signature, list) and len(signature) > 1:
        return signature[1]
    return None


def _axis_count(signature: Any) -> Any:
    if isinstance(signature, list) and len(signature) > 2:
        return signature[2]
    return None


def _load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data.get("signature_payload", {})
    signature = payload.get("signature")
    return {
        "path": str(path),
        "name": path.name,
        "kind": payload.get("kind") or _formula_kind(path),
        "schema_version": payload.get("schema_version"),
        "signature_version": _signature_version(signature),
        "axis_count": _axis_count(signature),
        "mode": data.get("mode"),
        "size_bytes": path.stat().st_size,
        "curated": "curated" in path.parts,
    }


def _iter_cache_files(root: Path) -> list[Path]:
    return sorted(
        [
            *root.glob("endpoint_projector_*.json"),
            *root.glob("regular_taylor_*.json"),
            *root.glob("curated/endpoint_projector_*.json"),
            *root.glob("curated/regular_taylor_*.json"),
        ]
    )


def _print_table(rows: list[dict[str, Any]], largest: int) -> None:
    total_size = sum(int(row["size_bytes"]) for row in rows)
    print(f"files: {len(rows)}")
    print(f"size:  {total_size / 1024 / 1024:.2f} MiB")
    print()

    counts: Counter[tuple[Any, ...]] = Counter()
    sizes: defaultdict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        key = (
            row["kind"],
            row["schema_version"],
            row["signature_version"],
            row["axis_count"],
            row["mode"],
            "curated" if row["curated"] else "generated",
        )
        counts[key] += 1
        sizes[key] += int(row["size_bytes"])
    print("by kind/schema/signature/axis/mode/location:")
    for key, count in sorted(counts.items(), key=lambda item: (str(item[0]), item[1])):
        print(f"  {key}: {count} files, {sizes[key] / 1024 / 1024:.2f} MiB")

    print()
    print(f"largest {min(largest, len(rows))} files:")
    for row in sorted(rows, key=lambda item: int(item["size_bytes"]), reverse=True)[:largest]:
        location = "curated" if row["curated"] else "generated"
        print(
            f"  {row['size_bytes'] / 1024 / 1024:8.2f} MiB "
            f"{location:9s} {row['kind']:18s} "
            f"schema={row['schema_version']} sig={row['signature_version']} "
            f"axis={row['axis_count']} mode={row['mode']} {row['name']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--largest", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = _cache_root(args.cache_dir)
    rows: list[dict[str, Any]] = []
    for path in _iter_cache_files(root):
        try:
            rows.append(_load_summary(path))
        except Exception as exc:
            rows.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "kind": _formula_kind(path),
                    "schema_version": None,
                    "signature_version": None,
                    "axis_count": None,
                    "mode": f"error: {exc}",
                    "size_bytes": path.stat().st_size,
                    "curated": "curated" in path.parts,
                }
            )
    if args.json:
        print(json.dumps({"cache_dir": str(root), "files": rows}, indent=2, sort_keys=True))
    else:
        print(f"cache: {root}")
        _print_table(rows, max(int(args.largest), 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
