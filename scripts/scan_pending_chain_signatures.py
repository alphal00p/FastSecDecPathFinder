#!/usr/bin/env python3
"""Summarize not-yet-finished cache shards and chain-rule signatures.

This is intentionally read-only and does not import the FastSecDec modules or
Symbolica.  It only inspects shard status/report/log files and, optionally,
brute-force matches logged ``chain_rule_<digest>`` names against the universal
rectangular chain-rule signature hash.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STATUS_PATH = Path("docs/cluster_cache_3l_shards_status.json")
REPORT_DIR = Path("docs/cache_shards/reports/triple-box-direct")
LOG_DIR = Path("docs/cache_shards/logs/triple-box-direct")
CACHE_DIR = Path("cache/subtraction_formulae")


@dataclass(frozen=True)
class SignatureSpec:
    active: int
    rank: int
    maxes: tuple[int, ...]
    output_len: int
    derivative_len: int


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _task_index(task_id: str) -> int | None:
    match = re.search(r":(\d{4})/", task_id)
    return int(match.group(1)) if match else None


def _report_statuses(report_dir: Path) -> dict[int, str]:
    statuses: dict[int, str] = {}
    for path in sorted(report_dir.glob("triple_box_shard_*_of_0099.json")):
        match = re.search(r"_shard_(\d{4})_of_", path.name)
        if not match:
            continue
        index = int(match.group(1))
        data = _load_json(path)
        cases = data.get("cases") or []
        case = cases[0] if cases and isinstance(cases[0], dict) else data
        statuses[index] = str(case.get("status") or data.get("status") or "")
    return statuses


def _schema_version(cache_dir: Path, root: Path) -> int:
    for path in sorted(cache_dir.glob("chain_rule_*.json")):
        data = _load_json(path)
        payload = data.get("signature_payload") or {}
        value = payload.get("schema_version")
        if isinstance(value, int):
            return value
    integrand = root / "integrand.py"
    try:
        text = integrand.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 4
    match = re.search(r"CHAIN_RULE_FORMULA_CACHE_VERSION\s*=\s*(\d+)", text)
    return int(match.group(1)) if match else 4


def _rectangular_shape(maxes: tuple[int, ...]) -> list[list[int]]:
    shape = [tuple(values) for values in itertools.product(*(range(m + 1) for m in maxes))]
    shape.sort(key=lambda item: (sum(item), item))
    return [list(values) for values in shape]


def _nonincreasing_tuples(rank: int, max_depth: int) -> Iterable[tuple[int, ...]]:
    def rec(prefix: list[int], max_next: int) -> Iterable[tuple[int, ...]]:
        if len(prefix) == rank:
            yield tuple(prefix)
            return
        for value in range(max_next, -1, -1):
            yield from rec([*prefix, value], value)

    yield from rec([], max_depth)


def _derivative_len(active: int, maxes: tuple[int, ...], max_degree_cap: int = 4) -> int:
    max_degree = min(max_degree_cap, sum(maxes))
    return sum(math.comb(active + degree - 1, degree) for degree in range(max_degree + 1))


def infer_signature_specs(
    digests: set[str],
    *,
    schema_version: int,
    max_active: int,
    max_rank: int,
    max_depth: int,
    max_output_len: int,
) -> dict[str, SignatureSpec]:
    """Infer canonical rectangular signature specs for known digest names."""

    remaining = set(digests)
    found: dict[str, SignatureSpec] = {}
    for rank in range(1, max_rank + 1):
        for maxes in _nonincreasing_tuples(rank, max_depth):
            output_len = math.prod(value + 1 for value in maxes)
            if output_len > max_output_len:
                continue
            shape = _rectangular_shape(maxes)
            for active in range(rank, max_active + 1):
                payload = {
                    "schema_version": schema_version,
                    "kind": "chain-rule",
                    "signature": [active, rank, shape],
                }
                digest = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                if digest not in remaining:
                    continue
                found[digest] = SignatureSpec(
                    active=active,
                    rank=rank,
                    maxes=maxes,
                    output_len=output_len,
                    derivative_len=_derivative_len(active, maxes),
                )
                remaining.remove(digest)
                if not remaining:
                    return found
    return found


def _scan_log(path: Path) -> tuple[set[str], dict[str, tuple[int, int]], list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    digests = set(re.findall(r"chain_rule_([0-9a-f]{64})", text))
    progress: dict[str, tuple[int, int]] = {}
    for match in re.finditer(
        r"\[fsd-chain-rule\]\s+([0-9a-f]{64})\s+compose_progress\s+(\d+)/(\d+)",
        text,
    ):
        digest = match.group(1)
        current = int(match.group(2))
        total = int(match.group(3))
        previous = progress.get(digest, (0, 0))
        progress[digest] = (max(previous[0], current), max(previous[1], total))
    panics = [
        match.group(1).strip()
        for match in re.finditer(
            r"(pyo3_runtime\.PanicException:[^\n]+|MemoryError|Killed|KeyboardInterrupt)",
            text,
        )
    ]
    return digests, progress, panics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-active", type=int, default=9)
    parser.add_argument("--max-rank", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-output-len", type=int, default=5000)
    parser.add_argument("--no-infer", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    status = _load_json(root / STATUS_PATH)
    report_status = _report_statuses(root / REPORT_DIR)
    running = sorted(
        index
        for task in status.get("running_tasks", [])
        if (index := _task_index(str(task.get("task_id", "")))) is not None
    )
    running_set = set(running)

    not_done: list[int] = []
    for index in range(1, int(status.get("task_count") or 99) + 1):
        if index in running_set:
            not_done.append(index)
            continue
        if report_status.get(index) not in {"ok", "valid_skipped"}:
            not_done.append(index)

    digest_shards: dict[str, set[int]] = defaultdict(set)
    digest_progress: dict[str, tuple[int, int]] = {}
    shard_rows: list[tuple[int, str, list[str], list[str]]] = []
    for index in not_done:
        log_path = root / LOG_DIR / f"triple_box_shard_{index:04d}_of_0099.log"
        digests, progress, panics = _scan_log(log_path)
        for digest in digests:
            digest_shards[digest].add(index)
        for digest, value in progress.items():
            previous = digest_progress.get(digest, (0, 0))
            digest_progress[digest] = (max(previous[0], value[0]), max(previous[1], value[1]))
        status_label = "running" if index in running_set else (report_status.get(index) or "no_report")
        shard_rows.append((index, status_label, sorted(digests), panics[-1:]))

    specs: dict[str, SignatureSpec] = {}
    if not args.no_infer and digest_shards:
        specs = infer_signature_specs(
            set(digest_shards),
            schema_version=_schema_version(root / CACHE_DIR, root),
            max_active=args.max_active,
            max_rank=args.max_rank,
            max_depth=args.max_depth,
            max_output_len=args.max_output_len,
        )

    print(
        "status",
        f"time={status.get('time_utc')}",
        f"running={status.get('running')}",
        f"pending={status.get('pending')}",
        f"skipped={status.get('skipped')}",
        f"failed={status.get('failed')}",
    )
    print("running_shards", ",".join(f"{index:04d}" for index in running) or "-")
    print("not_done_shards", ",".join(f"{index:04d}" for index in not_done) or "-")
    print()
    print("chain_rule_digests")
    for digest in sorted(digest_shards):
        progress = digest_progress.get(digest)
        progress_text = f"{progress[0]}/{progress[1]}" if progress else "-"
        spec = specs.get(digest)
        if spec is None:
            spec_text = "signature=?"
        else:
            spec_text = (
                f"active={spec.active} rank={spec.rank} maxes={spec.maxes} "
                f"outputs={spec.output_len} derivatives={spec.derivative_len}"
            )
        print(
            digest,
            f"shards={','.join(f'{index:04d}' for index in sorted(digest_shards[digest]))}",
            f"progress={progress_text}",
            spec_text,
        )
    print()
    print("not_done_by_shard")
    for index, status_label, digests, panics in shard_rows:
        sectors = f"{(index - 1) * 20}-{(index - 1) * 20 + 19}"
        digest_text = ",".join(digest[:8] for digest in digests) or "-"
        panic_text = panics[0][:90] if panics else "-"
        print(f"{index:04d} sectors={sectors:>9} status={status_label:<9} chain={digest_text:<20} latest_error={panic_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
