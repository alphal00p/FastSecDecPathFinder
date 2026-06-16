#!/usr/bin/env python3
"""Scan prepared-bundle sectors one point at a time with per-sector timeouts.

This diagnostic intentionally avoids the normal integrator.  It loads a
prepared DOT bundle once, then forks one sector evaluation at a time.  Each row
is appended to JSONL immediately, so a slow or timed-out sector does not discard
the sectors already scanned.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrand import SectorProcessor
from prepared_bundle import load_prepared_bundle


TOPOLOGY = None
SECTORS = None
PROCESSOR = None


def _json_complex(value: complex) -> dict[str, float]:
    """Return the JSON shape used by FSD result files."""
    return {"re": float(value.real), "im": float(value.imag)}


def _scan_one_sector(sector_id: int, seed: int, queue: Any) -> None:
    """Evaluate one deterministic point for a sector and send a JSON row."""
    assert TOPOLOGY is not None
    assert SECTORS is not None
    assert PROCESSOR is not None
    sector = SECTORS[sector_id]
    rng = np.random.default_rng(int(seed) + 1_000_003 * int(sector_id))
    coords = rng.random((1, sector.integration_dim), dtype=float)
    start = time.perf_counter()
    try:
        coeffs, _training, timing = PROCESSOR.evaluate_batch(sector, coords)
        elapsed = time.perf_counter() - start
        values = [complex(value) for value in coeffs[0]]
        row = {
            "sector_id": int(sector_id),
            "name": sector.name,
            "status": "ok",
            "elapsed_seconds": float(elapsed),
            "coords": [float(value) for value in coords[0]],
            "coefficients": [_json_complex(value) for value in values],
            "max_abs_coefficient": float(max((abs(value) for value in values), default=0.0)),
            "eval_seconds": float(timing.eval_seconds),
            "python_seconds": float(timing.python_seconds),
            "precision_counts": timing.precision_counts,
            "avg_eval_us_per_sample": float(timing.eval_seconds * 1.0e6),
            "avg_total_us_per_sample": float(timing.total_seconds * 1.0e6),
        }
    except BaseException as exc:  # pragma: no cover - diagnostics must not hide failures.
        elapsed = time.perf_counter() - start
        row = {
            "sector_id": int(sector_id),
            "name": sector.name,
            "status": "error",
            "elapsed_seconds": float(elapsed),
            "error": f"{type(exc).__name__}: {exc}",
        }
    queue.put(row)


def _parse_sector_ids(
    texts: list[str] | None,
    sector_count: int,
    start: int,
    stop: int | None,
    limit: int | None,
) -> list[int]:
    """Parse explicit sector ids or default to all sectors."""
    if texts:
        ids = [int(item) for item in texts]
    else:
        upper = sector_count if stop is None else min(int(stop), sector_count)
        ids = list(range(max(int(start), 0), upper))
    if limit is not None:
        ids = ids[: max(int(limit), 0)]
    for sector_id in ids:
        if sector_id < 0 or sector_id >= sector_count:
            raise ValueError(f"sector id {sector_id} outside [0,{sector_count})")
    return ids


def _load_done(path: Path) -> set[int]:
    """Return sector ids already present in an existing JSONL output."""
    done: set[int] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") in {"ok", "timeout", "error"}:
            done.add(int(row["sector_id"]))
    return done


def _write_row(handle: Any, row: dict[str, Any], ordinal: int, total: int) -> None:
    """Persist one sector row and print a compact progress line."""
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    max_abs = row.get("max_abs_coefficient")
    max_abs_text = "n/a" if max_abs is None or not math.isfinite(float(max_abs)) else f"{float(max_abs):.3e}"
    print(
        f"[{ordinal}/{total}] sector {row['sector_id']} {row['name']} "
        f"{row['status']} elapsed={row['elapsed_seconds']:.3g}s "
        f"max|c|={max_abs_text}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Prepared bundle directory.")
    parser.add_argument("--jsonl", required=True, help="Incremental JSONL output path.")
    parser.add_argument("--sectors", nargs="*", help="Optional sector ids to scan.")
    parser.add_argument("--start", type=int, default=0, help="First sector id when --sectors is omitted.")
    parser.add_argument("--stop", type=int, help="Exclusive final sector id when --sectors is omitted.")
    parser.add_argument("--limit", type=int, help="Maximum number of sector ids to scan.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-per-sector", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=1, help="Number of sector child processes to run concurrently.")
    parser.add_argument("--evaluator-lru-size", type=int, default=128)
    parser.add_argument("--subtraction-backend", default="projector-formula")
    parser.add_argument("--resume", action="store_true", help="Skip sector ids already in JSONL.")
    args = parser.parse_args()

    global TOPOLOGY, SECTORS, PROCESSOR
    TOPOLOGY, SECTORS, _manifest = load_prepared_bundle(
        args.output,
        lru_size=args.evaluator_lru_size,
    )
    PROCESSOR = SectorProcessor(TOPOLOGY, subtraction_backend=args.subtraction_backend)

    sector_ids = _parse_sector_ids(args.sectors, len(SECTORS), args.start, args.stop, args.limit)
    jsonl_path = Path(args.jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(jsonl_path) if args.resume else set()
    remaining = [sector_id for sector_id in sector_ids if sector_id not in done]
    ctx = mp.get_context("fork")
    worker_count = max(int(args.workers), 1)

    print(
        f"loaded {len(SECTORS)} sectors; scanning {len(remaining)} sector(s); "
        f"timeout={args.timeout_per_sector:g}s; workers={worker_count}; output={jsonl_path}",
        flush=True,
    )
    with jsonl_path.open("a", encoding="utf-8") as handle:
        if worker_count == 1:
            for ordinal, sector_id in enumerate(remaining, start=1):
                queue = ctx.Queue(maxsize=1)
                process = ctx.Process(target=_scan_one_sector, args=(sector_id, args.seed, queue))
                start = time.perf_counter()
                process.start()
                process.join(timeout=max(float(args.timeout_per_sector), 0.1))
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
                    row = {
                        "sector_id": int(sector_id),
                        "name": SECTORS[sector_id].name,
                        "status": "timeout",
                        "elapsed_seconds": float(time.perf_counter() - start),
                        "timeout_seconds": float(args.timeout_per_sector),
                    }
                else:
                    try:
                        row = queue.get_nowait()
                    except Exception as exc:
                        row = {
                            "sector_id": int(sector_id),
                            "name": SECTORS[sector_id].name,
                            "status": "error",
                            "elapsed_seconds": float(time.perf_counter() - start),
                            "error": f"missing worker row: {exc}",
                        }
                _write_row(handle, row, ordinal, len(remaining))
            return 0

        active: dict[int, tuple[mp.Process, Any, float]] = {}
        submitted = 0
        completed = 0
        remaining_iter = iter(remaining)

        def submit_until_full() -> None:
            nonlocal submitted
            while len(active) < worker_count:
                try:
                    sector_id = next(remaining_iter)
                except StopIteration:
                    return
                queue = ctx.Queue(maxsize=1)
                process = ctx.Process(target=_scan_one_sector, args=(sector_id, args.seed, queue))
                active[int(sector_id)] = (process, queue, time.perf_counter())
                process.start()
                submitted += 1

        submit_until_full()
        while active:
            now = time.perf_counter()
            finished_rows: list[dict[str, Any]] = []
            for sector_id, (process, queue, start) in list(active.items()):
                elapsed = now - start
                if process.is_alive() and elapsed > max(float(args.timeout_per_sector), 0.1):
                    process.terminate()
                    process.join(timeout=5.0)
                    finished_rows.append(
                        {
                            "sector_id": int(sector_id),
                            "name": SECTORS[sector_id].name,
                            "status": "timeout",
                            "elapsed_seconds": float(time.perf_counter() - start),
                            "timeout_seconds": float(args.timeout_per_sector),
                        }
                    )
                    del active[sector_id]
                    continue
                if process.is_alive():
                    continue
                process.join(timeout=0.0)
                try:
                    row = queue.get_nowait()
                except Exception as exc:
                    row = {
                        "sector_id": int(sector_id),
                        "name": SECTORS[sector_id].name,
                        "status": "error",
                        "elapsed_seconds": float(time.perf_counter() - start),
                        "error": f"missing worker row: {exc}",
                    }
                finished_rows.append(row)
                del active[sector_id]

            for row in sorted(finished_rows, key=lambda item: int(item["sector_id"])):
                completed += 1
                _write_row(handle, row, completed, len(remaining))
            submit_until_full()
            if active:
                time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
