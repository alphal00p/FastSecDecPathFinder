#!/usr/bin/env python3
"""Profile selected prepared-bundle sectors with multi-point batches.

The one-point sector scan is useful for finding outliers, but it can exaggerate
costs that amortize across a vectorized batch.  This script evaluates selected
sectors in the same prepared-bundle `SectorProcessor.evaluate_batch` path with
a configurable batch size and prints the wall/evaluator/Python timing split.
"""

from __future__ import annotations

import argparse
import json
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


def _complex_json(value: complex) -> dict[str, float]:
    """Return the complex-number JSON shape used in result files."""
    return {"re": float(value.real), "im": float(value.imag)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Prepared bundle directory.")
    parser.add_argument("--sectors", nargs="+", required=True, type=int, help="Sector ids to profile.")
    parser.add_argument("--points", type=int, default=1000, help="Batch size per sector.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeated batches per sector.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evaluator-lru-size", type=int, default=0)
    parser.add_argument("--subtraction-backend", default="projector-formula")
    parser.add_argument("--json", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    topology, sectors, _manifest = load_prepared_bundle(args.output, lru_size=args.evaluator_lru_size)
    processor = SectorProcessor(topology, subtraction_backend=args.subtraction_backend)
    rows: list[dict[str, Any]] = []

    print(
        f"loaded {len(sectors)} sectors; points={args.points}; repeats={args.repeats}; "
        f"lru={args.evaluator_lru_size}",
        flush=True,
    )
    for sector_id in args.sectors:
        if sector_id < 0 or sector_id >= len(sectors):
            raise ValueError(f"sector id {sector_id} outside [0,{len(sectors)})")
        sector = sectors[sector_id]
        print(f"\nsector {sector_id} {sector.name} dim={sector.integration_dim}", flush=True)
        for repeat in range(max(int(args.repeats), 1)):
            rng = np.random.default_rng(int(args.seed) + 1_000_003 * int(sector_id) + repeat)
            coords = rng.random((max(int(args.points), 1), sector.integration_dim), dtype=float)
            start = time.perf_counter()
            coeffs, training, timing = processor.evaluate_batch(sector, coords)
            wall = time.perf_counter() - start
            max_abs_coeff = float(np.max(np.abs(coeffs))) if coeffs.size else 0.0
            mean_abs_training = float(np.mean(np.abs(training))) if training.size else 0.0
            row = {
                "sector_id": int(sector_id),
                "name": sector.name,
                "repeat": int(repeat),
                "points": int(coords.shape[0]),
                "wall_seconds": float(wall),
                "eval_seconds": float(timing.eval_seconds),
                "python_seconds": float(timing.python_seconds),
                "total_profiled_seconds": float(timing.total_seconds),
                "wall_us_per_sample": float(wall * 1.0e6 / coords.shape[0]),
                "eval_us_per_sample": float(timing.eval_seconds * 1.0e6 / coords.shape[0]),
                "python_us_per_sample": float(timing.python_seconds * 1.0e6 / coords.shape[0]),
                "precision_counts": timing.precision_counts,
                "max_abs_coefficient": max_abs_coeff,
                "mean_abs_training": mean_abs_training,
                "mean_coefficients": [_complex_json(complex(value)) for value in np.mean(coeffs, axis=0)],
                "std_coefficients": [_complex_json(complex(value)) for value in np.std(coeffs, axis=0, ddof=1)],
            }
            rows.append(row)
            print(
                "  rep {repeat}: wall={wall:.3f}s "
                "wall_us={wall_us:.1f} eval_us={eval_us:.1f} py_us={py_us:.1f} "
                "max|c|={maxc:.3e} precision={precision}".format(
                    repeat=repeat,
                    wall=wall,
                    wall_us=row["wall_us_per_sample"],
                    eval_us=row["eval_us_per_sample"],
                    py_us=row["python_us_per_sample"],
                    maxc=max_abs_coeff,
                    precision=timing.precision_counts,
                ),
                flush=True,
            )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
