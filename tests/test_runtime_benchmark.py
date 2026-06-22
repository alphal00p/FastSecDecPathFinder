from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import runtime_benchmark


class DummyProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_batch(self, _sector, rows):
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt
        timing = SimpleNamespace(
            eval_seconds=1.0e-6,
            python_seconds=2.0e-6,
            ordinary_precision_samples=int(rows.shape[0]),
            stability_precision_samples=0,
            medium_precision_samples=0,
            high_precision_samples=0,
        )
        return np.zeros((rows.shape[0], 1), dtype=np.complex128), np.zeros(rows.shape[0]), timing


def test_runtime_benchmark_returns_partial_report_on_interrupt(monkeypatch, capsys) -> None:
    processor = DummyProcessor()
    monkeypatch.setattr(runtime_benchmark, "_make_sector_processor", lambda *_args: processor)
    request = SimpleNamespace(
        sectors=None,
        benchmark_samples_per_sector=3,
        seed=7,
        json=True,
        no_progress=True,
        show_stats=False,
    )
    sectors = [
        SimpleNamespace(name="S0", integration_dim=2),
        SimpleNamespace(name="S1", integration_dim=2),
    ]

    report = runtime_benchmark.run_sector_runtime_benchmark(
        request,
        topology=None,
        sectors=sectors,
        summary={},
    )

    assert report["status"] == "interrupted"
    assert report["completed_sector_count"] == 1
    assert report["requested_sector_count"] == 2
    assert report["sectors"][0]["sector_name"] == "S0"
    assert '"status": "interrupted"' in capsys.readouterr().out
