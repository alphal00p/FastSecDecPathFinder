"""Direct runtime benchmark for pySecDec generated sector kernels.

This module deliberately times the low-level generated C++ sector/order
integrand functions, not the public ``IntegralLibrary`` QMC driver.  It is used
as the fair runtime reference for FSD's sector-level ``benchmark`` command and
for the standalone MRE scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
from typing import Any

from colorama import Fore, Style
from prettytable import PrettyTable

from formatting import json_default


_KERNEL_RE = re.compile(r"^sector_(\d+)_order_(.+)$")


@dataclass(frozen=True)
class PySecDecKernelRecord:
    """One pySecDec generated sector timed after summing all requested orders."""

    sector_id: int
    orders: tuple[str, ...]
    samples: int
    seconds: float
    checksum: float

    @property
    def us_per_sector_point(self) -> float:
        """Return microseconds for one full sector point across all order kernels."""
        return 1.0e6 * self.seconds / max(int(self.samples), 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this timing record."""
        return {
            "sector_id": self.sector_id,
            "orders": list(self.orders),
            "samples": self.samples,
            "seconds": self.seconds,
            "us_per_sector_point": self.us_per_sector_point,
            "checksum": self.checksum,
        }


def _color(text: str, color: str) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


def _fmt_us(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:.3g} ms"
    return f"{value:.3g} μs"


def _order_sort_key(token: str) -> tuple[int, str]:
    """Sort pySecDec order tokens such as ``n4``, ``n1``, and ``0``."""
    if token.startswith("n") and token[1:].isdigit():
        return (int(token[1:]), token)
    try:
        return (1000 + int(token), token)
    except ValueError:
        return (2000, token)


def _pysecdec_contrib_dir() -> Path:
    """Return the installed pySecDecContrib directory."""
    try:
        module = importlib.import_module("pySecDecContrib")
    except Exception as exc:  # pragma: no cover - depends on external install.
        raise RuntimeError(
            "pySecDecContrib is required to compile the generated-kernel benchmark"
        ) from exc
    return Path(module.__file__).resolve().parent


def _metadata_path(integral_dir: Path) -> Path:
    name = integral_dir.name
    path = integral_dir / "disteval" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"pySecDec disteval metadata not found: {path}")
    return path


def _kernel_groups(metadata: dict[str, Any]) -> dict[int, list[str]]:
    """Return order tokens grouped by generated pySecDec sector id."""
    groups: dict[int, list[str]] = {}
    for kernel in metadata.get("kernels", []):
        match = _KERNEL_RE.match(str(kernel))
        if match is None:
            continue
        groups.setdefault(int(match.group(1)), []).append(match.group(2))
    for orders in groups.values():
        orders.sort(key=_order_sort_key)
    return groups


def _build_driver_source(
    *,
    namespace: str,
    dimension: int,
    grouped_orders: dict[int, list[str]],
    real_parameters: list[float],
    seed: int,
) -> str:
    """Build a self-contained C++ source timing selected pySecDec kernels."""
    includes: list[str] = []
    calls: list[str] = []
    for sector_id in sorted(grouped_orders):
        for token in grouped_orders[sector_id]:
            includes.append(f'#include "src/sector_{sector_id}_{token}.hpp"')
            calls.append(
                "        { "
                f"{sector_id}, \"{token}\", "
                "+[] (const double* x, const double* p, "
                "const std::complex<double>* cp) -> std::complex<double> { "
                f"return {namespace}::sector_{sector_id}_order_{token}_integrand"
                "(x, p, cp, nullptr); } },"
            )
    params = ", ".join(f"{float(value):.17g}" for value in real_parameters)
    return f"""
#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>
{chr(10).join(sorted(set(includes)))}

struct Kernel {{
    int sector;
    const char* order;
    std::complex<double> (*fn)(const double*, const double*, const std::complex<double>*);
}};

int main(int argc, char** argv) {{
    const int dim = {int(dimension)};
    const int samples = argc > 1 ? std::atoi(argv[1]) : 10000;
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 1;
    const unsigned long long seed = static_cast<unsigned long long>({int(seed)});
    std::vector<double> params = {{ {params} }};
    std::vector<std::complex<double>> cparams;
    std::vector<Kernel> kernels = {{
{chr(10).join(calls)}
    }};

    std::vector<double> points(static_cast<std::size_t>(samples) * dim);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.125, 0.875);
    for (int i = 0; i < samples; ++i) {{
        for (int j = 0; j < dim; ++j) {{
            points[static_cast<std::size_t>(i) * dim + j] = dist(rng);
        }}
    }}

    std::map<int, double> seconds;
    std::map<int, double> checksum;
    for (auto const& k : kernels) {{
        volatile double sink = 0.0;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeats; ++r) {{
            for (int i = 0; i < samples; ++i) {{
                const double* x = &points[static_cast<std::size_t>(i) * dim];
                auto val = k.fn(x, params.data(), cparams.data());
                sink += val.real() + 0.01 * val.imag();
            }}
        }}
        auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();
        seconds[k.sector] += elapsed;
        checksum[k.sector] += sink;
        std::cout << "kernel "
                  << k.sector << " "
                  << k.order << " "
                  << std::setprecision(17) << elapsed << "\\n";
    }}
    for (auto const& entry : seconds) {{
        const int sector = entry.first;
        std::cout << "sector_total "
                  << sector << " "
                  << std::setprecision(17) << entry.second << " "
                  << checksum[sector] << "\\n";
    }}
    return 0;
}}
"""


def _parse_driver_output(
    output: str,
    *,
    samples: int,
    repeats: int,
    grouped_orders: dict[int, list[str]],
) -> list[PySecDecKernelRecord]:
    """Parse machine-readable lines emitted by the temporary C++ driver."""
    records: list[PySecDecKernelRecord] = []
    effective_samples = max(int(samples) * int(repeats), 1)
    for line in output.splitlines():
        parts = line.split()
        if not parts or parts[0] != "sector_total":
            continue
        sector_id = int(parts[1])
        seconds = float(parts[2])
        checksum = float(parts[3])
        records.append(
            PySecDecKernelRecord(
                sector_id=sector_id,
                orders=tuple(grouped_orders.get(sector_id, ())),
                samples=effective_samples,
                seconds=seconds,
                checksum=checksum,
            )
        )
    return sorted(records, key=lambda item: item.sector_id)


def benchmark_pysecdec_generated_kernels(
    integral_dir: str | Path,
    *,
    samples_per_sector: int,
    sectors: list[int] | tuple[int, ...] | None = None,
    real_parameters: list[float] | tuple[float, ...] | None = None,
    repeats: int = 1,
    seed: int = 12345,
) -> dict[str, Any]:
    """Compile and run a temporary direct-C++ benchmark for pySecDec kernels."""
    integral_dir = Path(integral_dir).expanduser().resolve()
    metadata = json.loads(_metadata_path(integral_dir).read_text(encoding="utf-8"))
    namespace = str(metadata.get("name") or integral_dir.name)
    grouped = _kernel_groups(metadata)
    if sectors is not None:
        selected = {int(sector_id) for sector_id in sectors}
        grouped = {sector_id: orders for sector_id, orders in grouped.items() if sector_id in selected}
    if not grouped:
        raise ValueError("no generated pySecDec sector kernels selected for benchmarking")
    if real_parameters is None:
        parameter_count = len(metadata.get("realp", []))
        real_parameters = [0.0 for _ in range(parameter_count)]
    contrib = _pysecdec_contrib_dir()
    source = _build_driver_source(
        namespace=namespace,
        dimension=int(metadata["dimension"]),
        grouped_orders=grouped,
        real_parameters=[float(value) for value in real_parameters],
        seed=int(seed),
    )
    with tempfile.TemporaryDirectory(prefix="fsd-pysecdec-kernel-bench-") as tmp:
        tmpdir = Path(tmp)
        cpp = tmpdir / "bench_pysecdec_kernels.cpp"
        exe = tmpdir / "bench_pysecdec_kernels"
        cpp.write_text(source, encoding="utf-8")
        lib = integral_dir / f"lib{namespace}.a"
        if not lib.is_file():
            raise FileNotFoundError(f"pySecDec static library not found: {lib}")
        cmd = [
            "c++",
            "-std=c++17",
            "-O3",
            "-I",
            str(integral_dir),
            "-I",
            str(integral_dir / "src"),
            "-I",
            str(contrib / "include"),
            str(cpp),
            str(lib),
            "-L",
            str(contrib / "lib"),
            "-lgsl",
            "-lgslcblas",
            "-lcuba",
            "-lgmp",
            "-lm",
            "-pthread",
            "-o",
            str(exe),
        ]
        subprocess.run(cmd, check=True, text=True)
        output = subprocess.check_output(
            [str(exe), str(int(samples_per_sector)), str(int(repeats))],
            text=True,
        )
    records = _parse_driver_output(
        output,
        samples=int(samples_per_sector),
        repeats=int(repeats),
        grouped_orders=grouped,
    )
    values = [record.us_per_sector_point for record in records]
    min_record = min(records, key=lambda item: item.us_per_sector_point)
    max_record = max(records, key=lambda item: item.us_per_sector_point)
    return {
        "schema_version": 1,
        "engine": "pysecdec-generated-kernels",
        "integral_dir": str(integral_dir),
        "namespace": namespace,
        "dimension": int(metadata["dimension"]),
        "samples_per_sector": int(samples_per_sector),
        "repeats": int(repeats),
        "sector_count": len(records),
        "kernel_count": sum(len(orders) for orders in grouped.values()),
        "runtime_us_per_sector_point": {
            "min": min(values),
            "min_sector": {"id": min_record.sector_id, "name": f"sector_{min_record.sector_id}"},
            "max": max(values),
            "max_sector": {"id": max_record.sector_id, "name": f"sector_{max_record.sector_id}"},
            "average": statistics.fmean(values),
            "median": statistics.median(values),
        },
        "sectors": [record.to_dict() for record in records],
    }


def print_pysecdec_kernel_benchmark_report(report: dict[str, Any], *, show_all: bool = False) -> None:
    """Print a colored summary for a pySecDec generated-kernel benchmark."""
    stats = report["runtime_us_per_sector_point"]
    print(_color("\npySecDec generated-kernel runtime benchmark", Fore.CYAN))
    table = PrettyTable()
    table.field_names = [_color("statistic", Fore.CYAN), _color("value", Fore.CYAN)]
    table.add_row(["integral dir", report["integral_dir"]])
    table.add_row(["sectors measured", report["sector_count"]])
    table.add_row(["order kernels", report["kernel_count"]])
    table.add_row(["samples / sector", report["samples_per_sector"]])
    table.add_row(["repeats", report["repeats"]])
    table.add_row(["min μs/sector point", _fmt_us(stats["min"])])
    table.add_row(["max μs/sector point", _fmt_us(stats["max"])])
    table.add_row(["average μs/sector point", _fmt_us(stats["average"])])
    table.add_row(["median μs/sector point", _fmt_us(stats["median"])])
    print(table)

    extrema = PrettyTable()
    extrema.field_names = [
        _color("kind", Fore.CYAN),
        _color("sector", Fore.CYAN),
        _color("orders", Fore.CYAN),
        _color("μs/sector point", Fore.CYAN),
    ]
    records = [
        (entry["sector_id"], entry)
        for entry in report.get("sectors", [])
    ]
    by_id = {sector_id: entry for sector_id, entry in records}
    for kind, sector_info in (
        ("min", stats["min_sector"]),
        ("max", stats["max_sector"]),
    ):
        if not sector_info:
            continue
        entry = by_id[int(sector_info["id"])]
        color = Fore.GREEN if kind == "min" else Fore.YELLOW
        extrema.add_row(
            [
                _color(kind, color),
                f"sector_{entry['sector_id']}",
                ",".join(entry["orders"]),
                _fmt_us(entry["us_per_sector_point"]),
            ]
        )
    print(extrema)

    if show_all:
        all_rows = PrettyTable()
        all_rows.field_names = [
            _color("sector", Fore.CYAN),
            _color("orders", Fore.CYAN),
            _color("μs/sector point", Fore.CYAN),
            _color("samples", Fore.CYAN),
        ]
        for entry in report.get("sectors", []):
            all_rows.add_row(
                [
                    f"sector_{entry['sector_id']}",
                    ",".join(entry["orders"]),
                    _fmt_us(entry["us_per_sector_point"]),
                    entry["samples"],
                ]
            )
        print(all_rows)


def output_pysecdec_kernel_benchmark_json(report: dict[str, Any]) -> str:
    """Serialize a pySecDec kernel benchmark report."""
    return json.dumps(report, default=json_default, indent=2, sort_keys=True)
