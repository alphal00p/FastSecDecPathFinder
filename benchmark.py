"""OneLOopBridge benchmark access for the FastSecDec v2 CLI.

The bridge is required and external.  This module centralizes import failure
messages and keeps the rest of the code independent of the exact Python binding
API names.
"""

from __future__ import annotations

from typing import Any

from definitions import BenchmarkResult, IntegralRequest


SETUP_HINT = """OneLOopBridge is required for FSD_v2.

Install it with one of:
  ONELOOPBRIDGE_SRC=/path/to/OneLOopBridge ./install.sh
  ./install.sh --clone-oneloopbridge

The bridge is external and must not be vendored into this repository.
"""


def import_oneloop_bridge() -> Any:
    """Import the external bridge or raise setup instructions."""
    try:
        import oneloop_bridge
    except ImportError as exc:
        raise RuntimeError(SETUP_HINT) from exc
    return oneloop_bridge


def check_oneloop_bridge() -> None:
    """Validate that the required benchmark package is importable."""
    import_oneloop_bridge()


def compute_benchmark(request: IntegralRequest) -> BenchmarkResult:
    """Compute the OneLOopBridge Laurent coefficients for the requested case."""
    oneloop_bridge = import_oneloop_bridge()

    if request.mu is not None:
        oneloop_bridge.set_renormalization_scale(float(request.mu))
    if request.onshell_threshold is not None:
        oneloop_bridge.set_onshell_threshold(float(request.onshell_threshold))

    m2 = complex(request.m * request.m, 0.0)
    if request.integral == "triangle":
        if request.s is None:
            raise ValueError("triangle benchmark requires s")
        result = oneloop_bridge.three_point(0.0, 0.0, float(request.s), m2, m2, m2)
    elif request.integral == "box":
        if request.s12 is None or request.s23 is None:
            raise ValueError("box benchmark requires s12 and s23")
        result = oneloop_bridge.four_point(
            0.0,
            0.0,
            0.0,
            0.0,
            float(request.s12),
            float(request.s23),
            m2,
            m2,
            m2,
            m2,
        )
    else:
        raise ValueError(f"unsupported integral {request.integral!r}")

    return BenchmarkResult(
        raw=[result.epsilon_minus_2, result.epsilon_minus_1, result.epsilon_0],
        factor=oneloop_bridge.TO_FEYNMAN,
    )
