"""Direct Symanzik U/F topology input for pySecDec-backed FSD sectors."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from symbolica import E, S

from definitions import EpsilonExpansion, IntegralRequest
from generation_timing import GenerationProgress, GenerationTimings
from pysecdec_bridge import UFBuildBundle, UFTopologyData, build_uf_bundle

if TYPE_CHECKING:
    from integrand import TopologyDefinition
    from sectors_generator import SectorDefinition


def _key_variants(key: str) -> list[str]:
    """Return accepted nested YAML spellings for one semantic key."""
    text = str(key)
    return [
        text,
        text.replace("_", "-"),
        text.replace("-", "_"),
        text.lower(),
        text.lower().replace("_", "-"),
        text.lower().replace("-", "_"),
    ]


def _get(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    """Fetch a nested YAML value with kebab/snake/case-tolerant keys."""
    for variant in _key_variants(key):
        if variant in mapping:
            return mapping[variant]
    return default


def _required(mapping: dict[str, Any], key: str) -> Any:
    """Fetch a required nested YAML value."""
    value = _get(mapping, key)
    if value is None:
        raise ValueError(f"uf-topology is missing required key {key!r}")
    return value


def _affine_from_text(text: Any, *, key: str) -> EpsilonExpansion:
    """Parse ``base + eps_coeff*eps`` from a Symbolica expression string."""
    try:
        evaluator = E(str(text)).evaluator([S("eps")])
        base = float(evaluator.evaluate([[0.0]])[0][0])
        at_one = float(evaluator.evaluate([[1.0]])[0][0])
    except Exception as exc:
        raise ValueError(f"uf-topology key {key!r} must be affine in eps: {text!r}") from exc
    return EpsilonExpansion(base=base, eps_coeff=at_one - base)


def _float_tuple(values: Any, *, key: str, length: int | None = None) -> tuple[float, ...]:
    """Parse a numeric YAML sequence as a float tuple."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"uf-topology key {key!r} must be a sequence")
    out = tuple(float(value) for value in values)
    if length is not None and len(out) != length:
        raise ValueError(
            f"uf-topology key {key!r} has length {len(out)}, expected {length}"
        )
    return out


def _parameter_data(data: dict[str, Any]) -> tuple[list[str], list[float]]:
    """Return evaluator parameter order and numeric values from ``uf-topology``."""
    raw_values = _get(data, "values", {}) or {}
    if not isinstance(raw_values, dict):
        raise ValueError("uf-topology key 'values' must be a mapping when supplied")
    raw_parameters = _get(data, "parameters")
    if raw_parameters is None:
        names = [str(name) for name in raw_values.keys()]
    else:
        if not isinstance(raw_parameters, (list, tuple)):
            raise ValueError("uf-topology key 'parameters' must be a sequence")
        names = [str(name) for name in raw_parameters]
    missing = [name for name in names if name not in raw_values]
    if missing:
        raise ValueError(
            "uf-topology values are missing numeric entries for parameters: "
            + ", ".join(missing)
        )
    return names, [float(raw_values[name]) for name in names]


def uf_topology_data_from_request(request: IntegralRequest) -> UFTopologyData:
    """Normalize the direct U/F YAML block carried by the request."""
    data = request.uf_topology
    if not isinstance(data, dict):
        raise ValueError("direct U/F topology mode requires a uf-topology YAML mapping")
    raw_variables = _required(data, "variables")
    if not isinstance(raw_variables, (list, tuple)) or not raw_variables:
        raise ValueError("uf-topology key 'variables' must be a non-empty sequence")
    x_names = [str(name) for name in raw_variables]
    if len(set(x_names)) != len(x_names):
        raise ValueError("uf-topology variables must be unique")
    parameter_names, parameter_values = _parameter_data(data)
    propagator_powers = _float_tuple(
        _required(data, "propagator-powers"),
        key="propagator-powers",
        length=len(x_names),
    )
    invalid_powers = [
        power for power in propagator_powers if abs(power - round(power)) > 1.0e-12 or power <= 0.0
    ]
    if invalid_powers:
        raise ValueError(
            "uf-topology propagator-powers must be positive integers; "
            f"got {invalid_powers}"
        )
    raw_measure_powers = _get(data, "measure-powers")
    if raw_measure_powers is None:
        measure_powers = tuple(float(power - 1.0) for power in propagator_powers)
    else:
        measure_powers = _float_tuple(
            raw_measure_powers,
            key="measure-powers",
            length=len(x_names),
        )
    return UFTopologyData(
        family=str(_get(data, "family", _get(data, "label", "direct U/F topology"))),
        x_names=x_names,
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        u_expr_text=str(_required(data, "U")),
        f_expr_text=str(_required(data, "F")),
        loop_count=int(_required(data, "loop-count")),
        dimension=_affine_from_text(_required(data, "dimension"), key="dimension"),
        propagator_powers=tuple(float(power) for power in propagator_powers),
        measure_powers=tuple(float(power) for power in measure_powers),
        u_exponent=_affine_from_text(_required(data, "U-exponent"), key="U-exponent"),
        f_exponent=_affine_from_text(_required(data, "F-exponent"), key="F-exponent"),
        global_prefactor=str(_required(data, "global-prefactor")),
    )


_UF_BUNDLE_CACHE: dict[tuple[object, ...], UFBuildBundle] = {}


def clear_uf_bundle_cache() -> None:
    """Drop in-process U/F build bundles."""
    _UF_BUNDLE_CACHE.clear()


def _request_cache_key(request: IntegralRequest) -> tuple[object, ...]:
    """Return a stable cache key for direct U/F pySecDec generation."""
    source_payload = json.dumps(request.uf_topology or {}, sort_keys=True, default=str)
    return (
        source_payload,
        request.sector_method,
        request.normaliz_executable,
        request.prefactor_convention,
        request.jit_compile_evaluators,
        request.evaluator_compile_mode,
        request.real_evaluator,
        request.dual_evaluator_mode,
        request.subtraction_backend,
        request.sector_evaluator_backend,
        request.ibp_power_goal,
        request.max_eps_order,
    )


def get_uf_bundle(
    request: IntegralRequest,
    progress: GenerationProgress | None = None,
) -> UFBuildBundle:
    """Return the cached direct-U/F build bundle."""
    key = _request_cache_key(request)
    cached = _UF_BUNDLE_CACHE.get(key)
    if cached is not None:
        if progress is not None and progress.logger is not None:
            progress.logger.info("generation cache hit: reused direct U/F bundle")
        return cached
    timings = GenerationTimings()
    with timings.measure("U/F input load", progress=progress):
        source = uf_topology_data_from_request(request)
    bundle = build_uf_bundle(source, request, progress=progress)
    bundle.timings.records = [*timings.records, *bundle.timings.records]
    _UF_BUNDLE_CACHE[key] = bundle
    return bundle


def build_topology_from_uf_request(request: IntegralRequest) -> "TopologyDefinition":
    """Construct a topology from a direct U/F-backed request."""
    return get_uf_bundle(request).topology


def generate_sectors_from_uf_request(request: IntegralRequest) -> list["SectorDefinition"]:
    """Construct declarative sector definitions from direct U/F input."""
    return get_uf_bundle(request).sectors
