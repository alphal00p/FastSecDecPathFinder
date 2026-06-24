"""pySecDec make_package-style two-polynomial input for FSD sectors."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
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


def _required(mapping: dict[str, Any], key: str, *, context: str = "package-integrand") -> Any:
    """Fetch a required nested YAML value."""
    value = _get(mapping, key)
    if value is None:
        raise ValueError(f"{context} is missing required key {key!r}")
    return value


def _affine_from_text(text: Any, *, key: str) -> EpsilonExpansion:
    """Parse ``base + eps_coeff*eps`` from a Symbolica expression string."""
    try:
        evaluator = E(str(text)).evaluator([S("eps")])
        base = float(evaluator.evaluate([[0.0]])[0][0])
        at_one = float(evaluator.evaluate([[1.0]])[0][0])
    except Exception as exc:
        raise ValueError(f"package-integrand key {key!r} must be affine in eps: {text!r}") from exc
    return EpsilonExpansion(base=base, eps_coeff=at_one - base)


def _run_base_dir(request: IntegralRequest) -> Path:
    """Return the directory used for package-integrand relative paths."""
    if request.run_file is None:
        return Path.cwd()
    return Path(request.run_file).expanduser().resolve().parent


def _string_sequence(value: Any, *, key: str) -> list[str]:
    """Parse a YAML sequence as non-empty strings."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"package-integrand key {key!r} must be a non-empty sequence")
    out = [str(item) for item in value]
    if len(set(out)) != len(out):
        raise ValueError(f"package-integrand key {key!r} must contain unique entries")
    return out


def _read_polynomial_expression(entry: dict[str, Any], *, base_dir: Path, index: int) -> str:
    """Return one inline or file-backed polynomial expression."""
    context = f"package-integrand polynomials-to-decompose[{index}]"
    expression = _get(entry, "expression")
    expression_file = _get(entry, "expression-file")
    if expression is None and expression_file is None:
        raise ValueError(f"{context} needs 'expression' or 'expression-file'")
    if expression is not None and expression_file is not None:
        raise ValueError(f"{context} must use only one of 'expression' or 'expression-file'")
    if expression is not None:
        text = str(expression).strip()
    else:
        path = Path(str(expression_file)).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.is_file():
            raise ValueError(f"{context} expression-file does not exist: {path}")
        text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{context} expression must not be empty")
    return _pysecdec_to_symbolica_text(text)


def _pysecdec_to_symbolica_text(text: str) -> str:
    """Convert pySecDec/Python power syntax to the Symbolica syntax used by FSD."""
    return str(text).replace("**", "^")


def package_integrand_data_from_request(request: IntegralRequest) -> UFTopologyData:
    """Normalize the pySecDec-style package-integrand YAML block."""
    data = request.package_integrand
    if not isinstance(data, dict):
        raise ValueError("package topology mode requires a package-integrand YAML mapping")

    x_names = _string_sequence(_required(data, "integration-variables"), key="integration-variables")
    regulators = _string_sequence(_required(data, "regulators"), key="regulators")
    if regulators != ["eps"]:
        raise ValueError("package-integrand currently supports only regulators: [eps]")
    requested_orders = _required(data, "requested-orders")
    if not isinstance(requested_orders, (list, tuple)) or not requested_orders:
        raise ValueError("package-integrand key 'requested-orders' must be a non-empty sequence")
    try:
        [int(order) for order in requested_orders]
    except Exception as exc:
        raise ValueError("package-integrand requested-orders must be integers") from exc

    raw_polynomials = _required(data, "polynomials-to-decompose")
    if not isinstance(raw_polynomials, (list, tuple)):
        raise ValueError("package-integrand key 'polynomials-to-decompose' must be a sequence")
    if len(raw_polynomials) != 2:
        raise ValueError(
            "package-integrand currently requires exactly two polynomials-to-decompose"
        )

    base_dir = _run_base_dir(request)
    expressions: list[str] = []
    exponents: list[EpsilonExpansion] = []
    for index, raw_entry in enumerate(raw_polynomials):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"package-integrand polynomials-to-decompose[{index}] must be a mapping"
            )
        expressions.append(_read_polynomial_expression(raw_entry, base_dir=base_dir, index=index))
        exponents.append(
            _affine_from_text(
                _required(
                    raw_entry,
                    "exponent",
                    context=f"package-integrand polynomials-to-decompose[{index}]",
                ),
                key=f"polynomials-to-decompose[{index}].exponent",
            )
        )

    loop_count = int(_required(data, "loop-count"))
    if loop_count <= 0:
        raise ValueError("package-integrand key 'loop-count' must be positive")

    return UFTopologyData(
        family=str(_get(data, "family", _get(data, "name", "package integrand"))),
        x_names=x_names,
        parameter_names=[],
        parameter_values=[],
        u_expr_text=expressions[0],
        f_expr_text=expressions[1],
        loop_count=loop_count,
        dimension=_affine_from_text(_required(data, "dimension"), key="dimension"),
        propagator_powers=tuple(1.0 for _ in x_names),
        measure_powers=tuple(0.0 for _ in x_names),
        u_exponent=exponents[0],
        f_exponent=exponents[1],
        global_prefactor=str(_required(data, "global-prefactor")),
    )


_PACKAGE_BUNDLE_CACHE: dict[tuple[object, ...], UFBuildBundle] = {}


def clear_package_bundle_cache() -> None:
    """Drop in-process package-integrand build bundles."""
    _PACKAGE_BUNDLE_CACHE.clear()


def _request_cache_key(source: UFTopologyData, request: IntegralRequest) -> tuple[object, ...]:
    """Return a stable cache key for package-integrand generation."""
    source_payload = json.dumps(asdict(source), sort_keys=True, default=str)
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


def get_package_bundle(
    request: IntegralRequest,
    progress: GenerationProgress | None = None,
) -> UFBuildBundle:
    """Return the cached package-integrand build bundle."""
    timings = GenerationTimings()
    with timings.measure("package-integrand input load", progress=progress):
        source = package_integrand_data_from_request(request)
    key = _request_cache_key(source, request)
    cached = _PACKAGE_BUNDLE_CACHE.get(key)
    if cached is not None:
        if progress is not None and progress.logger is not None:
            progress.logger.info("generation cache hit: reused package-integrand bundle")
        return cached
    bundle = build_uf_bundle(source, request, progress=progress)
    bundle.timings.records = [*timings.records, *bundle.timings.records]
    _PACKAGE_BUNDLE_CACHE[key] = bundle
    return bundle


def build_topology_from_package_request(request: IntegralRequest) -> "TopologyDefinition":
    """Construct a topology from a package-integrand request."""
    return get_package_bundle(request).topology


def generate_sectors_from_package_request(request: IntegralRequest) -> list["SectorDefinition"]:
    """Construct declarative sector definitions from package-integrand input."""
    return get_package_bundle(request).sectors
