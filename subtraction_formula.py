"""Symbolica-owned endpoint-subtraction formula generation.

This module is intentionally self-contained: it constructs the complete
localized endpoint-subtraction formula from sector metadata and placeholder
Taylor coefficients, without substituting sector maps into U or F.

The generated formula depends only on:

* sector coordinates ``sf_y*``;
* black-box Taylor coefficients ``sf_u_*`` and ``sf_f_*``;
* regular-Jacobian Taylor coefficients ``sf_j_*``.

The algebraic work is delegated to Symbolica series/coefficient extraction and
replacement rules.  Python still enumerates the finite endpoint projectors and
Taylor multi-indices, but it no longer implements the regular-function Taylor
series, epsilon/log expansion, or Laurent coefficient convolution itself.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from functools import lru_cache
from itertools import product
import os
from pathlib import Path
import tempfile
from typing import Any

from cache_utils import formula_cache_dir, formula_cache_read_roots, mirror_cache_entry_to_primary
from symbolica import E, Evaluator, Replacement, S


ENDPOINT_PROJECTOR_CACHE_VERSION = 1
REGULAR_TAYLOR_CACHE_VERSION = 9
REGULAR_TAYLOR_COMPATIBLE_CACHE_VERSIONS = (9, 8)


class _RegularCacheEvaluatorRef:
    """Lazy proxy for a serialized regular-Taylor evaluator sidecar.

    Warm prepared-bundle generation should not deserialize hundreds of large
    Symbolica evaluators just to re-serialize them into the bundle.  This proxy
    keeps direct single-shot use working, while the bundle writer can copy the
    original sidecar bytes through ``RegularTaylorFormulaDefinition``'s
    ``cache_evaluator_files`` field.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._evaluator: Any | None = None

    def _load(self) -> Any:
        if self._evaluator is None:
            raw = self.path.read_bytes()
            self._evaluator = Evaluator.load(
                gzip.decompress(raw) if self.path.suffix == ".gz" else raw
            )
        return self._evaluator

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        return self._load().evaluate(*args, **kwargs)

    def evaluate_with_prec(self, *args: Any, **kwargs: Any) -> Any:
        return self._load().evaluate_with_prec(*args, **kwargs)

    def evaluate_complex(self, *args: Any, **kwargs: Any) -> Any:
        return self._load().evaluate_complex(*args, **kwargs)

    def evaluate_complex_with_prec(self, *args: Any, **kwargs: Any) -> Any:
        return self._load().evaluate_complex_with_prec(*args, **kwargs)


def build_subtraction_formula_symbolica(
    topology: Any,
    sector: Any,
    signature: tuple[Any, ...],
    formula_class: type,
) -> Any:
    """Build a pregenerated subtraction formula using Symbolica transformations.

    ``formula_class`` is injected by ``integrand.py`` to avoid a circular import:
    the class also owns the runtime evaluator helpers used by ``SectorProcessor``.
    """
    ctx = _FormulaContext(topology, sector, signature)
    outputs = ctx.build_outputs()
    evaluators = [
        expr.evaluator(ctx.input_symbols, jit_compile=topology.jit_compile_evaluators)
        for expr in outputs
    ]
    return formula_class(
        signature=signature,
        input_names=ctx.input_names,
        input_symbols=ctx.input_symbols,
        output_expressions=outputs,
        evaluators=evaluators,
        laurent_orders=topology.laurent_orders,
        zero_subsets=ctx.zero_subsets,
        dual_shape=list(sector.dual_shape),
    )


def build_endpoint_projector_formula_symbolica(
    topology: Any,
    sector: Any | None,
    signature: tuple[Any, ...],
    formula_class: type,
    ibp_reduce_to_log_endpoint: bool = False,
) -> Any:
    """Build the endpoint-only projector formula for a lower cache signature.

    This formula does not know how the regular function ``g_s`` is obtained.
    Its inputs are the sampled singular coordinates and precomputed
    ``g_{S,alpha,r}`` coefficients for every endpoint projector.  That makes
    the evaluator reusable across sectors that share only endpoint powers,
    Taylor orders, and Laurent range.
    """
    cached = _load_endpoint_projector_formula_from_cache(
        topology,
        signature,
        formula_class,
    )
    if cached is not None:
        _increment_topology_counter(topology, "endpoint_projector_formulas_from_cache")
        return cached

    ctx = _EndpointProjectorContext(
        topology,
        sector,
        signature,
        ibp_reduce_to_log_endpoint=ibp_reduce_to_log_endpoint,
    )
    outputs = ctx.build_outputs()
    evaluators = [
        expr.evaluator(ctx.input_symbols, jit_compile=topology.jit_compile_evaluators)
        for expr in outputs
    ]
    formula = formula_class(
        signature=signature,
        input_names=ctx.input_names,
        input_symbols=ctx.input_symbols,
        output_expressions=outputs,
        evaluators=evaluators,
        laurent_orders=topology.laurent_orders,
        zero_subsets=ctx.zero_subsets,
        taylor_orders=ctx.taylor_orders,
        coefficient_layout=ctx.coefficient_layout,
        ibp_reduce_to_log_endpoint=ibp_reduce_to_log_endpoint,
    )
    _write_endpoint_projector_formula_to_cache(formula)
    _increment_topology_counter(topology, "endpoint_projector_formulas_generated")
    return formula


def build_regular_taylor_formula_symbolica(
    topology: Any,
    sector: Any,
    signature: tuple[Any, ...],
    formula_class: type,
) -> Any:
    """Build a Symbolica evaluator for regular ``g_s`` Taylor coefficients.

    This is the companion to the lower-signature endpoint projector.  It still
    treats U and F as black boxes: the formula inputs are sector coordinates and
    already-computed Taylor coefficients of the mapped U, F, and regular
    Jacobian.  Symbolica owns only the algebra that combines those coefficients
    into the regular epsilon/Taylor coefficients ``g_{S,alpha,r}``.
    """
    cached = _load_regular_taylor_formula_from_cache(
        topology,
        signature,
        formula_class,
    )
    if cached is not None:
        _increment_topology_counter(topology, "regular_taylor_formulas_from_cache")
        return cached

    ctx = _RegularTaylorContext(topology, sector, signature)
    use_dualized_regular = ctx.uses_residual_inputs
    if use_dualized_regular and _regular_taylor_should_use_sparse_expression(ctx):
        formula = _build_regular_taylor_sparse_expression_formula(
            topology,
            ctx,
            signature,
            formula_class,
        )
        _write_regular_taylor_formula_to_cache(formula)
        _increment_topology_counter(topology, "regular_taylor_formulas_generated")
        return formula

    if use_dualized_regular:
        formula = _build_regular_taylor_dualized_formula(topology, ctx, signature, formula_class)
        _write_regular_taylor_formula_to_cache(formula)
        _increment_topology_counter(topology, "regular_taylor_formulas_generated")
        return formula

    outputs = ctx.build_outputs()
    evaluators = [
        expr.evaluator(ctx.input_symbols, jit_compile=topology.jit_compile_evaluators)
        for expr in outputs
    ]
    formula = formula_class(
        signature=signature,
        input_names=ctx.input_names,
        input_symbols=ctx.input_symbols,
        output_expressions=outputs,
        evaluators=evaluators,
        output_layout=ctx.output_layout,
        input_layout=ctx.input_layout,
        max_orders=ctx.max_orders,
        zero_positions=ctx.zero_positions,
    )
    _write_regular_taylor_formula_to_cache(formula)
    _increment_topology_counter(topology, "regular_taylor_formulas_generated")
    return formula


def _increment_topology_counter(topology: Any, name: str) -> None:
    """Increment an optional build/cache counter on ``TopologyDefinition``."""
    if hasattr(topology, name):
        setattr(topology, name, int(getattr(topology, name, 0)) + 1)


def _build_regular_taylor_dualized_formula(
    topology: Any,
    ctx: "_RegularTaylorContext",
    signature: tuple[Any, ...],
    formula_class: type,
) -> Any:
    """Build a v2 regular formula by dualizing one scalar Symbolica evaluator."""
    expr = ctx._regular_expression()
    dual_symbols = [ctx.eps, *ctx.taus]
    evaluator_symbols = [*dual_symbols, *ctx.input_symbols]
    output_layout: list[tuple[tuple[int, ...], int]] = []
    requested_dual_shape: list[tuple[int, ...]] = []
    for multi_index, regular_order in ctx.requested_outputs:
        output_layout.append((multi_index, regular_order))
        requested_dual_shape.append(
            (
                int(regular_order),
                *tuple(int(value) for value in multi_index),
                *tuple(0 for _ in ctx.input_symbols),
            )
        )
    dual_shape_set: set[tuple[int, ...]] = set()
    for target in requested_dual_shape:
        for ancestor in product(*[range(value + 1) for value in target]):
            dual_shape_set.add(tuple(int(value) for value in ancestor))
    zero = tuple(0 for _ in evaluator_symbols)
    dual_shape_set.add(zero)
    dual_shape = sorted(dual_shape_set, key=lambda item: (sum(item), item))
    if zero in dual_shape:
        dual_shape.remove(zero)
        dual_shape.insert(0, zero)
    dual_index = {multi: index for index, multi in enumerate(dual_shape)}
    output_indices = [dual_index[target] for target in requested_dual_shape]
    evaluator = expr.evaluator(
        evaluator_symbols,
        jit_compile=topology.jit_compile_evaluators,
    )
    evaluator.dualize([list(mi) for mi in dual_shape])
    return formula_class(
        signature=signature,
        input_names=ctx.input_names,
        input_symbols=ctx.input_symbols,
        output_expressions=[expr],
        evaluators=[evaluator],
        output_layout=output_layout,
        input_layout=ctx.input_layout,
        max_orders=ctx.max_orders,
        zero_positions=ctx.zero_positions,
        evaluator_input_symbols=evaluator_symbols,
        evaluator_dual_shape=dual_shape,
        evaluator_output_indices=output_indices,
        dual_variable_count=len(dual_symbols),
    )


def _regular_taylor_should_use_sparse_expression(ctx: "_RegularTaylorContext") -> bool:
    """Return whether to build residual-input coefficients explicitly.

    The dualized residual formula is compact to describe, but for high mixed
    derivatives it asks Symbolica for every ancestor coefficient of one scalar
    expression in ``eps`` and all Taylor variables.  Six-axis triple-box
    signatures such as ``(2,2,2,2,1,1)`` therefore spend offline cache time on a
    large dual box despite needing only a sparse set of output coefficients.
    The sparse expression path builds precisely those coefficient formulas.
    """
    return bool(ctx.version >= 3 and ctx.n_axes >= 6)


def _build_regular_taylor_sparse_expression_formula(
    topology: Any,
    ctx: "_RegularTaylorContext",
    signature: tuple[Any, ...],
    formula_class: type,
) -> Any:
    """Build sparse residual-input regular coefficients as Symbolica formulas."""
    allowed_multis = set(ctx.coefficient_multis)
    max_orders = list(ctx.max_orders)

    def coefficient_series(kind: str) -> ExprSeries:
        return {
            multi: ctx._coeff(kind, multi)
            for multi in ctx.coefficient_multis
        }

    j_series = coefficient_series("j")
    u_series = coefficient_series("u")
    f_series = coefficient_series("f")
    monomial_pref, monomial_log = ctx._regular_monomial_exprs()
    u_power, u_log = _expr_series_pow_real_and_log_allowed(
        u_series,
        topology.u_power_base,
        max_orders,
        allowed_multis,
    )
    f_power, f_log = _expr_series_pow_real_and_log_allowed(
        f_series,
        -topology.f_power_base,
        max_orders,
        allowed_multis,
    )
    pref_series = _expr_series_mul_allowed(
        _expr_series_constant(monomial_pref, max_orders),
        _expr_series_mul_allowed(
            j_series,
            _expr_series_mul_allowed(u_power, f_power, allowed_multis),
            allowed_multis,
        ),
        allowed_multis,
    )
    log_series = _expr_series_add(
        _expr_series_constant(monomial_log, max_orders),
        _expr_series_add(
            _expr_series_scale(u_log, topology.eps_log_u_coeff),
            _expr_series_scale(f_log, topology.eps_log_f_coeff),
        ),
    )

    by_eps_order: list[ExprSeries] = []
    log_power = _expr_series_constant(E("1"), max_orders)
    factorial = 1.0
    for regular_order in range(topology.coefficient_count):
        if regular_order > 0:
            factorial *= float(regular_order)
            log_power = _expr_series_mul_allowed(log_power, log_series, allowed_multis)
        by_eps_order.append(
            _expr_series_scale(
                _expr_series_mul_allowed(pref_series, log_power, allowed_multis),
                1.0 / factorial,
            )
        )

    outputs: list[Any] = []
    output_layout: list[tuple[tuple[int, ...], int]] = []
    for multi_index, regular_order in ctx.requested_outputs:
        outputs.append(
            _expr_series_coefficient(
                by_eps_order[int(regular_order)],
                tuple(int(value) for value in multi_index),
            )
        )
        output_layout.append((tuple(int(value) for value in multi_index), int(regular_order)))

    evaluators = [
        expr.evaluator(ctx.input_symbols, jit_compile=topology.jit_compile_evaluators)
        for expr in outputs
    ]
    return formula_class(
        signature=signature,
        input_names=ctx.input_names,
        input_symbols=ctx.input_symbols,
        output_expressions=outputs,
        evaluators=evaluators,
        output_layout=output_layout,
        input_layout=ctx.input_layout,
        max_orders=ctx.max_orders,
        zero_positions=ctx.zero_positions,
    )


def _endpoint_projector_cache_dir() -> Path:
    """Return the endpoint-projector formula cache directory."""
    configured = os.environ.get("FSD_SUBTRACTION_FORMULA_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return formula_cache_dir()


def _signature_payload(signature: tuple[Any, ...]) -> dict[str, Any]:
    """Return a JSON-stable topology-independent cache signature."""
    return {
        "schema_version": ENDPOINT_PROJECTOR_CACHE_VERSION,
        "kind": "endpoint-projector",
        "signature": _jsonable(signature),
    }


def _endpoint_projector_cache_path(signature: tuple[Any, ...]) -> Path:
    """Return the cache path for one endpoint-projector signature."""
    payload = _signature_payload(signature)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return _endpoint_projector_cache_dir() / f"endpoint_projector_{digest}.json"


def _cache_read_paths(path: Path) -> list[Path]:
    """Return generated and curated cache locations for one formula filename."""
    paths: list[Path] = []
    # Curated assets are treated as part of the FSD source/distribution cache.
    # They should take precedence over exploratory generated cache files with
    # the same signature, which may have been produced by older local runs.
    for root in formula_cache_read_roots():
        for candidate in (root / "curated" / path.name, root / path.name):
            if candidate not in paths:
                paths.append(candidate)
    return paths


def regular_taylor_formula_has_curated_cache(signature: tuple[Any, ...]) -> bool:
    """Return whether a vetted regular-Taylor asset exists for ``signature``."""
    path = _regular_taylor_cache_path(signature)
    curated = _endpoint_projector_cache_dir() / "curated" / path.name
    if not curated.is_file():
        return False
    try:
        data = json.loads(curated.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("signature_payload") == _regular_taylor_signature_payload(signature)


def regular_taylor_formula_has_cache(signature: tuple[Any, ...]) -> bool:
    """Return whether any readable generated or curated regular cache exists."""
    expected = _regular_taylor_signature_payload(signature)
    for path in _regular_taylor_cache_paths(signature):
        for candidate in _cache_read_paths(path):
            if not candidate.is_file():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            payload = data.get("signature_payload", {})
            if payload.get("kind") != expected["kind"] or payload.get("signature") != expected["signature"]:
                continue
            if (
                int(payload.get("schema_version", 0) or 0) < REGULAR_TAYLOR_CACHE_VERSION
                and len(signature) > 2
                and int(signature[1]) >= 3
                and int(signature[2]) >= 6
            ):
                continue
            return True
    return False


def endpoint_projector_formula_has_curated_cache(signature: tuple[Any, ...]) -> bool:
    """Return whether a vetted endpoint-projector asset exists for ``signature``."""
    return _endpoint_projector_formula_has_curated_cache(
        signature,
        str(_endpoint_projector_cache_dir()),
    )


@lru_cache(maxsize=None)
def _endpoint_projector_formula_has_curated_cache(
    signature: tuple[Any, ...],
    cache_dir: str,
) -> bool:
    """Cached implementation keyed by both signature and cache directory."""
    path = _endpoint_projector_cache_path(signature)
    curated = Path(cache_dir) / "curated" / path.name
    if not curated.is_file():
        return False
    try:
        data = json.loads(curated.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("signature_payload") == _signature_payload(signature)


endpoint_projector_formula_has_curated_cache.cache_clear = (  # type: ignore[attr-defined]
    _endpoint_projector_formula_has_curated_cache.cache_clear
)


def _regular_taylor_signature_payload(
    signature: tuple[Any, ...],
    schema_version: int | None = None,
) -> dict[str, Any]:
    """Return a JSON-stable cache signature for regular-Taylor formulae."""
    return {
        "schema_version": int(schema_version or REGULAR_TAYLOR_CACHE_VERSION),
        "kind": "regular-taylor",
        "signature": _jsonable(signature),
    }


def _regular_taylor_cache_path(
    signature: tuple[Any, ...],
    schema_version: int | None = None,
) -> Path:
    """Return the cache path for one regular-Taylor formula."""
    payload = _regular_taylor_signature_payload(signature, schema_version=schema_version)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return _endpoint_projector_cache_dir() / f"regular_taylor_{digest}.json"


def _regular_taylor_cache_paths(signature: tuple[Any, ...]) -> list[Path]:
    """Return current and compatible older cache paths for one signature."""
    paths: list[Path] = []
    for version in REGULAR_TAYLOR_COMPATIBLE_CACHE_VERSIONS:
        path = _regular_taylor_cache_path(signature, schema_version=version)
        if path not in paths:
            paths.append(path)
    return paths


def _jsonable(value: Any) -> Any:
    """Convert nested tuples and scalar values into stable JSON data."""
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tuple_deep(value: Any) -> Any:
    """Convert JSON lists back into tuples recursively for signatures/layouts."""
    if isinstance(value, list):
        return tuple(_tuple_deep(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuple_deep(item) for key, item in value.items()}
    return value


def _load_endpoint_projector_formula_from_cache(
    topology: Any,
    signature: tuple[Any, ...],
    formula_class: type,
) -> Any | None:
    """Load a cached endpoint-projector expression, if available."""
    path = _endpoint_projector_cache_path(signature)
    expected_orders: list[int] | None = None
    if (
        isinstance(signature, tuple)
        and len(signature) >= 7
        and signature[0] == "endpoint-projector"
    ):
        expected_orders = [int(order) for order in signature[6]]
    for candidate in _cache_read_paths(path):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if data.get("signature_payload") != _signature_payload(signature):
                continue
            laurent_orders = [int(order) for order in data["laurent_orders"]]
            if expected_orders is not None and laurent_orders != expected_orders:
                # Older exploratory cache files could carry the right signature
                # payload but only a truncated Laurent output range.  Accepting
                # those files makes high-order coefficients silently wrong, so
                # treat them as stale and continue to curated/generated hits.
                continue
            if len(data.get("output_expressions", [])) != len(laurent_orders):
                continue
            mirror_cache_entry_to_primary(candidate, data)
            input_names = [str(name) for name in data["input_names"]]
            input_symbols = [S(name) for name in input_names]
            outputs = [E(text) for text in data["output_expressions"]]
            evaluators = [
                expr.evaluator(input_symbols, jit_compile=topology.jit_compile_evaluators)
                for expr in outputs
            ]
            coefficient_layout = [
                _coefficient_key_from_json(item)
                for item in data["coefficient_layout"]
            ]
            return formula_class(
                signature=signature,
                input_names=input_names,
                input_symbols=input_symbols,
                output_expressions=outputs,
                evaluators=evaluators,
                laurent_orders=laurent_orders,
                zero_subsets=[tuple(int(x) for x in subset) for subset in data["zero_subsets"]],
                taylor_orders=[int(order) for order in data["taylor_orders"]],
                coefficient_layout=coefficient_layout,
                ibp_reduce_to_log_endpoint=bool(data.get("ibp_reduce_to_log_endpoint", False)),
            )
        except Exception:
            # A stale or hand-edited cache file should never make generation fail.
            # The freshly generated expression below will atomically replace it.
            continue
    return None


def _write_endpoint_projector_formula_to_cache(formula: Any) -> None:
    """Atomically store parseable expression strings for a projector formula."""
    path = _endpoint_projector_cache_path(formula.signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "signature_payload": _signature_payload(formula.signature),
        "input_names": list(formula.input_names),
        "output_expressions": [str(expr) for expr in formula.output_expressions],
        "laurent_orders": list(formula.laurent_orders),
        "zero_subsets": [list(subset) for subset in formula.zero_subsets],
        "taylor_orders": list(formula.taylor_orders),
        "coefficient_layout": [
            _coefficient_key_to_json(key) for key in formula.coefficient_layout
        ],
        "ibp_reduce_to_log_endpoint": bool(formula.ibp_reduce_to_log_endpoint),
    }
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _load_regular_taylor_formula_from_cache(
    topology: Any,
    signature: tuple[Any, ...],
    formula_class: type,
) -> Any | None:
    """Load a cached regular-Taylor expression, if available."""
    expected_signature = _regular_taylor_signature_payload(signature)["signature"]
    expected_kind = _regular_taylor_signature_payload(signature)["kind"]
    for path in _regular_taylor_cache_paths(signature):
        for candidate in _cache_read_paths(path):
            if not candidate.is_file():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                payload = data.get("signature_payload", {})
                if (
                    payload.get("kind") != expected_kind
                    or payload.get("signature") != expected_signature
                ):
                    continue
                if (
                    int(payload.get("schema_version", 0) or 0) < REGULAR_TAYLOR_CACHE_VERSION
                    and len(signature) > 2
                    and int(signature[1]) >= 3
                    and int(signature[2]) >= 6
                ):
                    # v8 six-axis residual caches used the dualized sparse
                    # expression.  v9 intentionally replaces those with
                    # direct sparse expressions so shipped caches do not pay
                    # the huge high-mixed-derivative dualization cost.
                    continue
                mirror_cache_entry_to_primary(
                    candidate,
                    data,
                    sidecar_fields=("evaluator_cache_files",),
                )
                mode = str(data.get("mode", "explicit"))
                input_names = [str(name) for name in data["input_names"]]
                input_symbols = [S(name) for name in input_names]
                cache_evaluator_files: list[str] = []
                if mode == "dualized":
                    evaluator_input_names = [
                        str(name) for name in data["evaluator_input_names"]
                    ]
                    evaluator_input_symbols = [S(name) for name in evaluator_input_names]
                    evaluator_dual_shape = [
                        tuple(int(value) for value in item)
                        for item in data["evaluator_dual_shape"]
                    ]
                    cached_evaluator_paths = _regular_evaluator_sidecar_paths(candidate, data)
                    if cached_evaluator_paths is not None:
                        outputs = []
                        evaluators = [
                            _RegularCacheEvaluatorRef(path)
                            for path in cached_evaluator_paths
                        ]
                        cache_evaluator_files = [
                            str(path) for path in cached_evaluator_paths
                        ]
                    else:
                        if "scalar_expression" not in data:
                            raise KeyError(
                                "regular Taylor cache has no scalar expression fallback"
                            )
                        outputs = [E(str(data["scalar_expression"]))]
                        evaluator = outputs[0].evaluator(
                            evaluator_input_symbols,
                            jit_compile=topology.jit_compile_evaluators,
                        )
                        evaluator.dualize([list(mi) for mi in evaluator_dual_shape])
                        evaluators = [evaluator]
                        _upgrade_regular_cache_with_evaluator_sidecars(
                            candidate,
                            data,
                            evaluators,
                        )
                else:
                    evaluator_input_symbols = []
                    evaluator_dual_shape = []
                    cached_evaluator_paths = _regular_evaluator_sidecar_paths(candidate, data)
                    if cached_evaluator_paths is not None:
                        outputs = []
                        evaluators = [
                            _RegularCacheEvaluatorRef(path)
                            for path in cached_evaluator_paths
                        ]
                        cache_evaluator_files = [
                            str(path) for path in cached_evaluator_paths
                        ]
                    else:
                        output_texts = _load_regular_output_expression_strings(candidate, data)
                        outputs = [E(text) for text in output_texts]
                        evaluators = [
                            expr.evaluator(
                                input_symbols,
                                jit_compile=topology.jit_compile_evaluators,
                            )
                            for expr in outputs
                        ]
                        _upgrade_regular_cache_with_evaluator_sidecars(
                            candidate,
                            data,
                            evaluators,
                        )
                return formula_class(
                    signature=signature,
                    input_names=input_names,
                    input_symbols=input_symbols,
                    output_expressions=outputs,
                    evaluators=evaluators,
                    output_layout=[
                        _regular_output_layout_from_json(item)
                        for item in data["output_layout"]
                    ],
                    input_layout=[
                        _regular_input_layout_from_json(item)
                        for item in data["input_layout"]
                    ],
                    max_orders=[int(order) for order in data["max_orders"]],
                    zero_positions=tuple(int(position) for position in data["zero_positions"]),
                    evaluator_input_symbols=evaluator_input_symbols,
                    evaluator_dual_shape=evaluator_dual_shape,
                    evaluator_output_indices=[
                        int(index) for index in data.get("evaluator_output_indices", [])
                    ],
                    dual_variable_count=int(data.get("dual_variable_count", 0)),
                    cache_evaluator_files=cache_evaluator_files,
                )
            except Exception:
                continue
    return None


def _regular_evaluator_sidecar_name(path: Path, index: int) -> str:
    """Return the generated evaluator-cache filename for one regular output."""
    return f"{path.stem}.eval_{int(index)}.bin.gz"


def _regular_expression_sidecar_name(path: Path) -> str:
    """Return the compressed reference-expression sidecar filename."""
    return f"{path.stem}.expr.json.gz"


def _load_regular_output_expression_strings(path: Path, data: dict[str, Any]) -> list[str]:
    """Load regular-output expression strings from JSON or a sidecar.

    Runtime-ready caches have serialized evaluator sidecars, so warm generation
    should not parse these strings.  They are retained as a rebuild/debug asset
    and loaded only when evaluator bytes are unavailable.
    """
    inline = data.get("output_expressions")
    if inline is not None:
        return [str(item) for item in inline]
    name = data.get("output_expression_cache_file")
    if not name:
        raise KeyError("regular Taylor cache has no output expressions")
    candidate = path.parent / str(name)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    raw = candidate.read_bytes()
    payload = gzip.decompress(raw) if candidate.suffix == ".gz" else raw
    decoded = json.loads(payload.decode("utf-8"))
    return [str(item) for item in decoded["output_expressions"]]


def _regular_evaluator_sidecar_paths(path: Path, data: dict[str, Any]) -> list[Path] | None:
    """Return serialized regular-Taylor evaluator sidecar paths if complete."""
    names = [str(name) for name in data.get("evaluator_cache_files", [])]
    if not names:
        return None
    paths: list[Path] = []
    for name in names:
        candidate = path.parent / name
        if not candidate.is_file() and candidate.suffix != ".gz":
            compressed = candidate.with_name(candidate.name + ".gz")
            if compressed.is_file():
                candidate = compressed
        if not candidate.is_file():
            return None
        paths.append(candidate)
    return paths


def _write_regular_evaluator_sidecars(path: Path, evaluators: list[Any]) -> list[str]:
    """Atomically store serialized regular-Taylor evaluator bytes.

    These sidecars are generated-cache artifacts, not curated source assets.
    They allow interrupted triple-box cache-warming runs to resume without
    paying the expensive Symbolica evaluator lowering cost again.
    """
    names: list[str] = []
    for index, evaluator in enumerate(evaluators):
        name = _regular_evaluator_sidecar_name(path, index)
        destination = path.parent / name
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(gzip.compress(evaluator.save(), compresslevel=6))
            os.replace(tmp_name, destination)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        names.append(name)
    return names


def _write_regular_expression_sidecar(path: Path, expressions: list[Any]) -> str:
    """Store large reference expression strings outside the metadata JSON."""
    name = _regular_expression_sidecar_name(path)
    destination = path.parent / name
    payload = json.dumps(
        {"output_expressions": [str(expr) for expr in expressions]},
        separators=(",", ":"),
    ).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(gzip.compress(payload, compresslevel=6))
        os.replace(tmp_name, destination)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return name


def _regular_cache_candidate_is_curated(path: Path) -> bool:
    """Return whether a cache path is part of the curated source assets."""
    return "curated" in {part.lower() for part in path.parts}


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Atomically write one JSON cache payload."""
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _upgrade_regular_cache_with_evaluator_sidecars(
    path: Path,
    data: dict[str, Any],
    evaluators: list[Any],
) -> None:
    """Persist evaluator bytes for an existing generated expression cache."""
    if data.get("evaluator_cache_files") or _regular_cache_candidate_is_curated(path):
        return
    try:
        data["evaluator_cache_files"] = _write_regular_evaluator_sidecars(path, evaluators)
        _write_json_atomic(path, data)
    except Exception:
        # The expression JSON remains a valid fallback cache even if evaluator
        # byte serialization is unavailable for this Symbolica build.
        data.pop("evaluator_cache_files", None)


def _write_regular_taylor_formula_to_cache(formula: Any) -> None:
    """Atomically store parseable expression strings for a regular formula."""
    path = _regular_taylor_cache_path(formula.signature)
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluator_cache_files = _write_regular_evaluator_sidecars(path, list(formula.evaluators))
    data = {
        "signature_payload": _regular_taylor_signature_payload(formula.signature),
        "mode": "dualized" if getattr(formula, "evaluator_dual_shape", None) else "explicit",
        "input_names": list(formula.input_names),
        "evaluator_cache_files": evaluator_cache_files,
        "output_layout": [
            _regular_output_layout_to_json(item) for item in formula.output_layout
        ],
        "input_layout": [
            _regular_input_layout_to_json(item) for item in formula.input_layout
        ],
        "max_orders": list(formula.max_orders),
        "zero_positions": list(formula.zero_positions),
    }
    if getattr(formula, "evaluator_dual_shape", None):
        data["scalar_expression_omitted"] = True
        data["scalar_expression_count"] = len(formula.output_expressions)
        data["evaluator_input_names"] = [
            str(symbol) for symbol in formula.evaluator_input_symbols
        ]
        data["evaluator_dual_shape"] = [
            [int(value) for value in multi_index]
            for multi_index in formula.evaluator_dual_shape
        ]
        data["evaluator_output_indices"] = [
            int(index) for index in getattr(formula, "evaluator_output_indices", [])
        ]
        data["dual_variable_count"] = int(formula.dual_variable_count)
    else:
        data["output_expression_omitted"] = True
        data["output_expression_count"] = len(formula.output_expressions)
    _write_json_atomic(path, data)


def _regular_output_layout_to_json(key: tuple[tuple[int, ...], int]) -> dict[str, Any]:
    """Serialize one regular-Taylor output descriptor."""
    multi, regular_order = key
    return {
        "multi": [int(value) for value in multi],
        "regular_order": int(regular_order),
    }


def _regular_output_layout_from_json(data: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    """Deserialize one regular-Taylor output descriptor."""
    return (
        tuple(int(value) for value in data.get("multi", [])),
        int(data.get("regular_order", 0)),
    )


def _regular_input_layout_to_json(key: tuple[str, tuple[int, ...]]) -> dict[str, Any]:
    """Serialize one regular-Taylor input coefficient descriptor."""
    kind, multi = key
    return {
        "kind": str(kind),
        "multi": [int(value) for value in multi],
    }


def _regular_input_layout_from_json(data: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    """Deserialize one regular-Taylor input coefficient descriptor."""
    return (
        str(data.get("kind", "")),
        tuple(int(value) for value in data.get("multi", [])),
    )


def _coefficient_key_to_json(key: tuple[Any, ...]) -> dict[str, Any]:
    """Serialize one regular-coefficient input descriptor."""
    boundary, zero, multi, regular_order = _normalise_coefficient_key(key)
    return {
        "boundary": list(boundary),
        "zero": list(zero),
        "multi": list(multi),
        "regular_order": int(regular_order),
    }


def _coefficient_key_from_json(data: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    """Deserialize one regular-coefficient input descriptor."""
    return (
        tuple(int(x) for x in data.get("boundary", [])),
        tuple(int(x) for x in data.get("zero", [])),
        tuple(int(x) for x in data.get("multi", [])),
        int(data.get("regular_order", 0)),
    )


def _normalise_coefficient_key(key: tuple[Any, ...]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    """Accept legacy three-field keys and return the four-field layout."""
    if len(key) == 3:
        zero, multi, regular_order = key
        boundary: tuple[int, ...] = ()
    elif len(key) == 4:
        boundary, zero, multi, regular_order = key
    else:
        raise ValueError(f"invalid endpoint-projector coefficient key: {key!r}")
    return (
        tuple(int(x) for x in boundary),
        tuple(int(x) for x in zero),
        tuple(int(x) for x in multi),
        int(regular_order),
    )


class _FormulaContext:
    """State container for one generated subtraction formula."""

    def __init__(self, topology: Any, sector: Any, signature: tuple[Any, ...]) -> None:
        self.topology = topology
        self.sector = sector
        self.signature = signature
        self.axes = list(sector.singular_axes)
        self.n_axes = len(self.axes)
        self.eps = S("sf_eps")
        self.taus = [S(f"sf_tau{position}") for position in range(self.n_axes)]
        self.y_symbols = [S(f"sf_y{axis}") for axis in range(sector.integration_dim)]
        self.bases, self.eps_coeffs, self.taylor_orders = self._endpoint_power_data()
        self.zero_subsets = [
            tuple(position for position, bit in enumerate(bits) if bit)
            for bits in product((False, True), repeat=self.n_axes)
        ]
        self.input_names = [f"sf_y{axis}" for axis in range(sector.integration_dim)]
        self.input_symbols = list(self.y_symbols)
        self.coeff_symbols: dict[tuple[str, tuple[int, ...], tuple[int, ...]], Any] = {}
        self._build_coefficient_symbols()
        self._g_cache: dict[tuple[int, ...], dict[tuple[int, ...], Any]] = {}

    def _endpoint_power_data(self) -> tuple[list[int], list[float], list[int]]:
        bases: list[int] = []
        eps_coeffs: list[float] = []
        taylor_orders: list[int] = []
        for axis in self.axes:
            endpoint_power = self.topology.endpoint_power(self.sector, axis)
            rounded_base = round(endpoint_power.base)
            if endpoint_power.base >= -1.0e-12:
                raise ValueError(
                    f"{self.sector.name}: declared singular axis "
                    f"{self.sector.variable_names[axis]} has non-singular endpoint "
                    f"power y^({endpoint_power.as_text()})"
                )
            if abs(endpoint_power.base - rounded_base) > 1.0e-12:
                raise ValueError(
                    f"{self.sector.name}: unsupported non-integer endpoint power "
                    f"y^({endpoint_power.as_text()})"
                )
            if abs(endpoint_power.eps_coeff) <= 1.0e-15:
                raise ValueError(
                    f"{self.sector.name}: endpoint power y^({endpoint_power.as_text()}) "
                    "has no epsilon regulator"
                )
            required_order = int(-rounded_base - 1)
            declared_order = int(self.sector.endpoint_taylor_orders[axis])
            if declared_order < required_order:
                raise ValueError(
                    f"{self.sector.name}: endpoint Taylor order {declared_order} on "
                    f"{self.sector.variable_names[axis]} is too small; need {required_order}"
                )
            bases.append(int(rounded_base))
            eps_coeffs.append(float(endpoint_power.eps_coeff))
            taylor_orders.append(required_order)
        return bases, eps_coeffs, taylor_orders

    def _build_coefficient_symbols(self) -> None:
        for subset in self.zero_subsets:
            mask = _subset_mask(subset)
            for kind in ("j", "u", "f"):
                for multi_index in self.sector.dual_shape:
                    name = f"sf_{kind}_{mask}_{_multi_suffix(multi_index)}"
                    symbol = S(name)
                    self.coeff_symbols[(kind, subset, multi_index)] = symbol
                    self.input_names.append(name)
                    self.input_symbols.append(symbol)

    def build_outputs(self) -> list[Any]:
        """Return Symbolica expressions for all requested Laurent coefficients."""
        total_expr = E("0")
        position_range = list(range(self.n_axes))
        for integrated_flags in product((False, True), repeat=self.n_axes):
            integrated_positions = [
                position for position, flag in enumerate(integrated_flags) if flag
            ]
            active_positions = [
                position for position, flag in enumerate(integrated_flags) if not flag
            ]
            active_base, active_eps_log = self._active_endpoint_factor(active_positions)
            for taylor_flags in product((False, True), repeat=len(active_positions)):
                projected_positions = [
                    position
                    for position, flag in zip(active_positions, taylor_flags)
                    if flag
                ]
                sign = -1 if len(projected_positions) % 2 else 1
                zero_positions = set(integrated_positions) | set(projected_positions)
                g_coefficients = self._regular_function_coefficients(tuple(sorted(zero_positions)))
                max_multi_orders = [
                    self.taylor_orders[position] if position in zero_positions else 0
                    for position in position_range
                ]
                for multi_index in _multi_indices(max_multi_orders):
                    term = _expr_number(sign) * active_base
                    term *= self._projected_coordinate_factor(projected_positions, multi_index)
                    term *= self._integrated_denominator_expr(integrated_positions, multi_index)
                    if active_positions:
                        term *= (self.eps * active_eps_log).exp()
                    regular_coeff = g_coefficients.get(multi_index)
                    if regular_coeff is None:
                        continue
                    term *= regular_coeff
                    total_expr += term

        eps_series = total_expr.series(
            self.eps,
            0,
            self.topology.coefficient_count,
            depth_is_absolute=False,
        )
        return [
            eps_series.get_coefficient(order)
            for order in self.topology.laurent_orders
        ]

    def _active_endpoint_factor(self, active_positions: list[int]) -> tuple[Any, Any]:
        base = E("1")
        eps_log = E("0")
        for position in active_positions:
            coord = self.y_symbols[self.axes[position]]
            base *= _expr_int_power(coord, self.bases[position])
            eps_log += _expr_number(self.eps_coeffs[position]) * coord.log()
        return base, eps_log

    def _projected_coordinate_factor(
        self,
        projected_positions: list[int],
        multi_index: tuple[int, ...],
    ) -> Any:
        out = E("1")
        for position in projected_positions:
            order = int(multi_index[position])
            if order:
                out *= _expr_int_power(self.y_symbols[self.axes[position]], order)
        return out

    def _integrated_denominator_expr(
        self,
        integrated_positions: list[int],
        multi_index: tuple[int, ...],
    ) -> Any:
        out = E("1")
        for position in integrated_positions:
            offset = self.bases[position] + int(multi_index[position]) + 1
            out /= _expr_number(offset) + _expr_number(self.eps_coeffs[position]) * self.eps
        return out

    def _regular_function_coefficients(self, subset: tuple[int, ...]) -> dict[tuple[int, ...], Any]:
        cached = self._g_cache.get(subset)
        if cached is not None:
            return cached
        max_orders = [
            self.taylor_orders[position] if position in subset else 0
            for position in range(self.n_axes)
        ]
        j_expr = self._jacobian_taylor_expr(subset, max_orders)
        u_expr = self._residual_taylor_expr(
            "u", subset, self.sector.u_monomial_powers, max_orders
        )
        f_expr = self._residual_taylor_expr(
            "f", subset, self.sector.f_monomial_powers, max_orders
        )
        monomial_pref, monomial_log = self._regular_monomial_exprs()
        expr = self._instantiate_regular_template(
            j_expr,
            u_expr,
            f_expr,
            monomial_pref,
            monomial_log,
        )
        expr = expr.series(
            self.eps,
            0,
            self.topology.coefficient_count - 1,
        ).to_expression()
        for tau, max_order in zip(self.taus, max_orders):
            if max_order:
                expr = expr.series(tau, 0, max_order).to_expression()
        coeffs = {
            multi: _coefficient_multi(expr, self.taus, multi)
            for multi in _multi_indices(max_orders)
        }
        self._g_cache[subset] = coeffs
        return coeffs

    def _instantiate_regular_template(
        self,
        j_expr: Any,
        u_expr: Any,
        f_expr: Any,
        monomial_pref: Any,
        monomial_log: Any,
    ) -> Any:
        placeholder_j = S("sf_template_J")
        placeholder_u = S("sf_template_U")
        placeholder_f = S("sf_template_F")
        placeholder_m = S("sf_template_M")
        placeholder_l = S("sf_template_L")
        template = placeholder_m * placeholder_j
        template *= _expr_real_power(placeholder_u, self.topology.u_power_base)
        template *= _expr_real_power(placeholder_f, -self.topology.f_power_base)
        epsilon_log = (
            placeholder_l
            + _expr_number(self.topology.eps_log_u_coeff) * placeholder_u.log()
            + _expr_number(self.topology.eps_log_f_coeff) * placeholder_f.log()
        )
        template *= (self.eps * epsilon_log).exp()
        return template.replace_multiple(
            [
                Replacement(placeholder_j, j_expr),
                Replacement(placeholder_u, u_expr),
                Replacement(placeholder_f, f_expr),
                Replacement(placeholder_m, monomial_pref),
                Replacement(placeholder_l, monomial_log),
            ]
        )

    def _jacobian_taylor_expr(self, subset: tuple[int, ...], max_orders: list[int]) -> Any:
        out = E("0")
        for multi in _multi_indices(max_orders):
            out += self._coeff("j", subset, multi) * _tau_monomial(self.taus, multi)
        return out

    def _residual_taylor_expr(
        self,
        kind: str,
        subset: tuple[int, ...],
        monomial_powers: list[int],
        max_orders: list[int],
    ) -> Any:
        axis_position = {axis: position for position, axis in enumerate(self.axes)}
        out = E("0")
        for residual_multi in _multi_indices(max_orders):
            polynomial_multi = [0 for _ in self.axes]
            denominator = E("1")
            for axis, power_value in enumerate(monomial_powers):
                position = axis_position.get(axis)
                power = int(power_value)
                if position is not None and position in subset:
                    polynomial_multi[position] = power + int(residual_multi[position])
                elif power:
                    denominator *= _expr_int_power(self.y_symbols[axis], power)
            out += (
                self._coeff(kind, subset, tuple(polynomial_multi))
                / denominator
                * _tau_monomial(self.taus, residual_multi)
            )
        return out

    def _regular_monomial_exprs(self) -> tuple[Any, Any]:
        singular = set(self.sector.singular_axes)
        base_value = E("1")
        eps_log = E("0")
        for axis in range(self.sector.integration_dim):
            if axis in singular:
                continue
            endpoint_power = self.topology.endpoint_power(self.sector, axis)
            coord = self.y_symbols[axis]
            if abs(endpoint_power.base) > 1.0e-15:
                base_value *= _expr_int_power(
                    coord,
                    _integer_coordinate_power(
                        endpoint_power.base,
                        f"{self.sector.name}:{self.sector.variable_names[axis]} regular monomial",
                    ),
                )
            if abs(endpoint_power.eps_coeff) > 1.0e-15:
                eps_log += _expr_number(endpoint_power.eps_coeff) * coord.log()
        return base_value, eps_log

    def _coeff(self, kind: str, subset: tuple[int, ...], multi_index: tuple[int, ...]) -> Any:
        return self.coeff_symbols.get((kind, subset, multi_index), E("0"))


class _EndpointProjectorContext:
    """State container for one endpoint-only subtraction projector."""

    def __init__(
        self,
        topology: Any,
        sector: Any | None,
        signature: tuple[Any, ...],
        ibp_reduce_to_log_endpoint: bool = False,
    ) -> None:
        self.topology = topology
        self.sector = sector
        self.signature = signature
        self.ibp_reduce_to_log_endpoint = bool(ibp_reduce_to_log_endpoint)
        if (
            isinstance(signature, tuple)
            and len(signature) >= 7
            and signature[0] == "endpoint-projector"
        ):
            self.n_axes = int(signature[3])
            self.axes = list(range(self.n_axes)) if sector is None else list(sector.singular_axes)
        else:
            if sector is None:
                raise ValueError("sector-free endpoint formula requires a v2 signature")
            self.axes = list(sector.singular_axes)
            self.n_axes = len(self.axes)
        self.eps = S("ep_eps")
        self.y_symbols = [S(f"ep_y{position}") for position in range(self.n_axes)]
        self.bases, self.eps_coeffs, self.taylor_orders = self._endpoint_power_data()
        self.zero_subsets = [
            tuple(position for position, bit in enumerate(bits) if bit)
            for bits in product((False, True), repeat=self.n_axes)
        ]
        self.input_names = [f"ep_y{position}" for position in range(self.n_axes)]
        self.input_symbols = list(self.y_symbols)
        self.coeff_symbols: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int],
            Any,
        ] = {}
        self.coefficient_layout: list[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]
        ] = []
        if not self.ibp_reduce_to_log_endpoint:
            self._build_coefficient_symbols()

    def _endpoint_power_data(self) -> tuple[list[int], list[float], list[int]]:
        bases: list[int] = []
        eps_coeffs: list[float] = []
        taylor_orders: list[int] = []
        if (
            isinstance(self.signature, tuple)
            and len(self.signature) >= 7
            and self.signature[0] == "endpoint-projector"
        ):
            endpoint_powers = self.signature[4]
            declared_orders = self.signature[5]
            for endpoint_power, declared_order in zip(endpoint_powers, declared_orders):
                base, eps_coeff = endpoint_power
                rounded_base = round(float(base))
                if float(base) >= -1.0e-12:
                    raise ValueError(
                        f"endpoint projector signature has non-singular power {base!r}"
                    )
                if abs(float(base) - rounded_base) > 1.0e-12:
                    raise ValueError(
                        f"endpoint projector signature has non-integer power {base!r}"
                    )
                if abs(float(eps_coeff)) <= 1.0e-15:
                    raise ValueError(
                        f"endpoint projector signature has no epsilon regulator: {endpoint_power!r}"
                    )
                required_order = int(-rounded_base - 1)
                if int(declared_order) < required_order:
                    raise ValueError(
                        f"endpoint projector signature Taylor order {declared_order} "
                        f"is too small; need {required_order}"
                    )
                bases.append(int(rounded_base))
                eps_coeffs.append(float(eps_coeff))
                taylor_orders.append(required_order)
            return bases, eps_coeffs, taylor_orders
        if self.sector is None:
            raise ValueError("sector-free endpoint formula requires endpoint powers in signature")
        for axis in self.axes:
            endpoint_power = self.topology.endpoint_power(self.sector, axis)
            rounded_base = round(endpoint_power.base)
            if endpoint_power.base >= -1.0e-12:
                raise ValueError(
                    f"{self.sector.name}: declared singular axis "
                    f"{self.sector.variable_names[axis]} has non-singular endpoint "
                    f"power y^({endpoint_power.as_text()})"
                )
            if abs(endpoint_power.base - rounded_base) > 1.0e-12:
                raise ValueError(
                    f"{self.sector.name}: unsupported non-integer endpoint power "
                    f"y^({endpoint_power.as_text()})"
                )
            if abs(endpoint_power.eps_coeff) <= 1.0e-15:
                raise ValueError(
                    f"{self.sector.name}: endpoint power y^({endpoint_power.as_text()}) "
                    "has no epsilon regulator"
                )
            required_order = int(-rounded_base - 1)
            declared_order = int(self.sector.endpoint_taylor_orders[axis])
            if declared_order < required_order:
                raise ValueError(
                    f"{self.sector.name}: endpoint Taylor order {declared_order} on "
                    f"{self.sector.variable_names[axis]} is too small; need {required_order}"
                )
            bases.append(int(rounded_base))
            eps_coeffs.append(float(endpoint_power.eps_coeff))
            taylor_orders.append(required_order)
        return bases, eps_coeffs, taylor_orders

    def _build_coefficient_symbols(self) -> None:
        for subset in self.zero_subsets:
            max_orders = [
                self.taylor_orders[position] if position in subset else 0
                for position in range(self.n_axes)
            ]
            for multi_index in _multi_indices(max_orders):
                for regular_order in range(self.topology.coefficient_count):
                    self._coeff((), subset, multi_index, regular_order)

    def build_outputs(self) -> list[Any]:
        """Return Symbolica expressions for all requested Laurent coefficients."""
        if self.ibp_reduce_to_log_endpoint:
            return self._build_outputs_ibp()

        total_expr = E("0")
        position_range = list(range(self.n_axes))
        for integrated_flags in product((False, True), repeat=self.n_axes):
            integrated_positions = [
                position for position, flag in enumerate(integrated_flags) if flag
            ]
            active_positions = [
                position for position, flag in enumerate(integrated_flags) if not flag
            ]
            active_base, active_eps_log = self._active_endpoint_factor(active_positions)
            for taylor_flags in product((False, True), repeat=len(active_positions)):
                projected_positions = [
                    position
                    for position, flag in zip(active_positions, taylor_flags)
                    if flag
                ]
                sign = -1 if len(projected_positions) % 2 else 1
                zero_positions = tuple(
                    sorted(set(integrated_positions) | set(projected_positions))
                )
                max_multi_orders = [
                    self.taylor_orders[position] if position in zero_positions else 0
                    for position in position_range
                ]
                for multi_index in _multi_indices(max_multi_orders):
                    term = _expr_number(sign) * active_base
                    term *= self._projected_coordinate_factor(projected_positions, multi_index)
                    term *= self._integrated_denominator_expr(integrated_positions, multi_index)
                    if active_positions:
                        term *= (self.eps * active_eps_log).exp()
                    term *= self._regular_eps_series((), zero_positions, multi_index)
                    total_expr += term

        eps_series = total_expr.series(
            self.eps,
            0,
            self.topology.coefficient_count,
            depth_is_absolute=False,
        )
        return [
            eps_series.get_coefficient(order)
            for order in self.topology.laurent_orders
        ]

    def _regular_eps_series(
        self,
        boundary_subset: tuple[int, ...],
        subset: tuple[int, ...],
        multi_index: tuple[int, ...],
    ) -> Any:
        out = E("0")
        for regular_order in range(self.topology.coefficient_count):
            symbol = self._coeff(boundary_subset, subset, multi_index, regular_order)
            if regular_order == 0:
                out += symbol
            else:
                out += symbol * _expr_int_power(self.eps, regular_order)
        return out

    def _coeff(
        self,
        boundary_subset: tuple[int, ...],
        zero_subset: tuple[int, ...],
        multi_index: tuple[int, ...],
        regular_order: int,
    ) -> Any:
        """Return or create one regular Taylor/Laurent coefficient symbol."""
        boundary = tuple(sorted(int(position) for position in boundary_subset))
        zero = tuple(sorted(int(position) for position in zero_subset))
        multi = tuple(int(value) for value in multi_index)
        key = (boundary, zero, multi, int(regular_order))
        symbol = self.coeff_symbols.get(key)
        if symbol is not None:
            return symbol
        name = (
            f"ep_g_b{_subset_mask(boundary)}"
            f"_z{_subset_mask(zero)}"
            f"_{_multi_suffix(multi)}"
            f"_{int(regular_order)}"
        )
        symbol = S(name)
        self.coeff_symbols[key] = symbol
        self.coefficient_layout.append(key)
        self.input_names.append(name)
        self.input_symbols.append(symbol)
        return symbol

    def _active_endpoint_factor(self, active_positions: list[int]) -> tuple[Any, Any]:
        base = E("1")
        eps_log = E("0")
        for position in active_positions:
            coord = self.y_symbols[position]
            base *= _expr_int_power(coord, self.bases[position])
            eps_log += _expr_number(self.eps_coeffs[position]) * coord.log()
        return base, eps_log

    def _projected_coordinate_factor(
        self,
        projected_positions: list[int],
        multi_index: tuple[int, ...],
    ) -> Any:
        out = E("1")
        for position in projected_positions:
            order = int(multi_index[position])
            if order:
                out *= _expr_int_power(self.y_symbols[position], order)
        return out

    def _integrated_denominator_expr(
        self,
        integrated_positions: list[int],
        multi_index: tuple[int, ...],
    ) -> Any:
        out = E("1")
        for position in integrated_positions:
            offset = self.bases[position] + int(multi_index[position]) + 1
            out /= _expr_number(offset) + _expr_number(self.eps_coeffs[position]) * self.eps
        return out

    def _build_outputs_ibp(self) -> list[Any]:
        """Return outputs after IBP-lowering all higher endpoint powers.

        Each original axis ``y^(-n+c eps)`` is analytically continued to a sum
        of boundary-at-one terms plus a logarithmic endpoint integral carrying
        ``n-1`` derivatives of the regular function.  The remaining logarithmic
        endpoints are then handled by the same localized plus-projector
        inclusion-exclusion used by the non-IBP formula.
        """
        total_expr = E("0")
        for prefactor, boundary_subset, derivative_multi, active_positions in self._ibp_terms():
            active_positions = list(active_positions)
            for integrated_flags in product((False, True), repeat=len(active_positions)):
                integrated_positions = [
                    position
                    for position, flag in zip(active_positions, integrated_flags)
                    if flag
                ]
                live_positions = [
                    position
                    for position, flag in zip(active_positions, integrated_flags)
                    if not flag
                ]
                active_base, active_eps_log = self._log_endpoint_factor(live_positions)
                for taylor_flags in product((False, True), repeat=len(live_positions)):
                    projected_positions = [
                        position
                        for position, flag in zip(live_positions, taylor_flags)
                        if flag
                    ]
                    sign = -1 if len(projected_positions) % 2 else 1
                    zero_positions = tuple(
                        sorted(set(integrated_positions) | set(projected_positions))
                    )
                    term = _expr_number(sign) * prefactor * active_base
                    term *= self._log_integrated_denominator_expr(integrated_positions)
                    if live_positions:
                        term *= (self.eps * active_eps_log).exp()
                    term *= _expr_number(_multi_factorial(derivative_multi))
                    term *= self._regular_eps_series(
                        tuple(sorted(boundary_subset)),
                        zero_positions,
                        derivative_multi,
                    )
                    total_expr += term

        eps_series = total_expr.series(
            self.eps,
            0,
            self.topology.coefficient_count,
            depth_is_absolute=False,
        )
        return [
            eps_series.get_coefficient(order)
            for order in self.topology.laurent_orders
        ]

    def _ibp_terms(
        self,
    ) -> list[tuple[Any, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """Enumerate topology-independent IBP terms for all endpoint axes."""
        zero_multi = tuple(0 for _ in range(self.n_axes))
        terms: list[tuple[Any, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = [
            (E("1"), (), zero_multi, tuple(range(self.n_axes)))
        ]
        for position, base in enumerate(self.bases):
            required_order = int(-int(base) - 1)
            if required_order <= 0:
                continue
            eps_coeff = self.eps_coeffs[position]
            next_terms: list[
                tuple[Any, tuple[int, ...], tuple[int, ...], tuple[int, ...]]
            ] = []
            for prefactor, boundary_subset, derivative_multi, active_positions in terms:
                if position not in active_positions:
                    next_terms.append((prefactor, boundary_subset, derivative_multi, active_positions))
                    continue
                denominators: list[Any] = []
                for shift in range(required_order):
                    offset = int(base) + shift + 1
                    denominators.append(
                        _expr_number(offset) + _expr_number(eps_coeff) * self.eps
                    )
                    boundary_derivative = list(derivative_multi)
                    boundary_derivative[position] += shift
                    boundary_prefactor = prefactor * _expr_number((-1) ** shift)
                    for denominator in denominators:
                        boundary_prefactor /= denominator
                    next_terms.append(
                        (
                            boundary_prefactor,
                            tuple(sorted((*boundary_subset, position))),
                            tuple(boundary_derivative),
                            tuple(active for active in active_positions if active != position),
                        )
                    )
                continuing_derivative = list(derivative_multi)
                continuing_derivative[position] += required_order
                continuing_prefactor = prefactor * _expr_number((-1) ** required_order)
                for denominator in denominators:
                    continuing_prefactor /= denominator
                next_terms.append(
                    (
                        continuing_prefactor,
                        boundary_subset,
                        tuple(continuing_derivative),
                        active_positions,
                    )
                )
            terms = next_terms
        return terms

    def _log_endpoint_factor(self, active_positions: list[int]) -> tuple[Any, Any]:
        """Return the sampled factor for already-lowered logarithmic endpoints."""
        base = E("1")
        eps_log = E("0")
        for position in active_positions:
            coord = self.y_symbols[position]
            base /= coord
            eps_log += _expr_number(self.eps_coeffs[position]) * coord.log()
        return base, eps_log

    def _log_integrated_denominator_expr(self, integrated_positions: list[int]) -> Any:
        """Return analytic denominators for logarithmic endpoint projectors."""
        out = E("1")
        for position in integrated_positions:
            out /= _expr_number(self.eps_coeffs[position]) * self.eps
        return out


class _RegularTaylorContext:
    """Symbolica expression builder for regular-function Taylor coefficients."""

    def __init__(self, topology: Any, sector: Any, signature: tuple[Any, ...]) -> None:
        self.topology = topology
        self.sector = sector
        self.signature = signature
        if len(signature) < 2 or signature[0] != "regular-taylor":
            raise ValueError(f"invalid regular Taylor signature: {signature!r}")
        self.version = int(signature[1])
        self.uses_residual_inputs = self.version >= 2
        if self.uses_residual_inputs:
            if len(signature) < 9:
                raise ValueError(f"invalid regular Taylor residual-input signature: {signature!r}")
            self.integration_dim = 0
            self.axes = list(range(int(signature[2])))
            self.n_axes = int(signature[2])
            self.zero_positions = ()
            if self.version >= 3:
                requested_outputs = [
                    (tuple(int(value) for value in multi), int(regular_order))
                    for multi, regular_order in signature[3]
                ]
                if not requested_outputs:
                    requested_outputs = [
                        (tuple(0 for _ in range(self.n_axes)), 0)
                    ]
                self.requested_outputs = sorted(
                    set(requested_outputs),
                    key=lambda item: (item[1], sum(item[0]), item[0]),
                )
                self.coefficient_multis = _ancestor_closed_multis(
                    [multi for multi, _regular_order in self.requested_outputs],
                    self.n_axes,
                )
                self.max_orders = [
                    max((multi[position] for multi in self.coefficient_multis), default=0)
                    for position in range(self.n_axes)
                ]
            else:
                self.max_orders = [int(order) for order in signature[3]]
                self.requested_outputs = [
                    (multi, int(regular_order))
                    for regular_order in range(topology.coefficient_count)
                    for multi in _multi_indices(self.max_orders)
                ]
                self.coefficient_multis = _multi_indices(self.max_orders)
            self.regular_endpoint_powers: list[tuple[float, float]] = []
        else:
            if len(signature) < 15:
                raise ValueError(f"invalid regular Taylor v1 signature: {signature!r}")
            self.integration_dim = int(signature[2])
            self.axes = [int(axis) for axis in signature[3]]
            self.n_axes = len(self.axes)
            self.zero_positions = tuple(int(position) for position in signature[11])
            self.max_orders = [int(order) for order in signature[12]]
            self.requested_outputs = [
                (multi, int(regular_order))
                for regular_order in range(topology.coefficient_count)
                for multi in _multi_indices(self.max_orders)
            ]
            self.coefficient_multis = _multi_indices(self.max_orders)
            self.regular_endpoint_powers = [
                (float(base), float(eps_coeff)) for base, eps_coeff in signature[13]
            ]
        self.eps = S("rg_eps")
        self.taus = [S(f"rg_tau{position}") for position in range(self.n_axes)]
        self.y_symbols = [S(f"rg_y{axis}") for axis in range(self.integration_dim)]
        self.input_names = [f"rg_y{axis}" for axis in range(self.integration_dim)]
        self.input_symbols = list(self.y_symbols)
        self.output_layout: list[tuple[tuple[int, ...], int]] = []
        self.input_layout: list[tuple[str, tuple[int, ...]]] = []
        self.coeff_symbols: dict[tuple[str, tuple[int, ...]], Any] = {}
        self.monomial_pref_symbol = S("rg_monomial_pref")
        self.monomial_log_symbol = S("rg_monomial_log")
        if self.uses_residual_inputs:
            self.input_names.extend(["rg_monomial_pref", "rg_monomial_log"])
            self.input_symbols.extend([self.monomial_pref_symbol, self.monomial_log_symbol])

    def build_outputs(self) -> list[Any]:
        """Return expressions for all requested regular Taylor coefficients."""
        expr = self._regular_expression()
        expr = expr.series(
            self.eps,
            0,
            self.topology.coefficient_count - 1,
        ).to_expression()
        for tau, max_order in zip(self.taus, self.max_orders):
            if max_order:
                expr = expr.series(tau, 0, max_order).to_expression()

        coefficient_map = _coefficient_list_map(
            expr,
            [self.eps, *self.taus],
            max_orders=[self.topology.coefficient_count - 1, *self.max_orders],
        )
        outputs: list[Any] = []
        for multi_index, regular_order in self.requested_outputs:
            outputs.append(
                coefficient_map.get(
                    (int(regular_order), *tuple(int(value) for value in multi_index)),
                    E("0"),
                )
            )
            self.output_layout.append((multi_index, regular_order))
        return outputs

    def _regular_expression(self) -> Any:
        j_expr = self._jacobian_taylor_expr()
        u_expr = self._residual_taylor_expr("u", self.sector.u_monomial_powers)
        f_expr = self._residual_taylor_expr("f", self.sector.f_monomial_powers)
        monomial_pref, monomial_log = self._regular_monomial_exprs()
        template_j = S("rg_template_J")
        template_u = S("rg_template_U")
        template_f = S("rg_template_F")
        template_m = S("rg_template_M")
        template_l = S("rg_template_L")
        expr = template_m * template_j
        expr *= _expr_real_power(template_u, self.topology.u_power_base)
        expr *= _expr_real_power(template_f, -self.topology.f_power_base)
        epsilon_log = (
            template_l
            + _expr_number(self.topology.eps_log_u_coeff) * template_u.log()
            + _expr_number(self.topology.eps_log_f_coeff) * template_f.log()
        )
        expr *= (self.eps * epsilon_log).exp()
        return expr.replace_multiple(
            [
                Replacement(template_j, j_expr),
                Replacement(template_u, u_expr),
                Replacement(template_f, f_expr),
                Replacement(template_m, monomial_pref),
                Replacement(template_l, monomial_log),
            ]
        )

    def _jacobian_taylor_expr(self) -> Any:
        out = E("0")
        for multi in self.coefficient_multis:
            out += self._coeff("j", multi) * _tau_monomial(self.taus, multi)
        return out

    def _residual_taylor_expr(self, kind: str, monomial_powers: list[int]) -> Any:
        if self.uses_residual_inputs:
            out = E("0")
            for multi in self.coefficient_multis:
                out += self._coeff(kind, multi) * _tau_monomial(self.taus, multi)
            return out

        axis_position = {axis: position for position, axis in enumerate(self.axes)}
        zero_positions = set(self.zero_positions)
        out = E("0")
        for residual_multi in _multi_indices(self.max_orders):
            polynomial_multi = [0 for _ in self.axes]
            denominator = E("1")
            for axis, power_value in enumerate(monomial_powers):
                position = axis_position.get(axis)
                power = int(power_value)
                if position is not None and position in zero_positions:
                    polynomial_multi[position] = power + int(residual_multi[position])
                elif power:
                    denominator *= _expr_int_power(self.y_symbols[axis], power)
            out += (
                self._coeff(kind, tuple(polynomial_multi))
                / denominator
                * _tau_monomial(self.taus, residual_multi)
            )
        return out

    def _regular_monomial_exprs(self) -> tuple[Any, Any]:
        if self.uses_residual_inputs:
            return self.monomial_pref_symbol, self.monomial_log_symbol

        singular = set(self.axes)
        base_value = E("1")
        eps_log = E("0")
        for axis, (base, eps_coeff) in enumerate(self.regular_endpoint_powers):
            if axis in singular:
                continue
            coord = self.y_symbols[axis]
            if abs(base) > 1.0e-15:
                base_value *= _expr_int_power(
                    coord,
                    _integer_coordinate_power(base, f"regular Taylor axis {axis}"),
                )
            if abs(eps_coeff) > 1.0e-15:
                eps_log += _expr_number(eps_coeff) * coord.log()
        return base_value, eps_log

    def _coeff(self, kind: str, multi_index: tuple[int, ...]) -> Any:
        multi = tuple(int(value) for value in multi_index)
        key = (kind, multi)
        symbol = self.coeff_symbols.get(key)
        if symbol is not None:
            return symbol
        name = f"rg_{kind}_{_multi_suffix(multi)}"
        symbol = S(name)
        self.coeff_symbols[key] = symbol
        self.input_layout.append(key)
        self.input_names.append(name)
        self.input_symbols.append(symbol)
        return symbol


def _multi_indices(max_orders: list[int]) -> list[tuple[int, ...]]:
    if not max_orders:
        return [()]
    ranges = [range(int(order) + 1) for order in max_orders]
    return [tuple(values) for values in product(*ranges)]


def _ancestor_closed_multis(
    multi_indices: list[tuple[int, ...]],
    rank: int,
) -> list[tuple[int, ...]]:
    """Return the component-wise ancestor closure of a sparse multi-index set."""
    closed: set[tuple[int, ...]] = set()
    zero = tuple(0 for _ in range(rank))
    closed.add(zero)
    for multi in multi_indices:
        if len(multi) != rank:
            raise ValueError(f"regular Taylor output rank mismatch: {multi!r}")
        for ancestor in product(*[range(int(value) + 1) for value in multi]):
            closed.add(tuple(int(value) for value in ancestor))
    ordered = sorted(closed, key=lambda item: (sum(item), item))
    if zero in ordered:
        ordered.remove(zero)
        ordered.insert(0, zero)
    return ordered


def _ancestor_closed_output_pairs(
    output_pairs: list[tuple[tuple[int, ...], int]],
    rank: int,
) -> list[tuple[tuple[int, ...], int]]:
    """Return ancestor closure in the combined ``(eps,tau...)`` dual shape."""
    closed: set[tuple[tuple[int, ...], int]] = set()
    for multi, regular_order in output_pairs:
        if len(multi) != rank:
            raise ValueError(f"regular Taylor output rank mismatch: {multi!r}")
        for eps_order in range(int(regular_order) + 1):
            for ancestor in product(*[range(int(value) + 1) for value in multi]):
                closed.add((tuple(int(value) for value in ancestor), int(eps_order)))
    zero = (tuple(0 for _ in range(rank)), 0)
    closed.add(zero)
    return sorted(closed, key=lambda item: (item[1], sum(item[0]), item[0]))


def _multi_factorial(multi_index: tuple[int, ...]) -> int:
    """Return the product of factorials for a multi-index."""
    out = 1
    for value in multi_index:
        factor = 1
        for integer in range(2, int(value) + 1):
            factor *= integer
        out *= factor
    return out


def _subset_mask(subset: tuple[int, ...]) -> str:
    return "none" if not subset else "_".join(str(index) for index in subset)


def _multi_suffix(multi_index: tuple[int, ...]) -> str:
    return "m" + "_".join(str(index) for index in multi_index)


def _expr_number(value: float | int | complex) -> Any:
    z = complex(value)
    if abs(z.imag) > 0.0:
        raise ValueError(f"complex constants are not supported in generated formulas: {value!r}")
    if abs(z.real) == 0.0:
        return E("0")
    if abs(z.real - 1.0) <= 0.0:
        return E("1")
    if abs(z.real + 1.0) <= 0.0:
        return E("-1")
    rounded = round(z.real)
    if abs(z.real - rounded) <= 1.0e-15:
        return E(str(int(rounded)))
    return E(repr(float(z.real)))


def _expr_int_power(base: Any, power: int) -> Any:
    integer_power = int(power)
    if integer_power == 0:
        return E("1")
    if integer_power > 0:
        return base ** integer_power
    return E("1") / (base ** abs(integer_power))


def _expr_real_power(base: Any, power: float) -> Any:
    if abs(power) <= 1.0e-15:
        return E("1")
    rounded = round(float(power))
    if abs(float(power) - rounded) <= 1.0e-12:
        return _expr_int_power(base, int(rounded))
    return (_expr_number(power) * base.log()).exp()


ExprSeries = dict[tuple[int, ...], Any]


def _zero_multi(rank: int) -> tuple[int, ...]:
    """Return the zero multi-index for a Taylor rank."""
    return tuple(0 for _ in range(int(rank)))


def _expr_series_constant(value: Any, max_orders: list[int]) -> ExprSeries:
    """Build a sparse Symbolica Taylor series with one constant term."""
    return {_zero_multi(len(max_orders)): value if hasattr(value, "evaluator") else _expr_number(value)}


def _expr_series_add(left: ExprSeries, right: ExprSeries) -> ExprSeries:
    """Add two sparse Symbolica Taylor series."""
    out = dict(left)
    for key, value in right.items():
        out[key] = out[key] + value if key in out else value
    return out


def _expr_series_scale(series: ExprSeries, factor: float | int | Any) -> ExprSeries:
    """Scale every coefficient of a sparse Symbolica Taylor series."""
    factor_expr = factor if hasattr(factor, "evaluator") else _expr_number(factor)
    return {key: value * factor_expr for key, value in series.items()}


def _expr_series_filter_allowed(
    series: ExprSeries,
    allowed_multis: set[tuple[int, ...]],
) -> ExprSeries:
    """Return a copy with coefficients outside ``allowed_multis`` removed."""
    return {key: value for key, value in series.items() if key in allowed_multis}


def _expr_series_mul_allowed(
    left: ExprSeries,
    right: ExprSeries,
    allowed_multis: set[tuple[int, ...]],
) -> ExprSeries:
    """Multiply sparse series and keep only declared ancestor support."""
    if not left or not right or not allowed_multis:
        return {}
    rank = len(next(iter(allowed_multis)))
    out: ExprSeries = {}
    for left_key, left_value in left.items():
        for right_key, right_value in right.items():
            key = tuple(
                int(left_key[index]) + int(right_key[index])
                for index in range(rank)
            )
            if key not in allowed_multis:
                continue
            term = left_value * right_value
            out[key] = out[key] + term if key in out else term
    return out


def _expr_series_log_allowed(
    series: ExprSeries,
    max_orders: list[int],
    allowed_multis: set[tuple[int, ...]],
) -> ExprSeries:
    """Compute ``log(series)`` on sparse ancestor support."""
    zero = _zero_multi(len(max_orders))
    constant = series[zero]
    out = _expr_series_constant(constant.log(), max_orders)
    h = {
        key: value / constant
        for key, value in series.items()
        if key != zero and key in allowed_multis
    }
    if not h:
        return _expr_series_filter_allowed(out, allowed_multis)
    h_power = h
    for order in range(1, sum(max_orders) + 1):
        sign = 1.0 if order % 2 == 1 else -1.0
        out = _expr_series_add(out, _expr_series_scale(h_power, sign / float(order)))
        h_power = _expr_series_mul_allowed(h_power, h, allowed_multis)
        if not h_power:
            break
    return _expr_series_filter_allowed(out, allowed_multis)


def _expr_series_exp_allowed(
    series: ExprSeries,
    max_orders: list[int],
    allowed_multis: set[tuple[int, ...]],
) -> ExprSeries:
    """Compute ``exp(series)`` on sparse ancestor support."""
    zero = _zero_multi(len(max_orders))
    constant = series.get(zero, E("0"))
    h = {
        key: value
        for key, value in series.items()
        if key != zero and key in allowed_multis
    }
    total = _expr_series_constant(E("1"), max_orders)
    if h:
        h_power = h
        factorial = 1.0
        for order in range(1, sum(max_orders) + 1):
            factorial *= float(order)
            total = _expr_series_add(
                total,
                _expr_series_scale(h_power, 1.0 / factorial),
            )
            h_power = _expr_series_mul_allowed(h_power, h, allowed_multis)
            if not h_power:
                break
    return _expr_series_mul_allowed(
        _expr_series_constant(constant.exp(), max_orders),
        _expr_series_filter_allowed(total, allowed_multis),
        allowed_multis,
    )


def _binomial_integer(exponent: int, order: int) -> float:
    """Return the generalized binomial coefficient for an integer exponent."""
    if order <= 0:
        return 1.0
    numerator = 1.0
    for step in range(int(order)):
        numerator *= float(int(exponent) - step)
    denominator = 1.0
    for step in range(1, int(order) + 1):
        denominator *= float(step)
    return numerator / denominator


def _expr_series_integer_power_allowed(
    series: ExprSeries,
    exponent: int,
    max_orders: list[int],
    allowed_multis: set[tuple[int, ...]],
) -> ExprSeries:
    """Raise a sparse series to an integer power on ancestor support."""
    zero = _zero_multi(len(max_orders))
    constant = series[zero]
    if exponent == 0:
        return _expr_series_filter_allowed(
            _expr_series_constant(E("1"), max_orders),
            allowed_multis,
        )
    if exponent > 0:
        out = _expr_series_filter_allowed(
            _expr_series_constant(E("1"), max_orders),
            allowed_multis,
        )
        base = _expr_series_filter_allowed(series, allowed_multis)
        remaining = int(exponent)
        while remaining:
            if remaining & 1:
                out = _expr_series_mul_allowed(out, base, allowed_multis)
            remaining >>= 1
            if remaining:
                base = _expr_series_mul_allowed(base, base, allowed_multis)
        return out

    h = {
        key: value / constant
        for key, value in series.items()
        if key != zero and key in allowed_multis
    }
    total = _expr_series_filter_allowed(
        _expr_series_constant(E("1"), max_orders),
        allowed_multis,
    )
    if h:
        h_power = h
        for order in range(1, sum(max_orders) + 1):
            coeff = _binomial_integer(int(exponent), order)
            if coeff:
                total = _expr_series_add(total, _expr_series_scale(h_power, coeff))
            h_power = _expr_series_mul_allowed(h_power, h, allowed_multis)
            if not h_power:
                break
    return _expr_series_mul_allowed(
        _expr_series_constant(_expr_int_power(constant, int(exponent)), max_orders),
        total,
        allowed_multis,
    )


def _expr_series_pow_real_and_log_allowed(
    series: ExprSeries,
    power: float,
    max_orders: list[int],
    allowed_multis: set[tuple[int, ...]],
) -> tuple[ExprSeries, ExprSeries]:
    """Return ``series**power`` and ``log(series)`` on sparse support."""
    log_series = _expr_series_log_allowed(series, max_orders, allowed_multis)
    rounded = round(float(power))
    if abs(float(power) - rounded) <= 1.0e-12:
        return (
            _expr_series_integer_power_allowed(
                series,
                int(rounded),
                max_orders,
                allowed_multis,
            ),
            log_series,
        )
    return (
        _expr_series_exp_allowed(
            _expr_series_scale(log_series, power),
            max_orders,
            allowed_multis,
        ),
        log_series,
    )


def _expr_series_coefficient(series: ExprSeries, multi_index: tuple[int, ...]) -> Any:
    """Return one sparse Symbolica coefficient, or zero if absent."""
    return series.get(tuple(int(value) for value in multi_index), E("0"))


def _integer_coordinate_power(value: float, label: str) -> int:
    rounded = round(float(value))
    if abs(float(value) - rounded) > 1.0e-12:
        raise ValueError(
            f"{label}: generated subtraction formula requires integer coordinate powers, "
            f"got {value!r}"
        )
    return int(rounded)


def _tau_monomial(tau_symbols: list[Any], multi_index: tuple[int, ...]) -> Any:
    out = E("1")
    for tau, power in zip(tau_symbols, multi_index):
        if power:
            out *= _expr_int_power(tau, int(power))
    return out


def _coefficient_multi(expr: Any, tau_symbols: list[Any], multi_index: tuple[int, ...]) -> Any:
    out = expr
    for tau, order in zip(tau_symbols, multi_index):
        order_int = int(order)
        out = out.series(tau, 0, order_int).get_coefficient(order_int)
    return out


def _coefficient_list_map(
    expr: Any,
    variables: list[Any],
    max_orders: list[int],
) -> dict[tuple[int, ...], Any]:
    """Return exact polynomial coefficients keyed by exponent tuple.

    ``Expression.coefficient_list`` extracts all monomial coefficients in one
    pass.  The generated regular-Taylor variables have simple ASCII names, so
    the returned monomial keys can be decoded without symbolic pattern matching.
    This avoids repeatedly calling ``series`` for every epsilon/Taylor output.
    """
    variable_names = [str(variable) for variable in variables]
    out: dict[tuple[int, ...], Any] = {}
    for monomial, coefficient in expr.coefficient_list(*variables):
        powers = _coefficient_list_monomial_powers(monomial, variable_names)
        if len(powers) != len(max_orders):
            continue
        if any(power < 0 or power > int(limit) for power, limit in zip(powers, max_orders)):
            continue
        out[powers] = coefficient
    return out


def _coefficient_list_monomial_powers(
    monomial: Any,
    variable_names: list[str],
) -> tuple[int, ...]:
    """Decode a Symbolica monomial key returned by ``coefficient_list``."""
    text = str(monomial)
    powers = [0 for _ in variable_names]
    if text == "1":
        return tuple(powers)
    index_by_name = {name: index for index, name in enumerate(variable_names)}
    for factor in text.split("*"):
        if "^" in factor:
            name, power_text = factor.split("^", 1)
            power = int(power_text)
        else:
            name = factor
            power = 1
        index = index_by_name.get(name)
        if index is None:
            raise ValueError(
                f"unexpected coefficient-list monomial factor {factor!r} in {text!r}"
            )
        powers[index] += power
    return tuple(powers)
