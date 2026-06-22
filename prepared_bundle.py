"""Prepared DOT bundle serialization for two-stage FSD runs.

The ``generate`` command writes a strict runtime bundle containing declarative
topology/sector metadata plus serialized Symbolica evaluator bytes.  The
``integrate`` command loads only those artifacts: it does not call pySecDec,
does not rebuild U/F expressions, and does not generate subtraction formulas.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

from symbolica import E, Evaluator, S

from definitions import EpsilonExpansion, IntegralRequest, ParametricRepresentation
from integrand import (
    ChainRuleFormulaDefinition,
    EndpointProjectorFormulaDefinition,
    ExplicitSectorFormulaDefinition,
    IBPEndpointProjectorTerm,
    RegularTaylorFormulaDefinition,
    SubtractionFormulaDefinition,
    TopologyDefinition,
    TwoStageSectorFormulaDefinition,
)
from sectors_generator import SectorDefinition


SCHEMA_VERSION = 2


def _json_default(value: Any) -> Any:
    """Encode values that the stdlib JSON module does not know about."""
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, tuple):
        return {"__tuple__": [_json_default(item) for item in value]}
    if isinstance(value, Path):
        return str(value)
    return value


def _encode(value: Any) -> Any:
    """Recursively encode tuples/complex values into stable JSON data."""
    if isinstance(value, complex):
        return {"__complex__": [float(value.real), float(value.imag)]}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    return value


def _decode(value: Any) -> Any:
    """Inverse of ``_encode`` for JSON payloads."""
    if isinstance(value, dict):
        if "__complex__" in value:
            real, imag = value["__complex__"]
            return complex(float(real), float(imag))
        if "__tuple__" in value:
            return tuple(_decode(item) for item in value["__tuple__"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _signature_key(signature: Any) -> str:
    """Return a deterministic short key for a formula/evaluator signature."""
    payload = json.dumps(_encode(signature), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON through a temporary file so incomplete bundles are obvious."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _collect_evaluator_paths(value: Any) -> list[str]:
    """Return all evaluator artifact paths mentioned in a JSON payload."""
    paths: list[str] = []
    if (
        isinstance(value, str)
        and value.startswith("evaluators/")
        and (value.endswith(".bin") or value.endswith(".bin.gz"))
    ):
        paths.append(value)
    elif isinstance(value, list):
        for item in value:
            paths.extend(_collect_evaluator_paths(item))
    elif isinstance(value, dict):
        for item in value.values():
            paths.extend(_collect_evaluator_paths(item))
    return paths


def _assert_evaluator_artifacts_exist(root: Path, *payloads: Any) -> None:
    """Fail before integration when a referenced evaluator byte file is absent."""
    missing: list[str] = []
    for payload in payloads:
        for relative_path in _collect_evaluator_paths(payload):
            if not (root / relative_path).is_file():
                missing.append(relative_path)
    if missing:
        preview = ", ".join(sorted(set(missing))[:10])
        more = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise RuntimeError(f"prepared bundle is missing evaluator artifact(s): {preview}{more}")


def _expr_text(expr: Any) -> str:
    """Serialize a Symbolica expression as a parseable reference string."""
    return str(expr)


def _expr_from_text(text: str) -> Any:
    """Parse a reference expression string back into a Symbolica expression."""
    return E(str(text))


def _expansion_to_json(value: EpsilonExpansion) -> dict[str, float]:
    return {"base": float(value.base), "eps_coeff": float(value.eps_coeff)}


def _expansion_from_json(value: dict[str, Any]) -> EpsilonExpansion:
    return EpsilonExpansion(float(value["base"]), float(value["eps_coeff"]))


def _parametric_to_json(value: ParametricRepresentation | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "loop_count": int(value.loop_count),
        "propagator_powers": list(value.propagator_powers),
        "dimension": _expansion_to_json(value.dimension),
        "gamma_argument": _expansion_to_json(value.gamma_argument),
        "u_exponent": _expansion_to_json(value.u_exponent),
        "f_exponent": _expansion_to_json(value.f_exponent),
        "parameter_weight_powers": list(value.parameter_weight_powers),
        "prefactor_description": value.prefactor_description,
        "convention_description": value.convention_description,
    }


def _parametric_from_json(value: dict[str, Any] | None) -> ParametricRepresentation | None:
    if value is None:
        return None
    return ParametricRepresentation(
        loop_count=int(value["loop_count"]),
        propagator_powers=tuple(float(item) for item in value["propagator_powers"]),
        dimension=_expansion_from_json(value["dimension"]),
        gamma_argument=_expansion_from_json(value["gamma_argument"]),
        u_exponent=_expansion_from_json(value["u_exponent"]),
        f_exponent=_expansion_from_json(value["f_exponent"]),
        parameter_weight_powers=tuple(float(item) for item in value["parameter_weight_powers"]),
        prefactor_description=str(value["prefactor_description"]),
        convention_description=str(value["convention_description"]),
    )


class PreparedEvaluatorStore:
    """Lazy LRU loader for serialized Symbolica evaluator artifacts."""

    def __init__(self, root: Path, lru_size: int = 128) -> None:
        self.root = Path(root)
        self.lru_size = int(lru_size)
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def load(self, relative_path: str) -> Any:
        """Load an evaluator from the bundle, evicting old entries if needed."""
        key = str(relative_path)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        path = self.root / key
        if not path.is_file():
            raise RuntimeError(f"prepared evaluator artifact is missing: {path}")
        raw = path.read_bytes()
        evaluator = Evaluator.load(gzip.decompress(raw) if path.suffix == ".gz" else raw)
        self._cache[key] = evaluator
        if self.lru_size > 0:
            while len(self._cache) > self.lru_size:
                self._cache.popitem(last=False)
        return evaluator


class LazyEvaluatorRef:
    """Small proxy exposing the Symbolica evaluator API from an artifact path."""

    def __init__(self, store: PreparedEvaluatorStore, relative_path: str) -> None:
        self.store = store
        self.relative_path = str(relative_path)

    def _evaluator(self) -> Any:
        return self.store.load(self.relative_path)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        return self._evaluator().evaluate(*args, **kwargs)

    def evaluate_with_prec(self, *args: Any, **kwargs: Any) -> Any:
        return self._evaluator().evaluate_with_prec(*args, **kwargs)

    def evaluate_complex(self, *args: Any, **kwargs: Any) -> Any:
        return self._evaluator().evaluate_complex(*args, **kwargs)

    def evaluate_complex_with_prec(self, *args: Any, **kwargs: Any) -> Any:
        return self._evaluator().evaluate_complex_with_prec(*args, **kwargs)

    def get_instructions(self, *args: Any, **kwargs: Any) -> Any:
        return self._evaluator().get_instructions(*args, **kwargs)


class _BundleWriter:
    """Helper owning evaluator file naming during bundle creation."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.evaluator_dir = root / "evaluators"
        self.expression_dir = root / "expressions"
        self.evaluator_dir.mkdir(parents=True, exist_ok=True)
        self.expression_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def save_evaluator(self, evaluator: Any, group: str, key: Any) -> str:
        """Serialize one evaluator and return a path relative to the bundle."""
        cached_path = getattr(evaluator, "cache_evaluator_file", None)
        if cached_path:
            return self.copy_evaluator_file(cached_path, group, key)
        self._counter += 1
        safe_group = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in group)
        digest = _signature_key((group, key, self._counter))
        rel = Path("evaluators") / f"{safe_group}_{digest}.bin.gz"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(evaluator.save(), compresslevel=6))
        return str(rel)

    def copy_evaluator_file(self, source: str | Path, group: str, key: Any) -> str:
        """Copy an already serialized evaluator artifact into the bundle."""
        source_path = Path(source)
        if not source_path.is_file():
            raise RuntimeError(f"cached evaluator artifact is missing: {source_path}")
        self._counter += 1
        safe_group = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in group)
        digest = _signature_key((group, key, self._counter, source_path.name))
        suffix = ".bin.gz" if source_path.suffix == ".gz" else ".bin"
        rel = Path("evaluators") / f"{safe_group}_{digest}{suffix}"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.exists():
                path.unlink()
            os.link(source_path, path)
        except OSError:
            shutil.copy2(source_path, path)
        return str(rel)

    def evaluator_refs(self, evaluators: list[Any], group: str, key: Any) -> list[str]:
        return [
            self.save_evaluator(evaluator, group=f"{group}_{index}", key=(key, index))
            for index, evaluator in enumerate(evaluators)
        ]


def _ref(store: PreparedEvaluatorStore, relative_path: str | None) -> Any | None:
    return LazyEvaluatorRef(store, relative_path) if relative_path else None


def _ref_list(store: PreparedEvaluatorStore, paths: list[str]) -> list[Any]:
    return [LazyEvaluatorRef(store, path) for path in paths]


def _topology_json(writer: _BundleWriter, topology: TopologyDefinition) -> dict[str, Any]:
    """Return serializable topology metadata and evaluator references."""
    data: dict[str, Any] = {
        "family": topology.family,
        "x_names": topology.x_names,
        "parameter_names": topology.parameter_names,
        "parameter_values": topology.parameter_values,
        "u_expr": _expr_text(topology.u_expr),
        "f_expr": _expr_text(topology.f_expr),
        "u_power_base": topology.u_power_base,
        "f_power_base": topology.f_power_base,
        "eps_log_u_coeff": topology.eps_log_u_coeff,
        "eps_log_f_coeff": topology.eps_log_f_coeff,
        "expected_laurent_orders": topology.expected_laurent_orders,
        "convention_note": topology.convention_note,
        "global_prefactor_coeffs": _encode(topology.global_prefactor_coeffs or []),
        "global_prefactor_min_order": int(getattr(topology, "global_prefactor_min_order", 0)),
        "jit_compile_evaluators": topology.jit_compile_evaluators,
        "dual_evaluator_mode": topology.dual_evaluator_mode,
        "ibp_reduce_to_log_endpoint": topology.ibp_reduce_to_log_endpoint,
        "ibp_power_goal": topology.ibp_power_goal,
        "parametric_representation": _parametric_to_json(topology.parametric_representation),
        "regular_taylor_signature_version": topology._regular_taylor_signature_version,
        "regular_taylor_formula_signature_limit": topology.regular_taylor_formula_signature_limit,
        "regular_taylor_formula_volume_limit": topology.regular_taylor_formula_volume_limit,
        "regular_taylor_formula_axis_limit": topology.regular_taylor_formula_axis_limit,
        "chain_rule_formula_signature_limit": topology.chain_rule_formula_signature_limit,
        "chain_rule_formula_output_length_limit": topology.chain_rule_formula_output_length_limit,
        "direct_projector_cache_term_threshold": topology.direct_projector_cache_term_threshold,
        "evaluators": {},
    }
    evaluators = data["evaluators"]
    evaluators["u_scalar"] = writer.save_evaluator(topology._u_evaluator, "topology_u", "scalar")
    evaluators["f_scalar"] = writer.save_evaluator(topology._f_evaluator, "topology_f", "scalar")
    evaluators["u_dual"] = [
        {
            "shape": _encode(shape),
            "path": writer.save_evaluator(ev, "topology_u_dual", shape),
        }
        for shape, ev in topology._u_dual_evaluators.items()
    ]
    evaluators["f_dual"] = [
        {
            "shape": _encode(shape),
            "path": writer.save_evaluator(ev, "topology_f_dual", shape),
        }
        for shape, ev in topology._f_dual_evaluators.items()
    ]
    evaluators["u_derivative_multi"] = [
        {
            "indices": _encode(indices),
            "path": writer.save_evaluator(ev, "topology_u_deriv_multi", indices),
        }
        for indices, ev in topology._u_derivative_multi_evaluators.items()
    ]
    evaluators["f_derivative_multi"] = [
        {
            "indices": _encode(indices),
            "path": writer.save_evaluator(ev, "topology_f_deriv_multi", indices),
        }
        for indices, ev in topology._f_derivative_multi_evaluators.items()
    ]
    evaluators["u_derivative_single"] = [
        {
            "index": _encode(index),
            "path": writer.save_evaluator(ev, "topology_u_deriv", index),
        }
        for index, ev in topology._u_derivative_evaluators.items()
    ]
    evaluators["f_derivative_single"] = [
        {
            "index": _encode(index),
            "path": writer.save_evaluator(ev, "topology_f_deriv", index),
        }
        for index, ev in topology._f_derivative_evaluators.items()
    ]
    data["derivative_indices_by_order"] = {
        "u": {str(order): _encode(indices) for order, indices in topology._u_derivative_indices_by_order.items()},
        "f": {str(order): _encode(indices) for order, indices in topology._f_derivative_indices_by_order.items()},
    }
    data["overall_dual_shapes"] = {
        str(dimension): _encode(shape)
        for dimension, shape in topology._overall_dual_shapes.items()
    }
    data["overall_dual_indices"] = [
        {"key": _encode(key), "indices": indices}
        for key, indices in topology._overall_dual_indices.items()
    ]
    return data


def _load_topology(data: dict[str, Any], store: PreparedEvaluatorStore) -> TopologyDefinition:
    """Reconstruct a strict topology object from JSON plus evaluator refs."""
    topology = TopologyDefinition(
        family=str(data["family"]),
        x_names=list(data["x_names"]),
        parameter_names=list(data["parameter_names"]),
        parameter_values=[float(value) for value in data["parameter_values"]],
        u_expr=_expr_from_text(data["u_expr"]),
        f_expr=_expr_from_text(data["f_expr"]),
        u_power_base=float(data["u_power_base"]),
        f_power_base=float(data["f_power_base"]),
        eps_log_u_coeff=float(data["eps_log_u_coeff"]),
        eps_log_f_coeff=float(data["eps_log_f_coeff"]),
        expected_laurent_orders=list(data["expected_laurent_orders"]),
        convention_note=str(data["convention_note"]),
        global_prefactor_coeffs=[complex(value) for value in _decode(data.get("global_prefactor_coeffs", []))],
        global_prefactor_min_order=int(data.get("global_prefactor_min_order", 0)),
        jit_compile_evaluators=bool(data.get("jit_compile_evaluators", False)),
        dual_evaluator_mode=str(data.get("dual_evaluator_mode", "pregenerate")),
        ibp_reduce_to_log_endpoint=bool(data.get("ibp_reduce_to_log_endpoint", False)),
        ibp_power_goal=(
            int(data["ibp_power_goal"])
            if data.get("ibp_power_goal") is not None
            else (-1 if bool(data.get("ibp_reduce_to_log_endpoint", False)) else None)
        ),
        skip_evaluator_build=True,
        strict_prepared_bundle=True,
        parametric_representation=_parametric_from_json(data.get("parametric_representation")),
    )
    topology._regular_taylor_signature_version = int(data.get("regular_taylor_signature_version", 1))
    topology.regular_taylor_formula_signature_limit = int(data.get("regular_taylor_formula_signature_limit", 256))
    topology.regular_taylor_formula_volume_limit = int(data.get("regular_taylor_formula_volume_limit", 64))
    topology.regular_taylor_formula_axis_limit = int(data.get("regular_taylor_formula_axis_limit", 5))
    topology.chain_rule_formula_signature_limit = int(data.get("chain_rule_formula_signature_limit", 256))
    topology.chain_rule_formula_output_length_limit = int(
        data.get("chain_rule_formula_output_length_limit", 0)
    )
    topology.direct_projector_cache_term_threshold = int(data.get("direct_projector_cache_term_threshold", 54))

    evaluators = data["evaluators"]
    topology._u_evaluator = _ref(store, evaluators["u_scalar"])
    topology._f_evaluator = _ref(store, evaluators["f_scalar"])
    topology._u_dual_evaluators = {
        tuple(_decode(item["shape"])): _ref(store, item["path"])
        for item in evaluators.get("u_dual", [])
    }
    topology._f_dual_evaluators = {
        tuple(_decode(item["shape"])): _ref(store, item["path"])
        for item in evaluators.get("f_dual", [])
    }
    topology._u_derivative_multi_evaluators = {
        tuple(_decode(item["indices"])): _ref(store, item["path"])
        for item in evaluators.get("u_derivative_multi", [])
    }
    topology._f_derivative_multi_evaluators = {
        tuple(_decode(item["indices"])): _ref(store, item["path"])
        for item in evaluators.get("f_derivative_multi", [])
    }
    topology._u_derivative_evaluators = {
        tuple(_decode(item["index"])): _ref(store, item["path"])
        for item in evaluators.get("u_derivative_single", [])
    }
    topology._f_derivative_evaluators = {
        tuple(_decode(item["index"])): _ref(store, item["path"])
        for item in evaluators.get("f_derivative_single", [])
    }
    indices = data.get("derivative_indices_by_order", {})
    topology._u_derivative_indices_by_order = {
        int(order): list(_decode(value))
        for order, value in indices.get("u", {}).items()
    }
    topology._f_derivative_indices_by_order = {
        int(order): list(_decode(value))
        for order, value in indices.get("f", {}).items()
    }
    topology._overall_dual_shapes = {
        int(dim): list(_decode(shape))
        for dim, shape in data.get("overall_dual_shapes", {}).items()
    }
    topology._overall_dual_indices = {
        tuple(_decode(item["key"])): list(item["indices"])
        for item in data.get("overall_dual_indices", [])
    }
    # Degree-zero symbolic derivatives are just the original scalar U/F
    # evaluators.  Older generated bundles do not store those as derivative
    # artifacts, but strict integrate mode still needs the derivative metadata
    # when a sector asks for a constant Taylor source.
    zero_multi = tuple(0 for _ in topology.x_names)
    topology._u_derivative_indices_by_order.setdefault(0, [zero_multi])
    topology._f_derivative_indices_by_order.setdefault(0, [zero_multi])
    topology._u_derivative_multi_evaluators.setdefault((zero_multi,), topology._u_evaluator)
    topology._f_derivative_multi_evaluators.setdefault((zero_multi,), topology._f_evaluator)
    _fill_missing_derivative_orders(topology._u_derivative_indices_by_order)
    _fill_missing_derivative_orders(topology._f_derivative_indices_by_order)
    return topology


def _fill_missing_derivative_orders(indices_by_order: dict[int, list[tuple[int, ...]]]) -> None:
    """Alias missing derivative orders to the smallest prepared superset.

    Symbolic-derivative generation stores the exact total-degree requests it
    saw during preparation.  Runtime sector paths can ask for a smaller exact
    degree.  Reusing a prepared higher-degree evaluator is algebraically safe:
    the extra derivatives have no contributing Taylor monomials in the smaller
    requested output shape, and it avoids any integrate-time evaluator build.
    """
    if not indices_by_order:
        return
    prepared = sorted(int(order) for order in indices_by_order)
    max_order = max(prepared)
    for order in range(max_order + 1):
        if order in indices_by_order:
            continue
        superset_order = next((candidate for candidate in prepared if candidate >= order), None)
        if superset_order is not None:
            indices_by_order[order] = list(indices_by_order[superset_order])


def _sector_json(writer: _BundleWriter, sector: SectorDefinition, sector_id: int) -> dict[str, Any]:
    """Return serializable sector metadata and evaluator references."""
    data: dict[str, Any] = {
        "id": int(sector_id),
        "name": sector.name,
        "integration_dim": sector.integration_dim,
        "variable_names": sector.variable_names,
        "map_exprs": [_expr_text(expr) for expr in sector.map_exprs],
        "regular_jacobian_expr": _expr_text(sector.regular_jacobian_expr),
        "numerator_expr": _expr_text(sector.numerator_expr),
        "numerator_eps_exprs": [_expr_text(expr) for expr in sector.numerator_eps_exprs or []],
        "f_monomial_powers": sector.f_monomial_powers,
        "jacobian_monomial_powers": sector.jacobian_monomial_powers,
        "singular_axes": sector.singular_axes,
        "subtraction_type": sector.subtraction_type,
        "description": sector.description,
        "jit_compile_evaluators": sector.jit_compile_evaluators,
        "u_monomial_powers": sector.u_monomial_powers,
        "measure_monomial_powers": sector.measure_monomial_powers,
        "numerator_monomial_powers": sector.numerator_monomial_powers,
        "endpoint_taylor_orders": sector.endpoint_taylor_orders,
        "dual_shape": _encode(tuple(sector.dual_shape)),
        "evaluators": {},
    }
    ev = data["evaluators"]
    ev["map"] = writer.evaluator_refs(sector._map_evaluators, f"sector_{sector_id}_map", sector.name)
    ev["jacobian"] = (
        writer.save_evaluator(sector._jacobian_evaluator, f"sector_{sector_id}_jac", sector.name)
        if sector._jacobian_evaluator is not None
        else None
    )
    ev["numerator"] = (
        writer.save_evaluator(sector._numerator_evaluator, f"sector_{sector_id}_num", sector.name)
        if sector._numerator_evaluator is not None
        else None
    )
    ev["numerator_eps"] = [
        (
            writer.save_evaluator(evaluator, f"sector_{sector_id}_num_eps_{index}", sector.name)
            if evaluator is not None
            else None
        )
        for index, evaluator in enumerate(sector._numerator_eps_evaluators)
    ]
    ev["map_dual"] = [
        {
            "shape": _encode(shape),
            "paths": writer.evaluator_refs(evaluators, f"sector_{sector_id}_map_dual", shape),
        }
        for shape, evaluators in sector._map_dual_evaluators_by_shape.items()
    ]
    ev["jacobian_dual"] = [
        {
            "shape": _encode(shape),
            "path": writer.save_evaluator(evaluator, f"sector_{sector_id}_jac_dual", shape),
        }
        for shape, evaluator in sector._jacobian_dual_evaluators_by_shape.items()
        if evaluator is not None
    ]
    ev["numerator_dual"] = [
        {
            "shape": _encode(shape),
            "path": writer.save_evaluator(evaluator, f"sector_{sector_id}_num_dual", shape),
        }
        for shape, evaluator in sector._numerator_dual_evaluators_by_shape.items()
        if evaluator is not None
    ]
    ev["numerator_eps_dual"] = [
        {
            "shape": _encode(shape),
            "paths": [
                (
                    writer.save_evaluator(
                        evaluator,
                        f"sector_{sector_id}_num_eps_dual_{index}",
                        shape,
                    )
                    if evaluator is not None
                    else None
                )
                for index, evaluator in enumerate(evaluators)
            ],
        }
        for shape, evaluators in sector._numerator_eps_dual_evaluators_by_shape.items()
    ]
    return data


def _load_sector(data: dict[str, Any], store: PreparedEvaluatorStore) -> SectorDefinition:
    """Reconstruct one strict sector object from JSON plus evaluator refs."""
    sector = SectorDefinition(
        name=str(data["name"]),
        integration_dim=int(data["integration_dim"]),
        variable_names=list(data["variable_names"]),
        map_exprs=[_expr_from_text(expr) for expr in data["map_exprs"]],
        regular_jacobian_expr=_expr_from_text(data["regular_jacobian_expr"]),
        numerator_expr=_expr_from_text(data.get("numerator_expr", "1")),
        numerator_eps_exprs=[
            _expr_from_text(expr)
            for expr in data.get("numerator_eps_exprs", [data.get("numerator_expr", "1")])
        ],
        f_monomial_powers=[int(value) for value in data["f_monomial_powers"]],
        jacobian_monomial_powers=[int(value) for value in data["jacobian_monomial_powers"]],
        singular_axes=[int(value) for value in data["singular_axes"]],
        subtraction_type=str(data["subtraction_type"]),
        description=str(data["description"]),
        jit_compile_evaluators=bool(data.get("jit_compile_evaluators", False)),
        u_monomial_powers=[int(value) for value in data.get("u_monomial_powers", [])],
        measure_monomial_powers=[float(value) for value in data.get("measure_monomial_powers", [])],
        numerator_monomial_powers=[float(value) for value in data.get("numerator_monomial_powers", [])],
        endpoint_taylor_orders=[int(value) for value in data.get("endpoint_taylor_orders", [])],
        strict_prepared_bundle=True,
    )
    ev = data["evaluators"]
    sector._map_evaluators = _ref_list(store, ev.get("map", []))
    sector._jacobian_evaluator = _ref(store, ev.get("jacobian"))
    sector._numerator_evaluator = _ref(store, ev.get("numerator"))
    sector._numerator_eps_evaluators = [
        _ref(store, item) for item in ev.get("numerator_eps", [])
    ]
    sector._map_dual_evaluators_by_shape = {
        tuple(_decode(item["shape"])): _ref_list(store, item["paths"])
        for item in ev.get("map_dual", [])
    }
    sector._jacobian_dual_evaluators_by_shape = {
        tuple(_decode(item["shape"])): _ref(store, item["path"])
        for item in ev.get("jacobian_dual", [])
    }
    sector._numerator_dual_evaluators_by_shape = {
        tuple(_decode(item["shape"])): _ref(store, item["path"])
        for item in ev.get("numerator_dual", [])
    }
    sector._numerator_eps_dual_evaluators_by_shape = {
        tuple(_decode(item["shape"])): [_ref(store, path) for path in item.get("paths", [])]
        for item in ev.get("numerator_eps_dual", [])
    }
    key = tuple(sector.dual_shape)
    sector._map_dual_evaluators = sector._map_dual_evaluators_by_shape.get(key, [])
    sector._jacobian_dual_evaluator = sector._jacobian_dual_evaluators_by_shape.get(key)
    sector._numerator_dual_evaluator = sector._numerator_dual_evaluators_by_shape.get(key)
    sector._evaluators_prepared = True
    return sector


def _subtraction_formula_json(writer: _BundleWriter, formula: SubtractionFormulaDefinition) -> dict[str, Any]:
    return {
        "signature": _encode(formula.signature),
        "input_names": formula.input_names,
        "output_expressions": [_expr_text(expr) for expr in formula.output_expressions],
        "evaluators": writer.evaluator_refs(formula.evaluators, "formula_subtraction", formula.signature),
        "laurent_orders": formula.laurent_orders,
        "zero_subsets": _encode(tuple(formula.zero_subsets)),
        "dual_shape": _encode(tuple(formula.dual_shape)),
        "build_seconds": formula.build_seconds,
    }


def _load_subtraction_formula(data: dict[str, Any], store: PreparedEvaluatorStore) -> SubtractionFormulaDefinition:
    input_names = list(data["input_names"])
    return SubtractionFormulaDefinition(
        signature=tuple(_decode(data["signature"])),
        input_names=input_names,
        input_symbols=[S(name) for name in input_names],
        output_expressions=list(data.get("output_expressions", [])),
        evaluators=_ref_list(store, data["evaluators"]),
        laurent_orders=[int(order) for order in data["laurent_orders"]],
        zero_subsets=[tuple(item) for item in _decode(data["zero_subsets"])],
        dual_shape=[tuple(item) for item in _decode(data["dual_shape"])],
        build_seconds=float(data.get("build_seconds", 0.0)),
    )


def _ibp_term_to_json(term: IBPEndpointProjectorTerm) -> dict[str, Any]:
    return {
        "prefactor_coeffs": _encode(term.prefactor_coeffs),
        "boundary_positions": _encode(term.boundary_positions),
        "derivative_multi": _encode(term.derivative_multi),
        "active_positions": _encode(term.active_positions),
        "child_signature": _encode(term.child_signature),
    }


def _ibp_term_from_json(data: dict[str, Any]) -> IBPEndpointProjectorTerm:
    return IBPEndpointProjectorTerm(
        prefactor_coeffs=[complex(value) for value in _decode(data["prefactor_coeffs"])],
        boundary_positions=tuple(_decode(data["boundary_positions"])),
        derivative_multi=tuple(_decode(data["derivative_multi"])),
        active_positions=tuple(_decode(data["active_positions"])),
        child_signature=tuple(_decode(data["child_signature"])),
    )


def _endpoint_formula_json(writer: _BundleWriter, formula: EndpointProjectorFormulaDefinition) -> dict[str, Any]:
    return {
        "signature": _encode(formula.signature),
        "input_names": formula.input_names,
        "output_expressions": [_expr_text(expr) for expr in formula.output_expressions],
        "evaluators": writer.evaluator_refs(formula.evaluators, "formula_endpoint", formula.signature),
        "laurent_orders": formula.laurent_orders,
        "zero_subsets": _encode(tuple(formula.zero_subsets)),
        "taylor_orders": formula.taylor_orders,
        "coefficient_layout": _encode(tuple(formula.coefficient_layout)),
        "ibp_reduce_to_log_endpoint": formula.ibp_reduce_to_log_endpoint,
        "ibp_power_goal": formula.ibp_power_goal,
        "ibp_terms": [_ibp_term_to_json(term) for term in formula.ibp_terms],
        "build_seconds": formula.build_seconds,
    }


def _load_endpoint_formula(data: dict[str, Any], store: PreparedEvaluatorStore) -> EndpointProjectorFormulaDefinition:
    input_names = list(data["input_names"])
    return EndpointProjectorFormulaDefinition(
        signature=tuple(_decode(data["signature"])),
        input_names=input_names,
        input_symbols=[S(name) for name in input_names],
        output_expressions=list(data.get("output_expressions", [])),
        evaluators=_ref_list(store, data.get("evaluators", [])),
        laurent_orders=[int(order) for order in data["laurent_orders"]],
        zero_subsets=[tuple(item) for item in _decode(data["zero_subsets"])],
        taylor_orders=[int(order) for order in data["taylor_orders"]],
        coefficient_layout=[tuple(item) for item in _decode(data["coefficient_layout"])],
        ibp_reduce_to_log_endpoint=bool(data.get("ibp_reduce_to_log_endpoint", False)),
        ibp_power_goal=(
            int(data["ibp_power_goal"])
            if data.get("ibp_power_goal") is not None
            else (-1 if bool(data.get("ibp_reduce_to_log_endpoint", False)) else None)
        ),
        ibp_terms=[_ibp_term_from_json(item) for item in data.get("ibp_terms", [])],
        build_seconds=float(data.get("build_seconds", 0.0)),
    )


def _regular_formula_json(writer: _BundleWriter, formula: RegularTaylorFormulaDefinition) -> dict[str, Any]:
    if formula.cache_evaluator_files:
        evaluator_refs = [
            writer.copy_evaluator_file(
                path,
                group=f"formula_regular_{index}",
                key=(formula.signature, index),
            )
            for index, path in enumerate(formula.cache_evaluator_files)
        ]
    else:
        evaluator_refs = writer.evaluator_refs(
            formula.evaluators,
            "formula_regular",
            formula.signature,
        )
    return {
        "signature": _encode(formula.signature),
        "input_names": formula.input_names,
        # Regular-Taylor expressions can be very large for six-axis three-loop
        # signatures.  Strict prepared bundles need the serialized evaluators
        # and layouts only; the cache signature is enough to regenerate the
        # reference expression offline if evaluator bytes are unavailable.
        "output_expressions": [],
        "output_expression_count": len(formula.output_expressions),
        "evaluators": evaluator_refs,
        "output_layout": _encode(tuple(formula.output_layout)),
        "input_layout": _encode(tuple(formula.input_layout)),
        "max_orders": formula.max_orders,
        "zero_positions": _encode(formula.zero_positions),
        "dual_shape": _encode(tuple(formula.dual_shape)),
        "evaluator_input_symbols": [str(symbol) for symbol in formula.evaluator_input_symbols],
        "evaluator_dual_shape": _encode(tuple(formula.evaluator_dual_shape)),
        "evaluator_output_indices": [
            int(index) for index in getattr(formula, "evaluator_output_indices", [])
        ],
        "dual_variable_count": formula.dual_variable_count,
        "build_seconds": formula.build_seconds,
    }


def _load_regular_formula(data: dict[str, Any], store: PreparedEvaluatorStore) -> RegularTaylorFormulaDefinition:
    input_names = list(data["input_names"])
    evaluator_input_names = list(data.get("evaluator_input_symbols", []))
    return RegularTaylorFormulaDefinition(
        signature=tuple(_decode(data["signature"])),
        input_names=input_names,
        input_symbols=[S(name) for name in input_names],
        output_expressions=list(data.get("output_expressions", [])),
        evaluators=_ref_list(store, data.get("evaluators", [])),
        output_layout=[tuple(item) for item in _decode(data["output_layout"])],
        input_layout=[tuple(item) for item in _decode(data["input_layout"])],
        max_orders=[int(order) for order in data["max_orders"]],
        zero_positions=tuple(_decode(data["zero_positions"])),
        dual_shape=[tuple(item) for item in _decode(data["dual_shape"])],
        evaluator_input_symbols=[S(name) for name in evaluator_input_names],
        evaluator_dual_shape=[tuple(item) for item in _decode(data.get("evaluator_dual_shape", []))],
        evaluator_output_indices=[
            int(index) for index in data.get("evaluator_output_indices", [])
        ],
        dual_variable_count=int(data.get("dual_variable_count", 0)),
        build_seconds=float(data.get("build_seconds", 0.0)),
    )


def _chain_formula_json(writer: _BundleWriter, formula: ChainRuleFormulaDefinition) -> dict[str, Any]:
    if formula.cache_evaluator_files:
        evaluator_refs = [
            writer.copy_evaluator_file(
                path,
                group=f"formula_chain_{index}",
                key=(formula.signature, index),
            )
            for index, path in enumerate(formula.cache_evaluator_files)
        ]
    else:
        evaluator_refs = writer.evaluator_refs(formula.evaluators, "formula_chain", formula.signature)
    return {
        "signature": _encode(formula.signature),
        "input_names": formula.input_names,
        # These expressions are universal but can be enormous for hard
        # three-loop signatures.  The strict integrate path only needs the
        # serialized evaluator plus the layouts, so keep prepared bundles
        # compact and inspectable through metadata rather than full formula
        # strings.
        "output_expressions": [],
        "output_expression_count": len(formula.output_expressions),
        "evaluators": evaluator_refs,
        "output_shape": _encode(tuple(formula.output_shape)),
        "derivative_indices": _encode(tuple(formula.derivative_indices)),
        "h_layout": _encode(tuple(formula.h_layout)),
        "evaluator_mode": formula.evaluator_mode,
        "build_seconds": formula.build_seconds,
    }


def _load_chain_formula(data: dict[str, Any], store: PreparedEvaluatorStore) -> ChainRuleFormulaDefinition:
    input_names = list(data["input_names"])
    return ChainRuleFormulaDefinition(
        signature=tuple(_decode(data["signature"])),
        input_names=input_names,
        input_symbols=[],
        output_expressions=[],
        evaluators=_ref_list(store, data.get("evaluators", [])),
        output_shape=[tuple(item) for item in _decode(data["output_shape"])],
        derivative_indices=[tuple(item) for item in _decode(data["derivative_indices"])],
        h_layout=[tuple(item) for item in _decode(data["h_layout"])],
        evaluator_mode=str(data.get("evaluator_mode", "separate")),
        build_seconds=float(data.get("build_seconds", 0.0)),
    )


def _two_stage_formula_json(
    writer: _BundleWriter,
    formula: TwoStageSectorFormulaDefinition,
) -> dict[str, Any]:
    """Serialize one sector-specific two-stage evaluator pair."""
    return {
        "sector_name": formula.sector_name,
        "source_input_names": formula.source_input_names,
        "assembler_input_names": formula.assembler_input_names,
        "coefficient_keys": _encode(tuple(formula.coefficient_keys)),
        "source_keys": _encode(tuple(formula.source_keys)),
        "laurent_orders": [int(order) for order in formula.laurent_orders],
        "source_evaluator": writer.save_evaluator(
            formula.source_evaluator,
            "two_stage_source",
            formula.sector_name,
        ),
        "assembler_evaluator": writer.save_evaluator(
            formula.assembler_evaluator,
            "two_stage_assembler",
            formula.sector_name,
        ),
        "source_expression_build_seconds": formula.source_expression_build_seconds,
        "source_evaluator_build_seconds": formula.source_evaluator_build_seconds,
        "assembler_expression_build_seconds": formula.assembler_expression_build_seconds,
        "assembler_evaluator_build_seconds": formula.assembler_evaluator_build_seconds,
        "source_expression_bytes": formula.source_expression_bytes,
        "assembler_expression_bytes": formula.assembler_expression_bytes,
        "source_evaluator_bytes": formula.source_evaluator_bytes,
        "assembler_evaluator_bytes": formula.assembler_evaluator_bytes,
        "source_kind": formula.source_kind,
    }


def _two_stage_key_from_json(value: Any) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    """Decode one two-stage coefficient-layout key."""
    decoded = _decode(value)
    return (
        tuple(int(item) for item in decoded[0]),
        tuple(int(item) for item in decoded[1]),
        tuple(int(item) for item in decoded[2]),
        int(decoded[3]),
    )


def _load_two_stage_formula(
    data: dict[str, Any],
    store: PreparedEvaluatorStore,
) -> TwoStageSectorFormulaDefinition:
    """Hydrate a lazy two-stage evaluator pair from a prepared bundle."""
    return TwoStageSectorFormulaDefinition(
        sector_name=str(data["sector_name"]),
        source_input_names=[str(name) for name in data["source_input_names"]],
        assembler_input_names=[str(name) for name in data["assembler_input_names"]],
        coefficient_keys=[
            _two_stage_key_from_json(item)
            for item in _decode(data.get("coefficient_keys", []))
        ],
        source_keys=list(_decode(data.get("source_keys", []))),
        laurent_orders=[int(order) for order in data["laurent_orders"]],
        source_evaluator=_ref(store, data["source_evaluator"]),
        assembler_evaluator=_ref(store, data["assembler_evaluator"]),
        source_expression_build_seconds=float(data.get("source_expression_build_seconds", 0.0)),
        source_evaluator_build_seconds=float(data.get("source_evaluator_build_seconds", 0.0)),
        assembler_expression_build_seconds=float(data.get("assembler_expression_build_seconds", 0.0)),
        assembler_evaluator_build_seconds=float(data.get("assembler_evaluator_build_seconds", 0.0)),
        source_expression_bytes=int(data.get("source_expression_bytes", 0)),
        assembler_expression_bytes=int(data.get("assembler_expression_bytes", 0)),
        source_evaluator_bytes=int(data.get("source_evaluator_bytes", 0)),
        assembler_evaluator_bytes=int(data.get("assembler_evaluator_bytes", 0)),
        source_kind=str(data.get("source_kind", "symbolic-derivative-source")),
    )


def _explicit_formula_json(
    writer: _BundleWriter,
    formula: ExplicitSectorFormulaDefinition,
) -> dict[str, Any]:
    """Serialize one sector-specific explicit evaluator."""
    return {
        "sector_name": formula.sector_name,
        "input_names": formula.input_names,
        "laurent_orders": [int(order) for order in formula.laurent_orders],
        "evaluator": writer.save_evaluator(
            formula.evaluator,
            "explicit_sector",
            formula.sector_name,
        ),
        "expression_build_seconds": formula.expression_build_seconds,
        "evaluator_build_seconds": formula.evaluator_build_seconds,
        "expression_bytes": formula.expression_bytes,
        "evaluator_bytes": formula.evaluator_bytes,
        "source_kind": formula.source_kind,
    }


def _load_explicit_formula(
    data: dict[str, Any],
    store: PreparedEvaluatorStore,
) -> ExplicitSectorFormulaDefinition:
    """Hydrate a lazy explicit sector evaluator from a prepared bundle."""
    return ExplicitSectorFormulaDefinition(
        sector_name=str(data["sector_name"]),
        input_names=[str(name) for name in data["input_names"]],
        laurent_orders=[int(order) for order in data["laurent_orders"]],
        evaluator=_ref(store, data["evaluator"]),
        expression_build_seconds=float(data.get("expression_build_seconds", 0.0)),
        evaluator_build_seconds=float(data.get("evaluator_build_seconds", 0.0)),
        expression_bytes=int(data.get("expression_bytes", 0)),
        evaluator_bytes=int(data.get("evaluator_bytes", 0)),
        source_kind=str(data.get("source_kind", "explicit-sector-expression")),
    )


def _formula_json(writer: _BundleWriter, topology: TopologyDefinition) -> dict[str, Any]:
    """Serialize all prepared formula families."""
    return {
        "subtraction": [
            _subtraction_formula_json(writer, formula)
            for formula in topology._subtraction_formulas.values()
        ],
        "endpoint_projector": [
            _endpoint_formula_json(writer, formula)
            for formula in topology._endpoint_projector_formulas.values()
        ],
        "regular_taylor": [
            _regular_formula_json(writer, formula)
            for formula in topology._regular_taylor_formulas.values()
        ],
        "chain_rule": [
            _chain_formula_json(writer, formula)
            for formula in topology._chain_rule_formulas.values()
        ],
        "two_stage_sector": [
            _two_stage_formula_json(writer, formula)
            for formula in getattr(topology, "_two_stage_sector_formulas", {}).values()
        ],
        "explicit_sector": [
            _explicit_formula_json(writer, formula)
            for formula in getattr(topology, "_explicit_sector_formulas", {}).values()
        ],
    }


def _load_formulas(data: dict[str, Any], topology: TopologyDefinition, store: PreparedEvaluatorStore) -> None:
    """Hydrate all formula caches onto a strict topology definition."""
    topology._subtraction_formulas = {
        formula.signature: formula
        for formula in (_load_subtraction_formula(item, store) for item in data.get("subtraction", []))
    }
    endpoint_formulas = [
        _load_endpoint_formula(item, store)
        for item in data.get("endpoint_projector", [])
    ]
    topology._endpoint_projector_formulas = {formula.signature: formula for formula in endpoint_formulas}
    for formula in endpoint_formulas:
        if formula.ibp_reduce_to_log_endpoint:
            formula.child_formulas = {
                term.child_signature: topology._endpoint_projector_formulas[term.child_signature]
                for term in formula.ibp_terms
                if term.child_signature in topology._endpoint_projector_formulas
            }
    topology._regular_taylor_formulas = {
        formula.signature: formula
        for formula in (_load_regular_formula(item, store) for item in data.get("regular_taylor", []))
    }
    topology._chain_rule_formulas = {
        formula.signature: formula
        for formula in (_load_chain_formula(item, store) for item in data.get("chain_rule", []))
    }
    topology._two_stage_sector_formulas = {
        formula.sector_name: formula
        for formula in (
            _load_two_stage_formula(item, store)
            for item in data.get("two_stage_sector", [])
        )
    }
    topology._explicit_sector_formulas = {
        formula.sector_name: formula
        for formula in (
            _load_explicit_formula(item, store)
            for item in data.get("explicit_sector", [])
        )
    }


def save_prepared_bundle(
    output_dir: str | Path,
    request: IntegralRequest,
    topology: TopologyDefinition,
    sectors: list[SectorDefinition],
    generation_timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a prepared DOT runtime bundle and return its manifest."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for subdir in ("evaluators", "expressions"):
        path = root / subdir
        if path.exists():
            shutil.rmtree(path)
    writer = _BundleWriter(root)
    start = time.perf_counter()

    topology_payload = _topology_json(writer, topology)
    sectors_payload = [_sector_json(writer, sector, index) for index, sector in enumerate(sectors)]
    formulas_payload = _formula_json(writer, topology)

    _atomic_write_json(root / "topology.json", topology_payload)
    _atomic_write_json(root / "sectors.json", {"sectors": sectors_payload})
    _atomic_write_json(root / "expressions" / "formulas.json", formulas_payload)
    serialization_seconds = time.perf_counter() - start
    timings_payload = _with_bundle_serialization_timing(
        generation_timings or {}, serialization_seconds
    )
    _atomic_write_json(root / "generation_timings.json", timings_payload)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "generation_options": {
            "sector_method": request.sector_method,
            "dual_evaluator_mode": request.dual_evaluator_mode,
            "subtraction_backend": request.subtraction_backend,
            "sector_evaluator_backend": request.sector_evaluator_backend,
            "ibp_reduce_to_log_endpoint": request.ibp_reduce_to_log_endpoint,
            "ibp_power_goal": request.ibp_power_goal,
            "direct_projector_cache_term_threshold": request.direct_projector_cache_term_threshold,
            "allow_fallback_for_missing_caches": request.allow_fallback_for_missing_caches,
            "chain_rule_formula_output_length_limit": request.chain_rule_formula_output_length_limit,
            "max_eps_order": request.max_eps_order,
        },
        "source_files": {
            "dot_file": request.dot_file,
            "kinematics_file": request.kinematics_file,
        },
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "symbolica": _symbolica_version(),
        },
        "laurent_range": {
            "min": topology.laurent_min_order,
            "max": topology.laurent_max_order,
            "labels": topology.expected_laurent_orders,
        },
        "prepared_sector_ids": [index for index in range(len(sectors))],
        "artifact_counts": {
            "sectors": len(sectors),
            "subtraction_formulas": len(topology._subtraction_formulas),
            "endpoint_projector_formulas": len(topology._endpoint_projector_formulas),
            "regular_taylor_formulas": len(topology._regular_taylor_formulas),
            "chain_rule_formulas": len(topology._chain_rule_formulas),
            "two_stage_sector_formulas": len(getattr(topology, "_two_stage_sector_formulas", {})),
            "explicit_sector_formulas": len(getattr(topology, "_explicit_sector_formulas", {})),
            "evaluator_files": len(list((root / "evaluators").glob("*.bin")))
            + len(list((root / "evaluators").glob("*.bin.gz"))),
        },
        "serialization_seconds": serialization_seconds,
        "files": {
            "topology": "topology.json",
            "sectors": "sectors.json",
            "formulas": "expressions/formulas.json",
            "generation_timings": "generation_timings.json",
        },
    }
    _atomic_write_json(root / "manifest.json", manifest)
    return manifest


def _with_bundle_serialization_timing(
    generation_timings: dict[str, Any],
    serialization_seconds: float,
) -> dict[str, Any]:
    """Return generation timings with the prepared-bundle write made explicit."""
    payload = dict(generation_timings)
    detail_record = {
        "name": "Prepared bundle serialization",
        "seconds": max(float(serialization_seconds), 0.0),
        "detail": "metadata and serialized evaluator artifacts",
    }
    details = list(payload.get("details", []))
    details = [record for record in details if record.get("name") != detail_record["name"]]
    details.append(detail_record)
    payload["details"] = details
    payload["total"] = float(payload.get("total", 0.0)) + detail_record["seconds"]
    return payload


def _symbolica_version() -> str:
    """Best-effort Symbolica version string for bundle diagnostics."""
    try:
        import symbolica  # type: ignore

        return str(getattr(symbolica, "__version__", "unknown"))
    except Exception:
        return "unknown"


def load_prepared_bundle(
    output_dir: str | Path,
    lru_size: int = 128,
) -> tuple[TopologyDefinition, list[SectorDefinition], dict[str, Any]]:
    """Load a strict prepared bundle from disk."""
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"prepared bundle is missing manifest.json: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported prepared bundle schema {manifest.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )
    files = manifest.get("files", {})
    required = {
        "topology": root / files.get("topology", "topology.json"),
        "sectors": root / files.get("sectors", "sectors.json"),
        "formulas": root / files.get("formulas", "expressions/formulas.json"),
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError("prepared bundle is incomplete; missing " + ", ".join(missing))

    store = PreparedEvaluatorStore(root, lru_size=lru_size)
    topology_data = json.loads(required["topology"].read_text(encoding="utf-8"))
    sectors_data = json.loads(required["sectors"].read_text(encoding="utf-8"))
    formulas_data = json.loads(required["formulas"].read_text(encoding="utf-8"))
    _assert_evaluator_artifacts_exist(root, topology_data, sectors_data, formulas_data)
    topology = _load_topology(topology_data, store)
    sectors = [_load_sector(item, store) for item in sectors_data.get("sectors", [])]
    _load_formulas(formulas_data, topology, store)
    return topology, sectors, manifest
