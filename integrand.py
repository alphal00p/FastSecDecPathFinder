"""Topology definitions and generic black-box sector processing.

The only symbolic expressions stored here are the topology-level U and F
polynomials used to build Symbolica evaluators and to print summaries.  The
``SectorProcessor`` never substitutes sector maps into U/F symbolically; it
only evaluates prepared sector callbacks and U/F callbacks on numeric batches.
"""

from __future__ import annotations

import copy
import cmath
from functools import lru_cache
import gc
import gzip
import hashlib
from itertools import product
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from symbolica import E, Evaluator, Expression, S

from cache_utils import formula_cache_dir, formula_cache_read_roots, mirror_cache_entry_to_primary
from definitions import EpsilonExpansion, HotPathTiming, IntegralRequest, ParametricRepresentation
from sectors_generator import SectorDefinition
from subtraction_formula import (
    build_endpoint_projector_formula_symbolica,
    build_regular_taylor_formula_symbolica,
    build_subtraction_formula_symbolica,
    endpoint_projector_formula_has_curated_cache,
    regular_taylor_formula_has_cache,
    regular_taylor_formula_has_curated_cache,
)
from utils import decimal_complex_with_precision, decimal_with_precision


ComplexPrecise = tuple[Any, Any]
ComplexPreciseRow = list[ComplexPrecise]

# The universal chain-rule formula path is currently scoped to scalar
# parameter integrals up to three loops, where U has degree L and F has degree
# L+1.  Dense derivative slots up to degree four therefore cover every
# topology-specific nonzero U/F derivative in the present DOT backend while
# keeping the formula signature independent of the number of original
# Feynman parameters and the sparse derivative support of one particular
# topology.
CHAIN_RULE_MAX_DERIVATIVE_DEGREE_1_TO_3_LOOPS = 4
CHAIN_RULE_FORMULA_CACHE_VERSION = 4


class _SerializedEvaluatorRef:
    """Lazy reference to an evaluator already serialized during generation.

    Prepared DOT generation can create many large topology-level dual
    evaluators.  Keeping every live Symbolica evaluator in memory until the
    bundle writer runs is exactly the wrong ownership boundary for the 3L
    triple-box path.  This proxy lets generation save one evaluator as soon as
    it is built, drop the live object, and still expose the same evaluator API
    to any later code that needs to inspect or copy the artifact.
    """

    def __init__(self, path: str | Path) -> None:
        self.cache_evaluator_file = str(path)
        self._loaded: Any | None = None

    def _raw_bytes(self) -> bytes:
        path = Path(self.cache_evaluator_file)
        raw = path.read_bytes()
        return gzip.decompress(raw) if path.suffix == ".gz" else raw

    def save(self) -> bytes:
        """Return the raw Symbolica evaluator bytes expected by bundle writers."""
        return self._raw_bytes()

    def _evaluator(self) -> Any:
        if self._loaded is None:
            self._loaded = Evaluator.load(self._raw_bytes())
        return self._loaded

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


def _decimal_real(value: Any, precision_digits: int) -> Decimal:
    """Convert a real scalar into a Decimal at the requested evaluator precision."""
    return decimal_with_precision(value, precision_digits)


def _decimal_complex(value: Any, precision_digits: int) -> tuple[Decimal, Decimal]:
    """Return Symbolica's ``(real, imag)`` multiprecision complex input shape."""
    return decimal_complex_with_precision(value, precision_digits)


@dataclass
class TopologyDefinition:
    """Retained U/F expressions plus their numeric and dual evaluators."""

    family: str
    x_names: list[str]
    parameter_names: list[str]
    parameter_values: list[float]
    u_expr: Any
    f_expr: Any
    u_power_base: float
    f_power_base: float
    eps_log_u_coeff: float
    eps_log_f_coeff: float
    expected_laurent_orders: list[str]
    convention_note: str
    global_prefactor_coeffs: list[complex] | None = None
    jit_compile_evaluators: bool = False
    dual_evaluator_mode: str = "pregenerate"
    ibp_reduce_to_log_endpoint: bool = False
    skip_evaluator_build: bool = False
    strict_prepared_bundle: bool = False
    chain_rule_metadata_only: bool = False
    parametric_representation: ParametricRepresentation | None = None
    streaming_evaluator_cache_dir: str | None = None
    _u_evaluator: Any = field(init=False, repr=False)
    _f_evaluator: Any = field(init=False, repr=False)
    _u_dual_evaluators: dict[tuple[tuple[int, ...], ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _f_dual_evaluators: dict[tuple[tuple[int, ...], ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _u_derivative_evaluators: dict[tuple[int, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _f_derivative_evaluators: dict[tuple[int, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _u_derivative_exprs: dict[tuple[int, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _f_derivative_exprs: dict[tuple[int, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _u_derivative_multi_evaluators: dict[tuple[tuple[int, ...], ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _f_derivative_multi_evaluators: dict[tuple[tuple[int, ...], ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _u_derivative_indices_by_order: dict[int, list[tuple[int, ...]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _f_derivative_indices_by_order: dict[int, list[tuple[int, ...]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _overall_dual_shapes: dict[int, list[tuple[int, ...]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _overall_dual_indices: dict[tuple[int, tuple[tuple[int, ...], ...]], list[int]] = field(
        default_factory=dict, init=False, repr=False
    )
    _subtraction_formulas: dict[tuple[Any, ...], "SubtractionFormulaDefinition"] = field(
        default_factory=dict, init=False, repr=False
    )
    _endpoint_projector_formulas: dict[
        tuple[Any, ...], "EndpointProjectorFormulaDefinition"
    ] = field(default_factory=dict, init=False, repr=False)
    _regular_taylor_formulas: dict[
        tuple[Any, ...], "RegularTaylorFormulaDefinition"
    ] = field(default_factory=dict, init=False, repr=False)
    _chain_rule_formulas: dict[
        tuple[Any, ...], "ChainRuleFormulaDefinition"
    ] = field(default_factory=dict, init=False, repr=False)
    _chain_rule_formula_lookup_cache: dict[
        tuple[str, str, tuple[tuple[int, ...], ...]], "ChainRuleFormulaDefinition"
    ] = field(default_factory=dict, init=False, repr=False)
    _chain_rule_h_layout_cache: dict[
        tuple[str, tuple[tuple[int, ...], ...]],
        tuple[tuple[int, tuple[int, ...]], ...],
    ] = field(default_factory=dict, init=False, repr=False)
    _regular_taylor_dual_signatures: set[tuple[Any, ...]] = field(
        default_factory=set, init=False, repr=False
    )
    _regular_taylor_source_shape_cache: dict[
        tuple[str, tuple[int, ...], tuple[int, ...]], list[tuple[int, ...]]
    ] = field(default_factory=dict, init=False, repr=False)
    _sparse_regular_source_shape_cache: dict[
        tuple[Any, ...], list[tuple[int, ...]]
    ] = field(default_factory=dict, init=False, repr=False)
    dual_evaluator_build_seconds: float = field(default=0.0, init=False)
    subtraction_formula_build_seconds: float = field(default=0.0, init=False)
    regular_taylor_formula_signature_limit: int = 256
    regular_taylor_formula_volume_limit: int = 8192
    regular_taylor_formula_axis_limit: int = 6
    regular_taylor_dual_shape_limit: int = 256
    regular_taylor_low_signature_sector_threshold: int = 0
    direct_projector_cache_term_threshold: int = 54
    _regular_taylor_signature_version: int = field(default=1, init=False, repr=False)
    regular_taylor_formulas_skipped: int = field(default=0, init=False)
    regular_taylor_formulas_from_curated_cache: int = field(default=0, init=False)
    endpoint_projector_formulas_from_cache: int = field(default=0, init=False)
    endpoint_projector_formulas_generated: int = field(default=0, init=False)
    regular_taylor_formulas_from_cache: int = field(default=0, init=False)
    regular_taylor_formulas_generated: int = field(default=0, init=False)
    endpoint_projector_direct_cache_override_sectors: int = field(default=0, init=False)
    endpoint_projector_direct_cache_override_signatures: int = field(default=0, init=False)
    chain_rule_formula_build_seconds: float = field(default=0.0, init=False)
    chain_rule_formula_signature_limit: int = 256
    chain_rule_formula_output_length_limit: int = 0
    chain_rule_formulas_skipped: int = field(default=0, init=False)
    chain_rule_formulas_from_cache: int = field(default=0, init=False)
    chain_rule_formulas_generated: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Build the scalar U and F evaluators in the declared row order."""
        allowed_modes = {"pregenerate", "lazy", "single-overall", "symbolic-derivatives"}
        if self.dual_evaluator_mode not in allowed_modes:
            raise ValueError(
                f"{self.family}: unsupported dual evaluator mode {self.dual_evaluator_mode!r}"
            )
        params = [S(name) for name in [*self.x_names, *self.parameter_names]]
        if self.skip_evaluator_build:
            self._u_evaluator = None
            self._f_evaluator = None
        else:
            self._u_evaluator = self.u_expr.evaluator(params, jit_compile=self.jit_compile_evaluators)
            self._f_evaluator = self.f_expr.evaluator(params, jit_compile=self.jit_compile_evaluators)
        if self.parametric_representation is None:
            propagator_powers = tuple(1.0 for _ in self.x_names)
            self.parametric_representation = ParametricRepresentation(
                loop_count=1,
                propagator_powers=propagator_powers,
                dimension=EpsilonExpansion(4.0, -2.0),
                gamma_argument=EpsilonExpansion(float(self.f_power_base), -float(self.eps_log_f_coeff)),
                u_exponent=EpsilonExpansion(float(self.u_power_base), float(self.eps_log_u_coeff)),
                f_exponent=EpsilonExpansion(-float(self.f_power_base), float(self.eps_log_f_coeff)),
                parameter_weight_powers=tuple(power - 1.0 for power in propagator_powers),
                prefactor_description="inferred one-loop scalar prefactor",
                convention_description=self.convention_note,
            )

    @property
    def evaluator_parameter_order(self) -> list[str]:
        """Return the input-column order expected by U/F evaluators."""
        return [*self.x_names, *self.parameter_names]

    @property
    def laurent_orders(self) -> list[int]:
        """Return integer epsilon powers covered by the stored coefficient array."""
        orders: list[int] = []
        for label in self.expected_laurent_orders:
            text = str(label).strip()
            if text == "eps^0":
                orders.append(0)
            elif text.startswith("eps^"):
                orders.append(int(text[4:]))
            else:
                raise ValueError(f"{self.family}: unsupported Laurent label {label!r}")
        return orders

    @property
    def laurent_min_order(self) -> int:
        """Return the lowest epsilon power represented by integration arrays."""
        return min(self.laurent_orders)

    @property
    def laurent_max_order(self) -> int:
        """Return the highest epsilon power represented by integration arrays."""
        return max(self.laurent_orders)

    @property
    def coefficient_count(self) -> int:
        """Return the number of Laurent coefficients represented."""
        return self.laurent_max_order - self.laurent_min_order + 1

    @property
    def finite_index(self) -> int:
        """Return the array index corresponding to the finite eps^0 coefficient."""
        return 0 - self.laurent_min_order

    @property
    def training_index(self) -> int:
        """Return the index used for Havana training."""
        return self.coefficient_count - 1

    @staticmethod
    def order_labels(min_order: int, max_order: int) -> list[str]:
        """Build display labels for a contiguous Laurent range."""
        return [f"eps^{order}" for order in range(min_order, max_order + 1)]

    def set_laurent_range(self, min_order: int, max_order: int = 0) -> None:
        """Set a contiguous Laurent range after sector pole depths are known."""
        if min_order > max_order:
            raise ValueError(f"{self.family}: invalid Laurent range {min_order}..{max_order}")
        self.expected_laurent_orders = self.order_labels(min_order, max_order)

    @staticmethod
    def _signature_laurent_orders(fragment: Any) -> tuple[int, ...]:
        """Parse the Laurent-order tail used by formula cache signatures."""
        orders: list[int] = []
        for item in fragment:
            text = str(item)
            if text == "eps^0":
                orders.append(0)
            elif text.startswith("eps^"):
                orders.append(int(text[4:]))
            else:
                orders.append(int(item))
        return tuple(orders)

    def _lookup_formula_with_laurent_superset(
        self,
        formulas: dict[tuple[Any, ...], Any],
        signature: tuple[Any, ...],
    ) -> Any | None:
        """Return a formula whose signature only differs by a larger Laurent range.

        Prepared bundles may be generated through a high order and later loaded
        for a leading-pole-only integration.  Formula signatures include their
        prepared Laurent orders, so strict prepared mode needs this lookup
        relaxation to reuse the serialized superset evaluator instead of trying
        to rebuild a smaller one.
        """
        formula = formulas.get(signature)
        if formula is not None or not self.strict_prepared_bundle or not signature:
            return formula
        prefix = signature[:-1]
        active_orders = set(self._signature_laurent_orders(signature[-1]))
        candidates: list[tuple[int, Any]] = []
        for prepared_signature, prepared_formula in formulas.items():
            if prepared_signature[:-1] != prefix:
                continue
            prepared_orders = set(self._signature_laurent_orders(prepared_signature[-1]))
            if active_orders.issubset(prepared_orders):
                candidates.append((len(prepared_orders), prepared_formula))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _row(self, x: list[float] | tuple[float, ...]) -> list[float]:
        """Build one evaluator row from Feynman parameters and invariants."""
        return [float(value) for value in x] + [float(value) for value in self.parameter_values]

    def _rows(self, x_values: np.ndarray) -> np.ndarray:
        """Build a batched evaluator matrix from mapped Feynman parameters."""
        x_array = np.asarray(x_values, dtype=float)
        if x_array.ndim != 2 or x_array.shape[1] != len(self.x_names):
            raise ValueError(
                f"{self.family}: expected Feynman-parameter array with shape (n,{len(self.x_names)})"
            )
        params = np.broadcast_to(
            np.asarray(self.parameter_values, dtype=float),
            (x_array.shape[0], len(self.parameter_values)),
        )
        return np.concatenate([x_array, params], axis=1)

    def _timed_evaluate(
        self,
        evaluator: Any,
        rows: np.ndarray,
        timing: HotPathTiming | None,
        allow_precision: bool = True,
    ) -> Any:
        """Evaluate a Symbolica evaluator and optionally charge EvalT."""
        precision_digits = None if timing is None or not allow_precision else timing.precision_digits
        start = time.perf_counter()
        if precision_digits is None:
            values = evaluator.evaluate(np.ascontiguousarray(rows))
        else:
            # Symbolica's multiprecision API is single-row oriented.  Keep the
            # ordinary hot path vectorized, and only route near-endpoint rows
            # through evaluate_with_prec when SectorProcessor requests it.
            row_matrix = np.asarray(rows, dtype=float)
            values = [
                evaluator.evaluate_with_prec(
                    [_decimal_real(value, precision_digits) for value in row],
                    precision_digits,
                )
                for row in row_matrix
            ]
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return values

    def _timed_evaluate_complex(
        self,
        evaluator: Any,
        rows: np.ndarray,
        timing: HotPathTiming | None,
    ) -> Any:
        """Evaluate a Symbolica callback with native complex inputs."""
        start = time.perf_counter()
        values = evaluator.evaluate_complex(np.ascontiguousarray(rows))
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return values

    def _timed_evaluate_complex_with_prec(
        self,
        evaluator: Any,
        row: ComplexPreciseRow,
        precision_digits: int,
        timing: HotPathTiming | None,
    ) -> list[ComplexPrecise]:
        """Evaluate one Symbolica callback with arbitrary-precision complex inputs."""
        start = time.perf_counter()
        values = evaluator.evaluate_complex_with_prec(row, precision_digits)
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return [
            (
                decimal_with_precision(value[0], precision_digits),
                decimal_with_precision(value[1], precision_digits),
            )
            for value in values
        ]

    def u_value(self, x: list[float] | tuple[float, ...]) -> complex:
        """Evaluate U at one Feynman-parameter point."""
        return complex(self._u_evaluator.evaluate([self._row(x)])[0][0])

    def f_value(self, x: list[float] | tuple[float, ...]) -> complex:
        """Evaluate F at one Feynman-parameter point."""
        return complex(self._f_evaluator.evaluate([self._row(x)])[0][0])

    def u_values(self, x_values: np.ndarray, timing: HotPathTiming | None = None) -> np.ndarray:
        """Evaluate U for a batch of Feynman-parameter points."""
        rows = self._rows(x_values)
        return np.asarray(self._timed_evaluate(self._u_evaluator, rows, timing), dtype=np.complex128)[:, 0]

    def f_values(self, x_values: np.ndarray, timing: HotPathTiming | None = None) -> np.ndarray:
        """Evaluate F for a batch of Feynman-parameter points."""
        rows = self._rows(x_values)
        return np.asarray(self._timed_evaluate(self._f_evaluator, rows, timing), dtype=np.complex128)[:, 0]

    def _cached_dual_evaluator(
        self,
        cache: dict[tuple[tuple[int, ...], ...], Any],
        scalar_evaluator: Any,
        dual_shape: list[tuple[int, ...]],
        label: str,
    ) -> Any:
        """Return a cached clone of a scalar evaluator dualized to ``dual_shape``."""
        key = tuple(dual_shape)
        evaluator = cache.get(key)
        if evaluator is None:
            if self.strict_prepared_bundle:
                raise RuntimeError(
                    f"{self.family}: missing prepared dual evaluator for shape {key}"
                )
            if self.streaming_evaluator_cache_dir:
                path = self._stream_dual_evaluator_path(label, key)
                if path.is_file():
                    evaluator = _SerializedEvaluatorRef(path)
                    cache[key] = evaluator
                    return evaluator
            start = time.perf_counter()
            evaluator = copy.copy(scalar_evaluator)
            evaluator.dualize([list(mi) for mi in dual_shape])
            if self.streaming_evaluator_cache_dir:
                path = self._stream_dual_evaluator(evaluator, label, key)
                evaluator = _SerializedEvaluatorRef(path)
                gc.collect()
            cache[key] = evaluator
            self.dual_evaluator_build_seconds += time.perf_counter() - start
        return evaluator

    def u_dual_evaluator(self, dual_shape: list[tuple[int, ...]]) -> Any:
        """Return a cached dualized U evaluator for the requested jet shape."""
        return self._cached_dual_evaluator(self._u_dual_evaluators, self._u_evaluator, dual_shape, "u")

    def f_dual_evaluator(self, dual_shape: list[tuple[int, ...]]) -> Any:
        """Return a cached dualized F evaluator for the requested jet shape."""
        # The heavy expression-to-evaluator lowering was already done in
        # __post_init__.  Symbolica evaluators support shallow copying, so we
        # clone the boot-time scalar evaluator and dualize the clone.
        return self._cached_dual_evaluator(self._f_dual_evaluators, self._f_evaluator, dual_shape, "f")

    def _stream_dual_evaluator(
        self,
        evaluator: Any,
        label: str,
        dual_shape: tuple[tuple[int, ...], ...],
    ) -> Path:
        """Persist one generated dual evaluator and return its sidecar path."""
        path = self._stream_dual_evaluator_path(label, dual_shape)
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(evaluator.save(), compresslevel=6))
        return path

    def _stream_dual_evaluator_path(
        self,
        label: str,
        dual_shape: tuple[tuple[int, ...], ...],
    ) -> Path:
        """Return the deterministic streaming sidecar path for one dual shape."""
        root = Path(str(self.streaming_evaluator_cache_dir)).expanduser()
        payload = {
            "family": self.family,
            "label": str(label),
            "shape": [list(multi) for multi in dual_shape],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return root / f"topology_{label}_dual_{digest}.bin.gz"

    @staticmethod
    def _pad_multi_index(multi_index: tuple[int, ...], rank: int) -> tuple[int, ...]:
        """Pad a sector-native multi-index into an envelope dual rank."""
        if len(multi_index) > rank:
            raise ValueError(f"cannot pad multi-index {multi_index} to rank {rank}")
        return tuple([*multi_index, *([0] * (rank - len(multi_index)))])

    def prepare_dual_evaluators(
        self,
        sectors: list[SectorDefinition],
        mode: str | None = None,
        progress: Any | None = None,
    ) -> None:
        """Pregenerate topology-level Taylor evaluators according to ``mode``."""
        selected_mode = mode or self.dual_evaluator_mode
        if selected_mode == "lazy":
            return
        dual_sectors = [sector for sector in sectors if sector.dual_shape]
        if selected_mode == "symbolic-derivatives":
            max_total = self._max_sector_taylor_total_degree(sectors)
            if max_total > 0:
                self._prepare_symbolic_derivative_evaluators("u", max_total)
                self._prepare_symbolic_derivative_evaluators("f", max_total)
            # In symbolic-derivative mode U/F Taylor coefficients are obtained
            # from shared x-space partial derivative evaluators and sector-map
            # Taylor coefficients.  DOT sectors have monomial maps/Jacobians,
            # so SectorDefinition can provide those jets analytically without
            # dualizing a large sector-local evaluator.  Non-monomial fallback
            # paths remain lazy and explicit.
            return
        if selected_mode == "pregenerate":
            unique_shapes: list[list[tuple[int, ...]]] = []
            seen: set[tuple[tuple[int, ...], ...]] = set()
            for sector in dual_sectors:
                key = tuple(sector.dual_shape)
                if key in seen:
                    continue
                seen.add(key)
                unique_shapes.append(list(sector.dual_shape))
            shape_total = len(unique_shapes)
            for shape_index, shape in enumerate(unique_shapes, start=1):
                if progress is not None:
                    progress.update(
                        shape_index - 1,
                        total=shape_total + len(dual_sectors),
                        detail=(
                            "U/F dual shape "
                            f"{shape_index}/{shape_total} len={len(shape)} "
                            f"rank={len(shape[0]) if shape else 0}"
                        ),
                    )
                self.u_dual_evaluator(shape)
                self.f_dual_evaluator(shape)
                if progress is not None:
                    progress.update(
                        shape_index,
                        total=shape_total + len(dual_sectors),
                        detail=(
                            "U/F dual shape "
                            f"{shape_index}/{shape_total} done"
                        ),
                    )
            self._prepare_sector_dual_evaluators(dual_sectors, progress=progress)
            return
        if selected_mode != "single-overall":
            raise ValueError(f"{self.family}: unsupported dual evaluator mode {selected_mode!r}")

        by_dimension: dict[int, list[SectorDefinition]] = {}
        for sector in dual_sectors:
            by_dimension.setdefault(sector.integration_dim, []).append(sector)
        for dimension, dim_sectors in by_dimension.items():
            rank = max(len(mi) for sector in dim_sectors for mi in sector.dual_shape)
            envelope_set = {
                self._pad_multi_index(mi, rank)
                for sector in dim_sectors
                for mi in sector.dual_shape
            }
            envelope = sorted(envelope_set)
            if tuple(0 for _ in range(rank)) in envelope:
                envelope.remove(tuple(0 for _ in range(rank)))
                envelope.insert(0, tuple(0 for _ in range(rank)))
            self._overall_dual_shapes[dimension] = envelope
            envelope_index = {mi: index for index, mi in enumerate(envelope)}
            total = len(dim_sectors) + 1
            for index, sector in enumerate(dim_sectors, start=1):
                if progress is not None and (index == 1 or index % 25 == 0 or index == len(dim_sectors)):
                    progress.update(
                        index - 1,
                        total=total,
                        detail=(
                            f"overall shape dim={dimension} sector {index}/{len(dim_sectors)} "
                            f"len={len(envelope)}"
                        ),
                    )
                key = (sector.integration_dim, tuple(sector.dual_shape))
                self._overall_dual_indices[key] = [
                    envelope_index[self._pad_multi_index(mi, rank)]
                    for mi in sector.dual_shape
                ]
                if not _sector_has_analytic_taylor_for_shape(sector):
                    sector.prepare_dual_evaluators_for_shape(envelope)
                if progress is not None and (index % 25 == 0 or index == len(dim_sectors)):
                    progress.update(
                        index,
                        total=total,
                        detail=(
                            f"overall sector duals done {index}/{len(dim_sectors)}"
                            if not _sector_has_analytic_taylor_for_shape(sector)
                            else f"overall sector duals analytic {index}/{len(dim_sectors)}"
                        ),
                    )
            if progress is not None:
                progress.update(
                    len(dim_sectors),
                    total=total,
                    detail=(
                        f"U/F overall dual shape dim={dimension} "
                        f"len={len(envelope)} rank={rank}"
                    ),
                )
            self.u_dual_evaluator(envelope)
            self.f_dual_evaluator(envelope)
            if progress is not None:
                progress.update(
                    total,
                    total=total,
                    detail=f"U/F overall dual shape dim={dimension} done",
                )

    def _prepare_sector_dual_evaluators(
        self,
        sectors: list[SectorDefinition],
        progress: Any | None = None,
    ) -> None:
        """Pregenerate sector-local map/Jacobian dual callbacks."""
        total = len(sectors)
        offset = 0
        if progress is not None:
            # In pregenerate mode the progress stage may already have consumed
            # one step per unique topology-level U/F dual shape.  Keep the
            # displayed counter monotonic by starting sector-local callbacks
            # after those shape steps when available.
            try:
                offset = int(getattr(progress, "_stage_current", 0))
            except Exception:
                offset = 0
        for index, sector in enumerate(sectors, start=1):
            analytic_sector_taylor = _sector_has_analytic_taylor_for_shape(sector)
            if progress is not None and (index == 1 or index % 25 == 0 or index == total):
                progress.update(
                    offset + index - 1,
                    total=max(offset + total, total),
                    detail=(
                        f"sector duals {sector.name} {index}/{total}"
                        if not analytic_sector_taylor
                        else f"sector duals {sector.name} analytic {index}/{total}"
                    ),
                )
            if not analytic_sector_taylor:
                sector.prepare_dual_evaluators_for_shape(sector.dual_shape)
            if progress is not None and (index % 25 == 0 or index == total):
                progress.update(
                    offset + index,
                    total=max(offset + total, total),
                    detail=(
                        f"sector duals {sector.name} done {index}/{total}"
                        if not analytic_sector_taylor
                        else f"sector duals {sector.name} analytic {index}/{total}"
                    ),
                )

    def prepare_subtraction_formulas(
        self,
        sectors: list[SectorDefinition],
        progress: Any | None = None,
    ) -> None:
        """Pregenerate Symbolica evaluators for all singular-sector formulas."""
        pending: list[tuple[SectorDefinition, tuple[Any, ...]]] = []
        seen: set[tuple[Any, ...]] = set()
        for sector in sectors:
            if not sector.singular_axes:
                continue
            signature = self.subtraction_formula_signature(sector)
            if signature in self._subtraction_formulas or signature in seen:
                continue
            seen.add(signature)
            pending.append((sector, signature))

        if progress is not None:
            progress.start_stage(
                "Symbolica subtraction formula build",
                detail=f"{len(pending)} formula signature(s)",
                total=len(pending),
            )
        start_all = time.perf_counter()
        try:
            for index, (sector, signature) in enumerate(pending, start=1):
                if progress is not None:
                    progress.update(
                        index - 1,
                        total=len(pending),
                        detail=f"{sector.name} signature {index}/{len(pending)}",
                    )
                start = time.perf_counter()
                formula = build_subtraction_formula(self, sector, signature)
                elapsed = time.perf_counter() - start
                formula.build_seconds = elapsed
                self._subtraction_formulas[signature] = formula
                self.subtraction_formula_build_seconds += elapsed
                if progress is not None:
                    progress.update(
                        index,
                        total=len(pending),
                        detail=f"{sector.name} done in {elapsed:.3g}s",
                    )
        finally:
            if progress is not None:
                progress.finish_stage(
                    "Symbolica subtraction formula build",
                    time.perf_counter() - start_all,
                    detail=f"{len(pending)} formula signature(s)",
                )

    def subtraction_formula_signature(self, sector: SectorDefinition) -> tuple[Any, ...]:
        """Return the formula cache key for one singular sector."""
        endpoint_powers = tuple(
            (self.endpoint_power(sector, axis).base, self.endpoint_power(sector, axis).eps_coeff)
            for axis in sector.singular_axes
        )
        return (
            sector.integration_dim,
            tuple(sector.singular_axes),
            tuple(sector.f_monomial_powers),
            tuple(sector.u_monomial_powers),
            tuple(sector.jacobian_monomial_powers),
            tuple(sector.measure_monomial_powers),
            tuple(sector.numerator_monomial_powers),
            tuple(sector.endpoint_taylor_orders),
            tuple(sector.variable_names),
            tuple(sector.dual_shape),
            endpoint_powers,
            self.u_power_base,
            self.f_power_base,
            self.eps_log_u_coeff,
            self.eps_log_f_coeff,
            tuple(self.expected_laurent_orders),
        )

    def subtraction_formula_for(self, sector: SectorDefinition) -> "SubtractionFormulaDefinition":
        """Return a pregenerated subtraction formula or fail clearly."""
        signature = self.subtraction_formula_signature(sector)
        formula = self._lookup_formula_with_laurent_superset(
            self._subtraction_formulas,
            signature,
        )
        if formula is None:
            raise RuntimeError(
                f"{sector.name}: missing pregenerated subtraction formula; "
                "call TopologyDefinition.prepare_subtraction_formulas(...) before integration"
            )
        return formula

    def prepare_endpoint_projector_formulas(
        self,
        sectors: list[SectorDefinition],
        progress: Any | None = None,
    ) -> None:
        """Pregenerate lower-signature endpoint projector evaluators.

        Unlike the full formula backend, these expressions only encode the
        plus/Taylor endpoint algebra.  Sector-specific U/F/J information is
        supplied at runtime as regular-function Taylor coefficients.
        """
        pending: list[tuple[SectorDefinition, tuple[Any, ...]]] = []
        seen: set[tuple[Any, ...]] = set()
        override_signatures: set[tuple[Any, ...]] = set()
        override_sector_count = 0
        for sector in sectors:
            if not sector.singular_axes:
                continue
            signature = self.endpoint_projector_signature(sector)
            if self.ibp_reduce_to_log_endpoint and not bool(signature[2]):
                override_sector_count += 1
                override_signatures.add(signature)
            if signature in self._endpoint_projector_formulas or signature in seen:
                continue
            seen.add(signature)
            pending.append((sector, signature))
        self.endpoint_projector_direct_cache_override_sectors = override_sector_count
        self.endpoint_projector_direct_cache_override_signatures = len(override_signatures)

        if progress is not None:
            progress.start_stage(
                "Symbolica endpoint projector build",
                detail=f"{len(pending)} endpoint signature(s)",
                total=len(pending),
            )
        start_all = time.perf_counter()
        try:
            for index, (sector, signature) in enumerate(pending, start=1):
                if progress is not None:
                    progress.update(
                        index - 1,
                        total=len(pending),
                        detail=f"{sector.name} endpoint signature {index}/{len(pending)}",
                    )
                start = time.perf_counter()
                formula = build_endpoint_projector_formula(self, sector, signature)
                elapsed = time.perf_counter() - start
                formula.build_seconds = elapsed
                self._endpoint_projector_formulas[signature] = formula
                self.subtraction_formula_build_seconds += elapsed
                if progress is not None:
                    progress.update(
                        index,
                        total=len(pending),
                        detail=f"{sector.name} endpoint projector done in {elapsed:.3g}s",
                    )
        finally:
            if progress is not None:
                progress.finish_stage(
                    "Symbolica endpoint projector build",
                    time.perf_counter() - start_all,
                    detail=(
                        f"{len(pending)} endpoint signature(s), "
                        f"{self.endpoint_projector_formulas_from_cache} cache hit(s), "
                        f"{self.endpoint_projector_formulas_generated} generated"
                    ),
                )

    def _endpoint_projector_signature_components(
        self,
        sector: SectorDefinition,
    ) -> tuple[list[tuple[int, float]], list[int]]:
        """Return validated endpoint powers and Taylor orders for one sector."""
        endpoint_powers: list[tuple[int, float]] = []
        taylor_orders: list[int] = []
        for axis in sector.singular_axes:
            endpoint_power = self.endpoint_power(sector, axis)
            rounded_base = round(endpoint_power.base)
            if endpoint_power.base >= -1.0e-12:
                raise ValueError(
                    f"{sector.name}: declared singular axis {sector.variable_names[axis]} "
                    f"has non-singular endpoint power y^({endpoint_power.as_text()})"
                )
            if abs(endpoint_power.base - rounded_base) > 1.0e-12:
                raise ValueError(
                    f"{sector.name}: unsupported non-integer endpoint power "
                    f"y^({endpoint_power.as_text()})"
                )
            if abs(endpoint_power.eps_coeff) <= 1.0e-15:
                raise ValueError(
                    f"{sector.name}: endpoint power y^({endpoint_power.as_text()}) "
                    "has no epsilon regulator"
                )
            required_order = int(-rounded_base - 1)
            declared_order = int(sector.endpoint_taylor_orders[axis])
            if declared_order < required_order:
                raise ValueError(
                    f"{sector.name}: endpoint Taylor order {declared_order} on "
                    f"{sector.variable_names[axis]} is too small; need {required_order}"
                )
            endpoint_powers.append((int(rounded_base), float(endpoint_power.eps_coeff)))
            taylor_orders.append(required_order)
        return endpoint_powers, taylor_orders

    def _endpoint_projector_signature_from_components(
        self,
        sector: SectorDefinition,
        endpoint_powers: list[tuple[int, float]],
        taylor_orders: list[int],
        *,
        use_ibp: bool,
    ) -> tuple[Any, ...]:
        """Build the topology-independent endpoint-projector signature."""
        return (
            "endpoint-projector",
            2,
            bool(use_ibp),
            len(sector.singular_axes),
            tuple(endpoint_powers),
            tuple(taylor_orders),
            tuple(self.laurent_orders),
        )

    def _effective_endpoint_projector_uses_ibp(
        self,
        sector: SectorDefinition,
        endpoint_powers: list[tuple[int, float]],
        taylor_orders: list[int],
    ) -> bool:
        """Choose IBP or a shipped direct projector for this endpoint signature."""
        if not self.ibp_reduce_to_log_endpoint:
            return False
        threshold = int(self.direct_projector_cache_term_threshold)
        if threshold <= 0:
            return True
        ibp_term_count = len(_ibp_endpoint_projector_terms(endpoint_powers, self.laurent_orders))
        if ibp_term_count < threshold:
            return True
        direct_signature = self._endpoint_projector_signature_from_components(
            sector,
            endpoint_powers,
            taylor_orders,
            use_ibp=False,
        )
        return not endpoint_projector_formula_has_curated_cache(direct_signature)

    def endpoint_projector_signature(self, sector: SectorDefinition) -> tuple[Any, ...]:
        """Return the cache key for the endpoint-only projector formula."""
        endpoint_powers, taylor_orders = self._endpoint_projector_signature_components(sector)
        use_ibp = self._effective_endpoint_projector_uses_ibp(
            sector,
            endpoint_powers,
            taylor_orders,
        )
        return self._endpoint_projector_signature_from_components(
            sector,
            endpoint_powers,
            taylor_orders,
            use_ibp=use_ibp,
        )

    def endpoint_projector_formula_for(
        self,
        sector: SectorDefinition,
    ) -> "EndpointProjectorFormulaDefinition":
        """Return a pregenerated endpoint projector formula or fail clearly."""
        signature = self.endpoint_projector_signature(sector)
        formula = self._lookup_formula_with_laurent_superset(
            self._endpoint_projector_formulas,
            signature,
        )
        if formula is None and self.strict_prepared_bundle:
            endpoint_powers, taylor_orders = self._endpoint_projector_signature_components(sector)
            for use_ibp in (False, True):
                alternate_signature = self._endpoint_projector_signature_from_components(
                    sector,
                    endpoint_powers,
                    taylor_orders,
                    use_ibp=use_ibp,
                )
                formula = self._lookup_formula_with_laurent_superset(
                    self._endpoint_projector_formulas,
                    alternate_signature,
                )
                if formula is not None:
                    break
        if formula is None:
            raise RuntimeError(
                f"{sector.name}: missing pregenerated endpoint projector formula; "
                "call TopologyDefinition.prepare_endpoint_projector_formulas(...) before integration"
            )
        return formula

    def prepare_regular_taylor_formulas(
        self,
        sectors: list[SectorDefinition],
        progress: Any | None = None,
    ) -> None:
        """Pregenerate Symbolica evaluators for regular ``g_s`` coefficients.

        Endpoint projectors are topology-independent, but their inputs are
        Taylor coefficients of the regular function multiplying the endpoint
        monomial.  This step prepares that coefficient algebra too, so the hot
        integration path only assembles black-box U/F/J Taylor values and calls
        pregenerated evaluators.
        """
        self._regular_taylor_signature_version = (
            3
            if len(sectors) > self.regular_taylor_low_signature_sector_threshold
            else 1
        )
        pending_by_signature: dict[tuple[Any, ...], SectorDefinition] = {}
        source_request_specs: list[
            tuple[SectorDefinition, tuple[Any, ...], tuple[int, ...], tuple[int, ...]]
        ] = []
        seen: set[tuple[Any, ...]] = set()
        seen_source_spec: set[tuple[str, tuple[int, ...], tuple[int, ...]]] = set()
        for sector in sectors:
            if not sector.singular_axes:
                continue
            for signature, zero_positions, max_orders in self.regular_taylor_requests_for_sector(sector):
                if signature not in self._regular_taylor_formulas and signature not in seen:
                    seen.add(signature)
                    pending_by_signature[signature] = sector
                source_key = (sector.name, tuple(zero_positions), tuple(max_orders))
                if source_key not in seen_source_spec:
                    seen_source_spec.add(source_key)
                    source_request_specs.append(
                        (sector, signature, tuple(zero_positions), tuple(max_orders))
                    )
        pending_all = [
            (sector, signature)
            for signature, sector in pending_by_signature.items()
        ]
        pending_all.sort(
            key=lambda item: (
                _regular_taylor_signature_volume(item[1]),
                int(item[1][2]) if len(item[1]) > 2 else 0,
                repr(item[1]),
            )
        )
        curated_signatures = {
            signature
            for _sector, signature in pending_all
            if regular_taylor_formula_has_curated_cache(signature)
        }
        cached_signatures = set(curated_signatures) | {
            signature
            for _sector, signature in pending_all
            if regular_taylor_formula_has_cache(signature)
        }
        cold_pending = [
            (sector, signature)
            for sector, signature in pending_all
            if signature not in cached_signatures
            and (
                _regular_taylor_signature_volume(signature)
                <= self.regular_taylor_formula_volume_limit
                and _regular_taylor_signature_axis_count(signature)
                <= self.regular_taylor_formula_axis_limit
            )
        ]
        cached_pending = [
            (sector, signature)
            for sector, signature in pending_all
            if signature in cached_signatures
        ]
        if len(cold_pending) > self.regular_taylor_formula_signature_limit:
            cold_pending = cold_pending[: self.regular_taylor_formula_signature_limit]
        pending = cached_pending + cold_pending
        prepared_signatures = {signature for _sector, signature in pending}
        skipped_pending = [
            (sector, signature)
            for sector, signature in pending_all
            if signature not in prepared_signatures
        ]
        prepared_source_request_specs = [
            spec for spec in source_request_specs if spec[1] in prepared_signatures
        ]
        if self.dual_evaluator_mode not in {"symbolic-derivatives", "lazy"}:
            # Skipped regular-Taylor formula signatures still use the same
            # black-box U/F Taylor coefficients in the fallback path.  Strict
            # prepared bundles therefore need their source dual evaluators too;
            # otherwise integration would have to generate them lazily.
            prepared_source_request_specs = list(source_request_specs)
        self.regular_taylor_formulas_skipped = len(skipped_pending)
        self.regular_taylor_formulas_from_curated_cache = len(
            prepared_signatures & curated_signatures
        )

        if not pending:
            if progress is not None:
                progress.start_stage(
                    "Symbolica regular Taylor build",
                    detail=(
                        f"skipped {len(skipped_pending)} regular signature(s) "
                        f"from {len(source_request_specs)} source request(s); "
                        f"axis limit={self.regular_taylor_formula_axis_limit}, "
                        f"volume limit={self.regular_taylor_formula_volume_limit}"
                    ),
                    total=1,
                )
                progress.update(
                    1,
                    total=1,
                    detail="falling back to Python regular Taylor assembly",
                )
                progress.finish_stage(
                    "Symbolica regular Taylor build",
                    0.0,
                    detail=(
                        f"skipped {len(skipped_pending)} regular signature(s) "
                        f"from {len(source_request_specs)} source request(s)"
                    ),
                )
            return

        dual_fast_path = (
            self.dual_evaluator_mode == "symbolic-derivatives"
            and len(pending) <= self.regular_taylor_dual_shape_limit
            and len(prepared_source_request_specs) <= self.regular_taylor_dual_shape_limit
        )
        prepare_source_duals = self.dual_evaluator_mode != "symbolic-derivatives" or dual_fast_path
        if progress is not None:
            progress.start_stage(
                "Symbolica regular Taylor build",
                detail=(
                    f"{len(pending)} regular signature(s)"
                    + (
                        f", {self.regular_taylor_formulas_from_curated_cache} curated"
                        if self.regular_taylor_formulas_from_curated_cache
                        else ""
                    )
                    + (
                        f", skipped {len(skipped_pending)} hard signature(s)"
                        if skipped_pending
                        else ""
                    )
                ),
                total=len(pending),
            )
        start_all = time.perf_counter()
        try:
            for index, (sector, signature) in enumerate(pending, start=1):
                if progress is not None:
                    progress.update(
                        index - 1,
                        total=len(pending),
                        detail=f"{sector.name} regular signature {index}/{len(pending)}",
                    )
                start = time.perf_counter()
                formula = build_regular_taylor_formula(self, sector, signature)
                formula.dual_shape = _regular_formula_dual_shape(formula)
                elapsed = time.perf_counter() - start
                formula.build_seconds = elapsed
                self._regular_taylor_formulas[signature] = formula
                self.subtraction_formula_build_seconds += elapsed
                if progress is not None:
                    progress.update(
                        index,
                        total=len(pending),
                        detail=f"{sector.name} regular Taylor done in {elapsed:.3g}s",
                    )
            if prepare_source_duals:
                for sector, signature, zero_positions, max_orders in prepared_source_request_specs:
                    formula = self._regular_taylor_formulas.get(signature)
                    formula_version = int(signature[1]) if len(signature) > 1 else 1
                    sector_shapes: list[list[tuple[int, ...]]] = []
                    topology_u_shapes: list[list[tuple[int, ...]]] = []
                    topology_f_shapes: list[list[tuple[int, ...]]] = []
                    if formula is not None and formula_version <= 1:
                        source_shape = list(formula.dual_shape)
                        sector_shapes.append(source_shape)
                        topology_u_shapes.append(source_shape)
                        topology_f_shapes.append(source_shape)
                    elif formula is not None and formula_version >= 3:
                        canonical_positions = _regular_taylor_canonical_positions(max_orders)
                        residual_multis = {
                            _regular_taylor_canonical_to_original(tuple(multi), canonical_positions)
                            for kind, multi in formula.input_layout
                            if kind in {"u", "f"}
                        }
                        jacobian_multis = {
                            _regular_taylor_canonical_to_original(tuple(multi), canonical_positions)
                            for kind, multi in formula.input_layout
                            if kind == "j"
                        }
                        source_shape = self.sparse_regular_source_shape_from_multis(
                            sector,
                            zero_positions,
                            residual_multis,
                            jacobian_multis,
                        )
                        sector_shapes.append(source_shape)
                        topology_u_shapes.append(source_shape)
                        topology_f_shapes.append(source_shape)
                    elif formula is None and formula_version >= 3:
                        # Hard six-axis signatures can deliberately skip the
                        # regular formula evaluator and use the generic sparse
                        # fallback at runtime.  Strict prepared bundles still
                        # need every source evaluator used by that fallback.
                        canonical_positions = _regular_taylor_canonical_positions(max_orders)
                        requested_multis = {
                            _regular_taylor_canonical_to_original(tuple(multi), canonical_positions)
                            for multi, _regular_order in signature[3]
                        }
                        residual_multis = _ancestor_closed_multi_set(
                            requested_multis,
                            len(max_orders),
                        )
                        jacobian_shape = _ordered_multi_shape(set(residual_multis), len(max_orders))
                        u_source_shape = self.sparse_regular_source_shape_for_monomial_powers(
                            sector,
                            zero_positions,
                            residual_multis,
                            sector.u_monomial_powers,
                        )
                        f_source_shape = self.sparse_regular_source_shape_for_monomial_powers(
                            sector,
                            zero_positions,
                            residual_multis,
                            sector.f_monomial_powers,
                        )
                        sector_shapes.extend([jacobian_shape, u_source_shape, f_source_shape])
                        topology_u_shapes.append(u_source_shape)
                        topology_f_shapes.append(f_source_shape)
                    else:
                        source_shape = self.regular_taylor_source_shape(
                            sector,
                            zero_positions,
                            max_orders,
                        )
                        sector_shapes.append(source_shape)
                        topology_u_shapes.append(source_shape)
                        topology_f_shapes.append(source_shape)
                    analytic_sector_taylor = _sector_has_analytic_taylor_for_shape(sector)
                    if (
                        self.dual_evaluator_mode != "symbolic-derivatives"
                        or dual_fast_path
                    ) and not analytic_sector_taylor:
                        seen_sector_shapes: set[tuple[tuple[int, ...], ...]] = set()
                        for shape in sector_shapes:
                            shape_key = tuple(shape)
                            if shape_key in seen_sector_shapes:
                                continue
                            seen_sector_shapes.add(shape_key)
                            sector.prepare_dual_evaluators_for_shape(shape)
                    if self.dual_evaluator_mode not in {"lazy", "symbolic-derivatives"}:
                        for shape in topology_u_shapes:
                            self.u_dual_evaluator(shape)
                        for shape in topology_f_shapes:
                            self.f_dual_evaluator(shape)
                    elif dual_fast_path:
                        for shape in topology_u_shapes:
                            self.u_dual_evaluator(shape)
                        for shape in topology_f_shapes:
                            self.f_dual_evaluator(shape)
                        self._regular_taylor_dual_signatures.add(signature)
        finally:
            if progress is not None:
                progress.finish_stage(
                    "Symbolica regular Taylor build",
                    time.perf_counter() - start_all,
                    detail=(
                        f"{len(pending)} regular signature(s)"
                        f", {self.regular_taylor_formulas_from_cache} cache hit(s)"
                        f", {self.regular_taylor_formulas_generated} generated"
                        + (
                            f", {self.regular_taylor_formulas_from_curated_cache} curated"
                            if self.regular_taylor_formulas_from_curated_cache
                            else ""
                        )
                        + (
                            f", skipped {len(skipped_pending)} hard signature(s)"
                            if skipped_pending
                            else ""
                        )
                    ),
                )

    def prepare_chain_rule_formulas(
        self,
        sectors: list[SectorDefinition],
        progress: Any | None = None,
    ) -> None:
        """Pregenerate mapped-derivative composition formulas.

        This applies only to symbolic-derivative mode.  U/F derivative
        evaluators are shared topology-level black boxes; these formulas own
        only the sector-map chain rule that turns those derivative values into
        Taylor coefficients in the sector variables.
        """
        if self.dual_evaluator_mode != "symbolic-derivatives":
            return

        requests: dict[
            tuple[Any, ...],
            tuple[str, str, tuple[tuple[int, ...], ...]],
        ] = {}
        request_limit = int(self.chain_rule_formula_signature_limit)
        sector_by_name = {sector.name: sector for sector in sectors}

        def add_chain_request(
            sector_for_request: SectorDefinition,
            polynomial: str,
            shape: list[tuple[int, ...]],
        ) -> None:
            shape_tuple = tuple(tuple(int(value) for value in multi) for multi in shape)
            signature = self._chain_rule_formula_signature(
                sector_for_request,
                polynomial,
                list(shape_tuple),
            )
            requests.setdefault(
                signature,
                (sector_for_request.name, polynomial, shape_tuple),
            )

        for sector in sectors:
            if not sector.singular_axes:
                continue
            try:
                formula = self.endpoint_projector_formula_for(sector)
            except RuntimeError:
                continue
            if formula.ibp_reduce_to_log_endpoint:
                shared_max_orders = _ibp_shared_max_orders_for_formula(sector, formula)
                shared_output_pairs = _ibp_shared_output_pairs_for_formula(sector, formula)
                fallback_by_zero: dict[
                    tuple[int, ...],
                    list[
                        tuple[
                            tuple[tuple[int, ...], tuple[int, ...]],
                            tuple[int, ...],
                            tuple[tuple[tuple[int, ...], int], ...],
                        ]
                    ],
                ] = {}
                for key, max_orders in shared_max_orders.items():
                    boundary, zero = key
                    output_pairs = shared_output_pairs.get(key, ())
                    signature = self.regular_taylor_signature(
                        sector,
                        zero_positions=zero,
                        max_orders=max_orders,
                        output_pairs=output_pairs,
                    )
                    if signature in self._regular_taylor_formulas:
                        formula_regular = self._regular_taylor_formulas[signature]
                        formula_version = int(signature[1]) if len(signature) > 1 else 1
                        if formula_version >= 3:
                            canonical_positions = _regular_taylor_canonical_positions(max_orders)
                            residual_multis = {
                                _regular_taylor_canonical_to_original(tuple(multi), canonical_positions)
                                for kind, multi in formula_regular.input_layout
                                if kind in {"u", "f"}
                            }
                            jacobian_multis = {
                                _regular_taylor_canonical_to_original(tuple(multi), canonical_positions)
                                for kind, multi in formula_regular.input_layout
                                if kind == "j"
                            }
                            source_shape = self.sparse_regular_source_shape_from_multis(
                                sector,
                                zero,
                                residual_multis,
                                jacobian_multis,
                            )
                        else:
                            source_shape = self.regular_taylor_source_shape(
                                sector,
                                zero,
                                max_orders,
                            )
                        add_chain_request(sector, "u", source_shape)
                        add_chain_request(sector, "f", source_shape)
                    else:
                        # Do not derive chain-rule formulas for regular
                        # Taylor signatures that were deliberately skipped.
                        # The fallback source shapes can be enormous for
                        # six-axis triple-box sectors.  In strict prepared
                        # bundles those sectors use the Python regular
                        # composer rather than triggering runtime formula
                        # generation.
                        continue
                iterable = []
                for zero, entries in fallback_by_zero.items():
                    entries.sort(key=lambda item: (len(item[0][0]), item[0][0], item[1]))
                    if not entries:
                        continue
                    envelope_orders = tuple(
                        max(int(max_orders[position]) for _key, max_orders, _pairs in entries)
                        for position in range(len(sector.singular_axes))
                    )
                    output_pair_set: set[tuple[tuple[int, ...], int]] = set()
                    for _key, _max_orders, output_pairs in entries:
                        output_pair_set.update(output_pairs)
                    union_output_pairs = tuple(
                        sorted(
                            output_pair_set,
                            key=lambda item: (item[1], sum(item[0]), item[0]),
                        )
                    )
                    iterable.append((zero, envelope_orders, union_output_pairs))
            else:
                groups: dict[
                    tuple[tuple[int, ...], tuple[int, ...]],
                    list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]],
                ] = {}
                for entry in formula.coefficient_layout:
                    boundary, zero, _multi_index, _regular_order = entry
                    groups.setdefault((boundary, zero), []).append(entry)
                iterable = []
                for (_boundary, zero), entries in groups.items():
                    max_orders = tuple(
                        max(int(entry[2][position]) for entry in entries)
                        for position in range(len(sector.singular_axes))
                    )
                    output_pairs = tuple(
                        sorted(
                            ((tuple(entry[2]), int(entry[3])) for entry in entries),
                            key=lambda item: (item[1], sum(item[0]), item[0]),
                        )
                    )
                    iterable.append((zero, max_orders, output_pairs))

            for zero, max_orders, output_pairs in iterable:
                signature = self.regular_taylor_signature(
                    sector,
                    zero_positions=zero,
                    max_orders=max_orders,
                    output_pairs=output_pairs,
                )
                if signature in self._regular_taylor_formulas:
                    formula_regular = self._regular_taylor_formulas[signature]
                    formula_version = int(signature[1]) if len(signature) > 1 else 1
                    if formula_version >= 3:
                        canonical_positions = _regular_taylor_canonical_positions(max_orders)
                        residual_multis = {
                            _regular_taylor_canonical_to_original(tuple(multi), canonical_positions)
                            for kind, multi in formula_regular.input_layout
                            if kind in {"u", "f"}
                        }
                        jacobian_multis = {
                            _regular_taylor_canonical_to_original(tuple(multi), canonical_positions)
                            for kind, multi in formula_regular.input_layout
                            if kind == "j"
                        }
                        source_shape = self.sparse_regular_source_shape_from_multis(
                            sector,
                            zero,
                            residual_multis,
                            jacobian_multis,
                        )
                    else:
                        source_shape = self.regular_taylor_source_shape(
                            sector,
                            zero,
                            max_orders,
                        )
                    add_chain_request(sector, "u", source_shape)
                    add_chain_request(sector, "f", source_shape)
                    continue

                # Chain-rule formulas are only useful when the regular
                # Taylor algebra that consumes them is also prepared.  Missing
                # regular signatures are handled by the Python fallback path;
                # generating chain formulas for those source shapes caused the
                # triple-box preparation to spend time and memory on formulas
                # that no prepared regular evaluator would actually call.
                continue
            if len(requests) > request_limit:
                # For huge all-sector triple-box runs, it is enough to know
                # that the runtime-ready chain-rule layer would be too large.
                # Stop collecting immediately rather than spending generation
                # time enumerating thousands of formulas that will be skipped.
                break

        output_length_limit = int(self.chain_rule_formula_output_length_limit)
        oversized_skipped = 0
        filtered_requests: list[
            tuple[str, str, tuple[tuple[int, ...], ...]]
        ] = []
        for signature, request in requests.items():
            _sector_name, _polynomial, shape_tuple = request
            if output_length_limit > 0 and len(shape_tuple) > output_length_limit:
                cached = _load_chain_rule_formula_from_cache(
                    signature,
                    load_evaluators=False,
                )
                if cached is None:
                    oversized_skipped += 1
                    continue
            filtered_requests.append(request)

        ordered_requests = sorted(
            filtered_requests,
            key=lambda item: (len(item[2]), len(item[2][0]) if item[2] else 0, item[0], item[1]),
        )
        self.chain_rule_formulas_skipped = oversized_skipped
        if not ordered_requests:
            if progress is not None and oversized_skipped:
                progress.start_stage(
                    "Symbolica chain-rule build",
                    detail=(
                        f"skipped {oversized_skipped} mapped derivative formula(s); "
                        f"output length limit={output_length_limit}"
                    ),
                    total=1,
                )
                progress.update(
                    1,
                    total=1,
                    detail="all cold chain-rule requests exceeded output-length guard",
                )
                progress.finish_stage(
                    "Symbolica chain-rule build",
                    0.0,
                    detail=(
                        f"skipped {oversized_skipped} mapped derivative formula(s); "
                        f"output length limit={output_length_limit}"
                    ),
                )
            return
        if len(ordered_requests) > request_limit:
            self.chain_rule_formulas_skipped = oversized_skipped + len(ordered_requests)
            if progress is not None:
                progress.start_stage(
                    "Symbolica chain-rule build",
                    detail=(
                        f"skipped {len(ordered_requests)} mapped derivative formula(s); "
                        f"limit={self.chain_rule_formula_signature_limit}"
                    ),
                    total=1,
                )
                progress.update(
                    1,
                    total=1,
                    detail="falling back to lazy chain-rule formula construction",
                )
                progress.finish_stage(
                    "Symbolica chain-rule build",
                    0.0,
                    detail=(
                        f"skipped {len(ordered_requests)} mapped derivative formula(s), "
                        f"{oversized_skipped} already skipped by output length"
                    ),
                )
            return
        if progress is not None:
            progress.start_stage(
                "Symbolica chain-rule build",
                detail=f"{len(ordered_requests)} mapped derivative formula(s)",
                total=len(ordered_requests),
            )
        start_all = time.perf_counter()
        try:
            for index, (sector_name, polynomial, shape_tuple) in enumerate(ordered_requests, start=1):
                sector = sector_by_name[sector_name]
                shape = [tuple(multi) for multi in shape_tuple]
                if progress is not None:
                    progress.update(
                        index - 1,
                        total=len(ordered_requests),
                        detail=(
                            f"{sector.name} {polynomial.upper()} chain "
                            f"{index}/{len(ordered_requests)} len={len(shape)}"
                        ),
                    )
                formula = self.chain_rule_formula_for(sector, polynomial, shape)
                if progress is not None:
                    progress.update(
                        index,
                        total=len(ordered_requests),
                        detail=(
                            f"{sector.name} {polynomial.upper()} chain done "
                            f"in {formula.build_seconds:.3g}s"
                        ),
                    )
        finally:
            if progress is not None:
                progress.finish_stage(
                    "Symbolica chain-rule build",
                    time.perf_counter() - start_all,
                    detail=(
                        f"{len(ordered_requests)} mapped derivative formula(s), "
                        f"{self.chain_rule_formulas_from_cache} cache hit(s), "
                        f"{self.chain_rule_formulas_generated} generated"
                        + (
                            f", {oversized_skipped} skipped by output length"
                            if oversized_skipped
                            else ""
                        )
                    ),
                )

    def regular_taylor_signatures_for_sector(
        self,
        sector: SectorDefinition,
    ) -> list[tuple[Any, ...]]:
        """Return every regular Taylor signature needed by one sector."""
        return sorted(
            {
                signature
                for signature, _zero_positions, _max_orders in self.regular_taylor_requests_for_sector(sector)
            },
            key=str,
        )

    def regular_taylor_requests_for_sector(
        self,
        sector: SectorDefinition,
    ) -> list[tuple[tuple[Any, ...], tuple[int, ...], tuple[int, ...]]]:
        """Return regular formula requests as ``(signature, zero, max_orders)``."""
        if not sector.singular_axes:
            return []
        formula = self.endpoint_projector_formula_for(sector)
        requests: set[tuple[tuple[Any, ...], tuple[int, ...], tuple[int, ...]]] = set()
        if formula.ibp_reduce_to_log_endpoint:
            shared_max_orders = _ibp_shared_max_orders_for_formula(sector, formula)
            output_pairs_by_key = _ibp_shared_output_pairs_for_formula(sector, formula)
            for (_boundary, zero), max_orders in shared_max_orders.items():
                output_pairs = output_pairs_by_key.get((_boundary, zero), ())
                signature = self.regular_taylor_signature(
                    sector,
                    zero_positions=zero,
                    max_orders=max_orders,
                    output_pairs=output_pairs,
                )
                requests.add((signature, tuple(zero), tuple(int(order) for order in max_orders)))
        else:
            groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[Any, ...]]] = {}
            for key in formula.coefficient_layout:
                boundary, zero, _multi_index, _regular_order = key
                groups.setdefault((boundary, zero), []).append(key)
            for (_boundary, zero), entries in groups.items():
                max_orders = tuple(
                    max(int(entry[2][position]) for entry in entries)
                    for position in range(len(sector.singular_axes))
                )
                output_pairs = tuple(
                    sorted(
                        ((tuple(entry[2]), int(entry[3])) for entry in entries),
                        key=lambda item: (item[1], sum(item[0]), item[0]),
                    )
                )
                signature = self.regular_taylor_signature(
                    sector,
                    zero_positions=zero,
                    max_orders=max_orders,
                    output_pairs=output_pairs,
                )
                requests.add((signature, tuple(zero), tuple(int(order) for order in max_orders)))
        return sorted(requests, key=str)

    def regular_taylor_signature(
        self,
        sector: SectorDefinition,
        zero_positions: tuple[int, ...],
        max_orders: tuple[int, ...] | list[int],
        output_pairs: tuple[tuple[tuple[int, ...], int], ...] | None = None,
    ) -> tuple[Any, ...]:
        """Return a cache key for the regular ``g_s`` coefficient algebra."""
        if self._regular_taylor_signature_version <= 1:
            regular_endpoint_powers = tuple(
                (self.endpoint_power(sector, axis).base, self.endpoint_power(sector, axis).eps_coeff)
                for axis in range(sector.integration_dim)
            )
            return (
                "regular-taylor",
                1,
                int(sector.integration_dim),
                tuple(int(axis) for axis in sector.singular_axes),
                tuple(int(power) for power in sector.u_monomial_powers),
                tuple(int(power) for power in sector.f_monomial_powers),
                tuple(int(power) for power in sector.jacobian_monomial_powers),
                float(self.u_power_base),
                float(self.f_power_base),
                float(self.eps_log_u_coeff),
                float(self.eps_log_f_coeff),
                tuple(int(position) for position in zero_positions),
                tuple(int(order) for order in max_orders),
                regular_endpoint_powers,
                tuple(self.laurent_orders),
            )

        if self._regular_taylor_signature_version >= 3:
            canonical_positions = _regular_taylor_canonical_positions(max_orders)
            if output_pairs is None:
                output_pairs = tuple(
                    (tuple(multi), int(regular_order))
                    for regular_order in range(self.coefficient_count)
                    for multi in _multi_indices([int(order) for order in max_orders])
                )
            canonical_pairs = tuple(
                sorted(
                    {
                        (
                            _regular_taylor_original_to_canonical(tuple(multi), canonical_positions),
                            int(regular_order),
                        )
                        for multi, regular_order in output_pairs
                    },
                    key=lambda item: (item[1], sum(item[0]), item[0]),
                )
            )
            return (
                "regular-taylor",
                3,
                int(len(sector.singular_axes)),
                canonical_pairs,
                float(self.u_power_base),
                float(self.f_power_base),
                float(self.eps_log_u_coeff),
                float(self.eps_log_f_coeff),
                tuple(self.laurent_orders),
            )

        # Version 2 of this signature is deliberately much lower than the
        # original sector-specific key.  The formula receives residual Taylor
        # coefficients of J, U/M_U, F/M_F plus the already evaluated
        # nonsingular monomial prefactor/log.  Therefore it no longer depends on
        # sector variable names, singular-axis positions, zero projectors, or
        # U/F/J monomial powers; those are handled while assembling the formula
        # inputs from black-box evaluator output.
        return (
            "regular-taylor",
            2,
            int(len(sector.singular_axes)),
            tuple(sorted((int(order) for order in max_orders), reverse=True)),
            float(self.u_power_base),
            float(self.f_power_base),
            float(self.eps_log_u_coeff),
            float(self.eps_log_f_coeff),
            tuple(self.laurent_orders),
        )

    def regular_taylor_formula_for(
        self,
        sector: SectorDefinition,
        zero_positions: tuple[int, ...],
        max_orders: tuple[int, ...] | list[int],
        output_pairs: tuple[tuple[tuple[int, ...], int], ...] | None = None,
    ) -> "RegularTaylorFormulaDefinition":
        """Return a pregenerated regular Taylor formula."""
        signature = self.regular_taylor_signature(
            sector,
            zero_positions,
            max_orders,
            output_pairs=output_pairs,
        )
        formula = self._lookup_formula_with_laurent_superset(
            self._regular_taylor_formulas,
            signature,
        )
        if formula is None:
            raise RuntimeError(
                f"{sector.name}: missing pregenerated regular Taylor formula; "
                "call TopologyDefinition.prepare_regular_taylor_formulas(...) before integration"
            )
        return formula

    def regular_taylor_source_shape(
        self,
        sector: SectorDefinition,
        zero_positions: tuple[int, ...] | set[int],
        max_orders: tuple[int, ...] | list[int],
    ) -> list[tuple[int, ...]]:
        """Return and cache the raw Taylor shape needed by a residual formula."""
        zero_tuple = tuple(sorted(int(position) for position in zero_positions))
        max_tuple = tuple(int(order) for order in max_orders)
        key = (
            tuple(int(sector.u_monomial_powers[axis]) for axis in sector.singular_axes),
            tuple(int(sector.f_monomial_powers[axis]) for axis in sector.singular_axes),
            zero_tuple,
            max_tuple,
        )
        cached = self._regular_taylor_source_shape_cache.get(key)
        if cached is None:
            cached = _regular_taylor_source_shape(sector, zero_tuple, max_tuple)
            self._regular_taylor_source_shape_cache[key] = cached
        return cached

    def sparse_regular_source_shape_from_multis(
        self,
        sector: SectorDefinition,
        zero_positions: tuple[int, ...] | set[int],
        residual_multis: set[tuple[int, ...]],
        jacobian_multis: set[tuple[int, ...]],
    ) -> list[tuple[int, ...]]:
        """Return cached sparse source shape for a regular formula input."""
        key = (
            "all",
            tuple(int(sector.u_monomial_powers[axis]) for axis in sector.singular_axes),
            tuple(int(sector.f_monomial_powers[axis]) for axis in sector.singular_axes),
            tuple(int(sector.jacobian_monomial_powers[axis]) for axis in sector.singular_axes),
            tuple(sorted(int(position) for position in zero_positions)),
            _multi_set_cache_key(residual_multis),
            _multi_set_cache_key(jacobian_multis),
        )
        cached = self._sparse_regular_source_shape_cache.get(key)
        if cached is None:
            cached = _regular_taylor_source_shape_from_multis(
                sector,
                zero_positions,
                residual_multis,
                jacobian_multis,
            )
            self._sparse_regular_source_shape_cache[key] = cached
        return cached

    def sparse_regular_source_shape_for_monomial_powers(
        self,
        sector: SectorDefinition,
        zero_positions: tuple[int, ...] | set[int],
        residual_multis: set[tuple[int, ...]],
        monomial_powers: list[int],
    ) -> list[tuple[int, ...]]:
        """Return cached sparse source shape for one residual polynomial."""
        key = (
            "one",
            tuple(int(monomial_powers[axis]) for axis in sector.singular_axes),
            tuple(sorted(int(position) for position in zero_positions)),
            _multi_set_cache_key(residual_multis),
        )
        cached = self._sparse_regular_source_shape_cache.get(key)
        if cached is None:
            cached = _regular_taylor_source_shape_for_monomial_powers(
                sector,
                zero_positions,
                residual_multis,
                monomial_powers,
            )
            self._sparse_regular_source_shape_cache[key] = cached
        return cached

    def _dual_evaluator_shape_and_columns(
        self,
        sector: SectorDefinition,
    ) -> tuple[list[tuple[int, ...]], list[int] | None]:
        """Return evaluator shape and output columns for one sector."""
        if self.dual_evaluator_mode != "single-overall":
            return sector.dual_shape, None
        envelope = self._overall_dual_shapes.get(sector.integration_dim)
        if envelope is None:
            # Lazy direct use of a sector before pregeneration still works by
            # falling back to the per-sector shape.
            return sector.dual_shape, None
        key = (sector.integration_dim, tuple(sector.dual_shape))
        columns = self._overall_dual_indices.get(key)
        if columns is None:
            rank = len(envelope[0]) if envelope else 0
            envelope_index = {mi: index for index, mi in enumerate(envelope)}
            columns = [envelope_index[self._pad_multi_index(mi, rank)] for mi in sector.dual_shape]
            self._overall_dual_indices[key] = columns
        return envelope, columns

    def _max_sector_taylor_total_degree(self, sectors: list[SectorDefinition]) -> int:
        """Return the largest total singular-variable Taylor degree requested."""
        max_total = 0
        for sector in sectors:
            if not sector.dual_shape:
                continue
            max_total = max(max_total, max(sum(multi_index) for multi_index in sector.dual_shape))
        return max_total

    def _candidate_derivative_multi_indices(self, expr: Any, max_total: int) -> list[tuple[int, ...]]:
        """Return non-impossible x-space partial derivatives up to ``max_total``.

        The symbolic-derivative mode differentiates U/F with respect to the
        original Feynman parameters, not the sector variables.  For polynomials
        we can read the monomial support from Symbolica and only build
        derivatives that can be non-zero.
        """
        n_vars = len(self.x_names)
        x_symbols = [S(name) for name in self.x_names]
        candidates: set[tuple[int, ...]] = {tuple(0 for _ in range(n_vars))}
        try:
            polynomial = expr.to_polynomial(vars=x_symbols)
            for exponents, _coefficient in polynomial.coefficient_list(vars=x_symbols):
                capped = [min(int(power), max_total) for power in exponents]
                for multi_index in product(*[range(power + 1) for power in capped]):
                    if sum(multi_index) <= max_total:
                        candidates.add(tuple(int(value) for value in multi_index))
        except Exception:
            # Fallback for non-polynomial expressions: still bounded by the
            # sector Taylor degree, but less selective than the polynomial path.
            for multi_index in product(*[range(max_total + 1) for _ in range(n_vars)]):
                if sum(multi_index) <= max_total:
                    candidates.add(tuple(int(value) for value in multi_index))
        return sorted(candidates, key=lambda item: (sum(item), item))

    def _differentiate_expr(self, expr: Any, multi_index: tuple[int, ...]) -> Any:
        """Differentiate ``expr`` by one original-parameter multi-index."""
        out = expr
        for name, count in zip(self.x_names, multi_index):
            symbol = S(name)
            for _ in range(int(count)):
                out = out.derivative(symbol)
                if str(out) == "0":
                    return out
        return out

    def _prepare_symbolic_derivative_evaluators(self, polynomial: str, max_total: int) -> None:
        """Build shared normal evaluators for U/F symbolic partial derivatives."""
        if polynomial == "u":
            expr = self.u_expr
            expr_cache = self._u_derivative_exprs
            multi_cache = self._u_derivative_multi_evaluators
            indices_by_order = self._u_derivative_indices_by_order
        elif polynomial == "f":
            expr = self.f_expr
            expr_cache = self._f_derivative_exprs
            multi_cache = self._f_derivative_multi_evaluators
            indices_by_order = self._f_derivative_indices_by_order
        else:
            raise ValueError(f"{self.family}: unknown polynomial {polynomial!r}")

        existing = indices_by_order.get(max_total)
        if existing is not None:
            return

        params = [S(name) for name in [*self.x_names, *self.parameter_names]]
        prepared: list[tuple[int, ...]] = []
        expressions: list[Any] = []
        start = time.perf_counter()
        for multi_index in self._candidate_derivative_multi_indices(expr, max_total):
            derivative_expr = expr_cache.get(multi_index)
            if derivative_expr is None:
                derivative_expr = self._differentiate_expr(expr, multi_index)
                if str(derivative_expr) == "0":
                    continue
                expr_cache[multi_index] = derivative_expr
            prepared.append(multi_index)
            expressions.append(derivative_expr)
        indices_by_order[max_total] = prepared
        if prepared:
            multi_cache[tuple(prepared)] = Expression.evaluator_multiple(
                expressions,
                params,
                jit_compile=self.jit_compile_evaluators,
            )
        self.dual_evaluator_build_seconds += time.perf_counter() - start

    def _symbolic_derivative_indices(self, polynomial: str, max_total: int) -> list[tuple[int, ...]]:
        """Return prepared symbolic derivative indices for U or F."""
        if polynomial == "u":
            indices_by_order = self._u_derivative_indices_by_order
        elif polynomial == "f":
            indices_by_order = self._f_derivative_indices_by_order
        else:
            raise ValueError(f"{self.family}: unknown polynomial {polynomial!r}")
        if max_total not in indices_by_order:
            if self.strict_prepared_bundle:
                raise RuntimeError(
                    f"{self.family}: missing prepared {polynomial} derivative index set "
                    f"for total degree {max_total}"
                )
            self._prepare_symbolic_derivative_evaluators(polynomial, max_total)
        return indices_by_order[max_total]

    def _symbolic_derivative_evaluator(self, polynomial: str, multi_index: tuple[int, ...]) -> Any:
        """Return a prepared evaluator for one x-space symbolic derivative."""
        if polynomial == "u":
            expr = self.u_expr
            cache = self._u_derivative_evaluators
            expr_cache = self._u_derivative_exprs
        elif polynomial == "f":
            expr = self.f_expr
            cache = self._f_derivative_evaluators
            expr_cache = self._f_derivative_exprs
        else:
            raise ValueError(f"{self.family}: unknown polynomial {polynomial!r}")
        evaluator = cache.get(multi_index)
        if evaluator is None:
            if self.strict_prepared_bundle:
                raise RuntimeError(
                    f"{self.family}: missing prepared {polynomial} derivative evaluator "
                    f"for {multi_index}"
                )
            derivative_expr = expr_cache.get(multi_index)
            if derivative_expr is None:
                derivative_expr = self._differentiate_expr(expr, multi_index)
                if str(derivative_expr) == "0":
                    raise KeyError(f"{polynomial} derivative {multi_index} is zero")
                expr_cache[multi_index] = derivative_expr
            params = [S(name) for name in [*self.x_names, *self.parameter_names]]
            evaluator = derivative_expr.evaluator(
                params,
                jit_compile=self.jit_compile_evaluators,
            )
            cache[multi_index] = evaluator
        return evaluator

    def _symbolic_derivative_multi_evaluator(
        self,
        polynomial: str,
        derivative_indices: list[tuple[int, ...]],
    ) -> Any:
        """Return a shared evaluator for a whole derivative index list."""
        key = tuple(tuple(int(value) for value in multi) for multi in derivative_indices)
        if polynomial == "u":
            expr = self.u_expr
            expr_cache = self._u_derivative_exprs
            multi_cache = self._u_derivative_multi_evaluators
        elif polynomial == "f":
            expr = self.f_expr
            expr_cache = self._f_derivative_exprs
            multi_cache = self._f_derivative_multi_evaluators
        else:
            raise ValueError(f"{self.family}: unknown polynomial {polynomial!r}")
        evaluator = multi_cache.get(key)
        if evaluator is not None:
            return evaluator
        if self.strict_prepared_bundle:
            raise RuntimeError(
                f"{self.family}: missing prepared {polynomial} multi-derivative evaluator "
                f"for {key}"
            )
        expressions: list[Any] = []
        for multi_index in key:
            derivative_expr = expr_cache.get(multi_index)
            if derivative_expr is None:
                derivative_expr = self._differentiate_expr(expr, multi_index)
                if str(derivative_expr) == "0":
                    raise KeyError(f"{polynomial} derivative {multi_index} is zero")
                expr_cache[multi_index] = derivative_expr
            expressions.append(derivative_expr)
        params = [S(name) for name in [*self.x_names, *self.parameter_names]]
        evaluator = Expression.evaluator_multiple(
            expressions,
            params,
            jit_compile=self.jit_compile_evaluators,
        )
        multi_cache[key] = evaluator
        return evaluator

    def _derivative_values_batch(
        self,
        polynomial: str,
        x_values: np.ndarray,
        derivative_indices: list[tuple[int, ...]],
        timing: HotPathTiming | None,
    ) -> dict[tuple[int, ...], np.ndarray]:
        """Evaluate shared symbolic derivative callbacks at mapped x-points."""
        rows = self._rows(x_values)
        if not derivative_indices:
            return {}
        requested_key = tuple(tuple(int(value) for value in multi) for multi in derivative_indices)
        actual_key = requested_key
        if polynomial == "u":
            multi_cache = self._u_derivative_multi_evaluators
        elif polynomial == "f":
            multi_cache = self._f_derivative_multi_evaluators
        else:
            raise ValueError(f"{self.family}: unknown polynomial {polynomial!r}")

        evaluator = multi_cache.get(requested_key)
        if evaluator is None and self.strict_prepared_bundle:
            # Prepared bundles often store one evaluator for a larger total
            # derivative order.  A sector may later need only a sparse subset
            # of those columns, especially after the universal chain-rule
            # formulas compress active coordinates.  Reusing the smallest
            # prepared superset keeps strict integrate mode disk-only: no
            # Symbolica derivative evaluator is built here.
            requested = set(requested_key)
            candidates = [
                key
                for key in multi_cache
                if requested.issubset(set(key))
            ]
            if candidates:
                actual_key = min(
                    candidates,
                    key=lambda key: (len(key), sum(sum(multi) for multi in key)),
                )
                evaluator = multi_cache[actual_key]
        if evaluator is None:
            evaluator = self._symbolic_derivative_multi_evaluator(polynomial, derivative_indices)
        start = time.perf_counter()
        matrix = np.asarray(
            evaluator.evaluate_complex(np.ascontiguousarray(rows)),
            dtype=np.complex128,
        )
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        actual_index = {tuple(multi_index): column for column, multi_index in enumerate(actual_key)}
        return {
            tuple(multi_index): matrix[:, actual_index[tuple(multi_index)]]
            for multi_index in derivative_indices
        }

    def _symbolic_derivative_taylor_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        polynomial: str,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Compose shared x-space derivative evaluators with sector-map jets.

        This is the explicit chain-rule alternative to dualizing the U/F
        evaluator.  The U/F expressions are differentiated once with respect to
        original Feynman parameters and reused for every sector.  Sector
        dependence only enters through the numerical map Taylor coefficients.
        """
        requested_shape = [tuple(int(value) for value in multi) for multi in sector.dual_shape]
        context = self._symbolic_derivative_taylor_context_batch(
            sector,
            y_values,
            timing,
            output_shape=requested_shape,
        )
        return self._compose_symbolic_derivative_taylor_batch(
            sector,
            context,
            polynomial,
            timing,
            output_shape=requested_shape,
        )

    def _symbolic_derivative_taylor_pair_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
        output_shape: list[tuple[int, ...]] | None = None,
        u_output_shape: list[tuple[int, ...]] | None = None,
        f_output_shape: list[tuple[int, ...]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return U and F Taylor coefficients while sharing sector-map jets.

        The symbolic-derivative backend used by heavy DOT sectors needs the
        same composed sector-map Taylor series for U and F.  Computing those
        jets twice is pure Python overhead, so this pair path builds the map
        context once and only runs the separate black-box derivative
        evaluators for U and F.
        """
        context = self._symbolic_derivative_taylor_context_batch(
            sector,
            y_values,
            timing,
            output_shape=(
                _merge_multi_shapes(u_output_shape or [], f_output_shape or [])
                if u_output_shape is not None or f_output_shape is not None
                else output_shape
            ),
        )
        return (
            self._compose_symbolic_derivative_taylor_batch(
                sector,
                context,
                "u",
                timing,
                output_shape=u_output_shape,
            ),
            self._compose_symbolic_derivative_taylor_batch(
                sector,
                context,
                "f",
                timing,
                output_shape=f_output_shape,
            ),
        )

    def _symbolic_derivative_taylor_context_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
        output_shape: list[tuple[int, ...]] | None = None,
    ) -> dict[str, Any]:
        """Build reusable sector-map Taylor data for symbolic derivatives."""
        requested_shape = [tuple(int(value) for value in multi) for multi in (output_shape or sector.dual_shape)]
        if not requested_shape:
            raise ValueError(f"{sector.name}: no dual shape declared")
        active_shape = _chain_rule_envelope_shape(requested_shape)

        rows_in = np.asarray(y_values, dtype=float)
        n_rows = rows_in.shape[0]
        rank = len(active_shape[0])
        max_orders = [
            max(multi_index[position] for multi_index in active_shape)
            for position in range(rank)
        ]
        max_total = max(sum(int(value) for value in multi_index) for multi_index in active_shape)

        x_jets = sector.map_dual_eval_batch_for_shape(rows_in, active_shape, timing)
        zero = _zero_multi(rank)
        shape_index = {multi_index: index for index, multi_index in enumerate(active_shape)}
        zero_column = shape_index[zero]
        x0 = x_jets[:, :, zero_column]

        # The sector map has a fixed sparse Taylor structure.  Use the same
        # structural layout that keys the pregenerated chain-rule evaluator
        # instead of probing every map-jet column with ``np.any`` for every
        # batch.  This keeps sample-dependent zeroes as explicit zero arrays,
        # which is algebraically harmless and avoids millions of Python-side
        # reductions in high-axis sectors such as the triple-box PSD649 class.
        h_series: list[MultiSeries] = [{} for _ in range(len(self.x_names))]
        for x_index, multi_index in self._chain_rule_h_layout(sector, active_shape):
            column = shape_index[tuple(multi_index)]
            h_series[int(x_index)][tuple(multi_index)] = x_jets[
                :,
                int(x_index),
                column,
            ].astype(np.complex128, copy=False)

        return {
            "n_rows": n_rows,
            "rank": rank,
            "max_orders": max_orders,
            "max_total": max_total,
            "x0": x0,
            "h_series": h_series,
            "output_shape": active_shape,
            "requested_output_shape": requested_shape,
        }

    def _chain_rule_h_layout(
        self,
        sector: SectorDefinition,
        output_shape: list[tuple[int, ...]],
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Return structurally nonzero sector-map Taylor inputs.

        Chain-rule composition formulas receive sector-map Taylor coefficients
        as numerical inputs.  They do not need sector names, sector variable
        names, or the map expressions themselves.  The structural nonzero
        layout is therefore the lowest safe key for reusing these evaluators
        without pretending that every map has the same sparse input contract.
        """
        if not output_shape:
            return ()
        rank = len(output_shape[0])
        zero = _zero_multi(rank)
        shape = [tuple(int(value) for value in multi) for multi in output_shape]
        cache_key = (sector.name, tuple(shape))
        cached = self._chain_rule_h_layout_cache.get(cache_key)
        if cached is not None:
            return cached
        shape_index = {multi: index for index, multi in enumerate(shape)}
        structural_row = np.full((1, sector.integration_dim), 0.37, dtype=float)
        x_jets = sector.map_dual_eval_batch_for_shape(structural_row, shape, None)
        layout: list[tuple[int, tuple[int, ...]]] = []
        for x_index in range(len(self.x_names)):
            for multi in shape:
                if multi == zero:
                    continue
                value = x_jets[:, x_index, shape_index[multi]]
                if value.any():
                    layout.append((int(x_index), tuple(int(v) for v in multi)))
        out = tuple(layout)
        self._chain_rule_h_layout_cache[cache_key] = out
        return out

    def _chain_rule_active_x_indices(
        self,
        sector: SectorDefinition,
        output_shape: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """Return original Feynman-parameter indices that vary in the map.

        The mapped-derivative chain rule only needs coordinates whose sector
        map has at least one non-constant Taylor coefficient.  Compressing to
        this active set removes the total topology arity from the universal
        formula signature.  The original indices are kept only at runtime to
        lift compressed derivative multi-indices back to U/F derivative
        evaluator inputs.
        """
        monomial_active = sector.structurally_active_map_indices()
        if monomial_active is not None:
            return tuple(int(index) for index in monomial_active)
        return tuple(
            sorted({int(x_index) for x_index, _multi in self._chain_rule_h_layout(sector, output_shape)})
        )

    def _chain_rule_compressed_derivative_indices(
        self,
        polynomial: str,
        max_total: int,
        active_x_indices: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Project nonzero U/F derivative indices onto active map coordinates."""
        derivative_indices = self._symbolic_derivative_indices(polynomial, max_total)
        active = set(int(index) for index in active_x_indices)
        compressed: set[tuple[int, ...]] = set()
        for multi_index in derivative_indices:
            # Derivatives with powers on inactive coordinates multiply a zero
            # sector-map Taylor series and cannot contribute to P(X_s(y)).
            if any(int(value) and position not in active for position, value in enumerate(multi_index)):
                continue
            compressed.add(tuple(int(multi_index[index]) for index in active_x_indices))
        return tuple(sorted(compressed, key=lambda item: (sum(item), item)))

    def _chain_rule_original_derivative_indices(
        self,
        compressed_derivative_indices: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
        active_x_indices: tuple[int, ...],
    ) -> list[tuple[int, ...]]:
        """Lift compressed derivative indices back to original x-space."""
        out: list[tuple[int, ...]] = []
        for compressed in compressed_derivative_indices:
            full = [0 for _ in self.x_names]
            if len(compressed) != len(active_x_indices):
                raise ValueError(
                    f"{self.family}: chain-rule derivative arity mismatch: "
                    f"{compressed!r} for active coordinates {active_x_indices!r}"
                )
            for active_position, x_index in enumerate(active_x_indices):
                full[int(x_index)] = int(compressed[active_position])
            out.append(tuple(full))
        return out

    def _chain_rule_dense_derivative_indices_for_signature(
        self,
        signature: tuple[Any, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Return the universal dense derivative slots for a chain formula.

        These slots intentionally depend only on the compressed active
        coordinate count and the requested Taylor order.  Topology-specific
        polynomial derivatives that are structurally zero are passed as zero
        numeric inputs when the formula is evaluated.
        """
        active_coordinate_count = int(signature[0])
        output_shape = tuple(tuple(int(value) for value in multi) for multi in signature[2])
        max_total = max((sum(multi) for multi in output_shape), default=0)
        max_degree = min(
            CHAIN_RULE_MAX_DERIVATIVE_DEGREE_1_TO_3_LOOPS,
            max_total,
        )
        return _dense_total_degree_multi_indices(active_coordinate_count, max_degree)

    def _chain_rule_formula_signature(
        self,
        sector: SectorDefinition,
        polynomial: str,
        output_shape: list[tuple[int, ...]],
    ) -> tuple[Any, ...]:
        """Return a cache key for a mapped-derivative composition formula."""
        loop_count = (
            int(self.parametric_representation.loop_count)
            if self.parametric_representation is not None
            else 1
        )
        if loop_count > 3:
            raise NotImplementedError(
                f"{self.family}: universal chain-rule formula cache is currently "
                "validated only through three loops"
            )
        original_envelope_shape = _chain_rule_envelope_shape(output_shape)
        canonical_envelope_shape = tuple(_chain_rule_canonical_envelope_shape(output_shape))
        active_x_indices = self._chain_rule_active_x_indices(sector, original_envelope_shape)
        rank = len(canonical_envelope_shape[0]) if canonical_envelope_shape else 0
        # The mathematical chain-rule formula is universal once the sector map
        # is compressed to its active coordinates.  U/F derivative support,
        # original Feynman-parameter arity, sector name, polynomial label, and
        # JIT policy are deliberately not part of the signature.  The Taylor
        # shape is the dense coordinate envelope requested by the caller, not a
        # sparse sector-specific coefficient list.
        return (len(active_x_indices), rank, canonical_envelope_shape)

    def chain_rule_formula_for(
        self,
        sector: SectorDefinition,
        polynomial: str,
        output_shape: list[tuple[int, ...]],
    ) -> "ChainRuleFormulaDefinition":
        """Return or build a Symbolica chain-rule composition evaluator."""
        requested_shape = [tuple(int(value) for value in multi) for multi in output_shape]
        if not requested_shape:
            raise ValueError(f"{sector.name}: empty chain-rule output shape")
        active_shape = _chain_rule_canonical_envelope_shape(requested_shape)
        lookup_key = (sector.name, polynomial, tuple(active_shape))
        direct_cached = self._chain_rule_formula_lookup_cache.get(lookup_key)
        if direct_cached is not None:
            return direct_cached
        signature = self._chain_rule_formula_signature(sector, polynomial, active_shape)
        cached = self._chain_rule_formulas.get(signature)
        if cached is not None:
            self._chain_rule_formula_lookup_cache[lookup_key] = cached
            return cached
        cached = _load_chain_rule_formula_from_cache(
            signature,
            load_evaluators=not self.chain_rule_metadata_only,
        )
        if cached is not None:
            self.chain_rule_formulas_from_cache += 1
            self._chain_rule_formulas[signature] = cached
            self._chain_rule_formula_lookup_cache[lookup_key] = cached
            return cached
        if self.strict_prepared_bundle:
            raise RuntimeError(
                f"{sector.name}: missing prepared chain-rule formula for "
                f"{polynomial} shape {tuple(active_shape)}"
            )

        start = time.perf_counter()
        formula = self._build_chain_rule_formula(sector, polynomial, active_shape, signature)
        formula.build_seconds = time.perf_counter() - start
        self.chain_rule_formula_build_seconds += formula.build_seconds
        self.chain_rule_formulas_generated += 1
        _write_chain_rule_formula_to_cache(formula)
        if self.chain_rule_metadata_only:
            metadata_formula = _load_chain_rule_formula_from_cache(
                signature,
                load_evaluators=False,
            )
            if metadata_formula is not None:
                formula = metadata_formula
        self._chain_rule_formulas[signature] = formula
        self._chain_rule_formula_lookup_cache[lookup_key] = formula
        return formula

    def _build_chain_rule_formula(
        self,
        sector: SectorDefinition,
        polynomial: str,
        output_shape: list[tuple[int, ...]],
        signature: tuple[Any, ...],
    ) -> "ChainRuleFormulaDefinition":
        """Build Symbolica expressions for ``P(X_s(y))`` Taylor composition."""
        rank = len(output_shape[0])
        zero = _zero_multi(rank)
        allowed_multis = {tuple(multi) for multi in output_shape}
        allowed_multis.add(zero)
        active_x_count = int(signature[0])
        derivative_indices = [
            tuple(int(value) for value in multi)
            for multi in self._chain_rule_dense_derivative_indices_for_signature(signature)
        ]
        h_multis = [multi for multi in output_shape if tuple(multi) != zero]

        input_names: list[str] = []
        input_symbols: list[Any] = []
        h_series: list[ExprSeries] = []
        h_layout: list[tuple[int, tuple[int, ...]]] = []
        for active_index in range(active_x_count):
            series: ExprSeries = {}
            for multi in h_multis:
                multi_tuple = tuple(int(value) for value in multi)
                name = f"ch_h_{active_index}_{_multi_suffix(multi_tuple)}"
                symbol = S(name)
                input_names.append(name)
                input_symbols.append(symbol)
                series[multi_tuple] = symbol
                h_layout.append((active_index, multi_tuple))
            h_series.append(series)

        derivative_symbols: dict[tuple[int, ...], Any] = {}
        for multi in derivative_indices:
            name = f"ch_d_{_multi_suffix(multi)}"
            symbol = S(name)
            input_names.append(name)
            input_symbols.append(symbol)
            derivative_symbols[tuple(multi)] = symbol

        power_cache: dict[tuple[int, int], ExprSeries] = {}

        def h_power(x_index: int, power: int) -> ExprSeries:
            key = (int(x_index), int(power))
            cached = power_cache.get(key)
            if cached is not None:
                return cached
            if power == 0:
                cached = _expr_series_constant(E("1"), [0 for _ in range(rank)])
            elif power == 1:
                cached = {
                    multi: value
                    for multi, value in h_series[x_index].items()
                    if multi in allowed_multis
                }
            else:
                cached = _expr_series_mul_allowed(
                    h_power(x_index, power - 1),
                    h_series[x_index],
                    allowed_multis,
                )
            power_cache[key] = cached
            return cached

        product_cache: dict[tuple[int, ...], ExprSeries] = {}

        def chain_product(x_multi_index: tuple[int, ...]) -> ExprSeries:
            cached = product_cache.get(x_multi_index)
            if cached is not None:
                return cached
            term = _expr_series_constant(E("1"), [0 for _ in range(rank)])
            factorial = 1
            for active_index, power in enumerate(x_multi_index):
                power_int = int(power)
                if not power_int:
                    continue
                factorial *= math.factorial(power_int)
                term = _expr_series_mul_allowed(
                    term,
                    h_power(active_index, power_int),
                    allowed_multis,
                )
                if not term:
                    break
            if term and factorial != 1:
                term = _expr_series_scale(term, E("1") / E(str(int(factorial))))
            product_cache[x_multi_index] = term
            return term

        composed: ExprSeries = {}
        for x_multi_index in derivative_indices:
            product_series = chain_product(tuple(int(value) for value in x_multi_index))
            if not product_series:
                continue
            composed = _expr_series_add(
                composed,
                _expr_series_scale(product_series, derivative_symbols[tuple(x_multi_index)]),
            )

        output_expressions = [
            _expr_series_coefficient(composed, tuple(multi))
            for multi in output_shape
        ]
        evaluators = (
            [
                Expression.evaluator_multiple(
                    output_expressions,
                    input_symbols,
                    jit_compile=self.jit_compile_evaluators,
                )
            ]
            if output_expressions
            else []
        )
        return ChainRuleFormulaDefinition(
            signature=signature,
            input_names=input_names,
            input_symbols=input_symbols,
            output_expressions=output_expressions,
            evaluators=evaluators,
            output_shape=list(output_shape),
            derivative_indices=[tuple(multi) for multi in derivative_indices],
            h_layout=list(h_layout),
            evaluator_mode="multiple",
        )

    def _compose_symbolic_derivative_taylor_batch(
        self,
        sector: SectorDefinition,
        context: dict[str, Any],
        polynomial: str,
        timing: HotPathTiming | None = None,
        output_shape: list[tuple[int, ...]] | None = None,
        use_chain_formula: bool = True,
    ) -> np.ndarray:
        """Compose one polynomial's symbolic derivatives with cached map jets."""
        n_rows = int(context["n_rows"])
        rank = int(context["rank"])
        max_orders = list(context["max_orders"])
        max_total = int(context["max_total"])
        x0 = np.asarray(context["x0"], dtype=float)
        h_series = list(context["h_series"])
        # ``output_shape`` can be much sparser than the rectangular
        # ``max_orders`` box.  This is especially important after the v3
        # regular-Taylor signature has closed only the actually requested
        # coefficients under ancestors.  Compose the chain rule directly in
        # that sparse space instead of materialising the full box.
        requested_output_shape = [
            tuple(int(value) for value in multi_index)
            for multi_index in (
                output_shape
                or context.get("requested_output_shape")
                or context.get("output_shape", [])
            )
        ]
        active_output_shape = list(requested_output_shape)
        allowed_multis = {
            tuple(int(value) for value in multi_index)
            for multi_index in active_output_shape
        }
        zero_multi = _zero_multi(rank)
        allowed_multis.add(zero_multi)
        derivative_indices = self._symbolic_derivative_indices(polynomial, max_total)
        if active_output_shape == [zero_multi] and max_total == 0:
            zero_x = tuple(0 for _ in self.x_names)
            derivative_values = self._derivative_values_batch(
                polynomial,
                x0,
                [zero_x],
                timing,
            )
            return derivative_values[zero_x][:, np.newaxis]

        if use_chain_formula:
            formula_output_shape = _chain_rule_canonical_envelope_shape(active_output_shape)
            canonical_positions = _chain_rule_canonical_positions(active_output_shape)
            signature = self._chain_rule_formula_signature(
                sector,
                polynomial,
                active_output_shape,
            )
            if signature in self._chain_rule_formulas or not self.strict_prepared_bundle:
                formula = self.chain_rule_formula_for(sector, polynomial, formula_output_shape)
            else:
                formula = None
        else:
            formula = None
        if formula is not None:
            active_x_indices = self._chain_rule_active_x_indices(
                sector,
                active_output_shape,
            )
            original_derivative_indices = self._chain_rule_original_derivative_indices(
                formula.derivative_indices,
                active_x_indices,
            )
            available_derivative_indices = set(derivative_indices)
            evaluated_derivative_indices = [
                multi_index
                for multi_index in original_derivative_indices
                if tuple(multi_index) in available_derivative_indices
            ]
            derivative_values = self._derivative_values_batch(
                polynomial,
                x0,
                evaluated_derivative_indices,
                timing,
            )
            input_matrix = np.zeros((n_rows, len(formula.input_names)), dtype=np.complex128)
            offset = 0
            for active_position, multi_index in formula.h_layout:
                x_index = active_x_indices[int(active_position)]
                original_multi = _chain_rule_canonical_to_original(
                    tuple(multi_index),
                    canonical_positions,
                )
                values = h_series[int(x_index)].get(original_multi)
                if values is not None:
                    input_matrix[:, offset] = values
                offset += 1
            for original_multi_index in original_derivative_indices:
                values = derivative_values.get(tuple(original_multi_index))
                if values is not None:
                    input_matrix[:, offset] = values
                offset += 1
            if offset != len(formula.input_names):
                raise RuntimeError(
                    f"{sector.name}: chain-rule formula input mismatch: filled {offset}, "
                    f"expected {len(formula.input_names)}"
                )
            formula_values = formula.evaluate_complex_batch(input_matrix, timing)
            if list(formula.output_shape) == active_output_shape:
                return formula_values
            formula_columns = {
                tuple(int(value) for value in multi): column
                for column, multi in enumerate(formula.output_shape)
            }
            try:
                selected_columns = [
                    formula_columns[
                        _chain_rule_original_to_canonical(tuple(multi), canonical_positions)
                    ]
                    for multi in active_output_shape
                ]
            except KeyError as exc:
                raise RuntimeError(
                    f"{sector.name}: chain-rule envelope does not cover requested "
                    f"coefficient {exc.args[0]!r}"
                ) from exc
            return formula_values[:, selected_columns]

        derivative_values = self._derivative_values_batch(polynomial, x0, derivative_indices, timing)

        allowed_key = _allowed_multi_key(allowed_multis)
        power_cache_by_shape: dict[
            tuple[tuple[int, ...], ...],
            dict[tuple[int, int], MultiSeries],
        ] = context.setdefault("_chain_rule_power_cache_by_shape", {})
        product_cache_by_shape: dict[
            tuple[tuple[int, ...], ...],
            dict[tuple[int, ...], MultiSeries],
        ] = context.setdefault("_chain_rule_product_cache_by_shape", {})
        power_cache = power_cache_by_shape.setdefault(allowed_key, {})
        product_cache = product_cache_by_shape.setdefault(allowed_key, {})

        def h_power(x_index: int, power: int) -> MultiSeries:
            key = (x_index, power)
            cached = power_cache.get(key)
            if cached is not None:
                return cached
            if power == 0:
                cached = _series_constant(1.0 + 0.0j, max_orders, n_rows)
            elif power == 1:
                cached = {
                    multi_index: values
                    for multi_index, values in h_series[x_index].items()
                    if multi_index in allowed_multis
                }
            else:
                cached = _series_mul_allowed(
                    h_power(x_index, power - 1),
                    h_series[x_index],
                    allowed_multis,
                )
            power_cache[key] = cached
            return cached

        def chain_product(x_multi_index: tuple[int, ...]) -> MultiSeries:
            """Return the map-chain product for one x-space derivative."""
            cached = product_cache.get(x_multi_index)
            if cached is not None:
                return cached
            term = _series_constant(1.0 + 0.0j, max_orders, n_rows)
            factorial = 1
            for x_index, power in enumerate(x_multi_index):
                power_int = int(power)
                if not power_int:
                    continue
                factorial *= math.factorial(power_int)
                term = _series_mul_allowed(
                    term,
                    h_power(x_index, power_int),
                    allowed_multis,
                )
                if not term:
                    break
            if term and factorial != 1:
                term = _series_scale(term, 1.0 / float(factorial))
            product_cache[x_multi_index] = term
            return term

        composed: MultiSeries = {}
        for x_multi_index in derivative_indices:
            derivative = derivative_values[x_multi_index]
            if not np.any(derivative):
                continue
            product_series = chain_product(tuple(int(value) for value in x_multi_index))
            if product_series:
                composed = _series_add(
                    composed,
                    _series_scale(product_series, derivative),
                )

        return np.stack(
            [
                _series_coefficient(composed, multi_index, n_rows)
                for multi_index in active_output_shape
            ],
            axis=1,
        )

    def _derivative_values_prec(
        self,
        polynomial: str,
        x0: list[ComplexPrecise],
        derivative_indices: list[tuple[int, ...]],
        precision_digits: int,
        timing: HotPathTiming | None,
    ) -> dict[tuple[int, ...], ComplexPrecise]:
        """Evaluate shared x-space derivative callbacks at one precise point."""
        row: ComplexPreciseRow = list(x0)
        row.extend(_decimal_complex(value, precision_digits) for value in self.parameter_values)
        values: dict[tuple[int, ...], ComplexPrecise] = {}
        for multi_index in derivative_indices:
            evaluator = self._symbolic_derivative_evaluator(polynomial, multi_index)
            start = time.perf_counter()
            result = evaluator.evaluate_complex_with_prec(row, precision_digits)[0]
            if timing is not None:
                timing.add_eval(time.perf_counter() - start)
            values[multi_index] = (
                decimal_with_precision(result[0], precision_digits),
                decimal_with_precision(result[1], precision_digits),
            )
        return values

    def _symbolic_derivative_taylor_complex_prec(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        polynomial: str,
        precision_digits: int,
        timing: HotPathTiming | None = None,
    ) -> list[ComplexPrecise]:
        """Compose symbolic x-space derivatives with precise sector-map jets.

        This is the high-precision analogue of
        ``_symbolic_derivative_taylor_batch``.  It is used only for endpoint
        rescue rows, so clarity and exact Decimal propagation matter more than
        vectorization.
        """
        if not sector.dual_shape:
            raise ValueError(f"{sector.name}: no dual shape declared")

        rank = len(sector.dual_shape[0])
        max_orders = [
            max(multi_index[position] for multi_index in sector.dual_shape)
            for position in range(rank)
        ]
        max_total = sum(max_orders)
        derivative_indices = self._symbolic_derivative_indices(polynomial, max_total)

        x_jets = sector.map_dual_complex_prec_for_shape(
            y,
            sector.dual_shape,
            precision_digits,
            timing,
        )
        zero = _zero_multi(rank)
        zero_column = sector.dual_index(zero)
        x0 = [jet[zero_column] for jet in x_jets]
        derivative_values = self._derivative_values_prec(
            polynomial,
            x0,
            derivative_indices,
            precision_digits,
            timing,
        )

        h_series: list[PrecSeries] = []
        for jet in x_jets:
            series: PrecSeries = {}
            for column, multi_index in enumerate(sector.dual_shape):
                if multi_index == zero:
                    continue
                value = jet[column]
                if not _pc_is_zero(value):
                    series[multi_index] = value
            h_series.append(series)

        power_cache: dict[tuple[int, int], PrecSeries] = {}

        def h_power(x_index: int, power: int) -> PrecSeries:
            key = (x_index, power)
            cached = power_cache.get(key)
            if cached is not None:
                return cached
            if power == 0:
                cached = _prec_series_constant(_pc_one(), max_orders)
            elif power == 1:
                cached = h_series[x_index]
            else:
                cached = _prec_series_mul(h_power(x_index, power - 1), h_series[x_index], max_orders)
            power_cache[key] = cached
            return cached

        composed: PrecSeries = {}
        for x_multi_index in derivative_indices:
            derivative = derivative_values[x_multi_index]
            factorial = 1
            for order in x_multi_index:
                factorial *= math.factorial(int(order))
            term = _prec_series_constant(
                _pc_scale(derivative, Decimal(1) / Decimal(factorial), precision_digits),
                max_orders,
            )
            for x_index, power in enumerate(x_multi_index):
                if power:
                    term = _prec_series_mul(term, h_power(x_index, int(power)), max_orders)
                    if not term:
                        break
            if term:
                composed = _prec_series_add(composed, term)

        return [
            _prec_series_coefficient(composed, multi_index)
            for multi_index in sector.dual_shape
        ]

    def _taylor_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        evaluator: Any,
        evaluator_shape: list[tuple[int, ...]] | None = None,
        output_columns: list[int] | None = None,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate Taylor coefficients of one black-box polynomial."""
        rows_in = np.asarray(y_values, dtype=float)
        active_shape = evaluator_shape or sector.dual_shape
        x_jets = sector.map_dual_eval_batch_for_shape(rows_in, active_shape, timing)
        n_rows = rows_in.shape[0]
        dual_len = len(active_shape)
        rows = np.zeros(
            (n_rows, (len(self.x_names) + len(self.parameter_values)) * dual_len),
            dtype=float,
        )
        offset = 0
        for x_index in range(len(self.x_names)):
            rows[:, offset : offset + dual_len] = x_jets[:, x_index, :]
            offset += dual_len
        for value in self.parameter_values:
            rows[:, offset] = float(value)
            offset += dual_len
        # Dualized evaluators currently have a stricter multiprecision API in
        # Symbolica than ordinary batched evaluators, especially for constant
        # expressions whose parameter arity is optimized.  Until that limitation
        # is lifted, endpoint Taylor coefficients stay on the ordinary dual API.
        values = self._timed_evaluate(evaluator, rows, timing, allow_precision=False)
        array = np.asarray(values, dtype=np.complex128)
        if output_columns is not None:
            array = array[:, output_columns]
        return array

    def _taylor_complex_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        evaluator: Any,
        evaluator_shape: list[tuple[int, ...]] | None = None,
        output_columns: list[int] | None = None,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate Taylor coefficients using Symbolica's complex evaluator."""
        rows_in = np.asarray(y_values, dtype=float)
        active_shape = evaluator_shape or sector.dual_shape
        x_jets = sector.map_dual_complex_batch_for_shape(rows_in, active_shape, timing)
        n_rows = rows_in.shape[0]
        dual_len = len(active_shape)
        rows = np.zeros(
            (n_rows, (len(self.x_names) + len(self.parameter_values)) * dual_len),
            dtype=np.complex128,
        )
        offset = 0
        for x_index in range(len(self.x_names)):
            rows[:, offset : offset + dual_len] = x_jets[:, x_index, :]
            offset += dual_len
        for value in self.parameter_values:
            rows[:, offset] = complex(float(value), 0.0)
            offset += dual_len
        values = self._timed_evaluate_complex(evaluator, rows, timing)
        array = np.asarray(values, dtype=np.complex128)
        if output_columns is not None:
            array = array[:, output_columns]
        return array

    def _taylor_complex_prec(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        evaluator: Any,
        precision_digits: int,
        evaluator_shape: list[tuple[int, ...]] | None = None,
        output_columns: list[int] | None = None,
        timing: HotPathTiming | None = None,
    ) -> list[ComplexPrecise]:
        """Evaluate one Taylor row with arbitrary-precision complex arithmetic."""
        active_shape = evaluator_shape or sector.dual_shape
        x_jets = sector.map_dual_complex_prec_for_shape(y, active_shape, precision_digits, timing)
        zero: ComplexPrecise = (Decimal(0), Decimal(0))
        row: ComplexPreciseRow = []
        for jet in x_jets:
            row.extend(jet)
        for value in self.parameter_values:
            param_jet = [zero for _ in active_shape]
            param_jet[0] = _decimal_complex(value, precision_digits)
            row.extend(param_jet)
        values = self._timed_evaluate_complex_with_prec(
            evaluator,
            row,
            precision_digits,
            timing,
        )
        if output_columns is not None:
            values = [values[index] for index in output_columns]
        return values

    def f_taylor(
        self, sector: SectorDefinition, y: list[float] | tuple[float, ...]
    ) -> dict[tuple[int, ...], complex]:
        """Evaluate F Taylor coefficients after composing map jets with F."""
        if not sector.dual_shape:
            raise ValueError(f"{sector.name}: no dual shape declared")

        # The sector owns X_s(y) and can therefore supply jets of x_i=X_i(y).
        # F remains a black-box evaluator that only sees those numeric jets.
        evaluator_shape, output_columns = self._dual_evaluator_shape_and_columns(sector)
        x_jets = sector.map_dual_eval_for_shape(y, evaluator_shape)
        zero = [0.0 for _ in evaluator_shape]
        row: list[float] = []
        for jet in x_jets:
            row.extend(jet)
        for value in self.parameter_values:
            # External invariants and masses are constants in the endpoint
            # Taylor expansion: only the zeroth dual component is non-zero.
            param_jet = zero.copy()
            param_jet[0] = float(value)
            row.extend(param_jet)

        values = self.f_dual_evaluator(evaluator_shape).evaluate([row])[0]
        if output_columns is not None:
            values = [values[index] for index in output_columns]
        return {mi: complex(values[i]) for i, mi in enumerate(sector.dual_shape)}

    def f_taylor_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Batch version of ``f_taylor`` for boundary samples."""
        if not sector.dual_shape:
            raise ValueError(f"{sector.name}: no dual shape declared")
        if self.dual_evaluator_mode == "symbolic-derivatives":
            return self._symbolic_derivative_taylor_batch(sector, y_values, "f", timing)

        evaluator_shape, output_columns = self._dual_evaluator_shape_and_columns(sector)
        evaluator = self.f_dual_evaluator(evaluator_shape)
        return self._taylor_batch(sector, y_values, evaluator, evaluator_shape, output_columns, timing)

    def f_taylor_complex_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Complex-evaluator batch version of F Taylor coefficients."""
        if self.dual_evaluator_mode == "symbolic-derivatives":
            return self._symbolic_derivative_taylor_batch(sector, y_values, "f", timing).astype(
                np.complex128,
                copy=False,
            )
        evaluator_shape, output_columns = self._dual_evaluator_shape_and_columns(sector)
        evaluator = self.f_dual_evaluator(evaluator_shape)
        return self._taylor_complex_batch(sector, y_values, evaluator, evaluator_shape, output_columns, timing)

    def f_taylor_complex_prec(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        precision_digits: int,
        timing: HotPathTiming | None = None,
    ) -> list[ComplexPrecise]:
        """Complex arbitrary-precision row version of F Taylor coefficients."""
        if self.dual_evaluator_mode == "symbolic-derivatives":
            return self._symbolic_derivative_taylor_complex_prec(
                sector,
                y,
                "f",
                precision_digits,
                timing,
            )
        evaluator_shape, output_columns = self._dual_evaluator_shape_and_columns(sector)
        evaluator = self.f_dual_evaluator(evaluator_shape)
        return self._taylor_complex_prec(
            sector,
            y,
            evaluator,
            precision_digits,
            evaluator_shape,
            output_columns,
            timing,
        )

    def u_taylor_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Batch Taylor coefficients of U after composing map jets with U."""
        if not sector.dual_shape:
            raise ValueError(f"{sector.name}: no dual shape declared")
        if self.dual_evaluator_mode == "symbolic-derivatives":
            return self._symbolic_derivative_taylor_batch(sector, y_values, "u", timing)

        evaluator_shape, output_columns = self._dual_evaluator_shape_and_columns(sector)
        evaluator = self.u_dual_evaluator(evaluator_shape)
        return self._taylor_batch(sector, y_values, evaluator, evaluator_shape, output_columns, timing)

    def u_taylor_complex_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Complex-evaluator batch version of U Taylor coefficients."""
        if self.dual_evaluator_mode == "symbolic-derivatives":
            return self._symbolic_derivative_taylor_batch(sector, y_values, "u", timing).astype(
                np.complex128,
                copy=False,
            )
        evaluator_shape, output_columns = self._dual_evaluator_shape_and_columns(sector)
        evaluator = self.u_dual_evaluator(evaluator_shape)
        return self._taylor_complex_batch(sector, y_values, evaluator, evaluator_shape, output_columns, timing)

    def u_taylor_complex_prec(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        precision_digits: int,
        timing: HotPathTiming | None = None,
    ) -> list[ComplexPrecise]:
        """Complex arbitrary-precision row version of U Taylor coefficients."""
        if self.dual_evaluator_mode == "symbolic-derivatives":
            return self._symbolic_derivative_taylor_complex_prec(
                sector,
                y,
                "u",
                precision_digits,
                timing,
            )
        evaluator_shape, output_columns = self._dual_evaluator_shape_and_columns(sector)
        evaluator = self.u_dual_evaluator(evaluator_shape)
        return self._taylor_complex_prec(
            sector,
            y,
            evaluator,
            precision_digits,
            evaluator_shape,
            output_columns,
            timing,
        )

    def endpoint_power(self, sector: SectorDefinition, axis: int) -> EpsilonExpansion:
        """Return the full endpoint power of one sector variable.

        The exponent of y_axis is assembled from every declared monomial source:
        the regularized measure/Jacobian, optional numerator weights, and the
        extracted U and F monomials with their topology-level epsilon-dependent
        powers.  This is the scalar quantity the subtraction algorithm needs.
        """
        parametric = self.parametric_representation
        if parametric is None:
            raise ValueError(f"{self.family}: missing parametric representation metadata")
        base = (
            float(sector.jacobian_monomial_powers[axis])
            + float(sector.measure_monomial_powers[axis])
            + float(sector.numerator_monomial_powers[axis])
            + parametric.u_exponent.base * float(sector.u_monomial_powers[axis])
            + parametric.f_exponent.base * float(sector.f_monomial_powers[axis])
        )
        eps_coeff = (
            parametric.u_exponent.eps_coeff * float(sector.u_monomial_powers[axis])
            + parametric.f_exponent.eps_coeff * float(sector.f_monomial_powers[axis])
        )
        return EpsilonExpansion(base=base, eps_coeff=eps_coeff)


def build_topology(request: IntegralRequest) -> TopologyDefinition:
    """Construct the U/F topology definition for the requested family."""
    m2 = request.m * request.m
    if request.integral == "dot":
        from dot_topology import build_topology_from_dot_request

        return build_topology_from_dot_request(request)
    if request.integral == "triangle":
        if request.s is None:
            raise ValueError("triangle topology requires s")
        return TopologyDefinition(
            family="C0(s;m^2)",
            x_names=["x0", "x1", "x2"],
            parameter_names=["s", "m2"],
            parameter_values=[float(request.s), float(m2)],
            u_expr=E("x0+x1+x2"),
            f_expr=E("m2*(x0+x1+x2)^2 - s*x1*x2"),
            u_power_base=-1,
            f_power_base=1,
            eps_log_u_coeff=2.0,
            eps_log_f_coeff=-1.0,
            expected_laurent_orders=["eps^-2", "eps^-1", "eps^0"],
            convention_note="triangle scalar integral in the OneLOop-compatible stripped convention",
            parametric_representation=ParametricRepresentation(
                loop_count=1,
                propagator_powers=(1.0, 1.0, 1.0),
                dimension=EpsilonExpansion(4.0, -2.0),
                gamma_argument=EpsilonExpansion(1.0, 1.0),
                u_exponent=EpsilonExpansion(-1.0, 2.0),
                f_exponent=EpsilonExpansion(-1.0, -1.0),
                parameter_weight_powers=(0.0, 0.0, 0.0),
                prefactor_description="-Gamma(1+eps) in the projective scalar-integral convention",
                convention_description="sector integrals are accumulated before the global prefactor/convention shift",
            ),
            jit_compile_evaluators=request.jit_compile_evaluators,
            dual_evaluator_mode=request.dual_evaluator_mode,
            ibp_reduce_to_log_endpoint=request.ibp_reduce_to_log_endpoint,
        )
    if request.integral == "box":
        if request.s12 is None or request.s23 is None:
            raise ValueError("box topology requires s12 and s23")
        return TopologyDefinition(
            family="D0(0,0,0,0,s12,s23;m^2)",
            x_names=["x0", "x1", "x2", "x3"],
            parameter_names=["s12", "s23", "m2"],
            parameter_values=[float(request.s12), float(request.s23), float(m2)],
            u_expr=E("x0+x1+x2+x3"),
            f_expr=E("m2*(x0+x1+x2+x3)^2 - s12*x0*x2 - s23*x1*x3"),
            u_power_base=0,
            f_power_base=2,
            eps_log_u_coeff=2.0,
            eps_log_f_coeff=-1.0,
            expected_laurent_orders=["eps^-2", "eps^-1", "eps^0"],
            convention_note="box scalar integral in the OneLOop-compatible stripped convention",
            parametric_representation=ParametricRepresentation(
                loop_count=1,
                propagator_powers=(1.0, 1.0, 1.0, 1.0),
                dimension=EpsilonExpansion(4.0, -2.0),
                gamma_argument=EpsilonExpansion(2.0, 1.0),
                u_exponent=EpsilonExpansion(0.0, 2.0),
                f_exponent=EpsilonExpansion(-2.0, -1.0),
                parameter_weight_powers=(0.0, 0.0, 0.0, 0.0),
                prefactor_description="Gamma(2+eps) in the projective scalar-integral convention",
                convention_description="sector integrals are accumulated before the global prefactor/convention shift",
            ),
            jit_compile_evaluators=request.jit_compile_evaluators,
            dual_evaluator_mode=request.dual_evaluator_mode,
            ibp_reduce_to_log_endpoint=request.ibp_reduce_to_log_endpoint,
        )
    raise ValueError(f"unsupported integral {request.integral!r}")


def feynman_log(value: complex) -> complex:
    """Logarithm with the scalar-integral ``-i0`` branch for negative reals."""
    z = complex(value)
    if abs(z.imag) < 1.0e-30 and z.real < 0.0:
        return complex(math.log(abs(z.real)), -math.pi)
    return cmath.log(z)


def feynman_log_array(values: np.ndarray) -> np.ndarray:
    """Vectorized version of ``feynman_log``."""
    z = np.asarray(values, dtype=np.complex128)
    logs = np.log(z)
    mask = (np.abs(z.imag) < 1.0e-30) & (z.real < 0.0)
    if np.any(mask):
        logs = logs.copy()
        logs[mask] = np.log(np.abs(z.real[mask])) - 1j * math.pi
    return logs


def complex_abs_for_training(value: complex) -> float:
    """Training weight used for one selected Laurent-coefficient sample."""
    return abs(complex(value))


def complex_abs_for_training_array(values: np.ndarray) -> np.ndarray:
    """Vectorized selected-coefficient training weight."""
    return np.abs(np.asarray(values, dtype=np.complex128))


MultiSeries = dict[tuple[int, ...], np.ndarray]
ExprSeries = dict[tuple[int, ...], Any]
PrecSeries = dict[tuple[int, ...], ComplexPrecise]


_DECIMAL_PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
)


def _multi_indices(max_orders: list[int]) -> list[tuple[int, ...]]:
    """Enumerate all Taylor multi-indices inside the requested truncation box."""
    if not max_orders:
        return [()]
    return [tuple(mi) for mi in product(*[range(order + 1) for order in max_orders])]


@lru_cache(maxsize=256)
def _dense_total_degree_multi_indices(rank: int, max_total: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate dense multi-indices with bounded total degree.

    The recursive enumeration avoids the large Cartesian product that would
    appear for signatures such as nine active coordinates and degree four.
    """
    rank = int(rank)
    max_total = int(max_total)
    if rank < 0 or max_total < 0:
        raise ValueError(f"invalid dense multi-index request rank={rank}, max_total={max_total}")
    if rank == 0:
        return ((),)

    out: list[tuple[int, ...]] = []

    def visit(prefix: list[int], remaining_rank: int, remaining_total: int) -> None:
        if remaining_rank == 0:
            out.append(tuple(prefix))
            return
        for value in range(remaining_total + 1):
            prefix.append(value)
            visit(prefix, remaining_rank - 1, remaining_total - value)
            prefix.pop()

    visit([], rank, max_total)
    return tuple(sorted(out, key=lambda item: (sum(item), item)))


def _zero_multi(dim: int) -> tuple[int, ...]:
    """Return the zero multi-index for a Taylor series dimension."""
    return tuple(0 for _ in range(dim))


def _series_constant(value: complex | np.ndarray, max_orders: list[int], n_rows: int) -> MultiSeries:
    """Build a truncated series with only a constant coefficient."""
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 0:
        array = np.full(n_rows, complex(array), dtype=np.complex128)
    return {_zero_multi(len(max_orders)): array.astype(np.complex128, copy=False)}


def _series_add(a: MultiSeries, b: MultiSeries) -> MultiSeries:
    """Add two sparse Taylor series."""
    out = {key: value.copy() for key, value in a.items()}
    for key, value in b.items():
        out[key] = out[key] + value if key in out else value.copy()
    return out


def _series_filter_allowed(
    a: MultiSeries,
    allowed_multis: set[tuple[int, ...]] | None,
) -> MultiSeries:
    """Return a copy containing only explicitly allowed Taylor coefficients."""
    if allowed_multis is None:
        return {key: value.copy() for key, value in a.items()}
    return {
        key: value.copy()
        for key, value in a.items()
        if key in allowed_multis
    }


def _series_scale(a: MultiSeries, factor: float | complex | np.ndarray) -> MultiSeries:
    """Scale every coefficient by a scalar or per-row array."""
    return {key: value * factor for key, value in a.items()}


def _series_mul(a: MultiSeries, b: MultiSeries, max_orders: list[int]) -> MultiSeries:
    """Multiply two truncated sparse Taylor series."""
    dim = len(max_orders)
    limits = tuple(int(order) for order in max_orders)
    out: MultiSeries = {}
    for key_a, value_a in a.items():
        for key_b, value_b in b.items():
            merged: list[int] = []
            valid = True
            for index in range(dim):
                value = int(key_a[index]) + int(key_b[index])
                if value > limits[index]:
                    valid = False
                    break
                merged.append(value)
            if not valid:
                continue
            key = tuple(merged)
            term = value_a * value_b
            out[key] = out[key] + term if key in out else term.copy()
    return out


def _allowed_multi_key(allowed_multis: set[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    """Return a stable cache key for an ancestor-closed sparse Taylor shape."""
    return tuple(sorted((tuple(int(value) for value in multi) for multi in allowed_multis)))


@lru_cache(maxsize=8192)
def _allowed_convolution_splits(
    allowed_key: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]], ...]:
    """Return valid sparse convolution splits for one allowed Taylor shape.

    The hard multi-axis fallback multiplies many series with the same sparse
    ancestor-closed support.  Precomputing ``left + right = result`` pairs moves
    tuple allocation and membership tests out of the hot multiplication loop.
    """
    by_result: list[
        tuple[tuple[int, ...], tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]]
    ] = []
    for result in allowed_key:
        split_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        ranges = [range(int(value) + 1) for value in result]
        for left in product(*ranges):
            left_tuple = tuple(int(value) for value in left)
            right_tuple = tuple(
                int(result[index]) - int(left_tuple[index])
                for index in range(len(result))
            )
            split_pairs.append((left_tuple, right_tuple))
        by_result.append((result, tuple(split_pairs)))
    return tuple(by_result)


@lru_cache(maxsize=8192)
def _allowed_convolution_dense_plan(
    allowed_key: tuple[tuple[int, ...], ...],
) -> tuple[
    tuple[tuple[int, ...], ...],
    dict[tuple[int, ...], int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return array indices for sparse-support convolution on ``allowed_key``."""
    rank = len(allowed_key[0]) if allowed_key else 0
    support_key = _allowed_multi_key(_ancestor_closed_multi_set(set(allowed_key), rank))
    key_to_index = {multi: index for index, multi in enumerate(support_key)}
    result_indices: list[int] = []
    left_indices: list[int] = []
    right_indices: list[int] = []
    for result, split_pairs in _allowed_convolution_splits(allowed_key):
        result_index = key_to_index[result]
        for left, right in split_pairs:
            result_indices.append(result_index)
            left_indices.append(key_to_index[left])
            right_indices.append(key_to_index[right])
    result_array = np.asarray(result_indices, dtype=np.int64)
    unique_results, starts = np.unique(result_array, return_index=True)
    return (
        support_key,
        key_to_index,
        result_array,
        np.asarray(left_indices, dtype=np.int64),
        np.asarray(right_indices, dtype=np.int64),
        unique_results.astype(np.int64, copy=False),
        starts.astype(np.int64, copy=False),
    )


@lru_cache(maxsize=8192)
def _allowed_convolution_split_count(
    allowed_key: tuple[tuple[int, ...], ...],
) -> int:
    """Return the number of dense convolution products for one support.

    For a full six-axis cubic support this is already ``10^6`` products for a
    single series multiplication.  Multiplying that by a large batch size
    creates multi-GB temporaries in the dense kernel, so the runtime uses this
    count to stay on the sparse direct path for hard endpoint sectors.
    """
    total = 0
    for result in allowed_key:
        count = 1
        for value in result:
            count *= int(value) + 1
        total += count
    return int(total)


def _series_mul_allowed_sparse_direct(
    a: MultiSeries,
    b: MultiSeries,
    allowed_multis: set[tuple[int, ...]],
    rank: int,
) -> MultiSeries:
    """Multiply two sparse Taylor series by direct dictionary convolution.

    Dense convolution is faster for wide batches and broad support, but the
    hard prepared-bundle diagnostics often evaluate one or a few points in a
    very sparse ancestor-closed shape.  In that regime repeatedly building and
    reducing dense support arrays dominates the runtime.  This direct path
    keeps the same algebra while avoiding the dense temporary layout.
    """
    out: MultiSeries = {}
    for key_a, value_a in a.items():
        value_a_array = np.asarray(value_a, dtype=np.complex128)
        for key_b, value_b in b.items():
            merged = tuple(int(key_a[index]) + int(key_b[index]) for index in range(rank))
            if merged not in allowed_multis:
                continue
            term = value_a_array * np.asarray(value_b, dtype=np.complex128)
            out[merged] = out[merged] + term if merged in out else term.copy()
    return out


def _series_mul_allowed(
    a: MultiSeries,
    b: MultiSeries,
    allowed_multis: set[tuple[int, ...]],
) -> MultiSeries:
    """Multiply sparse Taylor series while keeping only explicit outputs.

    The high-axis DOT sectors often need a sparse, ancestor-closed set of
    Taylor coefficients rather than the full rectangular box implied by the
    largest derivative in every axis.  Truncating by the explicit set avoids
    filling a large intermediate box in the symbolic-derivative chain-rule
    path while preserving every coefficient that can contribute to the
    requested outputs.
    """
    if not allowed_multis:
        return {}
    if not a or not b:
        return {}
    rank = len(next(iter(allowed_multis)))
    zero = _zero_multi(rank)
    if set(a) <= {zero}:
        factor = a.get(zero)
        if factor is None:
            return {}
        return {
            key: np.asarray(values, dtype=np.complex128) * factor
            for key, values in b.items()
            if key in allowed_multis
        }
    if set(b) <= {zero}:
        factor = b.get(zero)
        if factor is None:
            return {}
        return {
            key: np.asarray(values, dtype=np.complex128) * factor
            for key, values in a.items()
            if key in allowed_multis
        }
    if len(a) == 1:
        (key_a, value_a), = a.items()
        out: MultiSeries = {}
        for key_b, value_b in b.items():
            merged = tuple(int(key_a[index]) + int(key_b[index]) for index in range(rank))
            if merged in allowed_multis:
                out[merged] = np.asarray(value_a, dtype=np.complex128) * np.asarray(
                    value_b,
                    dtype=np.complex128,
                )
        return out
    if len(b) == 1:
        (key_b, value_b), = b.items()
        out = {}
        for key_a, value_a in a.items():
            merged = tuple(int(key_a[index]) + int(key_b[index]) for index in range(rank))
            if merged in allowed_multis:
                out[merged] = np.asarray(value_a, dtype=np.complex128) * np.asarray(
                    value_b,
                    dtype=np.complex128,
                )
        return out
    allowed_key = _allowed_multi_key(allowed_multis)
    n_rows = 0
    for values in a.values():
        n_rows = int(np.asarray(values).shape[0])
        break
    if n_rows <= 0:
        for values in b.values():
            n_rows = int(np.asarray(values).shape[0])
            break
    if n_rows <= 0:
        return {}
    split_count = _allowed_convolution_split_count(allowed_key)
    # The dense kernel materializes ``split_count * n_rows`` complex products.
    # It is excellent for modest supports, but pathological for the six-axis
    # triple-box chain-rule fallback.  Keep those hard cases sparse and let
    # NumPy vectorize over the sample axis inside the direct dictionary loop.
    if split_count * n_rows > 20_000_000:
        return _series_mul_allowed_sparse_direct(a, b, allowed_multis, rank)
    (
        support_key,
        key_to_index,
        result_indices,
        left_indices,
        right_indices,
        unique_results,
        starts,
    ) = _allowed_convolution_dense_plan(allowed_key)
    if not len(result_indices):
        return {}

    dense_a = np.zeros((len(support_key), n_rows), dtype=np.complex128)
    dense_b = np.zeros_like(dense_a)
    for key, values in a.items():
        index = key_to_index.get(tuple(key))
        if index is not None:
            dense_a[index, :] = np.asarray(values, dtype=np.complex128)
    for key, values in b.items():
        index = key_to_index.get(tuple(key))
        if index is not None:
            dense_b[index, :] = np.asarray(values, dtype=np.complex128)

    dense_out = np.zeros_like(dense_a)
    products = dense_a[left_indices, :] * dense_b[right_indices, :]
    dense_out[unique_results, :] = np.add.reduceat(products, starts, axis=0)
    active = np.any(dense_out != 0.0, axis=1)
    return {
        multi: dense_out[index, :].copy()
        for multi in allowed_key
        for index in (key_to_index[multi],)
        if active[index]
    }


def _series_without_constant(a: MultiSeries, n_rows: int) -> MultiSeries:
    """Return a copy with the constant coefficient removed."""
    zero = _zero_multi(len(next(iter(a.keys()))) if a else 0)
    return {key: value.copy() for key, value in a.items() if key != zero}


def _series_log(a: MultiSeries, max_orders: list[int], n_rows: int) -> MultiSeries:
    """Compute ``log(a)`` as a truncated Taylor series."""
    zero = _zero_multi(len(max_orders))
    constant = a[zero]
    out = _series_constant(feynman_log_array(constant), max_orders, n_rows)
    h = {
        key: value / constant
        for key, value in a.items()
        if key != zero
    }
    if not h:
        return out
    h_power = h
    for order in range(1, sum(max_orders) + 1):
        sign = 1.0 if order % 2 == 1 else -1.0
        out = _series_add(out, _series_scale(h_power, sign / float(order)))
        h_power = _series_mul(h_power, h, max_orders)
        if not h_power:
            break
    return out


def _series_log_allowed(
    a: MultiSeries,
    max_orders: list[int],
    n_rows: int,
    allowed_multis: set[tuple[int, ...]],
) -> MultiSeries:
    """Compute ``log(a)`` while retaining only an ancestor-closed sparse set."""
    zero = _zero_multi(len(max_orders))
    constant = a[zero]
    out = _series_constant(feynman_log_array(constant), max_orders, n_rows)
    h = {
        key: value / constant
        for key, value in a.items()
        if key != zero and key in allowed_multis
    }
    if not h:
        return _series_filter_allowed(out, allowed_multis)
    h_power = h
    for order in range(1, sum(max_orders) + 1):
        sign = 1.0 if order % 2 == 1 else -1.0
        out = _series_add(out, _series_scale(h_power, sign / float(order)))
        h_power = _series_mul_allowed(h_power, h, allowed_multis)
        if not h_power:
            break
    return _series_filter_allowed(out, allowed_multis)


def _series_exp(a: MultiSeries, max_orders: list[int], n_rows: int) -> MultiSeries:
    """Compute ``exp(a)`` as a truncated Taylor series."""
    zero = _zero_multi(len(max_orders))
    constant = a.get(zero, np.zeros(n_rows, dtype=np.complex128))
    h = _series_without_constant(a, n_rows)
    total = _series_constant(1.0 + 0.0j, max_orders, n_rows)
    if h:
        h_power = h
        factorial = 1.0
        for order in range(1, sum(max_orders) + 1):
            factorial *= float(order)
            total = _series_add(total, _series_scale(h_power, 1.0 / factorial))
            h_power = _series_mul(h_power, h, max_orders)
            if not h_power:
                break
    return _series_mul(
        _series_constant(np.exp(constant), max_orders, n_rows),
        total,
        max_orders,
    )


def _series_exp_allowed(
    a: MultiSeries,
    max_orders: list[int],
    n_rows: int,
    allowed_multis: set[tuple[int, ...]],
) -> MultiSeries:
    """Compute ``exp(a)`` while retaining only explicit sparse outputs."""
    zero = _zero_multi(len(max_orders))
    constant = a.get(zero, np.zeros(n_rows, dtype=np.complex128))
    h = {
        key: value.copy()
        for key, value in a.items()
        if key != zero and key in allowed_multis
    }
    total = _series_constant(1.0 + 0.0j, max_orders, n_rows)
    if h:
        h_power = h
        factorial = 1.0
        for order in range(1, sum(max_orders) + 1):
            factorial *= float(order)
            total = _series_add(total, _series_scale(h_power, 1.0 / factorial))
            h_power = _series_mul_allowed(h_power, h, allowed_multis)
            if not h_power:
                break
    return _series_mul_allowed(
        _series_constant(np.exp(constant), max_orders, n_rows),
        _series_filter_allowed(total, allowed_multis),
        allowed_multis,
    )


def _binomial_integer(exponent: int, order: int) -> float:
    """Return the generalized binomial coefficient for integer exponents."""
    if order < 0:
        return 0.0
    if order == 0:
        return 1.0
    numerator = 1.0
    for step in range(order):
        numerator *= float(exponent - step)
    denominator = 1.0
    for step in range(1, order + 1):
        denominator *= float(step)
    return numerator / denominator


def _series_integer_power(
    a: MultiSeries,
    exponent: int,
    max_orders: list[int],
    n_rows: int,
) -> MultiSeries:
    """Raise a Taylor series to an integer power without using log/exp."""
    if exponent == 0:
        return _series_constant(1.0 + 0.0j, max_orders, n_rows)
    if exponent > 0:
        result = _series_constant(1.0 + 0.0j, max_orders, n_rows)
        base = a
        remaining = int(exponent)
        while remaining:
            if remaining & 1:
                result = _series_mul(result, base, max_orders)
            remaining >>= 1
            if remaining:
                base = _series_mul(base, base, max_orders)
        return result

    zero = _zero_multi(len(max_orders))
    constant = a[zero]
    h = {
        key: value / constant
        for key, value in a.items()
        if key != zero
    }
    total = _series_constant(1.0 + 0.0j, max_orders, n_rows)
    if h:
        h_power = h
        for order in range(1, sum(max_orders) + 1):
            coeff = _binomial_integer(exponent, order)
            total = _series_add(total, _series_scale(h_power, coeff))
            h_power = _series_mul(h_power, h, max_orders)
            if not h_power:
                break
    return _series_mul(
        _series_constant(np.power(constant, exponent), max_orders, n_rows),
        total,
        max_orders,
    )


def _series_integer_power_allowed(
    a: MultiSeries,
    exponent: int,
    max_orders: list[int],
    n_rows: int,
    allowed_multis: set[tuple[int, ...]],
) -> MultiSeries:
    """Raise a sparse Taylor series to an integer power with explicit outputs."""
    if exponent == 0:
        return _series_filter_allowed(
            _series_constant(1.0 + 0.0j, max_orders, n_rows),
            allowed_multis,
        )
    if exponent > 0:
        result = _series_filter_allowed(
            _series_constant(1.0 + 0.0j, max_orders, n_rows),
            allowed_multis,
        )
        base = _series_filter_allowed(a, allowed_multis)
        remaining = int(exponent)
        while remaining:
            if remaining & 1:
                result = _series_mul_allowed(result, base, allowed_multis)
            remaining >>= 1
            if remaining:
                base = _series_mul_allowed(base, base, allowed_multis)
        return result

    zero = _zero_multi(len(max_orders))
    constant = a[zero]
    h = {
        key: value / constant
        for key, value in a.items()
        if key != zero and key in allowed_multis
    }
    total = _series_filter_allowed(
        _series_constant(1.0 + 0.0j, max_orders, n_rows),
        allowed_multis,
    )
    if h:
        h_power = h
        for order in range(1, sum(max_orders) + 1):
            coeff = _binomial_integer(exponent, order)
            total = _series_add(total, _series_scale(h_power, coeff))
            h_power = _series_mul_allowed(h_power, h, allowed_multis)
            if not h_power:
                break
    return _series_mul_allowed(
        _series_constant(np.power(constant, exponent), max_orders, n_rows),
        _series_filter_allowed(total, allowed_multis),
        allowed_multis,
    )


def _series_pow_real(a: MultiSeries, power: float, max_orders: list[int], n_rows: int) -> MultiSeries:
    """Raise a regular Taylor series to a real power using log/exp."""
    if abs(power) <= 1.0e-15:
        return _series_constant(1.0 + 0.0j, max_orders, n_rows)
    rounded = round(float(power))
    if abs(float(power) - rounded) <= 1.0e-12:
        return _series_integer_power(a, int(rounded), max_orders, n_rows)
    return _series_exp(_series_scale(_series_log(a, max_orders, n_rows), power), max_orders, n_rows)


def _series_pow_real_allowed(
    a: MultiSeries,
    power: float,
    max_orders: list[int],
    n_rows: int,
    allowed_multis: set[tuple[int, ...]],
) -> MultiSeries:
    """Raise a Taylor series to a real power with sparse truncation."""
    if abs(power) <= 1.0e-15:
        return _series_filter_allowed(
            _series_constant(1.0 + 0.0j, max_orders, n_rows),
            allowed_multis,
        )
    rounded = round(float(power))
    if abs(float(power) - rounded) <= 1.0e-12:
        return _series_integer_power_allowed(
            a,
            int(rounded),
            max_orders,
            n_rows,
            allowed_multis,
        )
    return _series_exp_allowed(
        _series_scale(_series_log_allowed(a, max_orders, n_rows, allowed_multis), power),
        max_orders,
        n_rows,
        allowed_multis,
    )


def _series_pow_real_and_log_allowed(
    a: MultiSeries,
    power: float,
    max_orders: list[int],
    n_rows: int,
    allowed_multis: set[tuple[int, ...]],
) -> tuple[MultiSeries, MultiSeries]:
    """Return ``a**power`` and ``log(a)`` sharing the same sparse ladder.

    Hard endpoint sectors repeatedly need both the regular prefactor
    ``U^p F^q`` and the epsilon logarithm ``p_eps log(U)+q_eps log(F)``.
    For integer powers, both series are functions of powers of
    ``h = a/a_0 - 1``.  Building that ladder once saves a sizeable amount of
    Python-side sparse convolution without changing the black-box U/F boundary.
    """
    rounded = round(float(power))
    if abs(float(power) - rounded) > 1.0e-12:
        log_series = _series_log_allowed(a, max_orders, n_rows, allowed_multis)
        return (
            _series_exp_allowed(
                _series_scale(log_series, power),
                max_orders,
                n_rows,
                allowed_multis,
            ),
            log_series,
        )

    exponent = int(rounded)
    zero = _zero_multi(len(max_orders))
    constant = a[zero]
    log_series = _series_constant(feynman_log_array(constant), max_orders, n_rows)
    if exponent == 0:
        power_series = _series_filter_allowed(
            _series_constant(1.0 + 0.0j, max_orders, n_rows),
            allowed_multis,
        )
    else:
        power_series = _series_filter_allowed(
            _series_constant(1.0 + 0.0j, max_orders, n_rows),
            allowed_multis,
        )

    h = {
        key: value / constant
        for key, value in a.items()
        if key != zero and key in allowed_multis
    }
    if h:
        h_power = h
        for order in range(1, sum(max_orders) + 1):
            if exponent:
                coeff = _binomial_integer(exponent, order)
                if coeff:
                    power_series = _series_add(
                        power_series,
                        _series_scale(h_power, coeff),
                    )
            sign = 1.0 if order % 2 == 1 else -1.0
            log_series = _series_add(log_series, _series_scale(h_power, sign / float(order)))
            h_power = _series_mul_allowed(h_power, h, allowed_multis)
            if not h_power:
                break

    if exponent:
        power_series = _series_mul_allowed(
            _series_constant(np.power(constant, exponent), max_orders, n_rows),
            _series_filter_allowed(power_series, allowed_multis),
            allowed_multis,
        )
    return (
        _series_filter_allowed(power_series, allowed_multis),
        _series_filter_allowed(log_series, allowed_multis),
    )


def _series_coefficient(series: MultiSeries, multi_index: tuple[int, ...], n_rows: int) -> np.ndarray:
    """Return one Taylor coefficient, or zeros if it is absent."""
    value = series.get(multi_index)
    if value is None:
        return np.zeros(n_rows, dtype=np.complex128)
    return value


def _slice_multi_series_list(
    series_by_order: list[MultiSeries],
    start: int,
    stop: int,
) -> list[MultiSeries]:
    """Slice every coefficient array in a list of sparse Taylor series."""
    return [
        {
            multi: np.asarray(values[start:stop], dtype=np.complex128).copy()
            for multi, values in series.items()
        }
        for series in series_by_order
    ]


def _pc_zero() -> ComplexPrecise:
    """Return the arbitrary-precision complex zero."""
    return (Decimal(0), Decimal(0))


def _pc_one() -> ComplexPrecise:
    """Return the arbitrary-precision complex one."""
    return (Decimal(1), Decimal(0))


def _pc_from_real(value: Any, precision_digits: int) -> ComplexPrecise:
    """Promote a real scalar to Symbolica's complex Decimal shape."""
    return (decimal_with_precision(value, precision_digits), Decimal(0))


def _pc_is_zero(value: ComplexPrecise) -> bool:
    """Return whether an arbitrary-precision complex value is exactly zero."""
    return Decimal(value[0]).is_zero() and Decimal(value[1]).is_zero()


def _pc_add(left: ComplexPrecise, right: ComplexPrecise) -> ComplexPrecise:
    """Add two arbitrary-precision complex values."""
    return (Decimal(left[0]) + Decimal(right[0]), Decimal(left[1]) + Decimal(right[1]))


def _pc_sub(left: ComplexPrecise, right: ComplexPrecise) -> ComplexPrecise:
    """Subtract two arbitrary-precision complex values."""
    return (Decimal(left[0]) - Decimal(right[0]), Decimal(left[1]) - Decimal(right[1]))


def _pc_mul(left: ComplexPrecise, right: ComplexPrecise) -> ComplexPrecise:
    """Multiply two arbitrary-precision complex values."""
    a, b = Decimal(left[0]), Decimal(left[1])
    c, d = Decimal(right[0]), Decimal(right[1])
    return (a * c - b * d, a * d + b * c)


def _pc_div(left: ComplexPrecise, right: ComplexPrecise) -> ComplexPrecise:
    """Divide two arbitrary-precision complex values."""
    a, b = Decimal(left[0]), Decimal(left[1])
    c, d = Decimal(right[0]), Decimal(right[1])
    denominator = c * c + d * d
    return ((a * c + b * d) / denominator, (b * c - a * d) / denominator)


def _pc_scale(value: ComplexPrecise, factor: Any, precision_digits: int) -> ComplexPrecise:
    """Scale an arbitrary-precision complex value by a real factor."""
    scalar = decimal_with_precision(factor, precision_digits)
    return (Decimal(value[0]) * scalar, Decimal(value[1]) * scalar)


def _pc_int_power(value: ComplexPrecise, power: int) -> ComplexPrecise:
    """Raise an arbitrary-precision complex value to an integer power."""
    exponent = int(power)
    if exponent == 0:
        return _pc_one()
    if exponent < 0:
        return _pc_div(_pc_one(), _pc_int_power(value, -exponent))
    result = _pc_one()
    base = value
    while exponent:
        if exponent & 1:
            result = _pc_mul(result, base)
        exponent >>= 1
        if exponent:
            base = _pc_mul(base, base)
    return result


def _pc_log(value: ComplexPrecise, precision_digits: int) -> ComplexPrecise:
    """Feynman-branch logarithm for arbitrary-precision complex values.

    The no-threshold FSD path samples Euclidean sectors where U/F residuals
    are positive reals.  That case stays fully Decimal.  A complex fallback is
    retained for diagnostics and non-default experiments, but it necessarily
    loses arbitrary precision because Python's standard library has no complex
    Decimal elementary functions.
    """
    real = Decimal(value[0])
    imag = Decimal(value[1])
    if imag.is_zero():
        if real > 0:
            return (real.ln(), Decimal(0))
        if real < 0:
            return ((-real).ln(), -decimal_with_precision(_DECIMAL_PI, precision_digits))
    approx = feynman_log(complex(float(real), float(imag)))
    return _decimal_complex(approx, precision_digits)


def _pc_exp(value: ComplexPrecise, precision_digits: int) -> ComplexPrecise:
    """Exponential for arbitrary-precision complex values."""
    real = Decimal(value[0])
    imag = Decimal(value[1])
    if imag.is_zero():
        return (real.exp(), Decimal(0))
    approx = cmath.exp(complex(float(real), float(imag)))
    return _decimal_complex(approx, precision_digits)


def _prec_series_constant(
    value: ComplexPrecise,
    max_orders: list[int],
) -> PrecSeries:
    """Build a single-row Decimal Taylor series with only a constant term."""
    return {_zero_multi(len(max_orders)): value}


def _prec_series_add(left: PrecSeries, right: PrecSeries) -> PrecSeries:
    """Add two single-row Decimal Taylor series."""
    out = dict(left)
    for key, value in right.items():
        out[key] = _pc_add(out[key], value) if key in out else value
    return out


def _prec_series_scale(series: PrecSeries, factor: Any, precision_digits: int) -> PrecSeries:
    """Scale every coefficient of a Decimal Taylor series."""
    return {
        key: _pc_scale(value, factor, precision_digits)
        for key, value in series.items()
    }


def _prec_series_mul(left: PrecSeries, right: PrecSeries, max_orders: list[int]) -> PrecSeries:
    """Multiply two single-row Decimal Taylor series."""
    dim = len(max_orders)
    out: PrecSeries = {}
    for key_left, value_left in left.items():
        for key_right, value_right in right.items():
            key = tuple(key_left[i] + key_right[i] for i in range(dim))
            if any(key[i] > max_orders[i] for i in range(dim)):
                continue
            term = _pc_mul(value_left, value_right)
            out[key] = _pc_add(out[key], term) if key in out else term
    return out


def _prec_series_coefficient(series: PrecSeries, multi_index: tuple[int, ...]) -> ComplexPrecise:
    """Return a Decimal Taylor coefficient or zero if absent."""
    return series.get(multi_index, _pc_zero())


def _prec_series_log(
    series: PrecSeries,
    max_orders: list[int],
    precision_digits: int,
) -> PrecSeries:
    """Compute the logarithm of a single-row Decimal Taylor series."""
    zero = _zero_multi(len(max_orders))
    constant = series[zero]
    out = _prec_series_constant(_pc_log(constant, precision_digits), max_orders)
    h = {
        key: _pc_div(value, constant)
        for key, value in series.items()
        if key != zero and not _pc_is_zero(value)
    }
    if not h:
        return out
    h_power = h
    for order in range(1, sum(max_orders) + 1):
        sign = 1 if order % 2 == 1 else -1
        out = _prec_series_add(
            out,
            _prec_series_scale(h_power, Decimal(sign) / Decimal(order), precision_digits),
        )
        h_power = _prec_series_mul(h_power, h, max_orders)
        if not h_power:
            break
    return out


def _prec_series_exp(
    series: PrecSeries,
    max_orders: list[int],
    precision_digits: int,
) -> PrecSeries:
    """Compute the exponential of a single-row Decimal Taylor series."""
    zero = _zero_multi(len(max_orders))
    constant = series.get(zero, _pc_zero())
    h = {key: value for key, value in series.items() if key != zero and not _pc_is_zero(value)}
    total = _prec_series_constant(_pc_one(), max_orders)
    if h:
        h_power = h
        factorial = Decimal(1)
        for order in range(1, sum(max_orders) + 1):
            factorial *= Decimal(order)
            total = _prec_series_add(
                total,
                _prec_series_scale(h_power, Decimal(1) / factorial, precision_digits),
            )
            h_power = _prec_series_mul(h_power, h, max_orders)
            if not h_power:
                break
    return _prec_series_mul(
        _prec_series_constant(_pc_exp(constant, precision_digits), max_orders),
        total,
        max_orders,
    )


def _prec_series_integer_power(
    series: PrecSeries,
    exponent: int,
    max_orders: list[int],
    precision_digits: int,
) -> PrecSeries:
    """Raise a Decimal Taylor series to an integer power without log/exp."""
    if exponent == 0:
        return _prec_series_constant(_pc_one(), max_orders)
    if exponent > 0:
        out = _prec_series_constant(_pc_one(), max_orders)
        base = series
        remaining = int(exponent)
        while remaining:
            if remaining & 1:
                out = _prec_series_mul(out, base, max_orders)
            remaining >>= 1
            if remaining:
                base = _prec_series_mul(base, base, max_orders)
        return out

    zero = _zero_multi(len(max_orders))
    constant = series[zero]
    h = {
        key: _pc_div(value, constant)
        for key, value in series.items()
        if key != zero and not _pc_is_zero(value)
    }
    total = _prec_series_constant(_pc_one(), max_orders)
    if h:
        h_power = h
        for order in range(1, sum(max_orders) + 1):
            coeff = Decimal(exponent)
            for step in range(1, order):
                coeff *= Decimal(exponent - step)
            factorial = Decimal(1)
            for step in range(1, order + 1):
                factorial *= Decimal(step)
            total = _prec_series_add(
                total,
                _prec_series_scale(h_power, coeff / factorial, precision_digits),
            )
            h_power = _prec_series_mul(h_power, h, max_orders)
            if not h_power:
                break
    return _prec_series_mul(
        _prec_series_constant(_pc_int_power(constant, exponent), max_orders),
        total,
        max_orders,
    )


def _prec_series_pow_real(
    series: PrecSeries,
    power: float,
    max_orders: list[int],
    precision_digits: int,
) -> PrecSeries:
    """Raise a Decimal Taylor series to a real power."""
    if abs(power) <= 1.0e-15:
        return _prec_series_constant(_pc_one(), max_orders)
    rounded = round(float(power))
    if abs(float(power) - rounded) <= 1.0e-12:
        return _prec_series_integer_power(
            series,
            int(rounded),
            max_orders,
            precision_digits,
        )
    return _prec_series_exp(
        _prec_series_scale(
            _prec_series_log(series, max_orders, precision_digits),
            power,
            precision_digits,
        ),
        max_orders,
        precision_digits,
    )


def _expr_number(value: float | int | complex) -> Any:
    """Create a Symbolica numeric expression from a Python scalar."""
    z = complex(value)
    if abs(z.imag) > 0.0:
        raise ValueError(f"complex constants are not supported in generated formula expressions: {value!r}")
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
    """Raise a Symbolica expression to an integer power without float exponents."""
    integer_power = int(power)
    if integer_power == 0:
        return E("1")
    if integer_power > 0:
        return base ** integer_power
    return E("1") / (base ** abs(integer_power))


def _integer_coordinate_power(value: float, label: str) -> int:
    """Return an integer coordinate exponent or reject unsupported powers."""
    rounded = round(float(value))
    if abs(float(value) - rounded) > 1.0e-12:
        raise ValueError(
            f"{label}: generated subtraction formula requires integer coordinate powers, "
            f"got {value!r}"
        )
    return int(rounded)


def _expr_series_constant(value: Any, max_orders: list[int]) -> ExprSeries:
    """Build a Symbolica Taylor series with only a constant coefficient."""
    return {_zero_multi(len(max_orders)): value if hasattr(value, "evaluator") else _expr_number(value)}


def _expr_series_add(a: ExprSeries, b: ExprSeries) -> ExprSeries:
    """Add two sparse Symbolica Taylor series."""
    out = dict(a)
    for key, value in b.items():
        out[key] = out[key] + value if key in out else value
    return out


def _expr_series_scale(a: ExprSeries, factor: float | int | Any) -> ExprSeries:
    """Scale every Symbolica Taylor coefficient."""
    factor_expr = factor if hasattr(factor, "evaluator") else _expr_number(factor)
    return {key: value * factor_expr for key, value in a.items()}


def _expr_series_mul(a: ExprSeries, b: ExprSeries, max_orders: list[int]) -> ExprSeries:
    """Multiply two truncated sparse Symbolica Taylor series."""
    dim = len(max_orders)
    out: ExprSeries = {}
    for key_a, value_a in a.items():
        for key_b, value_b in b.items():
            key = tuple(key_a[i] + key_b[i] for i in range(dim))
            if any(key[i] > max_orders[i] for i in range(dim)):
                continue
            term = value_a * value_b
            out[key] = out[key] + term if key in out else term
    return out


def _expr_series_mul_allowed(
    a: ExprSeries,
    b: ExprSeries,
    allowed_multis: set[tuple[int, ...]],
) -> ExprSeries:
    """Multiply Symbolica series while keeping only explicit coefficients.

    This is the expression-level analogue of ``_series_mul_allowed``.  It is
    used to build chain-rule composition evaluators without expanding to the
    full rectangular Taylor box of hard multi-axis sectors.
    """
    if not a or not b or not allowed_multis:
        return {}
    rank = len(next(iter(allowed_multis)))
    out: ExprSeries = {}
    for key_a, value_a in a.items():
        for key_b, value_b in b.items():
            key = tuple(int(key_a[index]) + int(key_b[index]) for index in range(rank))
            if key not in allowed_multis:
                continue
            term = value_a * value_b
            out[key] = out[key] + term if key in out else term
    return out


def _expr_series_without_constant(a: ExprSeries) -> ExprSeries:
    """Return a copy with the constant coefficient removed."""
    zero = _zero_multi(len(next(iter(a.keys()))) if a else 0)
    return {key: value for key, value in a.items() if key != zero}


def _expr_series_log(a: ExprSeries, max_orders: list[int]) -> ExprSeries:
    """Compute ``log(a)`` as a truncated Symbolica Taylor series."""
    zero = _zero_multi(len(max_orders))
    constant = a[zero]
    out = _expr_series_constant(constant.log(), max_orders)
    h = {
        key: value / constant
        for key, value in a.items()
        if key != zero
    }
    if not h:
        return out
    h_power = h
    for order in range(1, sum(max_orders) + 1):
        sign = 1.0 if order % 2 == 1 else -1.0
        out = _expr_series_add(out, _expr_series_scale(h_power, sign / float(order)))
        h_power = _expr_series_mul(h_power, h, max_orders)
        if not h_power:
            break
    return out


def _expr_series_exp(a: ExprSeries, max_orders: list[int]) -> ExprSeries:
    """Compute ``exp(a)`` as a truncated Symbolica Taylor series."""
    zero = _zero_multi(len(max_orders))
    constant = a.get(zero, E("0"))
    h = _expr_series_without_constant(a)
    total = _expr_series_constant(E("1"), max_orders)
    if h:
        h_power = h
        factorial = 1.0
        for order in range(1, sum(max_orders) + 1):
            factorial *= float(order)
            total = _expr_series_add(total, _expr_series_scale(h_power, 1.0 / factorial))
            h_power = _expr_series_mul(h_power, h, max_orders)
            if not h_power:
                break
    return _expr_series_mul(
        _expr_series_constant(constant.exp(), max_orders),
        total,
        max_orders,
    )


def _expr_series_pow_real(a: ExprSeries, power: float, max_orders: list[int]) -> ExprSeries:
    """Raise a Symbolica Taylor series to a real power."""
    if abs(power) <= 1.0e-15:
        return _expr_series_constant(E("1"), max_orders)
    return _expr_series_exp(_expr_series_scale(_expr_series_log(a, max_orders), power), max_orders)


def _expr_series_coefficient(series: ExprSeries, multi_index: tuple[int, ...]) -> Any:
    """Return a Symbolica Taylor coefficient, or zero if absent."""
    return series.get(multi_index, E("0"))


@dataclass
class SubtractionFormulaDefinition:
    """Pregenerated Symbolica formula evaluators for one endpoint signature."""

    signature: tuple[Any, ...]
    input_names: list[str]
    input_symbols: list[Any]
    output_expressions: list[Any]
    evaluators: list[Any]
    laurent_orders: list[int]
    zero_subsets: list[tuple[int, ...]]
    dual_shape: list[tuple[int, ...]]
    build_seconds: float = 0.0

    @property
    def output_labels(self) -> list[str]:
        """Return display labels for formula outputs."""
        return [f"eps^{order}" for order in self.laurent_orders]

    def evaluate_complex_batch(self, rows: np.ndarray, timing: HotPathTiming | None = None) -> np.ndarray:
        """Evaluate all Laurent outputs with Symbolica's complex batch API."""
        start = time.perf_counter()
        columns = [
            np.asarray(evaluator.evaluate_complex(rows), dtype=np.complex128)[:, 0]
            for evaluator in self.evaluators
        ]
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return np.stack(columns, axis=1)

    def evaluate_complex_prec(
        self,
        row: ComplexPreciseRow,
        precision_digits: int,
        timing: HotPathTiming | None = None,
    ) -> list[complex]:
        """Evaluate all Laurent outputs with complex multiprecision."""
        values: list[complex] = []
        start = time.perf_counter()
        for evaluator in self.evaluators:
            result = evaluator.evaluate_complex_with_prec(row, precision_digits)[0]
            values.append(complex(float(result[0]), float(result[1])))
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return values


@dataclass
class IBPEndpointProjectorTerm:
    """One IBP-lowered contribution feeding a logarithmic endpoint projector."""

    prefactor_coeffs: list[complex]
    boundary_positions: tuple[int, ...]
    derivative_multi: tuple[int, ...]
    active_positions: tuple[int, ...]
    child_signature: tuple[Any, ...]


@dataclass
class EndpointProjectorFormulaDefinition:
    """Endpoint-only Symbolica projector shared by many sectors."""

    signature: tuple[Any, ...]
    input_names: list[str]
    input_symbols: list[Any]
    output_expressions: list[Any]
    evaluators: list[Any]
    laurent_orders: list[int]
    zero_subsets: list[tuple[int, ...]]
    taylor_orders: list[int]
    coefficient_layout: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]]
    ibp_reduce_to_log_endpoint: bool = False
    ibp_terms: list[IBPEndpointProjectorTerm] = field(default_factory=list)
    child_formulas: dict[tuple[Any, ...], "EndpointProjectorFormulaDefinition"] = field(
        default_factory=dict
    )
    build_seconds: float = 0.0

    @property
    def output_labels(self) -> list[str]:
        """Return display labels for formula outputs."""
        return [f"eps^{order}" for order in self.laurent_orders]

    def evaluate_complex_batch(self, rows: np.ndarray, timing: HotPathTiming | None = None) -> np.ndarray:
        """Evaluate all Laurent outputs with Symbolica's complex batch API."""
        start = time.perf_counter()
        columns = [
            np.asarray(evaluator.evaluate_complex(rows), dtype=np.complex128)[:, 0]
            for evaluator in self.evaluators
        ]
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return np.stack(columns, axis=1)

    def evaluate_complex_prec(
        self,
        row: ComplexPreciseRow,
        precision_digits: int,
        timing: HotPathTiming | None = None,
    ) -> list[complex]:
        """Evaluate all Laurent outputs with complex multiprecision."""
        values: list[complex] = []
        start = time.perf_counter()
        for evaluator in self.evaluators:
            result = evaluator.evaluate_complex_with_prec(row, precision_digits)[0]
            values.append(complex(float(result[0]), float(result[1])))
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return values


@dataclass
class RegularTaylorFormulaDefinition:
    """Pregenerated Symbolica evaluator for regular ``g_s`` coefficients."""

    signature: tuple[Any, ...]
    input_names: list[str]
    input_symbols: list[Any]
    output_expressions: list[Any]
    evaluators: list[Any]
    output_layout: list[tuple[tuple[int, ...], int]]
    input_layout: list[tuple[str, tuple[int, ...]]]
    max_orders: list[int]
    zero_positions: tuple[int, ...]
    dual_shape: list[tuple[int, ...]] = field(default_factory=list)
    evaluator_input_symbols: list[Any] = field(default_factory=list)
    evaluator_dual_shape: list[tuple[int, ...]] = field(default_factory=list)
    evaluator_output_indices: list[int] = field(default_factory=list)
    dual_variable_count: int = 0
    cache_evaluator_files: list[str] = field(default_factory=list)
    build_seconds: float = 0.0

    def evaluate_complex_batch(self, rows: np.ndarray, timing: HotPathTiming | None = None) -> np.ndarray:
        """Evaluate all requested regular Taylor coefficients."""
        start = time.perf_counter()
        if self.evaluator_dual_shape:
            values = np.asarray(
                self.evaluators[0].evaluate_complex(self._dualized_input_matrix(rows)),
                dtype=np.complex128,
            )
            if self.evaluator_output_indices:
                values = values[:, self.evaluator_output_indices]
            if timing is not None:
                timing.add_eval(time.perf_counter() - start)
            return values

        columns = [
            np.asarray(evaluator.evaluate_complex(rows), dtype=np.complex128)[:, 0]
            for evaluator in self.evaluators
        ]
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        if not columns:
            return np.zeros((rows.shape[0], 0), dtype=np.complex128)
        return np.stack(columns, axis=1)

    def _dualized_input_matrix(self, rows: np.ndarray) -> np.ndarray:
        """Expand ordinary runtime inputs into Symbolica dual input storage."""
        sample_rows = np.asarray(rows, dtype=np.complex128)
        dual_shape = [tuple(int(value) for value in mi) for mi in self.evaluator_dual_shape]
        dual_len = len(dual_shape)
        param_count = len(self.evaluator_input_symbols)
        zero = tuple(0 for _ in range(param_count))
        zero_index = dual_shape.index(zero)
        unit_index: dict[int, int] = {}
        for param_index in range(int(self.dual_variable_count)):
            unit = tuple(1 if axis == param_index else 0 for axis in range(param_count))
            if unit in dual_shape:
                unit_index[param_index] = dual_shape.index(unit)

        expanded = np.zeros((sample_rows.shape[0], param_count * dual_len), dtype=np.complex128)
        for param_index in range(int(self.dual_variable_count)):
            index = unit_index.get(param_index)
            if index is not None:
                expanded[:, param_index * dual_len + index] = 1.0
        for column in range(sample_rows.shape[1]):
            param_index = int(self.dual_variable_count) + column
            expanded[:, param_index * dual_len + zero_index] = sample_rows[:, column]
        return expanded


@dataclass
class ChainRuleFormulaDefinition:
    """Symbolica evaluator for composing x-derivatives with sector-map jets.

    The formula is deliberately independent of U/F expressions.  It receives
    numerical values of original-parameter derivatives and numerical Taylor
    coefficients of the sector map, then returns Taylor coefficients of
    ``P(X_s(y))`` for one polynomial ``P``.
    """

    signature: tuple[Any, ...]
    input_names: list[str]
    input_symbols: list[Any]
    output_expressions: list[Any]
    evaluators: list[Any]
    output_shape: list[tuple[int, ...]]
    derivative_indices: list[tuple[int, ...]]
    h_layout: list[tuple[int, tuple[int, ...]]]
    evaluator_mode: str = "separate"
    build_seconds: float = 0.0
    cache_json_path: str | None = None
    cache_evaluator_files: list[str] = field(default_factory=list)

    def evaluate_complex_batch(self, rows: np.ndarray, timing: HotPathTiming | None = None) -> np.ndarray:
        """Evaluate all mapped Taylor coefficients for a complex batch."""
        start = time.perf_counter()
        if not self.evaluators:
            if self.output_shape:
                raise RuntimeError(
                    "chain-rule formula was loaded as metadata only; "
                    "it can be serialized into a prepared bundle but cannot be evaluated"
                )
            values = np.zeros((rows.shape[0], 0), dtype=np.complex128)
        elif self.evaluator_mode == "multiple":
            values = np.asarray(self.evaluators[0].evaluate_complex(rows), dtype=np.complex128)
            if values.ndim == 1:
                values = values.reshape(rows.shape[0], 1)
        else:
            values = np.column_stack(
                [
                    np.asarray(evaluator.evaluate_complex(rows), dtype=np.complex128)[:, 0]
                    for evaluator in self.evaluators
                ]
            )
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return values


def _chain_rule_formula_cache_dir() -> Path:
    """Return the local generated cache directory for universal chain formulas."""
    configured = os.environ.get("FSD_SUBTRACTION_FORMULA_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return formula_cache_dir()


def _chain_rule_jsonable(value: Any) -> Any:
    """Convert tuple-heavy signatures/layouts into deterministic JSON values."""
    if isinstance(value, tuple):
        return [_chain_rule_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_chain_rule_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _chain_rule_jsonable(item) for key, item in value.items()}
    return value


def _chain_rule_tupled(value: Any) -> Any:
    """Convert nested JSON lists back to tuples for cache signatures/layouts."""
    if isinstance(value, list):
        return tuple(_chain_rule_tupled(item) for item in value)
    if isinstance(value, dict):
        return {key: _chain_rule_tupled(item) for key, item in value.items()}
    return value


def _chain_rule_signature_payload(signature: tuple[Any, ...]) -> dict[str, Any]:
    """Return the topology-independent cache payload for a chain formula."""
    return {
        "schema_version": CHAIN_RULE_FORMULA_CACHE_VERSION,
        "kind": "chain-rule",
        "signature": _chain_rule_jsonable(signature),
    }


def _chain_rule_formula_cache_path(signature: tuple[Any, ...]) -> Path:
    """Return the generated JSON cache path for one universal chain formula."""
    payload = _chain_rule_signature_payload(signature)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return _chain_rule_formula_cache_dir() / f"chain_rule_{digest}.json"


def _chain_rule_formula_cache_read_paths(path: Path) -> list[Path]:
    """Return generated and curated cache candidates for one chain formula."""
    paths: list[Path] = []
    for root in formula_cache_read_roots():
        for candidate in (root / "curated" / path.name, root / path.name):
            if candidate not in paths:
                paths.append(candidate)
    return paths


def _chain_rule_evaluator_cache_name(path: Path, index: int) -> str:
    """Return the sidecar filename for one chain-rule evaluator."""
    return f"{path.stem}.eval_{int(index)}.bin.gz"


def _chain_rule_expression_cache_name(path: Path) -> str:
    """Return the compressed reference-expression sidecar filename."""
    return f"{path.stem}.expr.json.gz"


def _write_chain_rule_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one chain-rule cache JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_chain_rule_formula_to_cache(formula: ChainRuleFormulaDefinition) -> None:
    """Persist a generated universal chain-rule formula and evaluator sidecars."""
    path = _chain_rule_formula_cache_path(formula.signature)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_names: list[str] = []
        for index, evaluator in enumerate(formula.evaluators):
            name = _chain_rule_evaluator_cache_name(path, index)
            raw_bytes = evaluator.save()
            (path.parent / name).write_bytes(gzip.compress(raw_bytes, compresslevel=6))
            sidecar_names.append(name)
        expression_sidecar_name: str | None = None
        if formula.output_expressions:
            try:
                expression_sidecar_name = _chain_rule_expression_cache_name(path)
                expression_payload = json.dumps(
                    [str(expr) for expr in formula.output_expressions],
                    separators=(",", ":"),
                ).encode("utf-8")
                (path.parent / expression_sidecar_name).write_bytes(
                    gzip.compress(expression_payload, compresslevel=6)
                )
            except Exception:
                expression_sidecar_name = None
        payload = {
            "signature_payload": _chain_rule_signature_payload(formula.signature),
            "signature": _chain_rule_jsonable(formula.signature),
            "input_names": list(formula.input_names),
            # Runtime needs only the serialized evaluator plus the coefficient
            # layouts below.  Reference expressions, when written, live in a
            # compressed sidecar so the main metadata stays cheap to inspect.
            "output_expressions": [],
            "output_expression_count": len(formula.output_expressions),
            "expression_cache_file": expression_sidecar_name,
            "output_shape": _chain_rule_jsonable(tuple(formula.output_shape)),
            "derivative_indices": _chain_rule_jsonable(tuple(formula.derivative_indices)),
            "h_layout": _chain_rule_jsonable(tuple(formula.h_layout)),
            "evaluator_mode": str(formula.evaluator_mode),
            "evaluator_cache_files": sidecar_names,
            "build_seconds": float(formula.build_seconds),
        }
        _write_chain_rule_json_atomic(path, payload)
    except Exception:
        # The cache is an optimization only.  A full prepared bundle will still
        # serialize the in-memory evaluator if this local cache write fails.
        return


def _load_chain_rule_formula_from_cache(
    signature: tuple[Any, ...],
    *,
    load_evaluators: bool = True,
) -> ChainRuleFormulaDefinition | None:
    """Load a cached universal chain-rule formula if all sidecars are present.

    Prepared-bundle generation often only needs the formula metadata and the
    sidecar byte paths.  Avoiding ``Evaluator.load()`` there is important for
    large three-loop bundles, because the bundle writer can copy the sidecars
    directly.
    """
    path = _chain_rule_formula_cache_path(signature)
    for candidate in _chain_rule_formula_cache_read_paths(path):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if data.get("signature_payload") != _chain_rule_signature_payload(signature):
                continue
            sidecar_names = [str(name) for name in data.get("evaluator_cache_files", [])]
            if not sidecar_names:
                continue
            mirror_cache_entry_to_primary(
                candidate,
                data,
                sidecar_fields=("evaluator_cache_files", "expression_cache_file"),
            )
            sidecar_paths = []
            for name in sidecar_names:
                path = candidate.parent / name
                if not path.is_file() and path.suffix != ".gz":
                    compressed = path.with_name(path.name + ".gz")
                    if compressed.is_file():
                        path = compressed
                sidecar_paths.append(path)
            if any(not path.is_file() for path in sidecar_paths):
                continue
            evaluators = (
                [
                    Evaluator.load(
                        gzip.decompress(path.read_bytes())
                        if path.suffix == ".gz"
                        else path.read_bytes()
                    )
                    for path in sidecar_paths
                ]
                if load_evaluators
                else []
            )
            input_names = [str(name) for name in data["input_names"]]
            # Chain-rule formula evaluation goes through the serialized
            # evaluator; the symbolic expression list is reference-only and is
            # intentionally omitted from current caches.  Avoid parsing legacy
            # expression strings as that can be as expensive as rebuilding the
            # formula.
            input_symbols = []
            output_expressions = []
            return ChainRuleFormulaDefinition(
                signature=tuple(_chain_rule_tupled(data["signature"])),
                input_names=input_names,
                input_symbols=input_symbols,
                output_expressions=output_expressions,
                evaluators=evaluators,
                output_shape=[tuple(item) for item in _chain_rule_tupled(data["output_shape"])],
                derivative_indices=[
                    tuple(item) for item in _chain_rule_tupled(data["derivative_indices"])
                ],
                h_layout=[
                    (int(item[0]), tuple(item[1]))
                    for item in _chain_rule_tupled(data["h_layout"])
                ],
                evaluator_mode=str(data.get("evaluator_mode", "separate")),
                build_seconds=float(data.get("build_seconds", 0.0)),
                cache_json_path=str(candidate),
                cache_evaluator_files=[str(path) for path in sidecar_paths],
            )
        except Exception:
            continue
    return None


def _subset_mask(subset: tuple[int, ...]) -> int:
    """Encode a singular-position subset as a compact integer mask."""
    mask = 0
    for position in subset:
        mask |= 1 << int(position)
    return mask


def _multi_suffix(multi_index: tuple[int, ...]) -> str:
    """Encode a multi-index in a Symbolica-safe symbol name."""
    return "_".join(str(int(value)) for value in multi_index) if multi_index else "none"


def build_subtraction_formula(
    topology: TopologyDefinition,
    sector: SectorDefinition,
    signature: tuple[Any, ...],
) -> SubtractionFormulaDefinition:
    """Build a subtraction formula through the Symbolica-owned generator."""
    return build_subtraction_formula_symbolica(
        topology,
        sector,
        signature,
        SubtractionFormulaDefinition,
    )


def build_endpoint_projector_formula(
    topology: TopologyDefinition,
    sector: SectorDefinition,
    signature: tuple[Any, ...],
) -> EndpointProjectorFormulaDefinition:
    """Build a lower-signature endpoint projector through Symbolica."""
    if bool(signature[2]):
        return build_ibp_endpoint_projector_formula(topology, sector, signature)
    return build_endpoint_projector_formula_symbolica(
        topology,
        sector,
        signature,
        EndpointProjectorFormulaDefinition,
        ibp_reduce_to_log_endpoint=False,
    )


def build_regular_taylor_formula(
    topology: TopologyDefinition,
    sector: SectorDefinition,
    signature: tuple[Any, ...],
) -> RegularTaylorFormulaDefinition:
    """Build the regular ``g_s`` Taylor-combination formula."""
    return build_regular_taylor_formula_symbolica(
        topology,
        sector,
        signature,
        RegularTaylorFormulaDefinition,
    )


def build_ibp_endpoint_projector_formula(
    topology: TopologyDefinition,
    sector: SectorDefinition,
    signature: tuple[Any, ...],
) -> EndpointProjectorFormulaDefinition:
    """Build a compound IBP projector from cached logarithmic projectors."""
    endpoint_powers = [(int(base), float(coeff)) for base, coeff in signature[4]]
    terms = _ibp_endpoint_projector_terms(endpoint_powers, topology.laurent_orders)
    child_formulas: dict[tuple[Any, ...], EndpointProjectorFormulaDefinition] = {}
    for term in terms:
        if term.child_signature not in child_formulas:
            child = topology._endpoint_projector_formulas.get(term.child_signature)
            if child is None:
                child = build_endpoint_projector_formula_symbolica(
                    topology,
                    None,
                    term.child_signature,
                    EndpointProjectorFormulaDefinition,
                    ibp_reduce_to_log_endpoint=False,
                )
                topology._endpoint_projector_formulas[term.child_signature] = child
            child_formulas[term.child_signature] = child
    return EndpointProjectorFormulaDefinition(
        signature=signature,
        input_names=[],
        input_symbols=[],
        output_expressions=[],
        evaluators=[],
        laurent_orders=topology.laurent_orders,
        zero_subsets=[],
        taylor_orders=list(signature[5]),
        coefficient_layout=[],
        ibp_reduce_to_log_endpoint=True,
        ibp_terms=terms,
        child_formulas=child_formulas,
    )


def _ibp_shared_max_orders_for_formula(
    sector: SectorDefinition,
    formula: EndpointProjectorFormulaDefinition,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]]:
    """Return one Taylor envelope per IBP boundary/zero projector."""
    envelopes: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    n_axes = len(sector.singular_axes)
    for term in formula.ibp_terms:
        child = formula.child_formulas[term.child_signature]
        active_positions = tuple(int(position) for position in term.active_positions)
        for _child_boundary, child_zero, child_multi, _regular_order in child.coefficient_layout:
            original_zero = tuple(active_positions[position] for position in child_zero)
            original_multi = list(term.derivative_multi)
            for child_position, value in enumerate(child_multi):
                original_multi[active_positions[child_position]] += int(value)
            key = (tuple(term.boundary_positions), tuple(sorted(original_zero)))
            current = envelopes.setdefault(key, [0 for _ in range(n_axes)])
            for position, value in enumerate(original_multi):
                current[position] = max(current[position], int(value))
    return {key: tuple(values) for key, values in envelopes.items()}


def _ibp_shared_output_pairs_for_formula(
    sector: SectorDefinition,
    formula: EndpointProjectorFormulaDefinition,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[tuple[int, ...], int], ...]]:
    """Return sparse regular coefficients consumed by each IBP envelope."""
    pairs: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        set[tuple[tuple[int, ...], int]],
    ] = {}
    for term in formula.ibp_terms:
        child = formula.child_formulas[term.child_signature]
        active_positions = tuple(int(position) for position in term.active_positions)
        for _child_boundary, child_zero, child_multi, regular_order in child.coefficient_layout:
            original_zero = tuple(active_positions[position] for position in child_zero)
            original_multi = list(term.derivative_multi)
            for child_position, value in enumerate(child_multi):
                original_multi[active_positions[child_position]] += int(value)
            key = (tuple(term.boundary_positions), tuple(sorted(original_zero)))
            pairs.setdefault(key, set()).add((tuple(original_multi), int(regular_order)))
    return {
        key: tuple(
            sorted(values, key=lambda item: (item[1], sum(item[0]), item[0]))
        )
        for key, values in pairs.items()
    }


def _regular_formula_dual_shape(
    formula: RegularTaylorFormulaDefinition,
) -> list[tuple[int, ...]]:
    """Return the minimal dual shape needed by a regular Taylor formula."""
    if formula.dual_shape:
        return list(formula.dual_shape)
    rank = len(formula.max_orders)
    shape: set[tuple[int, ...]] = set()
    for _kind, multi_index in formula.input_layout:
        multi = tuple(int(value) for value in multi_index)
        if len(multi) != rank:
            raise ValueError(f"regular Taylor multi-index rank mismatch: {multi!r}")
        # Symbolica dual evaluators require ancestor-closed shapes.  If a
        # coefficient y^(2,1) is requested, every component-wise lower
        # coefficient must also be present in the shape passed to dualize.
        for ancestor in product(*[range(value + 1) for value in multi]):
            shape.add(tuple(int(value) for value in ancestor))
    zero = tuple(0 for _ in range(rank))
    shape.add(zero)
    ordered = sorted(shape, key=lambda item: (sum(item), item))
    if zero in ordered:
        ordered.remove(zero)
        ordered.insert(0, zero)
    return ordered


def _regular_taylor_signature_volume(signature: tuple[Any, ...]) -> int:
    """Return the Taylor-box volume implied by a regular formula signature."""
    if len(signature) < 4 or signature[0] != "regular-taylor":
        return 1
    version = int(signature[1])
    if version <= 1:
        if len(signature) < 13:
            return 1
        orders = signature[12]
    elif version >= 3:
        return max(len(signature[3]), 1)
    else:
        orders = signature[3]
    volume = 1
    for order in orders:
        volume *= int(order) + 1
    return int(volume)


def _regular_taylor_signature_axis_count(signature: tuple[Any, ...]) -> int:
    """Return the number of singular Taylor axes in a regular signature."""
    if len(signature) < 3 or signature[0] != "regular-taylor":
        return 0
    version = int(signature[1])
    if version <= 1 and len(signature) > 3:
        return len(signature[3])
    return int(signature[2])


def _regular_taylor_source_shape(
    sector: SectorDefinition,
    zero_positions: tuple[int, ...] | set[int],
    max_orders: tuple[int, ...] | list[int],
) -> list[tuple[int, ...]]:
    """Return the raw J/U/F Taylor shape needed to feed a residual formula.

    The reusable regular formula works with residual coefficients of
    ``U/M_U`` and ``F/M_F``.  Obtaining those coefficients from black-box U/F
    evaluators still requires shifted raw polynomial Taylor coefficients when
    a zeroed singular coordinate carries an extracted monomial.  This source
    shape is therefore sector-specific even though the downstream formula is
    not.
    """
    orders = [int(order) for order in max_orders]
    zero_set = {int(position) for position in zero_positions}
    axes = list(sector.singular_axes)
    axis_position = {axis: position for position, axis in enumerate(axes)}
    shape: set[tuple[int, ...]] = set()
    for residual_multi in _multi_indices(orders):
        shape.add(tuple(residual_multi))
        for monomial_powers in (sector.u_monomial_powers, sector.f_monomial_powers):
            polynomial_multi = list(residual_multi)
            for axis, power in enumerate(monomial_powers):
                position = axis_position.get(axis)
                if position is not None and position in zero_set:
                    polynomial_multi[position] += int(power)
            shape.add(tuple(polynomial_multi))
    closed: set[tuple[int, ...]] = set()
    for multi in shape:
        for ancestor in product(*[range(int(value) + 1) for value in multi]):
            closed.add(tuple(int(value) for value in ancestor))
    zero = tuple(0 for _ in orders)
    closed.add(zero)
    ordered = sorted(closed, key=lambda item: (sum(item), item))
    if zero in ordered:
        ordered.remove(zero)
        ordered.insert(0, zero)
    return ordered


def _ancestor_closed_multi_set(
    multi_indices: list[tuple[int, ...]] | set[tuple[int, ...]],
    rank: int,
) -> set[tuple[int, ...]]:
    """Return the component-wise ancestor closure of sparse Taylor outputs."""
    closed: set[tuple[int, ...]] = {tuple(0 for _ in range(rank))}
    for multi in multi_indices:
        multi_tuple = tuple(int(value) for value in multi)
        if len(multi_tuple) != rank:
            raise ValueError(f"regular Taylor output rank mismatch: {multi_tuple!r}")
        closed.update(_multi_index_ancestors(multi_tuple))
    return closed


def _multi_set_cache_key(
    multi_indices: set[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    """Return a canonical key for a sparse Taylor support set.

    High-axis endpoint sectors repeatedly ask for U, F, and Jacobian source
    shapes with the same ancestor-closed residual support.  Sorting hundreds of
    multi-indices every time was visible in the PSD649 profile.  A frozenset is
    still O(N) to build, but the expensive stable sort is then shared across
    equivalent requests.
    """
    return _cached_multi_set_cache_key(
        frozenset(tuple(int(value) for value in multi) for multi in multi_indices)
    )


@lru_cache(maxsize=8192)
def _cached_multi_set_cache_key(
    frozen_multi_indices: frozenset[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    """Sort one immutable sparse support set once."""
    return tuple(sorted(frozen_multi_indices, key=lambda item: (sum(item), item)))


@lru_cache(maxsize=131072)
def _multi_index_ancestors(multi_index: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return all component-wise ancestors of one Taylor multi-index.

    Hard high-axis sectors ask for many sparse Taylor source shapes that share
    the same few multi-indices.  Caching the one-index ancestor closure avoids
    rebuilding the Cartesian product in every source-shape request while
    keeping the cache key independent of topology and sector names.
    """
    return tuple(
        tuple(int(value) for value in ancestor)
        for ancestor in product(*[range(int(value) + 1) for value in multi_index])
    )


def _regular_taylor_source_shape_from_multis(
    sector: SectorDefinition,
    zero_positions: tuple[int, ...] | set[int],
    residual_multis: set[tuple[int, ...]],
    jacobian_multis: set[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Return source Taylor shape for a sparse regular-formula input layout."""
    zero_set = {int(position) for position in zero_positions}
    axes = list(sector.singular_axes)
    axis_position = {axis: position for position, axis in enumerate(axes)}
    rank = len(axes)
    shape: set[tuple[int, ...]] = {tuple(0 for _ in range(rank))}
    for multi in jacobian_multis:
        shape.add(tuple(int(value) for value in multi))
    for residual_multi in residual_multis:
        residual_tuple = tuple(int(value) for value in residual_multi)
        shape.add(residual_tuple)
        for monomial_powers in (sector.u_monomial_powers, sector.f_monomial_powers):
            polynomial_multi = list(residual_tuple)
            for axis, power in enumerate(monomial_powers):
                position = axis_position.get(axis)
                if position is not None and position in zero_set:
                    polynomial_multi[position] += int(power)
            shape.add(tuple(polynomial_multi))
    closed: set[tuple[int, ...]] = set()
    for multi in shape:
        closed.update(_multi_index_ancestors(tuple(int(value) for value in multi)))
    zero = tuple(0 for _ in range(rank))
    closed.add(zero)
    ordered = sorted(closed, key=lambda item: (sum(item), item))
    if zero in ordered:
        ordered.remove(zero)
        ordered.insert(0, zero)
    return ordered


def _regular_taylor_source_shape_for_monomial_powers(
    sector: SectorDefinition,
    zero_positions: tuple[int, ...] | set[int],
    residual_multis: set[tuple[int, ...]],
    monomial_powers: list[int],
) -> list[tuple[int, ...]]:
    """Return source Taylor shape needed by one polynomial residual only."""
    zero_set = {int(position) for position in zero_positions}
    axes = list(sector.singular_axes)
    axis_position = {axis: position for position, axis in enumerate(axes)}
    rank = len(axes)
    shape: set[tuple[int, ...]] = {tuple(0 for _ in range(rank))}
    for residual_multi in residual_multis:
        residual_tuple = tuple(int(value) for value in residual_multi)
        shape.add(residual_tuple)
        polynomial_multi = list(residual_tuple)
        for axis, power in enumerate(monomial_powers):
            position = axis_position.get(axis)
            if position is not None and position in zero_set:
                polynomial_multi[position] += int(power)
        shape.add(tuple(polynomial_multi))
    closed: set[tuple[int, ...]] = set()
    for multi in shape:
        closed.update(_multi_index_ancestors(tuple(int(value) for value in multi)))
    return _ordered_multi_shape(closed, rank)


def _ordered_multi_shape(
    multi_indices: set[tuple[int, ...]],
    rank: int,
) -> list[tuple[int, ...]]:
    """Return a stable Taylor-shape ordering with the zero coefficient first."""
    zero = tuple(0 for _ in range(rank))
    ordered = sorted(
        {tuple(int(value) for value in multi) for multi in multi_indices},
        key=lambda item: (sum(item), item),
    )
    if zero in ordered:
        ordered.remove(zero)
    ordered.insert(0, zero)
    return ordered


def _merge_multi_shapes(
    *shapes: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Merge Taylor coefficient shape lists while preserving stable ordering."""
    merged: set[tuple[int, ...]] = set()
    rank = 0
    for shape in shapes:
        for multi in shape:
            multi_tuple = tuple(int(value) for value in multi)
            rank = len(multi_tuple)
            merged.add(multi_tuple)
    if not merged:
        return []
    return _ordered_multi_shape(merged, rank)


def _chain_rule_envelope_shape(
    shape: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
) -> list[tuple[int, ...]]:
    """Return the dense Taylor box in the original sector-variable order.

    The symbolic chain-rule evaluator itself is independent of the sparse set
    of coefficients a sector happens to request.  Keying it by the rectangular
    Taylor envelope collapses many sector-specific requests onto one reusable
    evaluator; callers slice the requested sparse coefficients from the dense
    output after evaluation.
    """
    normalized = [tuple(int(value) for value in multi) for multi in shape]
    if not normalized:
        return []
    rank = len(normalized[0])
    if any(len(multi) != rank for multi in normalized):
        raise ValueError(f"inconsistent chain-rule Taylor shape ranks: {shape!r}")
    max_orders = [
        max(int(multi[position]) for multi in normalized)
        for position in range(rank)
    ]
    return _ordered_multi_shape(set(_multi_indices(max_orders)), rank)


def _chain_rule_canonical_positions(
    shape: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Return sector-variable positions ordered by descending Taylor depth."""
    normalized = [tuple(int(value) for value in multi) for multi in shape]
    if not normalized:
        return ()
    rank = len(normalized[0])
    max_orders = [
        max(int(multi[position]) for multi in normalized)
        for position in range(rank)
    ]
    return tuple(
        sorted(
            range(rank),
            key=lambda position: (-int(max_orders[position]), int(position)),
        )
    )


def _chain_rule_original_to_canonical(
    multi_index: tuple[int, ...],
    canonical_positions: tuple[int, ...],
) -> tuple[int, ...]:
    """Map a sector-order Taylor multi-index to canonical formula order."""
    return tuple(int(multi_index[position]) for position in canonical_positions)


def _chain_rule_canonical_to_original(
    multi_index: tuple[int, ...],
    canonical_positions: tuple[int, ...],
) -> tuple[int, ...]:
    """Map a canonical formula multi-index back to sector-variable order."""
    out = [0 for _ in canonical_positions]
    for canonical_position, original_position in enumerate(canonical_positions):
        out[int(original_position)] = int(multi_index[canonical_position])
    return tuple(out)


def _chain_rule_canonical_envelope_shape(
    shape: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
) -> list[tuple[int, ...]]:
    """Return the dense Taylor box in canonical axis order."""
    normalized = [tuple(int(value) for value in multi) for multi in shape]
    if not normalized:
        return []
    positions = _chain_rule_canonical_positions(normalized)
    canonical_shape = {
        _chain_rule_original_to_canonical(multi, positions)
        for multi in _chain_rule_envelope_shape(normalized)
    }
    return _ordered_multi_shape(canonical_shape, len(positions))


def _regular_taylor_canonical_positions(max_orders: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return original positions ordered by descending Taylor depth."""
    return tuple(
        sorted(
            range(len(max_orders)),
            key=lambda position: (-int(max_orders[position]), int(position)),
        )
    )


def _regular_taylor_canonical_to_original(
    multi_index: tuple[int, ...],
    canonical_positions: tuple[int, ...],
) -> tuple[int, ...]:
    """Map a canonical regular-formula multi-index to sector-axis order."""
    out = [0 for _ in canonical_positions]
    for canonical_position, original_position in enumerate(canonical_positions):
        out[int(original_position)] = int(multi_index[canonical_position])
    return tuple(out)


def _regular_taylor_original_to_canonical(
    multi_index: tuple[int, ...],
    canonical_positions: tuple[int, ...],
) -> tuple[int, ...]:
    """Map a sector-axis multi-index to canonical regular-formula order."""
    return tuple(int(multi_index[position]) for position in canonical_positions)


def _sector_has_analytic_taylor_for_shape(sector: SectorDefinition) -> bool:
    """Return whether sector map/Jacobian Taylor jets avoid runtime evaluators."""
    map_monomials = getattr(sector, "_map_monomials", None)
    jacobian_monomial = getattr(sector, "_jacobian_monomial", None)
    return bool(map_monomials) and all(item is not None for item in map_monomials) and jacobian_monomial is not None


def _regular_series_product(
    left: list[complex],
    right: list[complex],
    count: int,
) -> list[complex]:
    """Multiply two regular epsilon series and truncate to ``count`` terms."""
    out = [0.0 + 0.0j for _ in range(count)]
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if i + j >= count:
                break
            out[i + j] += a * b
    return out


def _inverse_affine_regular_series(offset: int, eps_coeff: float, count: int) -> list[complex]:
    """Expand ``1/(offset+eps_coeff*eps)`` for nonzero integer offset."""
    if offset == 0:
        raise ValueError("IBP lowering expected only nonzero affine denominators")
    return [
        ((-float(eps_coeff) / float(offset)) ** order) / float(offset)
        for order in range(count)
    ]


def _ibp_endpoint_projector_terms(
    endpoint_powers: list[tuple[int, float]],
    laurent_orders: list[int],
) -> list[IBPEndpointProjectorTerm]:
    """Enumerate the IBP-lowered terms for one endpoint signature."""
    n_axes = len(endpoint_powers)
    count = len(laurent_orders)
    zero_multi = tuple(0 for _ in range(n_axes))
    raw_terms: list[
        tuple[
            list[complex],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = [([1.0 + 0.0j] + [0.0 + 0.0j for _ in range(count - 1)], (), zero_multi, tuple(range(n_axes)))]
    for position, (base, eps_coeff) in enumerate(endpoint_powers):
        required_order = int(-int(base) - 1)
        if required_order <= 0:
            continue
        next_terms: list[
            tuple[
                list[complex],
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
            ]
        ] = []
        for prefactor, boundary_subset, derivative_multi, active_positions in raw_terms:
            if position not in active_positions:
                next_terms.append((prefactor, boundary_subset, derivative_multi, active_positions))
                continue
            denominator_product = [1.0 + 0.0j] + [0.0 + 0.0j for _ in range(count - 1)]
            for shift in range(required_order):
                offset = int(base) + shift + 1
                denominator_product = _regular_series_product(
                    denominator_product,
                    _inverse_affine_regular_series(offset, eps_coeff, count),
                    count,
                )
                boundary_derivative = list(derivative_multi)
                boundary_derivative[position] += shift
                boundary_prefactor = _regular_series_product(prefactor, denominator_product, count)
                if shift % 2:
                    boundary_prefactor = [-value for value in boundary_prefactor]
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
            continuing_prefactor = _regular_series_product(prefactor, denominator_product, count)
            if required_order % 2:
                continuing_prefactor = [-value for value in continuing_prefactor]
            next_terms.append(
                (
                    continuing_prefactor,
                    boundary_subset,
                    tuple(continuing_derivative),
                    active_positions,
                )
            )
        raw_terms = next_terms

    out: list[IBPEndpointProjectorTerm] = []
    for prefactor, boundary_subset, derivative_multi, active_positions in raw_terms:
        derivative_factor = 1
        for value in derivative_multi:
            derivative_factor *= math.factorial(int(value))
        prefactor = [derivative_factor * value for value in prefactor]
        child_signature = (
            "endpoint-projector",
            2,
            False,
            len(active_positions),
            tuple((-1, endpoint_powers[position][1]) for position in active_positions),
            tuple(0 for _ in active_positions),
            tuple(laurent_orders),
        )
        out.append(
            IBPEndpointProjectorTerm(
                prefactor_coeffs=prefactor,
                boundary_positions=tuple(sorted(boundary_subset)),
                derivative_multi=tuple(int(value) for value in derivative_multi),
                active_positions=tuple(active_positions),
                child_signature=child_signature,
            )
        )
    return out


def build_subtraction_formula_legacy(
    topology: TopologyDefinition,
    sector: SectorDefinition,
    signature: tuple[Any, ...],
) -> SubtractionFormulaDefinition:
    """Build the full Symbolica localized-subtraction formula for one signature."""
    axes = list(sector.singular_axes)
    n_axes = len(axes)
    bases: list[int] = []
    eps_coeffs: list[float] = []
    taylor_orders: list[int] = []
    for axis in axes:
        endpoint_power = topology.endpoint_power(sector, axis)
        rounded_base = round(endpoint_power.base)
        if endpoint_power.base >= -1.0e-12:
            raise ValueError(
                f"{sector.name}: declared singular axis {sector.variable_names[axis]} "
                f"has non-singular endpoint power y^({endpoint_power.as_text()})"
            )
        if abs(endpoint_power.base - rounded_base) > 1.0e-12:
            raise ValueError(
                f"{sector.name}: unsupported non-integer endpoint power y^({endpoint_power.as_text()})"
            )
        if abs(endpoint_power.eps_coeff) <= 1.0e-15:
            raise ValueError(
                f"{sector.name}: endpoint power y^({endpoint_power.as_text()}) has no epsilon regulator"
            )
        required_order = int(-rounded_base - 1)
        declared_order = int(sector.endpoint_taylor_orders[axis])
        if declared_order < required_order:
            raise ValueError(
                f"{sector.name}: endpoint Taylor order {declared_order} on "
                f"{sector.variable_names[axis]} is too small; need {required_order}"
            )
        bases.append(int(rounded_base))
        eps_coeffs.append(float(endpoint_power.eps_coeff))
        taylor_orders.append(required_order)

    zero_subsets = [
        tuple(position for position, bit in enumerate(bits) if bit)
        for bits in product((False, True), repeat=n_axes)
    ]
    y_symbols = [S(f"sf_y{axis}") for axis in range(sector.integration_dim)]
    input_names = [f"sf_y{axis}" for axis in range(sector.integration_dim)]
    input_symbols = list(y_symbols)
    coeff_symbols: dict[tuple[str, tuple[int, ...], tuple[int, ...]], Any] = {}
    for subset in zero_subsets:
        mask = _subset_mask(subset)
        for kind in ("j", "u", "f"):
            for multi_index in sector.dual_shape:
                name = f"sf_{kind}_{mask}_{_multi_suffix(multi_index)}"
                coeff_symbols[(kind, subset, multi_index)] = S(name)
                input_names.append(name)
                input_symbols.append(S(name))

    axis_position = {axis: position for position, axis in enumerate(axes)}

    def coeff(kind: str, subset: tuple[int, ...], multi_index: tuple[int, ...]) -> Any:
        symbol = coeff_symbols.get((kind, subset, multi_index))
        if symbol is None:
            return E("0")
        return symbol

    def residual_series(
        kind: str,
        subset: tuple[int, ...],
        monomial_powers: list[int],
        max_orders: list[int],
    ) -> ExprSeries:
        series: ExprSeries = {}
        for residual_multi in _multi_indices(max_orders):
            polynomial_multi = [0 for _ in axes]
            denominator = E("1")
            for axis, power_value in enumerate(monomial_powers):
                position = axis_position.get(axis)
                power = int(power_value)
                if position is not None and position in subset:
                    polynomial_multi[position] = power + int(residual_multi[position])
                elif power:
                    denominator = denominator * _expr_int_power(y_symbols[axis], power)
            series[residual_multi] = coeff(kind, subset, tuple(polynomial_multi)) / denominator
        return series

    def jacobian_series(subset: tuple[int, ...], max_orders: list[int]) -> ExprSeries:
        return {multi: coeff("j", subset, multi) for multi in _multi_indices(max_orders)}

    def regular_monomial_exprs() -> tuple[Any, Any]:
        singular = set(sector.singular_axes)
        base_value = E("1")
        eps_log = E("0")
        for axis in range(sector.integration_dim):
            if axis in singular:
                continue
            endpoint_power = topology.endpoint_power(sector, axis)
            coord = y_symbols[axis]
            if abs(endpoint_power.base) > 1.0e-15:
                base_value = base_value * _expr_int_power(
                    coord,
                    _integer_coordinate_power(
                        endpoint_power.base,
                        f"{sector.name}:{sector.variable_names[axis]} regular monomial",
                    ),
                )
            if abs(endpoint_power.eps_coeff) > 1.0e-15:
                eps_log = eps_log + _expr_number(endpoint_power.eps_coeff) * coord.log()
        return base_value, eps_log

    monomial_pref, monomial_log = regular_monomial_exprs()

    def g_for(subset: tuple[int, ...]) -> list[ExprSeries]:
        max_orders = [
            int(taylor_orders[position]) if position in subset else 0
            for position in range(n_axes)
        ]
        j_series = jacobian_series(subset, max_orders)
        u_series = residual_series("u", subset, sector.u_monomial_powers, max_orders)
        f_series = residual_series("f", subset, sector.f_monomial_powers, max_orders)
        pref_series = _expr_series_mul(
            j_series,
            _expr_series_mul(
                _expr_series_pow_real(u_series, topology.u_power_base, max_orders),
                _expr_series_pow_real(f_series, -topology.f_power_base, max_orders),
                max_orders,
            ),
            max_orders,
        )
        pref_series = _expr_series_mul(
            _expr_series_constant(monomial_pref, max_orders),
            pref_series,
            max_orders,
        )
        log_series = _expr_series_add(
            _expr_series_constant(monomial_log, max_orders),
            _expr_series_add(
                _expr_series_scale(_expr_series_log(u_series, max_orders), topology.eps_log_u_coeff),
                _expr_series_scale(_expr_series_log(f_series, max_orders), topology.eps_log_f_coeff),
            ),
        )
        out: list[ExprSeries] = []
        log_power = _expr_series_constant(E("1"), max_orders)
        factorial = 1.0
        for order in range(topology.coefficient_count):
            if order > 0:
                factorial *= float(order)
                log_power = _expr_series_mul(log_power, log_series, max_orders)
            out.append(
                _expr_series_scale(
                    _expr_series_mul(pref_series, log_power, max_orders),
                    1.0 / factorial,
                )
            )
        return out

    g_cache: dict[tuple[int, ...], list[ExprSeries]] = {}

    def cached_g(subset_set: set[int]) -> list[ExprSeries]:
        subset = tuple(sorted(subset_set))
        if subset not in g_cache:
            g_cache[subset] = g_for(subset)
        return g_cache[subset]

    outputs = [E("0") for _ in range(topology.coefficient_count)]
    min_order = topology.laurent_min_order
    max_order = topology.laurent_max_order
    regular_count = topology.coefficient_count
    position_range = list(range(n_axes))
    processor = SectorProcessor(topology)
    for integrated_flags in product((False, True), repeat=n_axes):
        integrated_positions = [pos for pos, flag in enumerate(integrated_flags) if flag]
        active_positions = [pos for pos, flag in enumerate(integrated_flags) if not flag]
        active_factor = E("1")
        active_log_sum = E("0")
        for position in active_positions:
            coord = y_symbols[axes[position]]
            active_factor = active_factor * _expr_int_power(coord, bases[position])
            active_log_sum = active_log_sum + _expr_number(eps_coeffs[position]) * coord.log()

        for taylor_flags in product((False, True), repeat=len(active_positions)):
            projected_positions = [
                position
                for position, flag in zip(active_positions, taylor_flags)
                if flag
            ]
            sign = -1.0 if len(projected_positions) % 2 else 1.0
            zero_positions = set(integrated_positions) | set(projected_positions)
            g_series_by_eps = cached_g(zero_positions)
            max_multi_orders = [
                taylor_orders[position] if position in zero_positions else 0
                for position in position_range
            ]
            for multi_index in _multi_indices(max_multi_orders):
                sample_factor = _expr_number(sign) * active_factor
                for position in projected_positions:
                    order = multi_index[position]
                    if order:
                        sample_factor = sample_factor * _expr_int_power(
                            y_symbols[axes[position]], int(order)
                        )
                denominator_series = processor._integrated_denominator_series(
                    bases,
                    eps_coeffs,
                    integrated_positions,
                    multi_index,
                )
                if not denominator_series:
                    continue
                active_log_power = E("1")
                active_log_factorial = 1.0
                for log_order in range(regular_count):
                    if log_order > 0:
                        active_log_factorial *= float(log_order)
                        active_log_power = active_log_power * active_log_sum
                    active_log_coeff = active_log_power / _expr_number(active_log_factorial)
                    for regular_order in range(regular_count):
                        regular_coeff = _expr_series_coefficient(
                            g_series_by_eps[regular_order],
                            multi_index,
                        )
                        for denom_order, denom_coeff in denominator_series.items():
                            eps_order = regular_order + log_order + denom_order
                            if eps_order < min_order or eps_order > max_order:
                                continue
                            outputs[eps_order - min_order] = outputs[eps_order - min_order] + (
                                sample_factor
                                * active_log_coeff
                                * _expr_number(denom_coeff)
                                * regular_coeff
                            )

    evaluators = [
        expr.evaluator(input_symbols, jit_compile=topology.jit_compile_evaluators)
        for expr in outputs
    ]
    return SubtractionFormulaDefinition(
        signature=signature,
        input_names=input_names,
        input_symbols=input_symbols,
        output_expressions=outputs,
        evaluators=evaluators,
        laurent_orders=topology.laurent_orders,
        zero_subsets=zero_subsets,
        dual_shape=list(sector.dual_shape),
    )


class SectorProcessor:
    """Generic sector application layer.

    This class deliberately knows nothing about triangle or box topology.  All
    topology-specific information is carried by TopologyDefinition and
    SectorDefinition.  The U/F polynomials are accessed only through evaluators.
    """

    def __init__(
        self,
        topology: TopologyDefinition,
        boundary_tol: float = 1.0e-14,
        stability_threshold: float = 1.0e-8,
        high_precision_stability_threshold: float = 1.0e-12,
        stability_precision: int = 32,
        high_precision_stability_precision: int = 1000,
        subtraction_backend: str = "formula",
    ) -> None:
        """Store topology evaluators and endpoint precision controls."""
        self.topology = topology
        self.boundary_tol = boundary_tol
        self.stability_threshold = float(stability_threshold)
        self.high_precision_stability_threshold = float(high_precision_stability_threshold)
        self.stability_precision = int(stability_precision)
        self.high_precision_stability_precision = int(high_precision_stability_precision)
        if subtraction_backend not in {"formula", "recursive", "projector-formula"}:
            raise ValueError(f"unsupported subtraction backend {subtraction_backend!r}")
        self.subtraction_backend = subtraction_backend
        self._endpoint_projector_plan_cache: dict[
            tuple[str, tuple[Any, ...]],
            tuple[
                list[
                    tuple[
                        tuple[tuple[int, ...], tuple[int, ...]],
                        tuple[int, ...],
                        tuple[tuple[tuple[int, ...], int], ...],
                    ]
                ],
                dict[
                    tuple[int, ...],
                    list[
                        tuple[
                            tuple[tuple[int, ...], tuple[int, ...]],
                            tuple[int, ...],
                            tuple[tuple[tuple[int, ...], int], ...],
                        ]
                    ],
                ],
            ],
        ] = {}

    def evaluate(self, sector: SectorDefinition, y: list[float] | tuple[float, ...]) -> tuple[list[complex], float]:
        """Evaluate one sector point through the batched implementation."""
        coords = np.asarray([y], dtype=float)
        coeffs, training, _ = self.evaluate_batch(sector, coords)
        return [complex(value) for value in coeffs[0]], float(training[0])

    @staticmethod
    def _charge_unprofiled_python(timing: HotPathTiming, start_time: float) -> None:
        """Charge batch wall time not already attributed to evaluator calls.

        The symbolic-derivative path composes Taylor jets through Python/NumPy
        chain-rule algebra.  That work is not visible to the narrow evaluator
        timers around Symbolica calls, so account for it at the sector-batch
        boundary.  Any helper that already reports Python time is respected by
        subtracting the existing profile before adding the residual.
        """
        elapsed = time.perf_counter() - start_time
        accounted = timing.eval_seconds + timing.python_seconds + timing.havana_seconds
        timing.add_python(max(elapsed - accounted, 0.0))

    def evaluate_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, HotPathTiming]:
        """Evaluate Laurent coefficients and training values for one sector batch."""
        rows = np.asarray(y_values, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != sector.integration_dim:
            raise ValueError(f"{sector.name}: expected coordinate array with shape (n,{sector.integration_dim})")

        # A sector with P endpoint axes can only start at eps^-P in the raw
        # sector convention.  When the user asks only for deeper poles, avoid
        # building regular Taylor data that is guaranteed to cancel to zero.
        # This matters for multi-loop DOT runs where leading-pole probes sample
        # many shallow sectors through the discrete Havana dimension.
        endpoint_pole_depth = sum(
            1
            for axis in sector.singular_axes
            if self.topology.endpoint_power(sector, axis).base < -1.0e-12
        )
        if self.topology.laurent_max_order < -endpoint_pole_depth:
            timing = HotPathTiming()
            timing.add_precision_samples(ordinary=rows.shape[0])
            zeros = np.zeros((rows.shape[0], self.topology.coefficient_count), dtype=np.complex128)
            return zeros, np.zeros(rows.shape[0], dtype=float), timing

        if (
            not sector.singular_axes
            or self.stability_threshold <= 0.0
            or rows.shape[0] == 0
        ):
            timing = HotPathTiming()
            timing.add_precision_samples(ordinary=rows.shape[0])
            start_time = time.perf_counter()
            coeffs, training = self._evaluate_batch_impl(sector, rows, timing)
            self._charge_unprofiled_python(timing, start_time)
            return coeffs, training, timing

        singular_coords = rows[:, sector.singular_axes]
        min_endpoint_distance = np.min(singular_coords, axis=1)
        high_mask = min_endpoint_distance <= self.high_precision_stability_threshold
        stable_mask = (
            (min_endpoint_distance <= self.stability_threshold)
            & ~high_mask
        )
        ordinary_mask = ~(high_mask | stable_mask)
        if np.all(ordinary_mask):
            timing = HotPathTiming()
            timing.add_precision_samples(ordinary=rows.shape[0])
            start_time = time.perf_counter()
            coeffs, training = self._evaluate_batch_impl(sector, rows, timing)
            self._charge_unprofiled_python(timing, start_time)
            return coeffs, training, timing

        coeffs = np.zeros((rows.shape[0], self.topology.coefficient_count), dtype=np.complex128)
        training = np.zeros(rows.shape[0], dtype=float)
        timing = HotPathTiming()
        for mask, precision in (
            (ordinary_mask, None),
            (stable_mask, self.stability_precision),
        ):
            if not np.any(mask):
                continue
            chunk_timing = HotPathTiming(precision_digits=precision)
            chunk_size = int(np.count_nonzero(mask))
            if precision is None:
                chunk_timing.add_precision_samples(ordinary=chunk_size)
            else:
                chunk_timing.add_precision_samples(stability=chunk_size)
            start_time = time.perf_counter()
            chunk_coeffs, chunk_training = self._evaluate_batch_impl(
                sector,
                rows[mask],
                chunk_timing,
            )
            self._charge_unprofiled_python(chunk_timing, start_time)
            timing.absorb(chunk_timing)
            coeffs[mask] = chunk_coeffs
            training[mask] = chunk_training
        if np.any(high_mask):
            chunk_timing = HotPathTiming(precision_digits=self.high_precision_stability_precision)
            chunk_size = int(np.count_nonzero(high_mask))
            chunk_timing.add_precision_samples(high=chunk_size)
            start_time = time.perf_counter()
            chunk_coeffs, chunk_training = self._evaluate_batch_impl(
                sector,
                rows[high_mask],
                chunk_timing,
            )
            self._charge_unprofiled_python(chunk_timing, start_time)
            timing.absorb(chunk_timing)
            coeffs[high_mask] = chunk_coeffs
            training[high_mask] = chunk_training
        return coeffs, training, timing

    def _evaluate_batch_impl(
        self,
        sector: SectorDefinition,
        rows: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one precision-homogeneous sector batch."""
        if len(sector.singular_axes) == 0:
            coeffs, training = self._finite_contribution_batch(sector, rows, timing)
        elif self.subtraction_backend == "recursive":
            coeffs, training = self._recursive_taylor_subtraction_batch(sector, rows, timing)
        elif self.subtraction_backend == "projector-formula":
            coeffs, training = self._endpoint_projector_subtraction_batch(sector, rows, timing)
        else:
            coeffs, training = self._formula_subtraction_batch(sector, rows, timing)
        return coeffs, training

    def _finite_contribution_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a sector with no singular endpoint axes.

        pySecDec can still factor positive powers of sector variables from the
        Jacobian, U, or F.  Those factors are regular, so they belong in the
        ordinary epsilon-expanded function g_s instead of the subtraction
        machinery.
        """
        regular_coeffs = self._g_coeffs_batch(sector, y_values, timing)
        coeffs = np.zeros((y_values.shape[0], self.topology.coefficient_count), dtype=np.complex128)
        for regular_order in range(regular_coeffs.shape[1]):
            eps_order = regular_order
            if self.topology.laurent_min_order <= eps_order <= self.topology.laurent_max_order:
                coeffs[:, eps_order - self.topology.laurent_min_order] += regular_coeffs[:, regular_order]
        return coeffs, complex_abs_for_training_array(coeffs[:, self.topology.training_index])

    def _phi_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        x_values: np.ndarray | None,
        f_values: np.ndarray | None,
        timing: HotPathTiming,
    ) -> np.ndarray:
        """Evaluate the regular residual ``phi = F(X(y))/M_F(y)``.

        Interior points use the scalar F evaluator and divide by the declared
        monomial.  Endpoint points use dual Taylor coefficients of the black-box
        F evaluator composed with sector-map dual jets.
        """
        return self._monomial_residual_batch(
            sector=sector,
            y_values=y_values,
            x_values=x_values,
            polynomial_values=f_values,
            monomial_powers=sector.f_monomial_powers,
            value_batch=self.topology.f_values,
            taylor_batch=self.topology.f_taylor_batch,
            timing=timing,
        )

    def _u_residual_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        x_values: np.ndarray | None,
        u_values: np.ndarray | None,
        timing: HotPathTiming,
    ) -> np.ndarray:
        """Evaluate the regular residual ``psi = U(X(y))/M_U(y)``."""
        return self._monomial_residual_batch(
            sector=sector,
            y_values=y_values,
            x_values=x_values,
            polynomial_values=u_values,
            monomial_powers=sector.u_monomial_powers,
            value_batch=self.topology.u_values,
            taylor_batch=self.topology.u_taylor_batch,
            timing=timing,
        )

    def _monomial_value_batch(self, y_values: np.ndarray, powers: list[int]) -> np.ndarray:
        """Evaluate a declared monomial for arbitrary power metadata."""
        rows = np.asarray(y_values, dtype=float)
        values = np.ones(rows.shape[0], dtype=float)
        for axis, power in enumerate(powers):
            if power:
                values *= rows[:, axis] ** power
        return values

    def _monomial_residual_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        x_values: np.ndarray | None,
        polynomial_values: np.ndarray | None,
        monomial_powers: list[int],
        value_batch: Any,
        taylor_batch: Any,
        timing: HotPathTiming,
    ) -> np.ndarray:
        """Evaluate ``P(X_s(y))/M_s(y)`` for P=U or F without opening P."""
        axes = sector.singular_axes
        axis_position = {axis: position for position, axis in enumerate(axes)}
        rows = np.asarray(y_values, dtype=float)
        residual = np.empty(rows.shape[0], dtype=np.complex128)
        if not any(monomial_powers):
            if polynomial_values is not None:
                return np.asarray(polynomial_values, dtype=np.complex128)
            if x_values is None:
                x_values = sector.map_eval_batch(rows, timing)
            return value_batch(x_values, timing)
        if not axes:
            if polynomial_values is None:
                if x_values is None:
                    x_values = sector.map_eval_batch(rows, timing)
                polynomial_values = value_batch(x_values, timing)
            return np.asarray(polynomial_values, dtype=np.complex128) / self._monomial_value_batch(
                rows, monomial_powers
            )

        axis_values = rows[:, axes]
        # Interior points implement the literal formula from the docs:
        #   residual_s(y) = P(X_s(y)) / M_s(y).
        # The caller often already evaluated X_s and F(X_s) while building g_s,
        # so we reuse those arrays to avoid extra evaluator calls.
        interior = np.all(axis_values > self.boundary_tol, axis=1)
        if np.any(interior):
            if polynomial_values is None:
                if x_values is None:
                    x_values = sector.map_eval_batch(rows[interior], timing)
                    values_interior = value_batch(x_values, timing)
                else:
                    values_interior = value_batch(x_values[interior], timing)
            else:
                values_interior = np.asarray(polynomial_values, dtype=np.complex128)[interior]
            residual[interior] = values_interior / self._monomial_value_batch(
                rows[interior], monomial_powers
            )

        boundary = ~interior
        if np.any(boundary):
            boundary_rows = rows[boundary]
            # Direct division by M_s would produce 0/0 at endpoints.  Instead,
            # request the Taylor coefficients of F(X_s(y)) from the dualized
            # black-box F evaluator.  The sector supplies only the known map
            # jets; F is still not opened or symbolically expanded.
            taylor = taylor_batch(sector, boundary_rows, timing)
            boundary_phi = np.empty(boundary_rows.shape[0], dtype=np.complex128)
            # boundary_flags marks which singular coordinates are actually at
            # the endpoint for each row.  For two singular axes this separates
            # edge limits, corner limits, and mixed cases in one vectorized pass.
            boundary_flags = boundary_rows[:, axes] <= self.boundary_tol
            for pattern in product((False, True), repeat=len(axes)):
                row_mask = np.all(boundary_flags == np.asarray(pattern, dtype=bool), axis=1)
                if not np.any(row_mask):
                    continue
                multi_index: list[int] = []
                denominator = np.ones(int(np.count_nonzero(row_mask)), dtype=float)
                for axis, power in enumerate(monomial_powers):
                    if power == 0:
                        continue
                    position = axis_position.get(axis)
                    is_boundary = position is not None and pattern[position]
                    power = monomial_powers[axis]
                    if is_boundary:
                        # If y_axis=0, dividing by y_axis^power is replaced by
                        # taking the matching Taylor coefficient of F(X_s).
                        pass
                    else:
                        # If y_axis is nonzero while another axis is at its
                        # endpoint, keep the ordinary quotient for this factor.
                        denominator *= boundary_rows[row_mask, axis] ** power
                for axis, is_boundary in zip(axes, pattern):
                    power = monomial_powers[axis]
                    multi_index.append(power if is_boundary else 0)
                # Symbolica dual coefficients are Taylor coefficients in the
                # declared multi-index basis, so this retrieves the finite
                # residual phi_s for the current boundary pattern.
                boundary_phi[row_mask] = (
                    taylor[row_mask, sector.dual_index(tuple(multi_index))] / denominator
                )
            residual[boundary] = boundary_phi

        return residual

    def _regular_monomial_base_log_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return regular monomial prefactor and epsilon-log for non-singular axes."""
        rows = np.asarray(y_values, dtype=float)
        singular = set(sector.singular_axes)
        base_value = np.ones(rows.shape[0], dtype=np.complex128)
        eps_log = np.zeros(rows.shape[0], dtype=float)
        for axis in range(sector.integration_dim):
            if axis in singular:
                continue
            endpoint_power = self.topology.endpoint_power(sector, axis)
            coord = rows[:, axis]
            if abs(endpoint_power.base) > 1.0e-15:
                base_value *= np.power(coord, endpoint_power.base)
            if abs(endpoint_power.eps_coeff) > 1.0e-15:
                eps_log += endpoint_power.eps_coeff * np.log(coord)
        return base_value, eps_log

    def _regular_monomial_base_log_prec(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        precision_digits: int,
    ) -> tuple[ComplexPrecise, ComplexPrecise]:
        """Return regular monomial pieces as Decimal complex values."""
        singular = set(sector.singular_axes)
        base_value = _pc_one()
        eps_log = _pc_zero()
        coords = [_decimal_complex(value, precision_digits) for value in np.asarray(y, dtype=float)]
        for axis in range(sector.integration_dim):
            if axis in singular:
                continue
            endpoint_power = self.topology.endpoint_power(sector, axis)
            coord = coords[axis]
            if abs(endpoint_power.base) > 1.0e-15:
                rounded = round(endpoint_power.base)
                if abs(endpoint_power.base - rounded) > 1.0e-12:
                    raise ValueError(
                        f"{sector.name}: high-precision regular monomial requires integer "
                        f"base power, got {endpoint_power.base!r}"
                    )
                base_value = _pc_mul(base_value, _pc_int_power(coord, int(rounded)))
            if abs(endpoint_power.eps_coeff) > 1.0e-15:
                eps_log = _pc_add(
                    eps_log,
                    _pc_scale(_pc_log(coord, precision_digits), endpoint_power.eps_coeff, precision_digits),
                )
        return base_value, eps_log

    def _g_coeffs_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> np.ndarray:
        """Build the epsilon-expanded regular function g_s(y)."""
        x = sector.map_eval_batch(y_values, timing)
        u = self.topology.u_values(x, timing)
        f = self.topology.f_values(x, timing)
        u_residual = self._u_residual_batch(sector, y_values, x, u, timing)
        phi = self._phi_batch(sector, y_values, x, f, timing)
        regular_j = sector.jacobian_eval_batch(y_values, timing).astype(np.complex128)
        monomial_pref, monomial_log = self._regular_monomial_base_log_batch(sector, y_values)
        # The monomial powers have already been extracted, so the integrable
        # endpoint structure lives outside g_s.  Here we build only the regular
        # coefficient multiplying the localized subtraction formula.
        pref = monomial_pref * regular_j * np.power(u_residual, self.topology.u_power_base) * np.power(
            phi, -self.topology.f_power_base
        )
        # Expand U^{a+b eps} phi^{c+d eps}.  If a sector has P logarithmic
        # endpoint axes, poles can shift regular eps^k terms down by eps^{-P};
        # therefore the regular series is needed through max_order-min_order.
        exponent_log = (
            monomial_log
            +
            self.topology.eps_log_u_coeff * feynman_log_array(u_residual)
            + self.topology.eps_log_f_coeff * feynman_log_array(phi)
        )
        regular_order_count = self.topology.coefficient_count
        coeffs = np.empty((y_values.shape[0], regular_order_count), dtype=np.complex128)
        coeffs[:, 0] = pref
        factorial = 1.0
        power = np.ones_like(exponent_log)
        for order in range(1, regular_order_count):
            factorial *= float(order)
            power = power * exponent_log
            coeffs[:, order] = pref * power / factorial
        return coeffs

    def _endpoint_power_data(
        self,
        sector: SectorDefinition,
    ) -> tuple[list[float], list[float], list[int]]:
        """Return base powers, epsilon slopes, and Taylor orders for endpoints."""
        bases: list[float] = []
        eps_coeffs: list[float] = []
        taylor_orders: list[int] = []
        for axis in sector.singular_axes:
            endpoint_power = self.topology.endpoint_power(sector, axis)
            rounded_base = round(endpoint_power.base)
            if endpoint_power.base >= -1.0e-12:
                raise ValueError(
                    f"{sector.name}: declared singular axis {sector.variable_names[axis]} "
                    f"has non-singular endpoint power y^({endpoint_power.as_text()})"
                )
            if abs(endpoint_power.base - rounded_base) > 1.0e-12:
                raise ValueError(
                    f"{sector.name}: unsupported non-integer endpoint power "
                    f"y^({endpoint_power.as_text()})"
                )
            if abs(endpoint_power.eps_coeff) <= 1.0e-15:
                raise ValueError(
                    f"{sector.name}: endpoint power y^({endpoint_power.as_text()}) "
                    "has no epsilon regulator"
                )
            required_order = int(-rounded_base - 1)
            declared_order = int(sector.endpoint_taylor_orders[axis])
            if declared_order < required_order:
                raise ValueError(
                    f"{sector.name}: endpoint Taylor order {declared_order} on "
                    f"{sector.variable_names[axis]} is too small; need {required_order}"
                )
            bases.append(float(rounded_base))
            eps_coeffs.append(float(endpoint_power.eps_coeff))
            taylor_orders.append(required_order)
        return bases, eps_coeffs, taylor_orders

    def _denominator_series(
        self,
        beta: float,
        order: int,
        eps_coeff: float,
    ) -> dict[int, complex]:
        """Expand ``1/(beta+order+1+eps_coeff*eps)`` as a Laurent series."""
        offset = beta + float(order) + 1.0
        max_terms = self.topology.coefficient_count + 2
        if abs(offset) <= 1.0e-14:
            return {-1: 1.0 / eps_coeff}
        return {
            eps_order: ((-eps_coeff / offset) ** eps_order) / offset
            for eps_order in range(max_terms)
        }

    def _multiply_laurent_series(
        self,
        left: dict[int, complex],
        right: dict[int, complex],
    ) -> dict[int, complex]:
        """Multiply scalar Laurent series and keep only potentially needed orders."""
        out: dict[int, complex] = {}
        min_order = self.topology.laurent_min_order
        max_order = self.topology.laurent_max_order
        for order_left, value_left in left.items():
            for order_right, value_right in right.items():
                order = order_left + order_right
                if order < min_order or order > max_order:
                    continue
                out[order] = out.get(order, 0.0 + 0.0j) + value_left * value_right
        return out

    def _integrated_denominator_series(
        self,
        bases: list[float],
        eps_coeffs: list[float],
        integrated_positions: list[int],
        multi_index: tuple[int, ...],
    ) -> dict[int, complex]:
        """Build the product of analytic endpoint-integral denominators."""
        out: dict[int, complex] = {0: 1.0 + 0.0j}
        # Positive orders from a finite denominator factor can later combine
        # with a pole from another axis, so do not truncate to eps^0 until all
        # factors have been multiplied.
        work_min = self.topology.laurent_min_order
        work_max = self.topology.coefficient_count
        for position in integrated_positions:
            factor = self._denominator_series(
                bases[position],
                multi_index[position],
                eps_coeffs[position],
            )
            product_series: dict[int, complex] = {}
            for order_left, value_left in out.items():
                for order_right, value_right in factor.items():
                    order = order_left + order_right
                    if order < work_min or order > work_max:
                        continue
                    product_series[order] = (
                        product_series.get(order, 0.0 + 0.0j)
                        + value_left * value_right
                    )
            out = product_series
            if not out:
                break
        return {
            order: value
            for order, value in out.items()
            if self.topology.laurent_min_order <= order <= self.topology.laurent_max_order
        }

    def _formula_endpoint_rows(
        self,
        sector: SectorDefinition,
        rows: np.ndarray,
        subset: tuple[int, ...],
    ) -> np.ndarray:
        """Return sampled sector coordinates with selected singular positions zeroed."""
        endpoint_rows = np.asarray(rows, dtype=float).copy()
        for position in subset:
            endpoint_rows[:, sector.singular_axes[position]] = 0.0
        return endpoint_rows

    def _select_active_laurent_columns(
        self,
        values: np.ndarray,
        prepared_orders: list[int],
        label: str,
    ) -> np.ndarray:
        """Project prepared formula outputs onto the requested Laurent range.

        Prepared bundles can be generated through a high order, for example
        ``eps^0``, and later integrated only through a lower order.  The
        serialized Symbolica evaluators still return the full prepared output
        vector, so the strict prepared path trims columns here instead of
        rebuilding a smaller evaluator.
        """
        active_orders = self.topology.laurent_orders
        if list(prepared_orders) == active_orders:
            return values
        index_by_order = {int(order): index for index, order in enumerate(prepared_orders)}
        try:
            columns = [index_by_order[int(order)] for order in active_orders]
        except KeyError as exc:
            raise RuntimeError(
                f"{label}: prepared formula does not contain Laurent order eps^{exc.args[0]}"
            ) from exc
        return values[:, columns]

    def _select_active_laurent_list(
        self,
        values: list[complex],
        prepared_orders: list[int],
        label: str,
    ) -> list[complex]:
        """List equivalent of ``_select_active_laurent_columns`` for prec rows."""
        active_orders = self.topology.laurent_orders
        if list(prepared_orders) == active_orders:
            return values
        index_by_order = {int(order): index for index, order in enumerate(prepared_orders)}
        try:
            return [values[index_by_order[int(order)]] for order in active_orders]
        except KeyError as exc:
            raise RuntimeError(
                f"{label}: prepared formula does not contain Laurent order eps^{exc.args[0]}"
            ) from exc

    def _subtraction_formula_input_matrix(
        self,
        sector: SectorDefinition,
        rows: np.ndarray,
        formula: SubtractionFormulaDefinition,
        timing: HotPathTiming,
    ) -> np.ndarray:
        """Assemble complex batch inputs for a pregenerated subtraction formula."""
        sample_rows = np.asarray(rows, dtype=float)
        input_matrix = np.zeros((sample_rows.shape[0], len(formula.input_names)), dtype=np.complex128)
        offset = 0
        input_matrix[:, offset : offset + sector.integration_dim] = sample_rows.astype(np.complex128)
        offset += sector.integration_dim
        for subset in formula.zero_subsets:
            endpoint_rows = self._formula_endpoint_rows(sector, sample_rows, subset)
            j_taylor = sector.jacobian_taylor_complex_batch(endpoint_rows, timing)
            u_taylor = self.topology.u_taylor_complex_batch(sector, endpoint_rows, timing)
            f_taylor = self.topology.f_taylor_complex_batch(sector, endpoint_rows, timing)
            for values in (j_taylor, u_taylor, f_taylor):
                width = len(sector.dual_shape)
                input_matrix[:, offset : offset + width] = values
                offset += width
        if offset != len(formula.input_names):
            raise RuntimeError(
                f"{sector.name}: subtraction formula input mismatch: filled {offset}, "
                f"expected {len(formula.input_names)}"
            )
        return input_matrix

    def _subtraction_formula_input_prec_row(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        formula: SubtractionFormulaDefinition,
        precision_digits: int,
        timing: HotPathTiming,
    ) -> ComplexPreciseRow:
        """Assemble one arbitrary-precision complex formula input row."""
        coords = np.asarray(y, dtype=float)
        input_row: ComplexPreciseRow = [
            _decimal_complex(value, precision_digits) for value in coords
        ]
        for subset in formula.zero_subsets:
            endpoint = coords.copy()
            for position in subset:
                endpoint[sector.singular_axes[position]] = 0.0
            input_row.extend(
                sector.jacobian_taylor_complex_prec(endpoint, precision_digits, timing)
            )
            input_row.extend(
                self.topology.u_taylor_complex_prec(sector, endpoint, precision_digits, timing)
            )
            input_row.extend(
                self.topology.f_taylor_complex_prec(sector, endpoint, precision_digits, timing)
            )
        if len(input_row) != len(formula.input_names):
            raise RuntimeError(
                f"{sector.name}: subtraction formula input mismatch: filled {len(input_row)}, "
                f"expected {len(formula.input_names)}"
            )
        return input_row

    def _formula_subtraction_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a singular sector through its pregenerated Symbolica formula."""
        formula = self.topology.subtraction_formula_for(sector)
        rows = np.asarray(y_values, dtype=float)
        precision_digits = timing.precision_digits
        if precision_digits is None:
            input_matrix = self._subtraction_formula_input_matrix(sector, rows, formula, timing)
            coeffs = self._select_active_laurent_columns(
                formula.evaluate_complex_batch(input_matrix, timing),
                formula.laurent_orders,
                sector.name,
            )
        else:
            coeffs = np.zeros((rows.shape[0], self.topology.coefficient_count), dtype=np.complex128)
            for row_index, row in enumerate(rows):
                input_row = self._subtraction_formula_input_prec_row(
                    sector,
                    row,
                    formula,
                    int(precision_digits),
                    timing,
                )
                coeffs[row_index, :] = self._select_active_laurent_list(
                    formula.evaluate_complex_prec(
                        input_row,
                        int(precision_digits),
                        timing,
                    ),
                    formula.laurent_orders,
                    sector.name,
                )
        return coeffs, complex_abs_for_training_array(coeffs[:, self.topology.training_index])

    def _endpoint_projector_input_matrix(
        self,
        sector: SectorDefinition,
        rows: np.ndarray,
        formula: EndpointProjectorFormulaDefinition,
        timing: HotPathTiming,
    ) -> np.ndarray:
        """Assemble inputs for the lower-signature endpoint projector.

        The first columns are the local singular coordinates.  All remaining
        columns are regular-function Taylor/Laurent coefficients
        ``g_{S,alpha,r}``, ordered exactly as declared by the formula layout.
        """
        sample_rows = np.asarray(rows, dtype=float)
        n_rows = sample_rows.shape[0]
        input_matrix = np.zeros((n_rows, len(formula.input_names)), dtype=np.complex128)
        n_axes = len(sector.singular_axes)
        offset = 0
        if n_axes:
            input_matrix[:, offset : offset + n_axes] = sample_rows[:, sector.singular_axes]
        offset += n_axes

        g_cache = self._precompute_endpoint_projector_g_cache(
            sector,
            sample_rows,
            formula,
            timing,
        )
        for boundary, zero, multi_index, regular_order in formula.coefficient_layout:
            cached = g_cache.get((boundary, zero))
            if cached is None:
                raise RuntimeError(f"{sector.name}: missing endpoint-projector coefficient cache")
            input_matrix[:, offset] = _series_coefficient(
                cached[regular_order],
                multi_index,
                n_rows,
            )
            offset += 1

        if offset != len(formula.input_names):
            raise RuntimeError(
                f"{sector.name}: endpoint projector input mismatch: filled {offset}, "
                f"expected {len(formula.input_names)}"
            )
        return input_matrix

    def _precompute_endpoint_projector_g_cache(
        self,
        sector: SectorDefinition,
        rows: np.ndarray,
        formula: EndpointProjectorFormulaDefinition,
        timing: HotPathTiming,
    ) -> dict[tuple[tuple[int, ...], tuple[int, ...]], list[MultiSeries]]:
        """Return all regular coefficients needed by a direct endpoint projector.

        A direct endpoint projector has one coefficient group for every
        boundary/zero projector in its inclusion-exclusion formula.  Building
        every group independently repeats the same sparse U/F/J source assembly
        for many boundary choices.  For groups that do not already have a
        pregenerated regular-Taylor evaluator, stack all boundary rows sharing a
        zero set and evaluate one union request.  This mirrors the IBP child
        cache while keeping curated regular formula groups on their specialized
        evaluator path.
        """
        sample_rows = np.asarray(rows, dtype=float)
        n_rows = int(sample_rows.shape[0])
        formula_groups, fallback_by_zero_template = self._endpoint_projector_plan(
            sector,
            formula,
        )

        g_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], list[MultiSeries]] = {}
        for group_key, max_orders, output_pairs in formula_groups:
            boundary, zero = group_key
            g_cache[group_key] = self._g_taylor_eps_series_batch(
                sector,
                sample_rows,
                set(zero),
                list(max_orders),
                timing,
                boundary_positions=set(boundary),
                max_orders_are_explicit=True,
                output_pairs=output_pairs,
            )

        max_stacked_rows = 4096
        for zero, entries in fallback_by_zero_template.items():
            keys_per_chunk = min(
                max(1, len(entries)),
                max(1, max_stacked_rows // max(1, n_rows)),
            )
            for start_index in range(0, len(entries), keys_per_chunk):
                chunk = entries[start_index : start_index + keys_per_chunk]
                envelope_orders = [
                    max(int(max_orders[position]) for _key, max_orders, _pairs in chunk)
                    for position in range(len(sector.singular_axes))
                ]
                output_pair_set: set[tuple[tuple[int, ...], int]] = set()
                stacked_rows: list[np.ndarray] = []
                slices: list[
                    tuple[
                        tuple[tuple[int, ...], tuple[int, ...]],
                        int,
                        int,
                    ]
                ] = []
                row_offset = 0
                for group_key, _max_orders, output_pairs in chunk:
                    boundary, _zero = group_key
                    output_pair_set.update(output_pairs)
                    endpoint_rows = sample_rows.copy()
                    for position in boundary:
                        endpoint_rows[:, sector.singular_axes[int(position)]] = 1.0
                    stacked_rows.append(endpoint_rows)
                    slices.append((group_key, row_offset, row_offset + n_rows))
                    row_offset += n_rows
                union_output_pairs = tuple(
                    sorted(
                        output_pair_set,
                        key=lambda item: (item[1], sum(item[0]), item[0]),
                    )
                )
                shared_series = self._g_taylor_eps_series_batch(
                    sector,
                    np.vstack(stacked_rows),
                    set(zero),
                    envelope_orders,
                    timing,
                    boundary_positions=set(),
                    max_orders_are_explicit=True,
                    output_pairs=union_output_pairs,
                )
                for group_key, row_start, row_stop in slices:
                    g_cache[group_key] = _slice_multi_series_list(
                        shared_series,
                        row_start,
                        row_stop,
                    )
        return g_cache

    def _endpoint_projector_plan(
        self,
        sector: SectorDefinition,
        formula: EndpointProjectorFormulaDefinition,
    ) -> tuple[
        list[
            tuple[
                tuple[tuple[int, ...], tuple[int, ...]],
                tuple[int, ...],
                tuple[tuple[tuple[int, ...], int], ...],
            ]
        ],
        dict[
            tuple[int, ...],
            list[
                tuple[
                    tuple[tuple[int, ...], tuple[int, ...]],
                    tuple[int, ...],
                    tuple[tuple[tuple[int, ...], int], ...],
                ]
            ],
        ],
    ]:
        """Return cached endpoint-projector grouping metadata.

        The grouping is independent of sample values.  Keeping it next to the
        processor avoids rebuilding sorted output-pair tuples and regular
        formula signatures for every batch of a hard sector.
        """
        cache_key = (sector.name, formula.signature)
        cached = self._endpoint_projector_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        n_axes = len(sector.singular_axes)
        groups: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]],
        ] = {}
        for key in formula.coefficient_layout:
            boundary, zero, _multi_index, _regular_order = key
            groups.setdefault((boundary, zero), []).append(key)

        formula_groups: list[
            tuple[
                tuple[tuple[int, ...], tuple[int, ...]],
                tuple[int, ...],
                tuple[tuple[tuple[int, ...], int], ...],
            ]
        ] = []
        fallback_by_zero: dict[
            tuple[int, ...],
            list[
                tuple[
                    tuple[tuple[int, ...], tuple[int, ...]],
                    tuple[int, ...],
                    tuple[tuple[tuple[int, ...], int], ...],
                ]
            ],
        ] = {}
        for group_key, entries in groups.items():
            boundary, zero = group_key
            max_orders = tuple(
                max(int(entry[2][position]) for entry in entries)
                for position in range(n_axes)
            )
            output_pairs = tuple(
                sorted(
                    ((tuple(entry[2]), int(entry[3])) for entry in entries),
                    key=lambda item: (item[1], sum(item[0]), item[0]),
                )
            )
            regular_signature = self.topology.regular_taylor_signature(
                sector,
                zero_positions=zero,
                max_orders=max_orders,
                output_pairs=output_pairs,
            )
            if regular_signature in self.topology._regular_taylor_formulas:
                formula_groups.append((group_key, max_orders, output_pairs))
            else:
                fallback_by_zero.setdefault(tuple(zero), []).append(
                    (group_key, max_orders, output_pairs)
                )

        for entries in fallback_by_zero.values():
            entries.sort(key=lambda item: (len(item[0][0]), item[0][0], item[1]))
        cached = (formula_groups, fallback_by_zero)
        self._endpoint_projector_plan_cache[cache_key] = cached
        return cached

    def _endpoint_projector_input_prec_row(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        formula: EndpointProjectorFormulaDefinition,
        precision_digits: int,
        timing: HotPathTiming,
    ) -> ComplexPreciseRow:
        """Assemble one fully multiprecision endpoint-projector input row.

        Near an endpoint the dangerous cancellation is not only in the final
        plus-distribution projector.  The regular Taylor coefficients
        ``g_{S,alpha,r}`` must also be obtained from high-precision sector-map,
        U, F, and Jacobian Taylor data.  This row therefore bypasses the double
        input matrix entirely and keeps Decimal arithmetic until the prepared
        Symbolica projector returns the final Laurent weight.
        """
        coords = np.asarray(y, dtype=float)
        input_row: ComplexPreciseRow = [
            _decimal_complex(coords[axis], precision_digits)
            for axis in sector.singular_axes
        ]

        with localcontext() as context:
            context.prec = int(precision_digits)
            groups: dict[
                tuple[tuple[int, ...], tuple[int, ...]],
                list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]],
            ] = {}
            for key in formula.coefficient_layout:
                boundary, zero, _multi_index, _regular_order = key
                groups.setdefault((boundary, zero), []).append(key)

            g_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], list[PrecSeries]] = {}
            for group_key, entries in groups.items():
                boundary, zero = group_key
                max_orders = [
                    max(int(entry[2][position]) for entry in entries)
                    for position in range(len(sector.singular_axes))
                ]
                g_cache[group_key] = self._g_taylor_eps_series_prec_row(
                    sector,
                    coords,
                    set(zero),
                    max_orders,
                    precision_digits,
                    timing,
                    boundary_positions=set(boundary),
                    max_orders_are_explicit=True,
                )

            for boundary, zero, multi_index, regular_order in formula.coefficient_layout:
                cached = g_cache.get((boundary, zero))
                if cached is None:
                    raise RuntimeError(f"{sector.name}: missing endpoint-projector coefficient cache")
                input_row.append(_prec_series_coefficient(cached[regular_order], multi_index))

        if len(input_row) != len(formula.input_names):
            raise RuntimeError(
                f"{sector.name}: endpoint projector input mismatch: filled {len(input_row)}, "
                f"expected {len(formula.input_names)}"
            )
        return input_row

    def _endpoint_projector_subtraction_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a singular sector through a reusable endpoint projector."""
        formula = self.topology.endpoint_projector_formula_for(sector)
        if formula.ibp_reduce_to_log_endpoint:
            return self._ibp_endpoint_projector_subtraction_batch(
                sector,
                y_values,
                formula,
                timing,
            )
        rows = np.asarray(y_values, dtype=float)
        precision_digits = timing.precision_digits
        if precision_digits is None:
            input_matrix = self._endpoint_projector_input_matrix(
                sector,
                rows,
                formula,
                timing,
            )
            coeffs = self._select_active_laurent_columns(
                formula.evaluate_complex_batch(input_matrix, timing),
                formula.laurent_orders,
                sector.name,
            )
        else:
            coeffs = np.zeros((rows.shape[0], self.topology.coefficient_count), dtype=np.complex128)
            for row_index, row in enumerate(rows):
                input_row = self._endpoint_projector_input_prec_row(
                    sector,
                    row,
                    formula,
                    int(precision_digits),
                    timing,
                )
                coeffs[row_index, :] = self._select_active_laurent_list(
                    formula.evaluate_complex_prec(
                        input_row,
                        int(precision_digits),
                        timing,
                    ),
                    formula.laurent_orders,
                    sector.name,
                )
        return coeffs, complex_abs_for_training_array(coeffs[:, self.topology.training_index])

    def _convolve_regular_prefactor_array(
        self,
        values: np.ndarray,
        prefactor_coeffs: list[complex],
    ) -> np.ndarray:
        """Multiply a Laurent array by a regular epsilon prefactor series."""
        out = np.zeros_like(values)
        count = values.shape[1]
        for value_index in range(count):
            for pref_index, prefactor in enumerate(prefactor_coeffs):
                out_index = value_index + pref_index
                if out_index >= count:
                    break
                out[:, out_index] += values[:, value_index] * prefactor
        return out

    def _convolve_regular_prefactor_list(
        self,
        values: list[complex],
        prefactor_coeffs: list[complex],
    ) -> list[complex]:
        """List analogue of ``_convolve_regular_prefactor_array``."""
        out = [0.0 + 0.0j for _ in values]
        count = len(values)
        for value_index, value in enumerate(values):
            for pref_index, prefactor in enumerate(prefactor_coeffs):
                out_index = value_index + pref_index
                if out_index >= count:
                    break
                out[out_index] += value * prefactor
        return out

    def _ibp_child_input_matrix(
        self,
        sector: SectorDefinition,
        rows: np.ndarray,
        child: EndpointProjectorFormulaDefinition,
        term: IBPEndpointProjectorTerm,
        timing: HotPathTiming,
        shared_g_cache: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], list[MultiSeries]] | None = None,
        shared_max_orders: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]] | None = None,
        shared_output_pairs: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[tuple[int, ...], int], ...],
        ] | None = None,
    ) -> np.ndarray:
        """Assemble inputs for one logarithmic child projector in IBP mode."""
        sample_rows = np.asarray(rows, dtype=float)
        n_rows = sample_rows.shape[0]
        input_matrix = np.zeros((n_rows, len(child.input_names)), dtype=np.complex128)
        active_positions = tuple(int(position) for position in term.active_positions)
        offset = 0
        if active_positions:
            active_axes = [sector.singular_axes[position] for position in active_positions]
            input_matrix[:, : len(active_positions)] = sample_rows[:, active_axes]
            offset = len(active_positions)

        groups: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int, tuple[int, ...]]],
        ] = {}
        expanded_entries: list[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int, tuple[int, ...]]
        ] = []
        for _child_boundary, child_zero, child_multi, regular_order in child.coefficient_layout:
            original_zero = tuple(active_positions[position] for position in child_zero)
            original_multi = list(term.derivative_multi)
            for child_position, value in enumerate(child_multi):
                original_multi[active_positions[child_position]] += int(value)
            original_multi_tuple = tuple(original_multi)
            entry = (
                tuple(term.boundary_positions),
                tuple(sorted(original_zero)),
                original_multi_tuple,
                int(regular_order),
                tuple(child_multi),
            )
            expanded_entries.append(entry)
            groups.setdefault((entry[0], entry[1]), []).append(entry)

        g_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], list[MultiSeries]] = {}
        for group_key, entries in groups.items():
            boundary, zero = group_key
            max_orders = [
                max(int(entry[2][position]) for entry in entries)
                for position in range(len(sector.singular_axes))
            ]
            if shared_max_orders is not None:
                # IBP decomposes one higher-power endpoint projector into many
                # logarithmic child projectors.  Those children often ask for
                # nested Taylor boxes for the same boundary/zero projector.  A
                # single larger box contains all smaller boxes, so build the
                # envelope once and reuse it across child terms.
                max_orders = list(shared_max_orders.get((boundary, zero), tuple(max_orders)))
            shared_key = (boundary, zero, tuple(max_orders))
            cached = shared_g_cache.get(shared_key) if shared_g_cache is not None else None
            if cached is None:
                if shared_output_pairs is not None:
                    output_pairs = shared_output_pairs.get((boundary, zero), ())
                else:
                    output_pairs = tuple(
                        sorted(
                            ((tuple(entry[2]), int(entry[3])) for entry in entries),
                            key=lambda item: (item[1], sum(item[0]), item[0]),
                        )
                    )
                cached = self._g_taylor_eps_series_batch(
                    sector,
                    sample_rows,
                    set(zero),
                    max_orders,
                    timing,
                    boundary_positions=set(boundary),
                    max_orders_are_explicit=True,
                    output_pairs=output_pairs,
                )
                if shared_g_cache is not None:
                    shared_g_cache[shared_key] = cached
            g_cache[group_key] = cached

        for boundary, zero, multi_index, regular_order, _child_multi in expanded_entries:
            cached = g_cache[(boundary, zero)]
            input_matrix[:, offset] = _series_coefficient(
                cached[regular_order],
                multi_index,
                n_rows,
            )
            offset += 1
        if offset != len(child.input_names):
            raise RuntimeError(
                f"{sector.name}: IBP child input mismatch: filled {offset}, "
                f"expected {len(child.input_names)}"
            )
        return input_matrix

    def _ibp_child_input_prec_row(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        child: EndpointProjectorFormulaDefinition,
        term: IBPEndpointProjectorTerm,
        precision_digits: int,
        timing: HotPathTiming,
        shared_g_cache: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], list[PrecSeries]] | None = None,
        shared_max_orders: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]] | None = None,
    ) -> ComplexPreciseRow:
        """Precision analogue of ``_ibp_child_input_matrix`` for one sample."""
        coords = np.asarray(y, dtype=float)
        active_positions = tuple(int(position) for position in term.active_positions)
        input_row: ComplexPreciseRow = [
            _decimal_complex(coords[sector.singular_axes[position]], precision_digits)
            for position in active_positions
        ]
        groups: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]],
        ] = {}
        expanded_entries: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]] = []
        for _child_boundary, child_zero, child_multi, regular_order in child.coefficient_layout:
            original_zero = tuple(active_positions[position] for position in child_zero)
            original_multi = list(term.derivative_multi)
            for child_position, value in enumerate(child_multi):
                original_multi[active_positions[child_position]] += int(value)
            entry = (
                tuple(term.boundary_positions),
                tuple(sorted(original_zero)),
                tuple(original_multi),
                int(regular_order),
            )
            expanded_entries.append(entry)
            groups.setdefault((entry[0], entry[1]), []).append(entry)

        with localcontext() as context:
            context.prec = int(precision_digits)
            g_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], list[PrecSeries]] = {}
            for group_key, entries in groups.items():
                boundary, zero = group_key
                max_orders = [
                    max(int(entry[2][position]) for entry in entries)
                    for position in range(len(sector.singular_axes))
                ]
                if shared_max_orders is not None:
                    max_orders = list(shared_max_orders.get((boundary, zero), tuple(max_orders)))
                shared_key = (boundary, zero, tuple(max_orders))
                cached = shared_g_cache.get(shared_key) if shared_g_cache is not None else None
                if cached is None:
                    cached = self._g_taylor_eps_series_prec_row(
                        sector,
                        coords,
                        set(zero),
                        max_orders,
                        precision_digits,
                        timing,
                        boundary_positions=set(boundary),
                        max_orders_are_explicit=True,
                    )
                    if shared_g_cache is not None:
                        shared_g_cache[shared_key] = cached
                g_cache[group_key] = cached

            for boundary, zero, multi_index, regular_order in expanded_entries:
                cached = g_cache[(boundary, zero)]
                input_row.append(_prec_series_coefficient(cached[regular_order], multi_index))
        if len(input_row) != len(child.input_names):
            raise RuntimeError(
                f"{sector.name}: IBP child input mismatch: filled {len(input_row)}, "
                f"expected {len(child.input_names)}"
            )
        return input_row

    def _ibp_shared_max_orders(
        self,
        sector: SectorDefinition,
        formula: EndpointProjectorFormulaDefinition,
    ) -> dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]]:
        """Return one Taylor envelope per IBP boundary/zero projector.

        The compound IBP projector is topology-independent, but applying it to a
        sector requires many regular-function Taylor coefficients.  Computing a
        coefficient box for every child term is correct but wasteful.  For a
        fixed boundary set and projector-zero set, the largest requested
        multi-index box contains every smaller box, so this envelope is the
        natural cache key for the runtime coefficient assembly.
        """
        return _ibp_shared_max_orders_for_formula(sector, formula)

    def _ibp_shared_output_pairs(
        self,
        sector: SectorDefinition,
        formula: EndpointProjectorFormulaDefinition,
    ) -> dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[tuple[int, ...], int], ...]]:
        """Return sparse regular coefficients consumed by each IBP envelope."""
        return _ibp_shared_output_pairs_for_formula(sector, formula)

    def _precompute_ibp_shared_batch_g_cache(
        self,
        sector: SectorDefinition,
        rows: np.ndarray,
        shared_max_orders: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]],
        shared_output_pairs: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[tuple[int, ...], int], ...],
        ],
        timing: HotPathTiming,
    ) -> dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], list[MultiSeries]]:
        """Batch fallback regular-Taylor assemblies across IBP boundaries.

        A hard six-axis IBP sector can request one regular Taylor object for
        every combination of "sampled", "zeroed", and "boundary at 1" endpoint
        state.  The regular algebra depends on the zero set and requested
        Taylor coefficients; the boundary set only changes the endpoint row.
        Grouping boundary rows by zero set therefore preserves the black-box
        U/F boundary while avoiding hundreds of repeated sparse-series builds.
        Pregenerated direct regular formulas are deliberately skipped here so
        they keep using their own evaluator path.
        """
        sample_rows = np.asarray(rows, dtype=float)
        n_rows = int(sample_rows.shape[0])
        if n_rows == 0:
            return {}

        fallback_by_zero: dict[
            tuple[int, ...],
            list[
                tuple[
                    tuple[tuple[int, ...], tuple[int, ...]],
                    tuple[int, ...],
                    tuple[tuple[tuple[int, ...], int], ...],
                ]
            ],
        ] = {}
        for key, max_orders in shared_max_orders.items():
            boundary, zero = key
            output_pairs = shared_output_pairs.get(key, ())
            signature = self.topology.regular_taylor_signature(
                sector,
                zero_positions=zero,
                max_orders=max_orders,
                output_pairs=output_pairs,
            )
            if signature in self.topology._regular_taylor_formulas:
                continue
            fallback_by_zero.setdefault(tuple(zero), []).append(
                (key, tuple(int(order) for order in max_orders), tuple(output_pairs))
            )

        cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            list[MultiSeries],
        ] = {}
        # Keep stacked evaluator inputs bounded for selected-sector runs with
        # large batches.  For the all-sector Havana path the per-sector batch is
        # usually small, so this still collapses the pathological 729 one-row
        # assemblies down to O(64) sparse builds.
        max_stacked_rows = 4096
        for zero, entries in fallback_by_zero.items():
            entries.sort(key=lambda item: (len(item[0][0]), item[0][0], item[1]))
            # After the symbolic chain-rule formula is pregenerated, the
            # dominant cost is the regular epsilon-series algebra, not the
            # mapped-derivative composition.  Therefore one stacked assembly
            # per zero projector is now preferable; the row cap still prevents
            # selected-sector runs with large batches from constructing
            # oversized temporary arrays.
            preferred_keys_per_chunk = len(entries)
            keys_per_chunk = min(
                max(1, int(preferred_keys_per_chunk)),
                max(1, max_stacked_rows // max(1, n_rows)),
            )
            for start_index in range(0, len(entries), keys_per_chunk):
                chunk = entries[start_index : start_index + keys_per_chunk]
                envelope_orders = [
                    max(int(max_orders[position]) for _key, max_orders, _pairs in chunk)
                    for position in range(len(sector.singular_axes))
                ]
                output_pair_set: set[tuple[tuple[int, ...], int]] = set()
                stacked_rows: list[np.ndarray] = []
                slices: list[
                    tuple[
                        tuple[tuple[int, ...], tuple[int, ...]],
                        tuple[int, ...],
                        int,
                        int,
                    ]
                ] = []
                row_offset = 0
                for key, max_orders, output_pairs in chunk:
                    boundary, _zero = key
                    output_pair_set.update(output_pairs)
                    endpoint_rows = sample_rows.copy()
                    for position in boundary:
                        endpoint_rows[:, sector.singular_axes[int(position)]] = 1.0
                    stacked_rows.append(endpoint_rows)
                    slices.append((key, max_orders, row_offset, row_offset + n_rows))
                    row_offset += n_rows
                if not stacked_rows:
                    continue
                union_output_pairs = tuple(
                    sorted(
                        output_pair_set,
                        key=lambda item: (item[1], sum(item[0]), item[0]),
                    )
                )
                stacked = np.vstack(stacked_rows)
                shared_series = self._g_taylor_eps_series_batch(
                    sector,
                    stacked,
                    set(zero),
                    envelope_orders,
                    timing,
                    boundary_positions=set(),
                    max_orders_are_explicit=True,
                    output_pairs=union_output_pairs,
                )
                for key, max_orders, row_start, row_stop in slices:
                    boundary, key_zero = key
                    cache[(boundary, key_zero, tuple(max_orders))] = _slice_multi_series_list(
                        shared_series,
                        row_start,
                        row_stop,
                    )
        return cache

    def _ibp_endpoint_projector_subtraction_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        formula: EndpointProjectorFormulaDefinition,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a compound IBP-lowered endpoint projector."""
        rows = np.asarray(y_values, dtype=float)
        coeffs = np.zeros((rows.shape[0], self.topology.coefficient_count), dtype=np.complex128)
        precision_digits = timing.precision_digits
        shared_batch_g_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            list[MultiSeries],
        ] = {}
        shared_prec_g_caches: list[
            dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], list[PrecSeries]]
        ] = [
            {}
            for _ in range(rows.shape[0])
        ] if precision_digits is not None else []
        shared_max_orders = self._ibp_shared_max_orders(sector, formula)
        shared_output_pairs = self._ibp_shared_output_pairs(sector, formula)
        if precision_digits is None:
            shared_batch_g_cache.update(
                self._precompute_ibp_shared_batch_g_cache(
                    sector,
                    rows,
                    shared_max_orders,
                    shared_output_pairs,
                    timing,
                )
            )
            # Many IBP decompositions contain hundreds of child terms, but only
            # a small number of distinct logarithmic endpoint-projector
            # signatures.  Build the child input rows term-by-term because the
            # derivative/boundary maps differ, then evaluate equal signatures in
            # one Symbolica batch.  This preserves the algebra while avoiding
            # hundreds of tiny evaluator calls in hard multi-axis sectors.
            terms_by_child: dict[
                tuple[Any, ...],
                list[tuple[IBPEndpointProjectorTerm, np.ndarray]],
            ] = {}
            for term in formula.ibp_terms:
                child = formula.child_formulas[term.child_signature]
                child_inputs = self._ibp_child_input_matrix(
                    sector,
                    rows,
                    child,
                    term,
                    timing,
                    shared_g_cache=shared_batch_g_cache,
                    shared_max_orders=shared_max_orders,
                    shared_output_pairs=shared_output_pairs,
                )
                terms_by_child.setdefault(term.child_signature, []).append(
                    (term, child_inputs)
                )
            for child_signature, entries in terms_by_child.items():
                child = formula.child_formulas[child_signature]
                stacked_inputs = np.vstack([inputs for _term, inputs in entries])
                stacked_values = self._select_active_laurent_columns(
                    child.evaluate_complex_batch(stacked_inputs, timing),
                    child.laurent_orders,
                    sector.name,
                )
                offset = 0
                for term, child_inputs in entries:
                    width = child_inputs.shape[0]
                    child_values = stacked_values[offset : offset + width, :]
                    offset += width
                    coeffs += self._convolve_regular_prefactor_array(
                        child_values,
                        term.prefactor_coeffs,
                    )
        else:
            for term in formula.ibp_terms:
                child = formula.child_formulas[term.child_signature]
                child_coeffs = np.zeros((rows.shape[0], self.topology.coefficient_count), dtype=np.complex128)
                for row_index, row in enumerate(rows):
                    child_input = self._ibp_child_input_prec_row(
                        sector,
                        row,
                        child,
                        term,
                        int(precision_digits),
                        timing,
                        shared_g_cache=shared_prec_g_caches[row_index],
                        shared_max_orders=shared_max_orders,
                    )
                    values = child.evaluate_complex_prec(
                        child_input,
                        int(precision_digits),
                        timing,
                    )
                    child_coeffs[row_index, :] = self._convolve_regular_prefactor_list(
                        self._select_active_laurent_list(
                            values,
                            child.laurent_orders,
                            sector.name,
                        ),
                        term.prefactor_coeffs,
                    )
                coeffs += child_coeffs
        return coeffs, complex_abs_for_training_array(coeffs[:, self.topology.training_index])

    def _residual_taylor_series_batch(
        self,
        sector: SectorDefinition,
        endpoint_rows: np.ndarray,
        monomial_powers: list[int],
        taylor_batch: Any,
        zero_positions: set[int],
        max_orders: list[int],
        timing: HotPathTiming,
    ) -> MultiSeries:
        """Taylor-expand ``P(X_s(y))/M_s(y)`` using only dualized P evaluators.

        ``zero_positions`` names the singular variables set to zero by a Taylor
        projector.  For those variables division by the extracted monomial is
        replaced by taking the matching black-box Taylor coefficient of P.  For
        every other variable, including non-singular monomial factors, the
        ordinary numerical quotient is used.
        """
        rows = np.asarray(endpoint_rows, dtype=float)
        taylor = taylor_batch(sector, rows, timing)
        return self._residual_taylor_series_from_values(
            sector,
            rows,
            monomial_powers,
            taylor,
            zero_positions,
            max_orders,
        )

    def _residual_taylor_series_from_values(
        self,
        sector: SectorDefinition,
        endpoint_rows: np.ndarray,
        monomial_powers: list[int],
        taylor: np.ndarray,
        zero_positions: set[int],
        max_orders: list[int],
        taylor_index: dict[tuple[int, ...], int] | None = None,
        residual_multis: set[tuple[int, ...]] | None = None,
    ) -> MultiSeries:
        """Taylor-expand a residual from already-composed polynomial jets."""
        rows = np.asarray(endpoint_rows, dtype=float)
        n_rows = rows.shape[0]
        axes = list(sector.singular_axes)
        axis_position = {axis: position for position, axis in enumerate(axes)}
        polynomial_series: MultiSeries = {}
        requested_multis = (
            sorted(residual_multis, key=lambda item: (sum(item), item))
            if residual_multis is not None
            else _multi_indices(max_orders)
        )
        for residual_multi in requested_multis:
            polynomial_multi = list(residual_multi)
            for axis, power in enumerate(monomial_powers):
                position = axis_position.get(axis)
                if position is not None and position in zero_positions:
                    # The requested coefficient is a Taylor coefficient of
                    # the residual P(X(y))/M(y).  At a zeroed singular axis,
                    # division by y^m shifts the polynomial coefficient by m,
                    # but the residual itself may still need derivatives even
                    # when m=0.  This is essential for multi-loop sectors where
                    # one polynomial creates the endpoint pole while the other
                    # remains regular but varies along the same coordinate.
                    polynomial_multi[position] += int(power)
            polynomial_multi_tuple = tuple(polynomial_multi)
            if taylor_index is None:
                column = sector.dual_index(polynomial_multi_tuple)
            else:
                column = taylor_index[polynomial_multi_tuple]
            polynomial_series[residual_multi] = taylor[:, column]
        denominator_series = self._monomial_taylor_series_batch(
            sector,
            rows,
            monomial_powers,
            zero_positions,
            max_orders,
        )
        if not denominator_series:
            return polynomial_series
        zero_multi = _zero_multi(len(max_orders))
        if set(denominator_series) <= {zero_multi}:
            denominator = denominator_series.get(zero_multi)
            if denominator is None:
                return polynomial_series
            return {
                multi_index: values / denominator
                for multi_index, values in polynomial_series.items()
            }
        return _series_mul(
            polynomial_series,
            _series_pow_real(denominator_series, -1.0, max_orders, n_rows),
            max_orders,
        )

    def _monomial_taylor_series_batch(
        self,
        sector: SectorDefinition,
        endpoint_rows: np.ndarray,
        monomial_powers: list[int],
        zero_positions: set[int],
        max_orders: list[int],
    ) -> MultiSeries:
        """Taylor-expand the nonzero part of an extracted monomial."""
        rows = np.asarray(endpoint_rows, dtype=float)
        n_rows = rows.shape[0]
        axes = list(sector.singular_axes)
        axis_position = {axis: position for position, axis in enumerate(axes)}
        series = _series_constant(1.0 + 0.0j, max_orders, n_rows)
        for axis, power_value in enumerate(monomial_powers):
            power = int(power_value)
            if power == 0:
                continue
            position = axis_position.get(axis)
            if position is not None and position in zero_positions:
                continue
            if position is None:
                factor = _series_constant(rows[:, axis] ** power, max_orders, n_rows)
            else:
                factor: MultiSeries = {}
                max_order = min(power, int(max_orders[position]))
                for order in range(max_order + 1):
                    multi = [0 for _ in max_orders]
                    multi[position] = order
                    coefficient = math.comb(power, order) * rows[:, axis] ** (power - order)
                    factor[tuple(multi)] = coefficient.astype(np.complex128)
            series = _series_mul(series, factor, max_orders)
        return series

    def _residual_taylor_series_prec_row(
        self,
        sector: SectorDefinition,
        endpoint_row: np.ndarray,
        monomial_powers: list[int],
        taylor_values: list[ComplexPrecise],
        zero_positions: set[int],
        max_orders: list[int],
        precision_digits: int,
    ) -> PrecSeries:
        """Decimal analogue of ``_residual_taylor_series_batch`` for one row."""
        axes = list(sector.singular_axes)
        axis_position = {axis: position for position, axis in enumerate(axes)}
        coords = [_decimal_complex(value, precision_digits) for value in np.asarray(endpoint_row, dtype=float)]
        polynomial_series: PrecSeries = {}
        for residual_multi in _multi_indices(max_orders):
            polynomial_multi = list(residual_multi)
            for axis, power_value in enumerate(monomial_powers):
                power = int(power_value)
                position = axis_position.get(axis)
                if position is not None and position in zero_positions:
                    polynomial_multi[position] += power
            coefficient = taylor_values[sector.dual_index(tuple(polynomial_multi))]
            polynomial_series[residual_multi] = coefficient
        denominator_series = self._monomial_taylor_series_prec_row(
            sector,
            endpoint_row,
            monomial_powers,
            zero_positions,
            max_orders,
            precision_digits,
        )
        return _prec_series_mul(
            polynomial_series,
            _prec_series_pow_real(
                denominator_series,
                -1.0,
                max_orders,
                precision_digits,
            ),
            max_orders,
        )

    def _monomial_taylor_series_prec_row(
        self,
        sector: SectorDefinition,
        endpoint_row: np.ndarray,
        monomial_powers: list[int],
        zero_positions: set[int],
        max_orders: list[int],
        precision_digits: int,
    ) -> PrecSeries:
        """Decimal Taylor expansion of the nonzero monomial denominator."""
        coords = [_decimal_complex(value, precision_digits) for value in np.asarray(endpoint_row, dtype=float)]
        axes = list(sector.singular_axes)
        axis_position = {axis: position for position, axis in enumerate(axes)}
        series = _prec_series_constant(_pc_one(), max_orders)
        for axis, power_value in enumerate(monomial_powers):
            power = int(power_value)
            if power == 0:
                continue
            position = axis_position.get(axis)
            if position is not None and position in zero_positions:
                continue
            factor: PrecSeries = {}
            if position is None:
                factor = _prec_series_constant(_pc_int_power(coords[axis], power), max_orders)
            else:
                for order in range(min(power, int(max_orders[position])) + 1):
                    multi = [0 for _ in max_orders]
                    multi[position] = order
                    coeff = _pc_scale(
                        _pc_int_power(coords[axis], power - order),
                        math.comb(power, order),
                        precision_digits,
                    )
                    factor[tuple(multi)] = coeff
            series = _prec_series_mul(series, factor, max_orders)
        return series

    def _jacobian_taylor_series_batch(
        self,
        sector: SectorDefinition,
        endpoint_rows: np.ndarray,
        max_orders: list[int],
        timing: HotPathTiming,
    ) -> MultiSeries:
        """Taylor-expand the regular sector Jacobian."""
        rows = np.asarray(endpoint_rows, dtype=float)
        n_rows = rows.shape[0]
        if not any(max_orders):
            return _series_constant(
                sector.jacobian_eval_batch(rows, timing).astype(np.complex128),
                max_orders,
                n_rows,
            )
        taylor = sector.jacobian_taylor_batch(rows, timing)
        return {
            multi: taylor[:, sector.dual_index(multi)]
            for multi in _multi_indices(max_orders)
        }

    def _jacobian_taylor_series_prec_row(
        self,
        sector: SectorDefinition,
        endpoint_row: np.ndarray,
        max_orders: list[int],
        precision_digits: int,
        timing: HotPathTiming,
    ) -> PrecSeries:
        """Taylor-expand the regular sector Jacobian at one precise point."""
        taylor = sector.jacobian_taylor_complex_prec(endpoint_row, precision_digits, timing)
        return {
            multi: taylor[sector.dual_index(multi)]
            for multi in _multi_indices(max_orders)
        }

    def _g_taylor_eps_series_formula_batch(
        self,
        sector: SectorDefinition,
        endpoint_rows: np.ndarray,
        zero_positions: set[int],
        max_orders: list[int],
        timing: HotPathTiming,
        output_pairs: tuple[tuple[tuple[int, ...], int], ...] | None = None,
    ) -> list[MultiSeries]:
        """Evaluate regular ``g_s`` Taylor coefficients with Symbolica.

        The inputs remain black-box Taylor coefficients of U, F, and the
        regular Jacobian.  The prepared formula owns the downstream algebra:
        monomial residuals, U/F powers, logs, and epsilon expansion.
        """
        rows = np.asarray(endpoint_rows, dtype=float)
        n_rows = rows.shape[0]
        zero_tuple = tuple(sorted(int(position) for position in zero_positions))
        max_tuple = tuple(int(order) for order in max_orders)
        formula = self.topology.regular_taylor_formula_for(
            sector,
            zero_positions=zero_tuple,
            max_orders=max_tuple,
            output_pairs=output_pairs,
        )
        formula_version = int(formula.signature[1]) if len(formula.signature) > 1 else 1
        canonical_positions = _regular_taylor_canonical_positions(max_tuple)
        formula_shape = _regular_formula_dual_shape(formula)
        if len(formula.signature) > 1 and int(formula.signature[1]) <= 1:
            source_shape = formula_shape
        elif formula_version >= 3:
            residual_multis = {
                _regular_taylor_canonical_to_original(tuple(multi), canonical_positions)
                for kind, multi in formula.input_layout
                if kind in {"u", "f"}
            }
            jacobian_multis = {
                _regular_taylor_canonical_to_original(tuple(multi), canonical_positions)
                for kind, multi in formula.input_layout
                if kind == "j"
            }
            source_shape = self.topology.sparse_regular_source_shape_from_multis(
                sector,
                zero_tuple,
                residual_multis,
                jacobian_multis,
            )
        else:
            source_shape = self.topology.regular_taylor_source_shape(
                sector,
                zero_tuple,
                max_tuple,
            )
        formula_index = {multi_index: index for index, multi_index in enumerate(formula_shape)}
        source_index = {multi_index: index for index, multi_index in enumerate(source_shape)}

        j_taylor = sector.jacobian_taylor_batch_for_shape(rows, source_shape, timing)
        if formula.signature in self.topology._regular_taylor_dual_signatures:
            u_taylor = self.topology._taylor_batch(
                sector,
                rows,
                self.topology.u_dual_evaluator(source_shape),
                evaluator_shape=source_shape,
                timing=timing,
            )
            f_taylor = self.topology._taylor_batch(
                sector,
                rows,
                self.topology.f_dual_evaluator(source_shape),
                evaluator_shape=source_shape,
                timing=timing,
            )
        elif self.topology.dual_evaluator_mode == "symbolic-derivatives":
            u_taylor, f_taylor = self.topology._symbolic_derivative_taylor_pair_batch(
                sector,
                rows,
                timing,
                output_shape=source_shape,
            )
        else:
            evaluator_shape, output_columns = self.topology._dual_evaluator_shape_and_columns(sector)
            if tuple(evaluator_shape) == tuple(source_shape):
                u_taylor = self.topology.u_taylor_batch(sector, rows, timing)
                f_taylor = self.topology.f_taylor_batch(sector, rows, timing)
            else:
                u_taylor = self.topology._taylor_batch(
                    sector,
                    rows,
                    self.topology.u_dual_evaluator(source_shape),
                    evaluator_shape=source_shape,
                    timing=timing,
                )
                f_taylor = self.topology._taylor_batch(
                    sector,
                    rows,
                    self.topology.f_dual_evaluator(source_shape),
                    evaluator_shape=source_shape,
                    timing=timing,
                )

        input_matrix = np.zeros((n_rows, len(formula.input_names)), dtype=np.complex128)
        offset = 0
        if len(formula.signature) > 1 and int(formula.signature[1]) >= 2:
            # The lower-signature regular formula is sector-agnostic.  It
            # receives residual Taylor coefficients that have already had the
            # declared U/F endpoint monomials divided out, plus the nonsingular
            # monomial prefactor/log evaluated at the endpoint row.
            monomial_pref, monomial_log = self._regular_monomial_base_log_batch(
                sector,
                endpoint_rows,
            )
            input_matrix[:, offset] = monomial_pref
            offset += 1
            input_matrix[:, offset] = monomial_log
            offset += 1
            u_series = self._residual_taylor_series_from_values(
                sector=sector,
                endpoint_rows=endpoint_rows,
                monomial_powers=sector.u_monomial_powers,
                taylor=u_taylor,
                zero_positions=zero_positions,
                max_orders=max_orders,
                taylor_index=source_index,
                residual_multis=(
                    residual_multis
                    if formula_version >= 3
                    else None
                ),
            )
            f_series = self._residual_taylor_series_from_values(
                sector=sector,
                endpoint_rows=endpoint_rows,
                monomial_powers=sector.f_monomial_powers,
                taylor=f_taylor,
                zero_positions=zero_positions,
                max_orders=max_orders,
                taylor_index=source_index,
                residual_multis=(
                    residual_multis
                    if formula_version >= 3
                    else None
                ),
            )

            for kind, multi_index in formula.input_layout:
                canonical_multi = tuple(multi_index)
                multi = (
                    _regular_taylor_canonical_to_original(canonical_multi, canonical_positions)
                    if formula_version >= 2
                    else canonical_multi
                )
                if kind == "j":
                    input_matrix[:, offset] = j_taylor[:, source_index[multi]]
                elif kind == "u":
                    input_matrix[:, offset] = _series_coefficient(
                        u_series,
                        multi,
                        n_rows,
                    )
                elif kind == "f":
                    input_matrix[:, offset] = _series_coefficient(
                        f_series,
                        multi,
                        n_rows,
                    )
                else:
                    raise RuntimeError(f"{sector.name}: unknown regular input kind {kind!r}")
                offset += 1
        else:
            input_matrix[:, : sector.integration_dim] = rows.astype(np.complex128)
            offset += sector.integration_dim
            source_by_kind = {
                "j": j_taylor,
                "u": u_taylor,
                "f": f_taylor,
            }
            for kind, multi_index in formula.input_layout:
                values = source_by_kind[kind]
                input_matrix[:, offset] = values[:, formula_index[tuple(multi_index)]]
                offset += 1
        if offset != len(formula.input_names):
            raise RuntimeError(
                f"{sector.name}: regular Taylor input mismatch: filled {offset}, "
                f"expected {len(formula.input_names)}"
            )

        values = formula.evaluate_complex_batch(input_matrix, timing)
        out: list[MultiSeries] = [
            {} for _ in range(self.topology.coefficient_count)
        ]
        for column, (multi_index, regular_order) in enumerate(formula.output_layout):
            canonical_multi = tuple(multi_index)
            multi = (
                _regular_taylor_canonical_to_original(canonical_multi, canonical_positions)
                if formula_version >= 2
                else canonical_multi
            )
            out[int(regular_order)][multi] = values[:, column]
        return out

    def _g_taylor_eps_series_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        zero_positions: set[int],
        taylor_orders: list[int],
        timing: HotPathTiming,
        boundary_positions: set[int] | None = None,
        max_orders_are_explicit: bool = False,
        output_pairs: tuple[tuple[tuple[int, ...], int], ...] | None = None,
    ) -> list[MultiSeries]:
        """Taylor-expand the regular function ``g_s(y,eps)`` at endpoints.

        The returned list is indexed by the non-negative epsilon order.  Each
        entry is a sparse Taylor series in the declared singular variables.
        """
        rows = np.asarray(y_values, dtype=float)
        n_rows = rows.shape[0]
        axes = list(sector.singular_axes)
        if max_orders_are_explicit:
            max_orders = [int(order) for order in taylor_orders]
        else:
            max_orders = [
                int(taylor_orders[position]) if position in zero_positions else 0
                for position in range(len(axes))
            ]
        endpoint_rows = rows.copy()
        for position in zero_positions:
            endpoint_rows[:, axes[position]] = 0.0
        for position in boundary_positions or set():
            endpoint_rows[:, axes[position]] = 1.0

        sparse_residual_multis: set[tuple[int, ...]] | None = None
        sparse_jacobian_multis: set[tuple[int, ...]] | None = None
        sparse_source_shape: list[tuple[int, ...]] | None = None
        sparse_jacobian_shape: list[tuple[int, ...]] | None = None
        sparse_jacobian_index: dict[tuple[int, ...], int] | None = None
        sparse_u_source_shape: list[tuple[int, ...]] | None = None
        sparse_u_source_index: dict[tuple[int, ...], int] | None = None
        sparse_f_source_shape: list[tuple[int, ...]] | None = None
        sparse_f_source_index: dict[tuple[int, ...], int] | None = None
        if output_pairs:
            requested_multis = [
                tuple(int(value) for value in multi_index)
                for multi_index, _regular_order in output_pairs
            ]
            sparse_residual_multis = _ancestor_closed_multi_set(
                requested_multis,
                len(max_orders),
            )
            sparse_jacobian_multis = set(sparse_residual_multis)
            if self.topology.dual_evaluator_mode != "symbolic-derivatives":
                sparse_source_shape = self.topology.sparse_regular_source_shape_from_multis(
                    sector,
                    tuple(sorted(int(position) for position in zero_positions)),
                    sparse_residual_multis,
                    sparse_jacobian_multis,
                )
            sparse_jacobian_shape = _ordered_multi_shape(
                set(sparse_jacobian_multis),
                len(max_orders),
            )
            sparse_jacobian_index = {
                tuple(multi_index): index
                for index, multi_index in enumerate(sparse_jacobian_shape)
            }
            sparse_u_source_shape = self.topology.sparse_regular_source_shape_for_monomial_powers(
                sector,
                tuple(sorted(int(position) for position in zero_positions)),
                sparse_residual_multis,
                sector.u_monomial_powers,
            )
            sparse_u_source_index = {
                tuple(multi_index): index
                for index, multi_index in enumerate(sparse_u_source_shape)
            }
            sparse_f_source_shape = self.topology.sparse_regular_source_shape_for_monomial_powers(
                sector,
                tuple(sorted(int(position) for position in zero_positions)),
                sparse_residual_multis,
                sector.f_monomial_powers,
            )
            sparse_f_source_index = {
                tuple(multi_index): index
                for index, multi_index in enumerate(sparse_f_source_shape)
            }

        if self.subtraction_backend == "projector-formula":
            signature = self.topology.regular_taylor_signature(
                sector,
                zero_positions=tuple(sorted(int(position) for position in zero_positions)),
                max_orders=tuple(int(order) for order in max_orders),
                output_pairs=output_pairs,
            )
            if signature in self.topology._regular_taylor_formulas:
                return self._g_taylor_eps_series_formula_batch(
                    sector,
                    endpoint_rows,
                    zero_positions,
                    max_orders,
                    timing,
                    output_pairs=output_pairs,
                )

        if not zero_positions and not any(max_orders) and not boundary_positions:
            coeffs = self._g_coeffs_batch(sector, endpoint_rows, timing)
            return [
                _series_constant(coeffs[:, order], max_orders, n_rows)
                for order in range(self.topology.coefficient_count)
            ]

        if sparse_jacobian_shape is not None and sparse_jacobian_index is not None:
            j_taylor = sector.jacobian_taylor_batch_for_shape(
                endpoint_rows,
                sparse_jacobian_shape,
                timing,
            )
            jacobian_series = {
                multi_index: j_taylor[:, sparse_jacobian_index[multi_index]]
                for multi_index in sparse_jacobian_multis or set()
            }
        else:
            jacobian_series = self._jacobian_taylor_series_batch(
                sector, endpoint_rows, max_orders, timing
            )
        if self.topology.dual_evaluator_mode == "symbolic-derivatives":
            u_taylor, f_taylor = self.topology._symbolic_derivative_taylor_pair_batch(
                sector,
                endpoint_rows,
                timing,
                output_shape=sparse_source_shape,
                u_output_shape=sparse_u_source_shape,
                f_output_shape=sparse_f_source_shape,
            )
            u_series = self._residual_taylor_series_from_values(
                sector=sector,
                endpoint_rows=endpoint_rows,
                monomial_powers=sector.u_monomial_powers,
                taylor=u_taylor,
                zero_positions=zero_positions,
                max_orders=max_orders,
                taylor_index=sparse_u_source_index,
                residual_multis=sparse_residual_multis,
            )
            f_series = self._residual_taylor_series_from_values(
                sector=sector,
                endpoint_rows=endpoint_rows,
                monomial_powers=sector.f_monomial_powers,
                taylor=f_taylor,
                zero_positions=zero_positions,
                max_orders=max_orders,
                taylor_index=sparse_f_source_index,
                residual_multis=sparse_residual_multis,
            )
        else:
            if (
                sparse_u_source_shape is not None
                and sparse_u_source_index is not None
                and sparse_f_source_shape is not None
                and sparse_f_source_index is not None
            ):
                u_taylor = self.topology._taylor_batch(
                    sector,
                    endpoint_rows,
                    self.topology.u_dual_evaluator(sparse_u_source_shape),
                    evaluator_shape=sparse_u_source_shape,
                    timing=timing,
                )
                f_taylor = self.topology._taylor_batch(
                    sector,
                    endpoint_rows,
                    self.topology.f_dual_evaluator(sparse_f_source_shape),
                    evaluator_shape=sparse_f_source_shape,
                    timing=timing,
                )
                u_series = self._residual_taylor_series_from_values(
                    sector=sector,
                    endpoint_rows=endpoint_rows,
                    monomial_powers=sector.u_monomial_powers,
                    taylor=u_taylor,
                    zero_positions=zero_positions,
                    max_orders=max_orders,
                    taylor_index=sparse_u_source_index,
                    residual_multis=sparse_residual_multis,
                )
                f_series = self._residual_taylor_series_from_values(
                    sector=sector,
                    endpoint_rows=endpoint_rows,
                    monomial_powers=sector.f_monomial_powers,
                    taylor=f_taylor,
                    zero_positions=zero_positions,
                    max_orders=max_orders,
                    taylor_index=sparse_f_source_index,
                    residual_multis=sparse_residual_multis,
                )
            else:
                u_series = self._residual_taylor_series_batch(
                    sector=sector,
                    endpoint_rows=endpoint_rows,
                    monomial_powers=sector.u_monomial_powers,
                    taylor_batch=self.topology.u_taylor_batch,
                    zero_positions=zero_positions,
                    max_orders=max_orders,
                    timing=timing,
                )
                f_series = self._residual_taylor_series_batch(
                    sector=sector,
                    endpoint_rows=endpoint_rows,
                    monomial_powers=sector.f_monomial_powers,
                    taylor_batch=self.topology.f_taylor_batch,
                    zero_positions=zero_positions,
                    max_orders=max_orders,
                    timing=timing,
                )
        allowed_multis = sparse_residual_multis
        if allowed_multis is not None:
            u_power_series, u_log_series = _series_pow_real_and_log_allowed(
                u_series,
                self.topology.u_power_base,
                max_orders,
                n_rows,
                allowed_multis,
            )
            f_power_series, f_log_series = _series_pow_real_and_log_allowed(
                f_series,
                -self.topology.f_power_base,
                max_orders,
                n_rows,
                allowed_multis,
            )
            pref_series = _series_mul_allowed(
                _series_filter_allowed(jacobian_series, allowed_multis),
                _series_mul_allowed(
                    u_power_series,
                    f_power_series,
                    allowed_multis,
                ),
                allowed_multis,
            )
        else:
            pref_series = _series_mul(
                jacobian_series,
                _series_mul(
                    _series_pow_real(u_series, self.topology.u_power_base, max_orders, n_rows),
                    _series_pow_real(f_series, -self.topology.f_power_base, max_orders, n_rows),
                    max_orders,
                ),
                max_orders,
            )
        monomial_pref, monomial_log = self._regular_monomial_base_log_batch(sector, endpoint_rows)
        if allowed_multis is not None:
            pref_series = _series_mul_allowed(
                _series_constant(monomial_pref, max_orders, n_rows),
                pref_series,
                allowed_multis,
            )
            log_series = _series_filter_allowed(
                _series_add(
                    _series_constant(monomial_log, max_orders, n_rows),
                    _series_add(
                        _series_scale(
                            u_log_series,
                            self.topology.eps_log_u_coeff,
                        ),
                        _series_scale(
                            f_log_series,
                            self.topology.eps_log_f_coeff,
                        ),
                    ),
                ),
                allowed_multis,
            )
        else:
            pref_series = _series_mul(
                _series_constant(monomial_pref, max_orders, n_rows),
                pref_series,
                max_orders,
            )
            log_series = _series_add(
                _series_constant(monomial_log, max_orders, n_rows),
                _series_add(
                    _series_scale(_series_log(u_series, max_orders, n_rows), self.topology.eps_log_u_coeff),
                    _series_scale(_series_log(f_series, max_orders, n_rows), self.topology.eps_log_f_coeff),
                ),
            )

        requested_multis_by_order: dict[int, set[tuple[int, ...]]] | None = None
        if output_pairs is not None:
            requested_multis_by_order = {}
            for multi_index, regular_order in output_pairs:
                requested_multis_by_order.setdefault(int(regular_order), set()).add(
                    tuple(int(value) for value in multi_index)
                )

        out: list[MultiSeries] = []
        log_power = _series_constant(1.0 + 0.0j, max_orders, n_rows)
        factorial = 1.0
        for order in range(self.topology.coefficient_count):
            if order > 0:
                factorial *= float(order)
                log_power = (
                    _series_mul_allowed(log_power, log_series, allowed_multis)
                    if allowed_multis is not None
                    else _series_mul(log_power, log_series, max_orders)
                )
            final_allowed = (
                requested_multis_by_order.get(order, set())
                if requested_multis_by_order is not None
                else allowed_multis
            )
            if requested_multis_by_order is not None and not final_allowed:
                out.append({})
                continue
            product_series = (
                _series_mul_allowed(pref_series, log_power, final_allowed)
                if final_allowed is not None
                else _series_mul(pref_series, log_power, max_orders)
            )
            out.append(
                _series_scale(
                    product_series,
                    1.0 / factorial,
                )
            )
        return out

    def _g_taylor_eps_series_prec_row(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        zero_positions: set[int],
        taylor_orders: list[int],
        precision_digits: int,
        timing: HotPathTiming,
        boundary_positions: set[int] | None = None,
        max_orders_are_explicit: bool = False,
    ) -> list[PrecSeries]:
        """Decimal Taylor/Laurent expansion of the regular function ``g_s``.

        This mirrors ``_g_taylor_eps_series_batch`` but every ingredient is
        obtained from Symbolica's complex multiprecision evaluator APIs and
        combined in Decimal arithmetic.  It is intentionally single-row only:
        the ordinary path remains vectorized and this path is reserved for
        near-endpoint stability rescue samples.
        """
        axes = list(sector.singular_axes)
        if max_orders_are_explicit:
            max_orders = [int(order) for order in taylor_orders]
        else:
            max_orders = [
                int(taylor_orders[position]) if position in zero_positions else 0
                for position in range(len(axes))
            ]
        endpoint_row = np.asarray(y, dtype=float).copy()
        for position in zero_positions:
            endpoint_row[axes[position]] = 0.0
        for position in boundary_positions or set():
            endpoint_row[axes[position]] = 1.0

        if not zero_positions and not any(max_orders) and not boundary_positions:
            coeffs = self._g_coeffs_prec_row(sector, endpoint_row, precision_digits, timing)
            return [
                _prec_series_constant(coeff, max_orders)
                for coeff in coeffs
            ]

        jacobian_series = self._jacobian_taylor_series_prec_row(
            sector,
            endpoint_row,
            max_orders,
            precision_digits,
            timing,
        )
        u_taylor = self.topology.u_taylor_complex_prec(
            sector,
            endpoint_row,
            precision_digits,
            timing,
        )
        f_taylor = self.topology.f_taylor_complex_prec(
            sector,
            endpoint_row,
            precision_digits,
            timing,
        )
        u_series = self._residual_taylor_series_prec_row(
            sector=sector,
            endpoint_row=endpoint_row,
            monomial_powers=sector.u_monomial_powers,
            taylor_values=u_taylor,
            zero_positions=zero_positions,
            max_orders=max_orders,
            precision_digits=precision_digits,
        )
        f_series = self._residual_taylor_series_prec_row(
            sector=sector,
            endpoint_row=endpoint_row,
            monomial_powers=sector.f_monomial_powers,
            taylor_values=f_taylor,
            zero_positions=zero_positions,
            max_orders=max_orders,
            precision_digits=precision_digits,
        )
        pref_series = _prec_series_mul(
            jacobian_series,
            _prec_series_mul(
                _prec_series_pow_real(
                    u_series,
                    self.topology.u_power_base,
                    max_orders,
                    precision_digits,
                ),
                _prec_series_pow_real(
                    f_series,
                    -self.topology.f_power_base,
                    max_orders,
                    precision_digits,
                ),
                max_orders,
            ),
            max_orders,
        )
        monomial_pref, monomial_log = self._regular_monomial_base_log_prec(
            sector,
            endpoint_row,
            precision_digits,
        )
        pref_series = _prec_series_mul(
            _prec_series_constant(monomial_pref, max_orders),
            pref_series,
            max_orders,
        )
        log_series = _prec_series_add(
            _prec_series_constant(monomial_log, max_orders),
            _prec_series_add(
                _prec_series_scale(
                    _prec_series_log(u_series, max_orders, precision_digits),
                    self.topology.eps_log_u_coeff,
                    precision_digits,
                ),
                _prec_series_scale(
                    _prec_series_log(f_series, max_orders, precision_digits),
                    self.topology.eps_log_f_coeff,
                    precision_digits,
                ),
            ),
        )

        out: list[PrecSeries] = []
        log_power = _prec_series_constant(_pc_one(), max_orders)
        factorial = Decimal(1)
        for order in range(self.topology.coefficient_count):
            if order > 0:
                factorial *= Decimal(order)
                log_power = _prec_series_mul(log_power, log_series, max_orders)
            out.append(
                _prec_series_scale(
                    _prec_series_mul(pref_series, log_power, max_orders),
                    Decimal(1) / factorial,
                    precision_digits,
                )
            )
        return out

    def _g_coeffs_prec_row(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        precision_digits: int,
        timing: HotPathTiming,
    ) -> list[ComplexPrecise]:
        """High-precision scalar regular-function coefficients for one point."""
        max_orders = [0 for _ in sector.singular_axes]
        jacobian_series = self._jacobian_taylor_series_prec_row(
            sector,
            np.asarray(y, dtype=float),
            max_orders,
            precision_digits,
            timing,
        )
        u_taylor = self.topology.u_taylor_complex_prec(sector, y, precision_digits, timing)
        f_taylor = self.topology.f_taylor_complex_prec(sector, y, precision_digits, timing)
        u_series = self._residual_taylor_series_prec_row(
            sector,
            np.asarray(y, dtype=float),
            sector.u_monomial_powers,
            u_taylor,
            set(),
            max_orders,
            precision_digits,
        )
        f_series = self._residual_taylor_series_prec_row(
            sector,
            np.asarray(y, dtype=float),
            sector.f_monomial_powers,
            f_taylor,
            set(),
            max_orders,
            precision_digits,
        )
        pref_series = _prec_series_mul(
            jacobian_series,
            _prec_series_mul(
                _prec_series_pow_real(
                    u_series,
                    self.topology.u_power_base,
                    max_orders,
                    precision_digits,
                ),
                _prec_series_pow_real(
                    f_series,
                    -self.topology.f_power_base,
                    max_orders,
                    precision_digits,
                ),
                max_orders,
            ),
            max_orders,
        )
        monomial_pref, monomial_log = self._regular_monomial_base_log_prec(
            sector,
            y,
            precision_digits,
        )
        pref = _pc_mul(
            monomial_pref,
            _prec_series_coefficient(pref_series, _zero_multi(len(max_orders))),
        )
        u_const = _prec_series_coefficient(u_series, _zero_multi(len(max_orders)))
        f_const = _prec_series_coefficient(f_series, _zero_multi(len(max_orders)))
        exponent_log = _pc_add(
            monomial_log,
            _pc_add(
                _pc_scale(_pc_log(u_const, precision_digits), self.topology.eps_log_u_coeff, precision_digits),
                _pc_scale(_pc_log(f_const, precision_digits), self.topology.eps_log_f_coeff, precision_digits),
            ),
        )
        coeffs: list[ComplexPrecise] = [pref]
        power = _pc_one()
        factorial = Decimal(1)
        for order in range(1, self.topology.coefficient_count):
            factorial *= Decimal(order)
            power = _pc_mul(power, exponent_log)
            coeffs.append(_pc_mul(pref, _pc_scale(power, Decimal(1) / factorial, precision_digits)))
        return coeffs

    def _recursive_taylor_subtraction_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Apply localized Taylor endpoint subtraction for all singular axes.

        For an endpoint factor ``y_a^(beta_a+c_a eps)`` with negative integer
        ``beta_a``, the Taylor polynomial through ``N_a=-beta_a-1`` is
        integrated analytically.  The remainder is localized at the sampled
        point.  Logarithmic plus distributions are the special case ``N_a=0``.
        """
        axes = list(sector.singular_axes)
        n_axes = len(axes)
        bases, eps_coeffs, taylor_orders = self._endpoint_power_data(sector)
        if self.topology.laurent_min_order > -n_axes:
            raise ValueError(
                f"{sector.name}: topology Laurent range starts at eps^{self.topology.laurent_min_order}, "
                f"but this sector has endpoint pole depth {n_axes}"
            )

        rows = np.asarray(y_values, dtype=float)
        n_rows = rows.shape[0]
        coeffs = np.zeros((n_rows, self.topology.coefficient_count), dtype=np.complex128)
        g_cache: dict[frozenset[int], list[MultiSeries]] = {}

        def g_for(zero_positions: set[int]) -> list[MultiSeries]:
            key = frozenset(zero_positions)
            cached = g_cache.get(key)
            if cached is None:
                cached = self._g_taylor_eps_series_batch(
                    sector, rows, set(zero_positions), taylor_orders, timing
                )
                g_cache[key] = cached
            return cached

        min_order = self.topology.laurent_min_order
        max_order = self.topology.laurent_max_order
        regular_count = self.topology.coefficient_count
        position_range = list(range(n_axes))
        for integrated_flags in product((False, True), repeat=n_axes):
            integrated_positions = [pos for pos, flag in enumerate(integrated_flags) if flag]
            active_positions = [pos for pos, flag in enumerate(integrated_flags) if not flag]
            active_factor = np.ones(n_rows, dtype=np.complex128)
            active_log_sum = np.zeros(n_rows, dtype=float)
            for position in active_positions:
                coord = rows[:, axes[position]]
                active_factor *= np.power(coord, bases[position])
                active_log_sum += eps_coeffs[position] * np.log(coord)

            for taylor_flags in product((False, True), repeat=len(active_positions)):
                projected_positions = [
                    position
                    for position, flag in zip(active_positions, taylor_flags)
                    if flag
                ]
                sign = -1.0 if len(projected_positions) % 2 else 1.0
                zero_positions = set(integrated_positions) | set(projected_positions)
                g_series_by_eps = g_for(zero_positions)

                max_multi_orders = [
                    taylor_orders[position] if position in zero_positions else 0
                    for position in position_range
                ]
                for multi_index in _multi_indices(max_multi_orders):
                    sample_factor = sign * active_factor.copy()
                    for position in projected_positions:
                        order = multi_index[position]
                        if order:
                            sample_factor *= rows[:, axes[position]] ** order
                    denominator_series = self._integrated_denominator_series(
                        bases, eps_coeffs, integrated_positions, multi_index
                    )
                    if not denominator_series:
                        continue

                    active_log_power = np.ones(n_rows, dtype=np.complex128)
                    active_log_factorial = 1.0
                    for log_order in range(regular_count):
                        if log_order > 0:
                            active_log_factorial *= float(log_order)
                            active_log_power = active_log_power * active_log_sum
                        active_log_coeff = active_log_power / active_log_factorial
                        for regular_order in range(regular_count):
                            regular_coeff = _series_coefficient(
                                g_series_by_eps[regular_order],
                                multi_index,
                                n_rows,
                            )
                            if not np.any(regular_coeff):
                                continue
                            for denom_order, denom_coeff in denominator_series.items():
                                eps_order = regular_order + log_order + denom_order
                                if eps_order < min_order or eps_order > max_order:
                                    continue
                                coeffs[:, eps_order - min_order] += (
                                    sample_factor
                                    * active_log_coeff
                                    * denom_coeff
                                    * regular_coeff
                                )
        return coeffs, complex_abs_for_training_array(coeffs[:, self.topology.training_index])

    def _recursive_log_subtraction_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backward-compatible name for the generic Taylor subtraction path."""
        return self._recursive_taylor_subtraction_batch(sector, y_values, timing)

    def _one_axis_subtraction_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility wrapper for one-axis logarithmic endpoint subtraction."""
        return self._recursive_log_subtraction_batch(sector, y_values, timing)

    def _two_axis_subtraction_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility wrapper for two-axis logarithmic endpoint subtraction."""
        return self._recursive_log_subtraction_batch(sector, y_values, timing)

    def _finite_contribution(
        self, sector: SectorDefinition, y: list[float] | tuple[float, ...]
    ) -> tuple[list[complex], float]:
        """Scalar compatibility wrapper for finite sectors."""
        coeffs, training, _ = self.evaluate_batch(sector, np.asarray([y], dtype=float))
        return [complex(value) for value in coeffs[0]], float(training[0])

    def _legacy_evaluate(self, sector: SectorDefinition, y: list[float] | tuple[float, ...]) -> tuple[list[complex], float]:
        """Older scalar dispatch kept for debugging against the vectorized path."""
        if len(sector.singular_axes) == 0:
            return self._finite_contribution(sector, y)
        if len(sector.singular_axes) == 1:
            return self._one_axis_subtraction(sector, y)
        if len(sector.singular_axes) == 2:
            return self._two_axis_subtraction(sector, y)
        raise ValueError(
            f"{sector.name}: only zero, one, and two singular axes are currently supported"
        )

    def _log_endpoint_eps_coeff(self, sector: SectorDefinition, axis: int) -> float:
        """Return c for a supported endpoint factor y^(-1+c eps)."""
        endpoint_power = self.topology.endpoint_power(sector, axis)
        if abs(endpoint_power.base + 1.0) > 1.0e-12:
            raise ValueError(
                f"{sector.name}: unsupported endpoint power y^({endpoint_power.as_text()}); "
                "only logarithmic y^(-1+c*eps) factors are implemented"
            )
        if abs(endpoint_power.eps_coeff) <= 1.0e-15:
            raise ValueError(
                f"{sector.name}: endpoint power y^({endpoint_power.as_text()}) has no epsilon regulator"
            )
        return endpoint_power.eps_coeff

    def _check_supported_singular_powers(self, sector: SectorDefinition) -> None:
        """Ensure declared endpoint powers are logarithmic after extraction."""
        for axis in sector.singular_axes:
            self._log_endpoint_eps_coeff(sector, axis)

    def _phi(self, sector: SectorDefinition, y: list[float] | tuple[float, ...]) -> complex:
        """Scalar residual evaluator kept for legacy/debug comparisons."""
        axes = sector.singular_axes
        if all(float(y[axis]) > self.boundary_tol for axis in axes):
            f = self.topology.f_value(sector.map_eval(y))
            return f / complex(sector.f_monomial_value(y))

        taylor = self.topology.f_taylor(sector, y)
        multi_index: list[int] = []
        denominator = 1.0
        for axis in axes:
            power = sector.f_monomial_powers[axis]
            if float(y[axis]) <= self.boundary_tol:
                multi_index.append(power)
            else:
                multi_index.append(0)
                denominator *= float(y[axis]) ** power
        return taylor[tuple(multi_index)] / denominator

    def _g_coeffs(
        self, sector: SectorDefinition, y: list[float] | tuple[float, ...]
    ) -> tuple[complex, complex, complex]:
        """Scalar regular-function coefficient builder."""
        x = sector.map_eval(y)
        u = self.topology.u_value(x)
        if any(sector.u_monomial_powers):
            raise NotImplementedError("scalar debug path does not implement U residual limits")
        phi = self._phi(sector, y)
        regular_j = complex(sector.jacobian_eval(y))
        pref = regular_j * (u ** self.topology.u_power_base) * (phi ** (-self.topology.f_power_base))
        exponent_log = (
            self.topology.eps_log_u_coeff * feynman_log(u)
            + self.topology.eps_log_f_coeff * feynman_log(phi)
        )
        return (pref, pref * exponent_log, 0.5 * pref * exponent_log * exponent_log)

    def _with_axis_value(
        self, y: list[float] | tuple[float, ...], axis: int, value: float
    ) -> list[float]:
        """Return a copy of a point with one coordinate replaced."""
        out = [float(coord) for coord in y]
        out[axis] = value
        return out

    def _one_axis_subtraction(
        self, sector: SectorDefinition, y: list[float] | tuple[float, ...]
    ) -> tuple[list[complex], float]:
        """Scalar one-axis subtraction kept for legacy/debug comparisons."""
        self._check_supported_singular_powers(sector)
        axis = sector.singular_axes[0]
        eps_coeff = self._log_endpoint_eps_coeff(sector, axis)
        coord = float(y[axis])

        g_y = self._g_coeffs(sector, y)
        y0 = self._with_axis_value(y, axis, 0.0)
        g_0 = self._g_coeffs(sector, y0)

        coeff_m2 = 0.0 + 0.0j
        coeff_m1 = g_0[0] / eps_coeff
        coeff_0 = g_0[1] / eps_coeff + (g_y[0] - g_0[0]) / coord
        coeffs = [coeff_m2, coeff_m1, coeff_0]
        return coeffs, complex_abs_for_training(coeffs[self.topology.training_index])

    def _two_axis_subtraction(
        self, sector: SectorDefinition, y: list[float] | tuple[float, ...]
    ) -> tuple[list[complex], float]:
        """Scalar two-axis subtraction kept for legacy/debug comparisons."""
        self._check_supported_singular_powers(sector)
        axis_a, axis_b = sector.singular_axes
        eps_a = self._log_endpoint_eps_coeff(sector, axis_a)
        eps_b = self._log_endpoint_eps_coeff(sector, axis_b)
        ya = float(y[axis_a])
        yb = float(y[axis_b])

        y_a0 = self._with_axis_value(y, axis_b, 0.0)
        y_0b = self._with_axis_value(y, axis_a, 0.0)
        y_00 = self._with_axis_value(y_0b, axis_b, 0.0)

        g_ab = self._g_coeffs(sector, y)
        g_a0 = self._g_coeffs(sector, y_a0)
        g_0b = self._g_coeffs(sector, y_0b)
        g_00 = self._g_coeffs(sector, y_00)

        remainder0 = g_ab[0] - g_0b[0] - g_a0[0] + g_00[0]
        edge_b0 = g_0b[0] - g_00[0]
        edge_b1 = g_0b[1] - g_00[1]
        edge_a0 = g_a0[0] - g_00[0]
        edge_a1 = g_a0[1] - g_00[1]

        coeff_m2 = g_00[0] / (eps_a * eps_b)
        coeff_m1 = (
            g_00[1] / (eps_a * eps_b)
            + edge_b0 / (eps_a * yb)
            + edge_a0 / (eps_b * ya)
        )
        coeff_0 = (
            g_00[2] / (eps_a * eps_b)
            + (edge_b1 + eps_b * edge_b0 * math.log(yb)) / (eps_a * yb)
            + (edge_a1 + eps_a * edge_a0 * math.log(ya)) / (eps_b * ya)
            + remainder0 / (ya * yb)
        )

        coeffs = [coeff_m2, coeff_m1, coeff_0]
        return coeffs, complex_abs_for_training(coeffs[self.topology.training_index])
