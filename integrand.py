"""Topology definitions and generic black-box sector processing.

The only symbolic expressions stored here are the topology-level U and F
polynomials used to build Symbolica evaluators and to print summaries.  The
``SectorProcessor`` never substitutes sector maps into U/F symbolically; it
only evaluates prepared sector callbacks and U/F callbacks on numeric batches.
"""

from __future__ import annotations

import copy
import cmath
from itertools import product
import math
from dataclasses import dataclass, field
from decimal import Decimal
import time
from typing import Any

import numpy as np
from symbolica import E, S

from definitions import EpsilonExpansion, HotPathTiming, IntegralRequest, ParametricRepresentation
from sectors_generator import SectorDefinition
from subtraction_formula import (
    build_endpoint_projector_formula_symbolica,
    build_subtraction_formula_symbolica,
)
from utils import decimal_complex_with_precision, decimal_with_precision


ComplexPrecise = tuple[Any, Any]
ComplexPreciseRow = list[ComplexPrecise]


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
    parametric_representation: ParametricRepresentation | None = None
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
    dual_evaluator_build_seconds: float = field(default=0.0, init=False)
    subtraction_formula_build_seconds: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Build the scalar U and F evaluators in the declared row order."""
        allowed_modes = {"pregenerate", "lazy", "single-overall", "symbolic-derivatives"}
        if self.dual_evaluator_mode not in allowed_modes:
            raise ValueError(
                f"{self.family}: unsupported dual evaluator mode {self.dual_evaluator_mode!r}"
            )
        params = [S(name) for name in [*self.x_names, *self.parameter_names]]
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
            values = evaluator.evaluate(rows)
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
        values = evaluator.evaluate_complex(rows)
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
    ) -> Any:
        """Return a cached clone of a scalar evaluator dualized to ``dual_shape``."""
        key = tuple(dual_shape)
        evaluator = cache.get(key)
        if evaluator is None:
            start = time.perf_counter()
            evaluator = copy.copy(scalar_evaluator)
            evaluator.dualize([list(mi) for mi in dual_shape])
            cache[key] = evaluator
            self.dual_evaluator_build_seconds += time.perf_counter() - start
        return evaluator

    def u_dual_evaluator(self, dual_shape: list[tuple[int, ...]]) -> Any:
        """Return a cached dualized U evaluator for the requested jet shape."""
        return self._cached_dual_evaluator(self._u_dual_evaluators, self._u_evaluator, dual_shape)

    def f_dual_evaluator(self, dual_shape: list[tuple[int, ...]]) -> Any:
        """Return a cached dualized F evaluator for the requested jet shape."""
        # The heavy expression-to-evaluator lowering was already done in
        # __post_init__.  Symbolica evaluators support shallow copying, so we
        # clone the boot-time scalar evaluator and dualize the clone.
        return self._cached_dual_evaluator(self._f_dual_evaluators, self._f_evaluator, dual_shape)

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
                sector.prepare_dual_evaluators_for_shape(envelope)
                if progress is not None and (index % 25 == 0 or index == len(dim_sectors)):
                    progress.update(
                        index,
                        total=total,
                        detail=f"overall sector duals done {index}/{len(dim_sectors)}",
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
            if progress is not None and (index == 1 or index % 25 == 0 or index == total):
                progress.update(
                    offset + index - 1,
                    total=max(offset + total, total),
                    detail=f"sector duals {sector.name} {index}/{total}",
                )
            sector.prepare_dual_evaluators_for_shape(sector.dual_shape)
            if progress is not None and (index % 25 == 0 or index == total):
                progress.update(
                    offset + index,
                    total=max(offset + total, total),
                    detail=f"sector duals {sector.name} done {index}/{total}",
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
        formula = self._subtraction_formulas.get(signature)
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
        for sector in sectors:
            if not sector.singular_axes:
                continue
            signature = self.endpoint_projector_signature(sector)
            if signature in self._endpoint_projector_formulas or signature in seen:
                continue
            seen.add(signature)
            pending.append((sector, signature))

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
                    detail=f"{len(pending)} endpoint signature(s)",
                )

    def endpoint_projector_signature(self, sector: SectorDefinition) -> tuple[Any, ...]:
        """Return the cache key for the endpoint-only projector formula."""
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
        return (
            len(sector.singular_axes),
            tuple(endpoint_powers),
            tuple(taylor_orders),
            tuple(self.laurent_orders),
        )

    def endpoint_projector_formula_for(
        self,
        sector: SectorDefinition,
    ) -> "EndpointProjectorFormulaDefinition":
        """Return a pregenerated endpoint projector formula or fail clearly."""
        signature = self.endpoint_projector_signature(sector)
        formula = self._endpoint_projector_formulas.get(signature)
        if formula is None:
            raise RuntimeError(
                f"{sector.name}: missing pregenerated endpoint projector formula; "
                "call TopologyDefinition.prepare_endpoint_projector_formulas(...) before integration"
            )
        return formula

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
            cache = self._u_derivative_evaluators
            indices_by_order = self._u_derivative_indices_by_order
        elif polynomial == "f":
            expr = self.f_expr
            cache = self._f_derivative_evaluators
            indices_by_order = self._f_derivative_indices_by_order
        else:
            raise ValueError(f"{self.family}: unknown polynomial {polynomial!r}")

        existing = indices_by_order.get(max_total)
        if existing is not None and all(multi_index in cache for multi_index in existing):
            return

        params = [S(name) for name in [*self.x_names, *self.parameter_names]]
        prepared: list[tuple[int, ...]] = []
        start = time.perf_counter()
        for multi_index in self._candidate_derivative_multi_indices(expr, max_total):
            if multi_index not in cache:
                derivative_expr = self._differentiate_expr(expr, multi_index)
                if str(derivative_expr) == "0":
                    continue
                cache[multi_index] = derivative_expr.evaluator(
                    params,
                    jit_compile=self.jit_compile_evaluators,
                )
            if multi_index in cache:
                prepared.append(multi_index)
        indices_by_order[max_total] = prepared
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
            self._prepare_symbolic_derivative_evaluators(polynomial, max_total)
        return indices_by_order[max_total]

    def _symbolic_derivative_evaluator(self, polynomial: str, multi_index: tuple[int, ...]) -> Any:
        """Return a prepared evaluator for one x-space symbolic derivative."""
        cache = self._u_derivative_evaluators if polynomial == "u" else self._f_derivative_evaluators
        return cache[multi_index]

    def _derivative_values_batch(
        self,
        polynomial: str,
        x_values: np.ndarray,
        derivative_indices: list[tuple[int, ...]],
        timing: HotPathTiming | None,
    ) -> dict[tuple[int, ...], np.ndarray]:
        """Evaluate shared symbolic derivative callbacks at mapped x-points."""
        rows = self._rows(x_values)
        values: dict[tuple[int, ...], np.ndarray] = {}
        for multi_index in derivative_indices:
            evaluator = self._symbolic_derivative_evaluator(polynomial, multi_index)
            values[multi_index] = np.asarray(
                self._timed_evaluate(evaluator, rows, timing),
                dtype=np.complex128,
            )[:, 0]
        return values

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
        if not sector.dual_shape:
            raise ValueError(f"{sector.name}: no dual shape declared")

        rows_in = np.asarray(y_values, dtype=float)
        n_rows = rows_in.shape[0]
        rank = len(sector.dual_shape[0])
        max_orders = [
            max(multi_index[position] for multi_index in sector.dual_shape)
            for position in range(rank)
        ]
        max_total = sum(max_orders)
        derivative_indices = self._symbolic_derivative_indices(polynomial, max_total)

        x_jets = sector.map_dual_eval_batch_for_shape(rows_in, sector.dual_shape, timing)
        zero = _zero_multi(rank)
        zero_column = sector.dual_index(zero)
        x0 = x_jets[:, :, zero_column]
        derivative_values = self._derivative_values_batch(polynomial, x0, derivative_indices, timing)

        h_series: list[MultiSeries] = []
        for x_index in range(len(self.x_names)):
            series: MultiSeries = {}
            for column, multi_index in enumerate(sector.dual_shape):
                if multi_index == zero:
                    continue
                values = x_jets[:, x_index, column].astype(np.complex128, copy=False)
                if np.any(values):
                    series[multi_index] = values
            h_series.append(series)

        power_cache: dict[tuple[int, int], MultiSeries] = {}

        def h_power(x_index: int, power: int) -> MultiSeries:
            key = (x_index, power)
            cached = power_cache.get(key)
            if cached is not None:
                return cached
            if power == 0:
                cached = _series_constant(1.0 + 0.0j, max_orders, n_rows)
            elif power == 1:
                cached = h_series[x_index]
            else:
                cached = _series_mul(h_power(x_index, power - 1), h_series[x_index], max_orders)
            power_cache[key] = cached
            return cached

        composed: MultiSeries = {}
        for x_multi_index in derivative_indices:
            derivative = derivative_values[x_multi_index]
            factorial = 1
            for order in x_multi_index:
                factorial *= math.factorial(int(order))
            term = _series_constant(derivative / float(factorial), max_orders, n_rows)
            for x_index, power in enumerate(x_multi_index):
                if power:
                    term = _series_mul(term, h_power(x_index, int(power)), max_orders)
                    if not term:
                        break
            if term:
                composed = _series_add(composed, term)

        return np.stack(
            [
                _series_coefficient(composed, multi_index, n_rows)
                for multi_index in sector.dual_shape
            ],
            axis=1,
        )

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
            values = self._symbolic_derivative_taylor_batch(
                sector,
                np.asarray([y], dtype=float),
                "f",
                timing,
            )[0]
            return [_decimal_complex(value, precision_digits) for value in values]
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
            values = self._symbolic_derivative_taylor_batch(
                sector,
                np.asarray([y], dtype=float),
                "u",
                timing,
            )[0]
            return [_decimal_complex(value, precision_digits) for value in values]
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


def _multi_indices(max_orders: list[int]) -> list[tuple[int, ...]]:
    """Enumerate all Taylor multi-indices inside the requested truncation box."""
    if not max_orders:
        return [()]
    return [tuple(mi) for mi in product(*[range(order + 1) for order in max_orders])]


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


def _series_scale(a: MultiSeries, factor: float | complex | np.ndarray) -> MultiSeries:
    """Scale every coefficient by a scalar or per-row array."""
    return {key: value * factor for key, value in a.items()}


def _series_mul(a: MultiSeries, b: MultiSeries, max_orders: list[int]) -> MultiSeries:
    """Multiply two truncated sparse Taylor series."""
    dim = len(max_orders)
    out: MultiSeries = {}
    for key_a, value_a in a.items():
        for key_b, value_b in b.items():
            key = tuple(key_a[i] + key_b[i] for i in range(dim))
            if any(key[i] > max_orders[i] for i in range(dim)):
                continue
            term = value_a * value_b
            out[key] = out[key] + term if key in out else term.copy()
    return out


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


def _series_pow_real(a: MultiSeries, power: float, max_orders: list[int], n_rows: int) -> MultiSeries:
    """Raise a regular Taylor series to a real power using log/exp."""
    if abs(power) <= 1.0e-15:
        return _series_constant(1.0 + 0.0j, max_orders, n_rows)
    return _series_exp(_series_scale(_series_log(a, max_orders, n_rows), power), max_orders, n_rows)


def _series_coefficient(series: MultiSeries, multi_index: tuple[int, ...], n_rows: int) -> np.ndarray:
    """Return one Taylor coefficient, or zeros if it is absent."""
    value = series.get(multi_index)
    if value is None:
        return np.zeros(n_rows, dtype=np.complex128)
    return value


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
    coefficient_layout: list[tuple[tuple[int, ...], tuple[int, ...], int]]
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
    return build_endpoint_projector_formula_symbolica(
        topology,
        sector,
        signature,
        EndpointProjectorFormulaDefinition,
    )


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

    def evaluate(self, sector: SectorDefinition, y: list[float] | tuple[float, ...]) -> tuple[list[complex], float]:
        """Evaluate one sector point through the batched implementation."""
        coords = np.asarray([y], dtype=float)
        coeffs, training, _ = self.evaluate_batch(sector, coords)
        return [complex(value) for value in coeffs[0]], float(training[0])

    def evaluate_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, HotPathTiming]:
        """Evaluate Laurent coefficients and training values for one sector batch."""
        rows = np.asarray(y_values, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != sector.integration_dim:
            raise ValueError(f"{sector.name}: expected coordinate array with shape (n,{sector.integration_dim})")

        if (
            not sector.singular_axes
            or self.stability_threshold <= 0.0
            or rows.shape[0] == 0
        ):
            timing = HotPathTiming()
            timing.add_precision_samples(ordinary=rows.shape[0])
            coeffs, training = self._evaluate_batch_impl(sector, rows, timing)
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
            coeffs, training = self._evaluate_batch_impl(sector, rows, timing)
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
            chunk_coeffs, chunk_training = self._evaluate_batch_impl(
                sector,
                rows[mask],
                chunk_timing,
            )
            timing.absorb(chunk_timing)
            coeffs[mask] = chunk_coeffs
            training[mask] = chunk_training
        if np.any(high_mask):
            chunk_timing = HotPathTiming(precision_digits=self.high_precision_stability_precision)
            chunk_size = int(np.count_nonzero(high_mask))
            chunk_timing.add_precision_samples(high=chunk_size)
            chunk_coeffs, chunk_training = self._evaluate_batch_impl(
                sector,
                rows[high_mask],
                chunk_timing,
            )
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
            coeffs = formula.evaluate_complex_batch(input_matrix, timing)
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
                coeffs[row_index, :] = formula.evaluate_complex_prec(
                    input_row,
                    int(precision_digits),
                    timing,
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

        g_cache: dict[tuple[int, ...], list[MultiSeries]] = {}
        for subset, multi_index, regular_order in formula.coefficient_layout:
            cached = g_cache.get(subset)
            if cached is None:
                cached = self._g_taylor_eps_series_batch(
                    sector,
                    sample_rows,
                    set(subset),
                    formula.taylor_orders,
                    timing,
                )
                g_cache[subset] = cached
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

    def _endpoint_projector_input_prec_row(
        self,
        sector: SectorDefinition,
        y: np.ndarray,
        formula: EndpointProjectorFormulaDefinition,
        precision_digits: int,
        timing: HotPathTiming,
    ) -> ComplexPreciseRow:
        """Assemble one multiprecision endpoint-projector input row.

        The current regular-coefficient layer is still shared with the
        recursive backend and returns complex doubles.  The endpoint projector
        itself receives padded Decimal inputs, so the unstable inclusion-
        exclusion algebra is evaluated by Symbolica at the requested precision.
        """
        matrix = self._endpoint_projector_input_matrix(
            sector,
            np.asarray([y], dtype=float),
            formula,
            timing,
        )
        return [_decimal_complex(value, precision_digits) for value in matrix[0]]

    def _endpoint_projector_subtraction_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a singular sector through a reusable endpoint projector."""
        formula = self.topology.endpoint_projector_formula_for(sector)
        rows = np.asarray(y_values, dtype=float)
        precision_digits = timing.precision_digits
        if precision_digits is None:
            input_matrix = self._endpoint_projector_input_matrix(
                sector,
                rows,
                formula,
                timing,
            )
            coeffs = formula.evaluate_complex_batch(input_matrix, timing)
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
                coeffs[row_index, :] = formula.evaluate_complex_prec(
                    input_row,
                    int(precision_digits),
                    timing,
                )
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
        n_rows = rows.shape[0]
        axes = list(sector.singular_axes)
        axis_position = {axis: position for position, axis in enumerate(axes)}
        taylor = taylor_batch(sector, rows, timing)
        series: MultiSeries = {}
        for residual_multi in _multi_indices(max_orders):
            polynomial_multi = [0 for _ in axes]
            denominator = np.ones(n_rows, dtype=float)
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
                    polynomial_multi[position] = int(power) + int(residual_multi[position])
                else:
                    if power:
                        denominator *= rows[:, axis] ** int(power)
            series[residual_multi] = (
                taylor[:, sector.dual_index(tuple(polynomial_multi))] / denominator
            )
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

    def _g_taylor_eps_series_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        zero_positions: set[int],
        taylor_orders: list[int],
        timing: HotPathTiming,
    ) -> list[MultiSeries]:
        """Taylor-expand the regular function ``g_s(y,eps)`` at endpoints.

        The returned list is indexed by the non-negative epsilon order.  Each
        entry is a sparse Taylor series in the declared singular variables.
        """
        rows = np.asarray(y_values, dtype=float)
        n_rows = rows.shape[0]
        axes = list(sector.singular_axes)
        max_orders = [
            int(taylor_orders[position]) if position in zero_positions else 0
            for position in range(len(axes))
        ]
        endpoint_rows = rows.copy()
        for position in zero_positions:
            endpoint_rows[:, axes[position]] = 0.0

        if not zero_positions and not any(max_orders):
            coeffs = self._g_coeffs_batch(sector, endpoint_rows, timing)
            return [
                _series_constant(coeffs[:, order], max_orders, n_rows)
                for order in range(self.topology.coefficient_count)
            ]

        jacobian_series = self._jacobian_taylor_series_batch(
            sector, endpoint_rows, max_orders, timing
        )
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

        out: list[MultiSeries] = []
        log_power = _series_constant(1.0 + 0.0j, max_orders, n_rows)
        factorial = 1.0
        for order in range(self.topology.coefficient_count):
            if order > 0:
                factorial *= float(order)
                log_power = _series_mul(log_power, log_series, max_orders)
            out.append(
                _series_scale(
                    _series_mul(pref_series, log_power, max_orders),
                    1.0 / factorial,
                )
            )
        return out

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
