"""Declarative sector definitions for the supported triangle and box cases.

The sector generator is the only place where the current prototype hard-codes
maps, Jacobians, endpoint monomials, and subtraction axes.  It deliberately does
not touch the Symanzik polynomials: it only prepares data and evaluators that
the generic processor can later combine with black-box U/F callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from itertools import product
import math
import time
from typing import Any

import numpy as np
from symbolica import E, S

from definitions import HotPathTiming, IntegralRequest
from utils import decimal_complex_with_precision, decimal_with_precision


def _decimal_real(value: Any, precision_digits: int) -> Decimal:
    """Convert a sampled real scalar into a Decimal at evaluator precision."""
    return decimal_with_precision(value, precision_digits)


def _decimal_complex(value: Any, precision_digits: int) -> tuple[Decimal, Decimal]:
    """Return the complex tuple shape required by Symbolica multiprecision."""
    return decimal_complex_with_precision(value, precision_digits)


def _expr(text: str) -> Any:
    """Build a Symbolica expression from a compact string."""
    return E(text)


def _symbols(names: list[str]) -> list[Any]:
    """Build Symbolica symbols in the order expected by evaluator rows."""
    return [S(name) for name in names]


def _monomial_expr(variable_names: list[str], powers: list[int]) -> Any:
    """Create the display expression for a declared endpoint monomial."""
    factors: list[str] = []
    for name, power in zip(variable_names, powers):
        if power == 0:
            continue
        if power == 1:
            factors.append(name)
        else:
            factors.append(f"{name}^{power}")
    return _expr("*".join(factors) if factors else "1")


def dual_shape_from_powers(powers: list[int]) -> list[tuple[int, ...]]:
    """Return all derivative multi-indices required by the declared powers."""
    if not powers:
        return []
    return [tuple(mi) for mi in product(*[range(power + 1) for power in powers])]


def _parse_numeric_coefficient(value: Any) -> complex | None:
    """Parse a Symbolica polynomial coefficient if it is a plain number."""
    text = str(value).strip()
    try:
        return complex(float(Fraction(text)))
    except Exception:
        pass
    try:
        return complex(float(text))
    except Exception:
        return None


def _monomial_data(expr: Any, variable_names: list[str]) -> tuple[complex, list[int]] | None:
    """Return ``(coefficient, powers)`` when ``expr`` is one numeric monomial.

    pySecDec DOT sectors recovered from identity Feynman parameters are
    monomial maps.  Recognizing that declaratively lets the symbolic-derivative
    runtime obtain map and Jacobian Taylor jets by a direct binomial formula,
    avoiding heavy Symbolica dualization of large endpoint shapes.
    """
    variables = _symbols(variable_names)
    try:
        polynomial = expr.to_polynomial(vars=variables)
        terms = polynomial.coefficient_list(vars=variables)
    except Exception:
        return None
    if len(terms) != 1:
        return None
    powers, coefficient = terms[0]
    coeff_value = _parse_numeric_coefficient(coefficient)
    if coeff_value is None:
        return None
    int_powers = [int(power) for power in powers]
    if any(power < 0 for power in int_powers):
        return None
    return coeff_value, int_powers


def _pow_decimal(value: Decimal, power: int) -> Decimal:
    """Integer Decimal power with the ``0^0 = 1`` Taylor convention."""
    if power == 0:
        return Decimal(1)
    return value ** int(power)


@dataclass
class SectorDefinition:
    """Prepared sector map plus endpoint-subtraction metadata."""

    name: str
    integration_dim: int
    variable_names: list[str]
    map_exprs: list[Any]
    regular_jacobian_expr: Any
    f_monomial_powers: list[int]
    jacobian_monomial_powers: list[int]
    singular_axes: list[int]
    subtraction_type: str
    description: str
    jit_compile_evaluators: bool = False
    u_monomial_powers: list[int] | None = None
    measure_monomial_powers: list[float] | None = None
    numerator_monomial_powers: list[float] | None = None
    endpoint_taylor_orders: list[int] | None = None
    f_monomial_expr: Any = field(init=False)
    u_monomial_expr: Any = field(init=False)
    dual_shape: list[tuple[int, ...]] = field(init=False)
    _map_evaluators: list[Any] = field(default_factory=list, init=False, repr=False)
    _jacobian_evaluator: Any | None = field(default=None, init=False, repr=False)
    _jacobian_dual_evaluator: Any | None = field(init=False, repr=False)
    _jacobian_dual_evaluators_by_shape: dict[tuple[tuple[int, ...], ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _map_dual_evaluators: list[Any] = field(default_factory=list, init=False, repr=False)
    _map_dual_evaluators_by_shape: dict[tuple[tuple[int, ...], ...], list[Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _dual_index_by_multi_index: dict[tuple[int, ...], int] = field(init=False, repr=False)
    _evaluators_prepared: bool = field(default=False, init=False, repr=False)
    _map_monomials: list[tuple[complex, list[int]] | None] = field(
        default_factory=list, init=False, repr=False
    )
    _jacobian_monomial: tuple[complex, list[int]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the declarative sector metadata."""
        if len(self.variable_names) != self.integration_dim:
            raise ValueError(f"{self.name}: variable_names/integration_dim mismatch")
        if len(self.f_monomial_powers) != self.integration_dim:
            raise ValueError(f"{self.name}: f_monomial_powers has wrong length")
        if len(self.jacobian_monomial_powers) != self.integration_dim:
            raise ValueError(f"{self.name}: jacobian_monomial_powers has wrong length")
        if len(self.map_exprs) == 0:
            raise ValueError(f"{self.name}: empty sector map")
        if self.u_monomial_powers is None:
            self.u_monomial_powers = [0 for _ in range(self.integration_dim)]
        if self.measure_monomial_powers is None:
            self.measure_monomial_powers = [0.0 for _ in range(self.integration_dim)]
        if self.numerator_monomial_powers is None:
            self.numerator_monomial_powers = [0.0 for _ in range(self.integration_dim)]
        if self.endpoint_taylor_orders is None:
            self.endpoint_taylor_orders = [0 for _ in range(self.integration_dim)]
        if len(self.u_monomial_powers) != self.integration_dim:
            raise ValueError(f"{self.name}: u_monomial_powers has wrong length")
        if len(self.measure_monomial_powers) != self.integration_dim:
            raise ValueError(f"{self.name}: measure_monomial_powers has wrong length")
        if len(self.numerator_monomial_powers) != self.integration_dim:
            raise ValueError(f"{self.name}: numerator_monomial_powers has wrong length")
        if len(self.endpoint_taylor_orders) != self.integration_dim:
            raise ValueError(f"{self.name}: endpoint_taylor_orders has wrong length")
        if any(order < 0 for order in self.endpoint_taylor_orders):
            raise ValueError(f"{self.name}: endpoint_taylor_orders must be non-negative")

        self.f_monomial_expr = _monomial_expr(self.variable_names, self.f_monomial_powers)
        self.u_monomial_expr = _monomial_expr(self.variable_names, self.u_monomial_powers)
        # The docs describe each endpoint sector by a known monomial M_s(y).
        # The dual shape is exactly the set of Taylor coefficients needed to
        # recover U(X_s(y))/M_U(y) or F(X_s(y))/M_F(y) when one or more
        # monomial variables vanish.  Overall-dual mode may request a larger
        # envelope shape, but the sector's native declaration remains minimal.
        self.dual_shape = dual_shape_from_powers(
            [
                max(self.u_monomial_powers[axis], self.f_monomial_powers[axis])
                + self.endpoint_taylor_orders[axis]
                for axis in self.singular_axes
            ]
        )
        self._jacobian_dual_evaluator = None
        self._dual_index_by_multi_index = {multi_index: i for i, multi_index in enumerate(self.dual_shape)}
        self._map_monomials = [_monomial_data(expr, self.variable_names) for expr in self.map_exprs]
        self._jacobian_monomial = _monomial_data(self.regular_jacobian_expr, self.variable_names)

    def prepare_evaluators(self, include_dual: bool = True) -> None:
        """Build map/Jacobian Symbolica callbacks for this sector.

        Sector conversion from pySecDec is deliberately declarative.  This
        method marks the boundary where Symbolica evaluator generation starts,
        so timing reports can separate sector construction from callback
        lowering.  Repeated calls are no-ops.
        """
        if self._evaluators_prepared:
            return
        params = _symbols(self.variable_names)
        # Runtime map/Jacobian evaluation is done through generated callbacks.
        # These expressions are never substituted into the U/F expressions.
        self._map_evaluators = [
            expr.evaluator(params, jit_compile=self.jit_compile_evaluators)
            for expr in self.map_exprs
        ]
        self._jacobian_evaluator = self.regular_jacobian_expr.evaluator(
            params, jit_compile=self.jit_compile_evaluators
        )
        self._map_dual_evaluators = []
        self._map_dual_evaluators_by_shape = {}
        self._jacobian_dual_evaluators_by_shape = {}
        self._jacobian_dual_evaluator = None
        self._evaluators_prepared = True
        if include_dual:
            self.prepare_dual_evaluators_for_shape(self.dual_shape)

    def _ensure_evaluators(self) -> None:
        """Prepare callbacks for direct unit-test construction paths."""
        if not self._evaluators_prepared:
            self.prepare_evaluators()

    def _timed_evaluate(self, evaluator: Any, rows: np.ndarray, timing: HotPathTiming | None) -> Any:
        """Evaluate a Symbolica callback and optionally charge it to EvalT."""
        precision_digits = None if timing is None else timing.precision_digits
        start = time.perf_counter()
        if precision_digits is None:
            values = evaluator.evaluate(rows)
        else:
            # evaluate_with_prec currently accepts one input row at a time.  It
            # is intentionally used only for small near-boundary subsets.
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

    def _timed_evaluate_complex(self, evaluator: Any, rows: np.ndarray, timing: HotPathTiming | None) -> Any:
        """Evaluate a Symbolica callback with native complex inputs."""
        start = time.perf_counter()
        values = evaluator.evaluate_complex(rows)
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return values

    def _timed_evaluate_complex_with_prec(
        self,
        evaluator: Any,
        row: list[tuple[Any, Any]],
        precision_digits: int,
        timing: HotPathTiming | None,
    ) -> list[tuple[Any, Any]]:
        """Evaluate one complex row with Symbolica multiprecision arithmetic."""
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

    def map_eval(self, y: list[float] | tuple[float, ...]) -> list[float]:
        """Evaluate the sector map at one point."""
        self._ensure_evaluators()
        row = [float(value) for value in y]
        return [float(evaluator.evaluate([row])[0][0]) for evaluator in self._map_evaluators]

    def map_eval_batch(
        self,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate all mapped Feynman parameters for a batch of points."""
        self._ensure_evaluators()
        rows = np.asarray(y_values, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != self.integration_dim:
            raise ValueError(f"{self.name}: expected coordinate array with shape (n,{self.integration_dim})")
        columns = [
            np.asarray(self._timed_evaluate(evaluator, rows, timing), dtype=float)[:, 0]
            for evaluator in self._map_evaluators
        ]
        return np.stack(columns, axis=1)

    def jacobian_eval(self, y: list[float] | tuple[float, ...]) -> float:
        """Evaluate the regular Jacobian at one point."""
        self._ensure_evaluators()
        if self._jacobian_evaluator is None:
            raise ValueError(f"{self.name}: missing Jacobian evaluator")
        row = [float(value) for value in y]
        return float(self._jacobian_evaluator.evaluate([row])[0][0])

    def jacobian_eval_batch(
        self,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate the regular Jacobian for a batch."""
        self._ensure_evaluators()
        if self._jacobian_evaluator is None:
            raise ValueError(f"{self.name}: missing Jacobian evaluator")
        rows = np.asarray(y_values, dtype=float)
        values = self._timed_evaluate(self._jacobian_evaluator, rows, timing)
        return np.asarray(values, dtype=float)[:, 0]

    def jacobian_taylor_batch(
        self,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate Taylor coefficients of the regular Jacobian.

        The regular Jacobian is sector data, not part of the black-box U/F
        topology.  For higher-order endpoint subtractions it must be Taylor
        expanded alongside the U/F residuals, so the same dual-shape convention
        is used for its generated evaluator.
        """
        return self.jacobian_taylor_batch_for_shape(y_values, self.dual_shape, timing)

    def jacobian_taylor_batch_for_shape(
        self,
        y_values: np.ndarray,
        dual_shape: list[tuple[int, ...]],
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate regular-Jacobian Taylor coefficients for an explicit shape."""
        self._ensure_evaluators()
        rows_in = np.asarray(y_values, dtype=float)
        if not dual_shape:
            return self.jacobian_eval_batch(rows_in, timing)[:, np.newaxis]
        monomial_values = self._monomial_taylor_batch(
            [self._jacobian_monomial],
            rows_in,
            dual_shape,
            dtype=np.complex128,
        )
        if monomial_values is not None:
            return monomial_values[:, 0, :]
        if dual_shape == self.dual_shape:
            evaluator = self._jacobian_dual_evaluator
            if evaluator is None:
                evaluator = self.prepare_jacobian_dual_evaluator(dual_shape)
        else:
            evaluator = self.prepare_jacobian_dual_evaluator(dual_shape)
        if evaluator is None:
            raise ValueError(f"{self.name}: missing dualized Jacobian evaluator")
        n_rows = rows_in.shape[0]
        dual_len = len(dual_shape)
        dual_rank = len(dual_shape[0])
        zero_mi = tuple(0 for _ in range(dual_rank))
        unit_by_axis = {
            axis: tuple(1 if i == pos else 0 for i in range(len(self.singular_axes)))
            for pos, axis in enumerate(self.singular_axes)
            if pos < dual_rank
        }
        rows = np.zeros((n_rows, self.integration_dim * dual_len), dtype=float)
        for axis in range(self.integration_dim):
            unit = unit_by_axis.get(axis)
            offset = axis * dual_len
            for j, mi in enumerate(dual_shape):
                if mi == zero_mi:
                    rows[:, offset + j] = rows_in[:, axis]
                elif unit is not None and mi == unit:
                    rows[:, offset + j] = 1.0
        values = self._timed_evaluate(evaluator, rows, timing)
        return np.asarray(values, dtype=np.complex128)

    def jacobian_taylor_complex_batch(
        self,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate Jacobian Taylor coefficients through the complex API."""
        self._ensure_evaluators()
        rows_in = np.asarray(y_values, dtype=float)
        if not self.dual_shape:
            return self.jacobian_eval_batch(rows_in, timing)[:, np.newaxis].astype(np.complex128)
        monomial_values = self._monomial_taylor_batch(
            [self._jacobian_monomial],
            rows_in,
            self.dual_shape,
            dtype=np.complex128,
        )
        if monomial_values is not None:
            return monomial_values[:, 0, :]
        evaluator = self._jacobian_dual_evaluator
        if evaluator is None:
            evaluator = self.prepare_jacobian_dual_evaluator(self.dual_shape)
        rows = self._dual_input_matrix(rows_in, self.dual_shape).astype(np.complex128)
        values = self._timed_evaluate_complex(evaluator, rows, timing)
        return np.asarray(values, dtype=np.complex128)

    def jacobian_taylor_complex_prec(
        self,
        y: np.ndarray,
        precision_digits: int,
        timing: HotPathTiming | None = None,
    ) -> list[tuple[Any, Any]]:
        """Evaluate one Jacobian Taylor row with complex multiprecision."""
        self._ensure_evaluators()
        if not self.dual_shape:
            if self._jacobian_evaluator is None:
                raise ValueError(f"{self.name}: missing Jacobian evaluator")
            row = [_decimal_complex(value, precision_digits) for value in np.asarray(y, dtype=float)]
            return self._timed_evaluate_complex_with_prec(
                self._jacobian_evaluator,
                row,
                precision_digits,
                timing,
            )
        monomial_values = self._monomial_taylor_prec(
            [self._jacobian_monomial],
            np.asarray(y, dtype=float),
            self.dual_shape,
            precision_digits,
        )
        if monomial_values is not None:
            return monomial_values[0]
        evaluator = self._jacobian_dual_evaluator
        if evaluator is None:
            evaluator = self.prepare_jacobian_dual_evaluator(self.dual_shape)
        row = self._dual_input_prec_row(np.asarray(y, dtype=float), self.dual_shape, precision_digits)
        return self._timed_evaluate_complex_with_prec(evaluator, row, precision_digits, timing)

    def f_monomial_value(self, y: list[float] | tuple[float, ...]) -> float:
        """Evaluate the declared F monomial at one point."""
        value = 1.0
        for coord, power in zip(y, self.f_monomial_powers):
            if power:
                value *= float(coord) ** power
        return value

    def f_monomial_value_batch(self, y_values: np.ndarray) -> np.ndarray:
        """Evaluate the declared F monomial for a batch."""
        rows = np.asarray(y_values, dtype=float)
        values = np.ones(rows.shape[0], dtype=float)
        for axis, power in enumerate(self.f_monomial_powers):
            if power:
                values *= rows[:, axis] ** power
        return values

    def map_dual_eval(self, y: list[float] | tuple[float, ...]) -> list[list[float]]:
        """Evaluate sector-map dual jets for one endpoint point."""
        return self.map_dual_eval_for_shape(y, self.dual_shape)

    def map_dual_eval_for_shape(
        self,
        y: list[float] | tuple[float, ...],
        dual_shape: list[tuple[int, ...]],
    ) -> list[list[float]]:
        """Evaluate sector-map dual jets using an explicit dual shape."""
        self._ensure_evaluators()
        if not dual_shape:
            return [[value] for value in self.map_eval(y)]
        monomial_values = self._monomial_taylor_batch(
            self._map_monomials,
            np.asarray([y], dtype=float),
            dual_shape,
            dtype=float,
        )
        if monomial_values is not None:
            return monomial_values[0].tolist()

        dual_rank = len(dual_shape[0])
        zero_mi = tuple(0 for _ in range(dual_rank))
        unit_by_axis = {
            axis: tuple(1 if i == pos else 0 for i in range(dual_rank))
            for pos, axis in enumerate(self.singular_axes)
            if pos < dual_rank
        }

        row: list[float] = []
        for axis, coord in enumerate(y):
            unit = unit_by_axis.get(axis)
            for mi in dual_shape:
                # Coordinates are encoded as dual variables only along declared
                # singular axes.  Non-singular coordinates stay ordinary
                # constants in the Taylor expansion.
                if mi == zero_mi:
                    row.append(float(coord))
                elif unit is not None and mi == unit:
                    row.append(1.0)
                else:
                    row.append(0.0)

        return [
            [float(value) for value in evaluator.evaluate([row])[0]]
            for evaluator in (
                self._map_dual_evaluators
                if dual_shape == self.dual_shape
                else self.prepare_map_dual_evaluators(dual_shape)
            )
        ]

    def map_dual_eval_batch(
        self,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate sector-map dual jets for endpoint batches."""
        return self.map_dual_eval_batch_for_shape(y_values, self.dual_shape, timing)

    def prepare_map_dual_evaluators(self, dual_shape: list[tuple[int, ...]]) -> list[Any]:
        """Build or return map evaluators for a requested dual shape."""
        if not dual_shape:
            return []
        key = tuple(dual_shape)
        existing = self._map_dual_evaluators_by_shape.get(key)
        if existing is not None:
            return existing
        params = _symbols(self.variable_names)
        evaluators: list[Any] = []
        for expr in self.map_exprs:
            evaluator = expr.evaluator(params, jit_compile=self.jit_compile_evaluators)
            evaluator.dualize([list(mi) for mi in dual_shape])
            evaluators.append(evaluator)
        self._map_dual_evaluators_by_shape[key] = evaluators
        if key == tuple(self.dual_shape):
            self._map_dual_evaluators = evaluators
        return evaluators

    def prepare_jacobian_dual_evaluator(self, dual_shape: list[tuple[int, ...]]) -> Any | None:
        """Build or return a dualized regular-Jacobian evaluator."""
        if not dual_shape:
            return None
        self._ensure_evaluators()
        key = tuple(dual_shape)
        if key == tuple(self.dual_shape) and self._jacobian_dual_evaluator is not None:
            return self._jacobian_dual_evaluator
        existing = self._jacobian_dual_evaluators_by_shape.get(key)
        if existing is not None:
            return existing
        params = _symbols(self.variable_names)
        evaluator = self.regular_jacobian_expr.evaluator(
            params,
            jit_compile=self.jit_compile_evaluators,
        )
        evaluator.dualize([list(mi) for mi in dual_shape])
        if key == tuple(self.dual_shape):
            self._jacobian_dual_evaluator = evaluator
        self._jacobian_dual_evaluators_by_shape[key] = evaluator
        return evaluator

    def prepare_dual_evaluators_for_shape(self, dual_shape: list[tuple[int, ...]]) -> None:
        """Pregenerate all sector-local dual callbacks for one shape."""
        if not dual_shape:
            return
        self.prepare_jacobian_dual_evaluator(dual_shape if dual_shape == self.dual_shape else self.dual_shape)
        self.prepare_map_dual_evaluators(dual_shape)

    def map_dual_eval_batch_for_shape(
        self,
        y_values: np.ndarray,
        dual_shape: list[tuple[int, ...]],
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate sector-map dual jets for endpoint batches with an explicit shape."""
        self._ensure_evaluators()
        rows_in = np.asarray(y_values, dtype=float)
        if not dual_shape:
            return self.map_eval_batch(rows_in, timing)[:, :, np.newaxis]
        monomial_values = self._monomial_taylor_batch(
            self._map_monomials,
            rows_in,
            dual_shape,
            dtype=float,
        )
        if monomial_values is not None:
            return monomial_values.astype(float, copy=False)

        n_rows = rows_in.shape[0]
        dual_len = len(dual_shape)
        dual_rank = len(dual_shape[0])
        zero_mi = tuple(0 for _ in range(dual_rank))
        unit_by_axis = {
            axis: tuple(1 if i == pos else 0 for i in range(dual_rank))
            for pos, axis in enumerate(self.singular_axes)
            if pos < dual_rank
        }
        rows = np.zeros((n_rows, self.integration_dim * dual_len), dtype=float)
        for axis in range(self.integration_dim):
            unit = unit_by_axis.get(axis)
            offset = axis * dual_len
            for j, mi in enumerate(dual_shape):
                # Row layout: [y0 jets][y1 jets]... in the same dual-shape
                # order later used by TopologyDefinition.f_taylor_batch.
                if mi == zero_mi:
                    rows[:, offset + j] = rows_in[:, axis]
                elif unit is not None and mi == unit:
                    rows[:, offset + j] = 1.0

        evaluators = (
            self._map_dual_evaluators
            if dual_shape == self.dual_shape
            else self.prepare_map_dual_evaluators(dual_shape)
        )
        if dual_shape == self.dual_shape and not evaluators:
            evaluators = self.prepare_map_dual_evaluators(dual_shape)
        columns = [
            np.asarray(self._timed_evaluate(evaluator, rows, timing), dtype=float)
            for evaluator in evaluators
        ]
        return np.stack(columns, axis=1)

    def _dual_input_matrix(
        self,
        rows_in: np.ndarray,
        dual_shape: list[tuple[int, ...]],
    ) -> np.ndarray:
        """Build the dual-input matrix for sector-coordinate evaluators."""
        n_rows = rows_in.shape[0]
        dual_len = len(dual_shape)
        dual_rank = len(dual_shape[0])
        zero_mi = tuple(0 for _ in range(dual_rank))
        unit_by_axis = {
            axis: tuple(1 if i == pos else 0 for i in range(dual_rank))
            for pos, axis in enumerate(self.singular_axes)
            if pos < dual_rank
        }
        rows = np.zeros((n_rows, self.integration_dim * dual_len), dtype=float)
        for axis in range(self.integration_dim):
            unit = unit_by_axis.get(axis)
            offset = axis * dual_len
            for j, mi in enumerate(dual_shape):
                if mi == zero_mi:
                    rows[:, offset + j] = rows_in[:, axis]
                elif unit is not None and mi == unit:
                    rows[:, offset + j] = 1.0
        return rows

    def _dual_input_prec_row(
        self,
        y: np.ndarray,
        dual_shape: list[tuple[int, ...]],
        precision_digits: int,
    ) -> list[tuple[Any, Any]]:
        """Build one arbitrary-precision complex dual row."""
        dual_rank = len(dual_shape[0])
        zero_mi = tuple(0 for _ in range(dual_rank))
        unit_by_axis = {
            axis: tuple(1 if i == pos else 0 for i in range(dual_rank))
            for pos, axis in enumerate(self.singular_axes)
            if pos < dual_rank
        }
        row: list[tuple[Any, Any]] = []
        for axis, coord in enumerate(np.asarray(y, dtype=float)):
            unit = unit_by_axis.get(axis)
            for mi in dual_shape:
                if mi == zero_mi:
                    row.append(_decimal_complex(coord, precision_digits))
                elif unit is not None and mi == unit:
                    row.append(
                        (
                            decimal_with_precision(1.0, precision_digits),
                            decimal_with_precision(0.0, precision_digits),
                        )
                    )
                else:
                    row.append(
                        (
                            decimal_with_precision(0.0, precision_digits),
                            decimal_with_precision(0.0, precision_digits),
                        )
                    )
        return row

    def map_dual_complex_batch_for_shape(
        self,
        y_values: np.ndarray,
        dual_shape: list[tuple[int, ...]],
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate sector-map dual jets through Symbolica's complex API."""
        self._ensure_evaluators()
        rows_in = np.asarray(y_values, dtype=float)
        if not dual_shape:
            return self.map_eval_batch(rows_in, timing)[:, :, np.newaxis].astype(np.complex128)
        monomial_values = self._monomial_taylor_batch(
            self._map_monomials,
            rows_in,
            dual_shape,
            dtype=np.complex128,
        )
        if monomial_values is not None:
            return monomial_values
        rows = self._dual_input_matrix(rows_in, dual_shape).astype(np.complex128)
        evaluators = (
            self._map_dual_evaluators
            if dual_shape == self.dual_shape
            else self.prepare_map_dual_evaluators(dual_shape)
        )
        if dual_shape == self.dual_shape and not evaluators:
            evaluators = self.prepare_map_dual_evaluators(dual_shape)
        columns = [
            np.asarray(self._timed_evaluate_complex(evaluator, rows, timing), dtype=np.complex128)
            for evaluator in evaluators
        ]
        return np.stack(columns, axis=1)

    def map_dual_complex_prec_for_shape(
        self,
        y: np.ndarray,
        dual_shape: list[tuple[int, ...]],
        precision_digits: int,
        timing: HotPathTiming | None = None,
    ) -> list[list[tuple[Any, Any]]]:
        """Evaluate one sector-map dual jet row with complex multiprecision."""
        self._ensure_evaluators()
        if not dual_shape:
            return [[_decimal_complex(value, precision_digits)] for value in self.map_eval(np.asarray(y, dtype=float))]
        monomial_values = self._monomial_taylor_prec(
            self._map_monomials,
            np.asarray(y, dtype=float),
            dual_shape,
            precision_digits,
        )
        if monomial_values is not None:
            return monomial_values
        row = self._dual_input_prec_row(np.asarray(y, dtype=float), dual_shape, precision_digits)
        evaluators = (
            self._map_dual_evaluators
            if dual_shape == self.dual_shape
            else self.prepare_map_dual_evaluators(dual_shape)
        )
        return [
            self._timed_evaluate_complex_with_prec(evaluator, row, precision_digits, timing)
            for evaluator in evaluators
        ]

    def dual_index(self, multi_index: tuple[int, ...]) -> int:
        """Return the column of a stored dual Taylor coefficient."""
        return self._dual_index_by_multi_index[multi_index]

    def _monomial_taylor_batch(
        self,
        monomials: list[tuple[complex, list[int]] | None],
        y_values: np.ndarray,
        dual_shape: list[tuple[int, ...]],
        dtype: Any,
    ) -> np.ndarray | None:
        """Evaluate Taylor jets of monomial sector expressions analytically."""
        if not monomials or any(monomial is None for monomial in monomials):
            return None
        rows = np.asarray(y_values, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != self.integration_dim:
            raise ValueError(f"{self.name}: expected coordinate array with shape (n,{self.integration_dim})")
        output = np.zeros((rows.shape[0], len(monomials), len(dual_shape)), dtype=np.complex128)
        singular_position = {axis: position for position, axis in enumerate(self.singular_axes)}
        dual_rank = len(dual_shape[0]) if dual_shape else 0
        for expr_index, monomial in enumerate(monomials):
            if monomial is None:
                return None
            coefficient, powers = monomial
            for column, multi_index in enumerate(dual_shape):
                if any(multi_index[position] != 0 for position in range(len(self.singular_axes), dual_rank)):
                    continue
                values = np.full(rows.shape[0], coefficient, dtype=np.complex128)
                for axis, power in enumerate(powers):
                    position = singular_position.get(axis)
                    if position is None or position >= dual_rank:
                        if power:
                            values *= rows[:, axis] ** int(power)
                        continue
                    order = int(multi_index[position])
                    if order > power:
                        values[:] = 0.0
                        break
                    values *= float(math.comb(int(power), order))
                    remaining_power = int(power) - order
                    if remaining_power:
                        values *= rows[:, axis] ** remaining_power
                output[:, expr_index, column] = values
        if dtype is float:
            return output.real.astype(float, copy=False)
        return output.astype(dtype, copy=False)

    def _monomial_taylor_prec(
        self,
        monomials: list[tuple[complex, list[int]] | None],
        y: np.ndarray,
        dual_shape: list[tuple[int, ...]],
        precision_digits: int,
    ) -> list[list[tuple[Any, Any]]] | None:
        """Evaluate monomial Taylor jets in Symbolica's complex Decimal shape."""
        if not monomials or any(monomial is None for monomial in monomials):
            return None
        coords = [
            decimal_with_precision(value, precision_digits)
            for value in np.asarray(y, dtype=float)
        ]
        singular_position = {axis: position for position, axis in enumerate(self.singular_axes)}
        dual_rank = len(dual_shape[0]) if dual_shape else 0
        output: list[list[tuple[Any, Any]]] = []
        for monomial in monomials:
            if monomial is None:
                return None
            coefficient, powers = monomial
            coeff_re = decimal_with_precision(coefficient.real, precision_digits)
            coeff_im = decimal_with_precision(coefficient.imag, precision_digits)
            jet: list[tuple[Any, Any]] = []
            for multi_index in dual_shape:
                if any(multi_index[position] != 0 for position in range(len(self.singular_axes), dual_rank)):
                    jet.append((Decimal(0), Decimal(0)))
                    continue
                real = coeff_re
                imag = coeff_im
                zero = False
                for axis, power in enumerate(powers):
                    position = singular_position.get(axis)
                    if position is None or position >= dual_rank:
                        factor = _pow_decimal(coords[axis], int(power))
                    else:
                        order = int(multi_index[position])
                        if order > power:
                            zero = True
                            break
                        factor = Decimal(math.comb(int(power), order)) * _pow_decimal(
                            coords[axis],
                            int(power) - order,
                        )
                    real *= factor
                    imag *= factor
                jet.append((Decimal(0), Decimal(0)) if zero else (real, imag))
            output.append(jet)
        return output


def _triangle_sector(
    name: str,
    swapped: bool,
    mode: str,
    jit_compile_evaluators: bool,
) -> SectorDefinition:
    """Construct one of the two triangle sectors."""
    x_forward = "t/(1+z)"
    x_backward = "t*z/(1+z)"
    x1, x2 = (x_backward, x_forward) if swapped else (x_forward, x_backward)
    if mode == "massless":
        regular_jacobian = "1/(1+z)^2"
        f_monomial_powers = [2, 1]
        jacobian_monomial_powers = [1, 0]
        singular_axes = [0, 1]
        subtraction_type = "two-axis endpoint subtraction"
    else:
        regular_jacobian = "t/(1+z)^2"
        f_monomial_powers = [0, 0]
        jacobian_monomial_powers = [0, 0]
        singular_axes = []
        subtraction_type = "finite"
    return SectorDefinition(
        name=name,
        integration_dim=2,
        variable_names=["t", "z"],
        map_exprs=[_expr("1-t"), _expr(x1), _expr(x2)],
        regular_jacobian_expr=_expr(regular_jacobian),
        f_monomial_powers=f_monomial_powers,
        jacobian_monomial_powers=jacobian_monomial_powers,
        singular_axes=singular_axes,
        subtraction_type=subtraction_type,
        description="triangle endpoint sector with x1/x2 swapped" if swapped else "triangle endpoint sector",
        jit_compile_evaluators=jit_compile_evaluators,
    )


def _box_primary_sector(dominant_index: int, jit_compile_evaluators: bool) -> SectorDefinition:
    """Construct a finite massive box primary sector."""
    variable_names = ["y0", "y1", "y2"]
    denom = "1+y0+y1+y2"
    others = [i for i in range(4) if i != dominant_index]
    map_texts = ["0", "0", "0", "0"]
    map_texts[dominant_index] = f"1/({denom})"
    for slot, original_index in enumerate(others):
        map_texts[original_index] = f"{variable_names[slot]}/({denom})"
    return SectorDefinition(
        name=f"B{dominant_index}",
        integration_dim=3,
        variable_names=variable_names,
        map_exprs=[_expr(text) for text in map_texts],
        regular_jacobian_expr=_expr(f"1/({denom})^4"),
        f_monomial_powers=[0, 0, 0],
        jacobian_monomial_powers=[0, 0, 0],
        singular_axes=[],
        subtraction_type="finite",
        description=f"box primary sector with x{dominant_index} dominant",
        jit_compile_evaluators=jit_compile_evaluators,
    )


def _box_massless_sector(
    dominant_index: int,
    kind: str,
    linear_slot: int,
    left_slot: int,
    right_slot: int,
    jit_compile_evaluators: bool,
) -> SectorDefinition:
    """Construct one massless box secondary sector inside a primary sector."""
    variable_names = ["u", "v", "w"]
    primary_values = ["0", "0", "0"]
    if kind == "single":
        primary_values[left_slot] = "u*v"
        primary_values[linear_slot] = "u"
        primary_values[right_slot] = "w"
        f_monomial_powers = [1, 0, 0]
        jacobian_monomial_powers = [1, 0, 0]
        singular_axes = [0]
        suffix = "L"
        subtraction_type = "one-axis endpoint subtraction"
    elif kind == "product":
        primary_values[left_slot] = "u"
        primary_values[right_slot] = "v"
        primary_values[linear_slot] = "u*v*w"
        f_monomial_powers = [1, 1, 0]
        jacobian_monomial_powers = [1, 1, 0]
        singular_axes = [0, 1]
        suffix = "P"
        subtraction_type = "two-axis endpoint subtraction"
    elif kind == "complement":
        primary_values[left_slot] = "u"
        primary_values[linear_slot] = "u*v"
        primary_values[right_slot] = "v*w"
        f_monomial_powers = [1, 1, 0]
        jacobian_monomial_powers = [1, 1, 0]
        singular_axes = [0, 1]
        suffix = "Q"
        subtraction_type = "two-axis endpoint subtraction"
    else:
        raise ValueError(f"unknown massless box sector kind {kind!r}")

    denom = "1+" + "+".join(primary_values)
    others = [i for i in range(4) if i != dominant_index]
    map_texts = ["0", "0", "0", "0"]
    map_texts[dominant_index] = f"1/({denom})"
    for slot, original_index in enumerate(others):
        map_texts[original_index] = f"({primary_values[slot]})/({denom})"

    return SectorDefinition(
        name=f"B{dominant_index}{suffix}",
        integration_dim=3,
        variable_names=variable_names,
        map_exprs=[_expr(text) for text in map_texts],
        regular_jacobian_expr=_expr(f"1/({denom})^4"),
        f_monomial_powers=f_monomial_powers,
        jacobian_monomial_powers=jacobian_monomial_powers,
        singular_axes=singular_axes,
        subtraction_type=subtraction_type,
        description=f"massless box secondary sector {kind} from primary B{dominant_index}",
        jit_compile_evaluators=jit_compile_evaluators,
    )


def _box_massless_sectors(jit_compile_evaluators: bool) -> list[SectorDefinition]:
    """Enumerate all twelve massless box sectors."""
    sectors: list[SectorDefinition] = []
    pair_by_primary = {
        0: (2, (1, 3)),
        2: (0, (1, 3)),
        1: (3, (0, 2)),
        3: (1, (0, 2)),
    }
    for dominant_index in range(4):
        others = [i for i in range(4) if i != dominant_index]
        linear_index, product_indices = pair_by_primary[dominant_index]
        linear_slot = others.index(linear_index)
        left_slot = others.index(product_indices[0])
        right_slot = others.index(product_indices[1])
        sectors.append(
            _box_massless_sector(
                dominant_index,
                "single",
                linear_slot,
                left_slot,
                right_slot,
                jit_compile_evaluators,
            )
        )
        sectors.append(
            _box_massless_sector(
                dominant_index,
                "product",
                linear_slot,
                left_slot,
                right_slot,
                jit_compile_evaluators,
            )
        )
        sectors.append(
            _box_massless_sector(
                dominant_index,
                "complement",
                linear_slot,
                left_slot,
                right_slot,
                jit_compile_evaluators,
            )
        )
    return sectors


def prepare_sector_evaluators(
    sectors: list[SectorDefinition],
    progress: Any | None = None,
    include_dual: bool = True,
) -> None:
    """Build all sector-local Symbolica callbacks."""
    total = len(sectors)
    for index, sector in enumerate(sectors, start=1):
        if progress is not None and (index == 1 or index % 25 == 0 or index == total):
            progress.update(
                index - 1,
                total=total,
                detail=f"{sector.name} {index}/{total}",
            )
        sector.prepare_evaluators(include_dual=include_dual)
        if progress is not None and (index % 25 == 0 or index == total):
            progress.update(
                index,
                total=total,
                detail=f"{sector.name} done {index}/{total}",
            )


def generate_sectors(request: IntegralRequest) -> list[SectorDefinition]:
    """Return all prepared sectors for the requested supported integral."""
    if request.integral == "dot":
        from dot_topology import generate_sectors_from_dot_request

        return generate_sectors_from_dot_request(request)
    if request.integral == "triangle":
        sectors = [
            _triangle_sector(
                "S1",
                swapped=False,
                mode=request.mode,
                jit_compile_evaluators=request.jit_compile_evaluators,
            ),
            _triangle_sector(
                "S2",
                swapped=True,
                mode=request.mode,
                jit_compile_evaluators=request.jit_compile_evaluators,
            ),
        ]
        prepare_sector_evaluators(sectors)
        return sectors
    if request.integral == "box":
        if request.mode == "massless":
            sectors = _box_massless_sectors(request.jit_compile_evaluators)
        else:
            sectors = [_box_primary_sector(i, request.jit_compile_evaluators) for i in range(4)]
        prepare_sector_evaluators(sectors)
        return sectors
    raise ValueError(f"unsupported integral {request.integral!r}")
