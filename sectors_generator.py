"""Declarative sector definitions for the supported triangle and box cases.

The sector generator is the only place where the current prototype hard-codes
maps, Jacobians, endpoint monomials, and subtraction axes.  It deliberately does
not touch the Symanzik polynomials: it only prepares data and evaluators that
the generic processor can later combine with black-box U/F callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import time
from typing import Any

import numpy as np
from symbolica import E, S

from definitions import HotPathTiming, IntegralRequest


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
    f_monomial_expr: Any = field(init=False)
    u_monomial_expr: Any = field(init=False)
    dual_shape: list[tuple[int, ...]] = field(init=False)
    _map_evaluators: list[Any] = field(init=False, repr=False)
    _jacobian_evaluator: Any = field(init=False, repr=False)
    _map_dual_evaluators: list[Any] = field(init=False, repr=False)
    _dual_index_by_multi_index: dict[tuple[int, ...], int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the sector declaration and build Symbolica evaluators."""
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
        if len(self.u_monomial_powers) != self.integration_dim:
            raise ValueError(f"{self.name}: u_monomial_powers has wrong length")
        if len(self.measure_monomial_powers) != self.integration_dim:
            raise ValueError(f"{self.name}: measure_monomial_powers has wrong length")
        if len(self.numerator_monomial_powers) != self.integration_dim:
            raise ValueError(f"{self.name}: numerator_monomial_powers has wrong length")

        params = _symbols(self.variable_names)
        self.f_monomial_expr = _monomial_expr(self.variable_names, self.f_monomial_powers)
        self.u_monomial_expr = _monomial_expr(self.variable_names, self.u_monomial_powers)
        # The docs describe each endpoint sector by a known monomial M_s(y).
        # The dual shape is exactly the set of Taylor coefficients needed to
        # recover U(X_s(y))/M_U(y) or F(X_s(y))/M_F(y) when one or more
        # monomial variables vanish.
        self.dual_shape = dual_shape_from_powers(
            [
                max(self.u_monomial_powers[axis], self.f_monomial_powers[axis])
                for axis in self.singular_axes
            ]
        )
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
        self._dual_index_by_multi_index = {multi_index: i for i, multi_index in enumerate(self.dual_shape)}
        if self.dual_shape:
            for expr in self.map_exprs:
                # Dualized map evaluators produce the chain-rule input jets for
                # the black-box F evaluator, matching the construction in the
                # implementation notes.
                evaluator = expr.evaluator(params, jit_compile=self.jit_compile_evaluators)
                evaluator.dualize([list(mi) for mi in self.dual_shape])
                self._map_dual_evaluators.append(evaluator)

    def _timed_evaluate(self, evaluator: Any, rows: np.ndarray, timing: HotPathTiming | None) -> Any:
        """Evaluate a Symbolica callback and optionally charge it to EvalT."""
        start = time.perf_counter()
        values = evaluator.evaluate(rows)
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return values

    def map_eval(self, y: list[float] | tuple[float, ...]) -> list[float]:
        """Evaluate the sector map at one point."""
        row = [float(value) for value in y]
        return [float(evaluator.evaluate([row])[0][0]) for evaluator in self._map_evaluators]

    def map_eval_batch(
        self,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate all mapped Feynman parameters for a batch of points."""
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
        row = [float(value) for value in y]
        return float(self._jacobian_evaluator.evaluate([row])[0][0])

    def jacobian_eval_batch(
        self,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate the regular Jacobian for a batch."""
        rows = np.asarray(y_values, dtype=float)
        values = self._timed_evaluate(self._jacobian_evaluator, rows, timing)
        return np.asarray(values, dtype=float)[:, 0]

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
        if not self.dual_shape:
            return [[value] for value in self.map_eval(y)]

        zero_mi = tuple(0 for _ in self.singular_axes)
        unit_by_axis = {
            axis: tuple(1 if i == pos else 0 for i in range(len(self.singular_axes)))
            for pos, axis in enumerate(self.singular_axes)
        }

        row: list[float] = []
        for axis, coord in enumerate(y):
            unit = unit_by_axis.get(axis)
            for mi in self.dual_shape:
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
            for evaluator in self._map_dual_evaluators
        ]

    def map_dual_eval_batch(
        self,
        y_values: np.ndarray,
        timing: HotPathTiming | None = None,
    ) -> np.ndarray:
        """Evaluate sector-map dual jets for endpoint batches."""
        rows_in = np.asarray(y_values, dtype=float)
        if not self.dual_shape:
            return self.map_eval_batch(rows_in, timing)[:, :, np.newaxis]

        n_rows = rows_in.shape[0]
        dual_len = len(self.dual_shape)
        zero_mi = tuple(0 for _ in self.singular_axes)
        unit_by_axis = {
            axis: tuple(1 if i == pos else 0 for i in range(len(self.singular_axes)))
            for pos, axis in enumerate(self.singular_axes)
        }
        rows = np.zeros((n_rows, self.integration_dim * dual_len), dtype=float)
        for axis in range(self.integration_dim):
            unit = unit_by_axis.get(axis)
            offset = axis * dual_len
            for j, mi in enumerate(self.dual_shape):
                # Row layout: [y0 jets][y1 jets]... in the same dual-shape
                # order later used by TopologyDefinition.f_taylor_batch.
                if mi == zero_mi:
                    rows[:, offset + j] = rows_in[:, axis]
                elif unit is not None and mi == unit:
                    rows[:, offset + j] = 1.0

        columns = [
            np.asarray(self._timed_evaluate(evaluator, rows, timing), dtype=float)
            for evaluator in self._map_dual_evaluators
        ]
        return np.stack(columns, axis=1)

    def dual_index(self, multi_index: tuple[int, ...]) -> int:
        """Return the column of a stored dual Taylor coefficient."""
        return self._dual_index_by_multi_index[multi_index]


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


def generate_sectors(request: IntegralRequest) -> list[SectorDefinition]:
    """Return all prepared sectors for the requested supported integral."""
    if request.integral == "dot":
        from dot_topology import generate_sectors_from_dot_request

        return generate_sectors_from_dot_request(request)
    if request.integral == "triangle":
        return [
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
    if request.integral == "box":
        if request.mode == "massless":
            return _box_massless_sectors(request.jit_compile_evaluators)
        return [_box_primary_sector(i, request.jit_compile_evaluators) for i in range(4)]
    raise ValueError(f"unsupported integral {request.integral!r}")
