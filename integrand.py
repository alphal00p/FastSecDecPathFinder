"""Topology definitions and generic black-box sector processing.

The only symbolic expressions stored here are the topology-level U and F
polynomials used to build Symbolica evaluators and to print summaries.  The
``SectorProcessor`` never substitutes sector maps into U/F symbolically; it
only evaluates prepared sector callbacks and U/F callbacks on numeric batches.
"""

from __future__ import annotations

import cmath
from itertools import product
import math
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
from symbolica import E, S

from definitions import HotPathTiming, IntegralRequest
from sectors_generator import SectorDefinition


@dataclass
class TopologyDefinition:
    """Retained U/F expressions plus their numeric and dual evaluators."""

    family: str
    x_names: list[str]
    parameter_names: list[str]
    parameter_values: list[float]
    u_expr: Any
    f_expr: Any
    u_power_base: int
    f_power_base: int
    eps_log_u_coeff: float
    eps_log_f_coeff: float
    expected_laurent_orders: list[str]
    convention_note: str
    jit_compile_evaluators: bool = False
    _u_evaluator: Any = field(init=False, repr=False)
    _f_evaluator: Any = field(init=False, repr=False)
    _f_dual_evaluators: dict[tuple[tuple[int, ...], ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Build the scalar U and F evaluators in the declared row order."""
        params = [S(name) for name in [*self.x_names, *self.parameter_names]]
        self._u_evaluator = self.u_expr.evaluator(params, jit_compile=self.jit_compile_evaluators)
        self._f_evaluator = self.f_expr.evaluator(params, jit_compile=self.jit_compile_evaluators)

    @property
    def evaluator_parameter_order(self) -> list[str]:
        """Return the input-column order expected by U/F evaluators."""
        return [*self.x_names, *self.parameter_names]

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

    def _timed_evaluate(self, evaluator: Any, rows: np.ndarray, timing: HotPathTiming | None) -> Any:
        """Evaluate a Symbolica evaluator and optionally charge EvalT."""
        start = time.perf_counter()
        values = evaluator.evaluate(rows)
        if timing is not None:
            timing.add_eval(time.perf_counter() - start)
        return values

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

    def f_dual_evaluator(self, dual_shape: list[tuple[int, ...]]) -> Any:
        """Return a cached dualized F evaluator for the requested jet shape."""
        key = tuple(dual_shape)
        evaluator = self._f_dual_evaluators.get(key)
        if evaluator is None:
            params = [S(name) for name in [*self.x_names, *self.parameter_names]]
            evaluator = self.f_expr.evaluator(params, jit_compile=self.jit_compile_evaluators)
            evaluator.dualize([list(mi) for mi in dual_shape])
            self._f_dual_evaluators[key] = evaluator
        return evaluator

    def f_taylor(
        self, sector: SectorDefinition, y: list[float] | tuple[float, ...]
    ) -> dict[tuple[int, ...], complex]:
        """Evaluate F Taylor coefficients after composing map jets with F."""
        if not sector.dual_shape:
            raise ValueError(f"{sector.name}: no dual shape declared")

        x_jets = sector.map_dual_eval(y)
        zero = [0.0 for _ in sector.dual_shape]
        row: list[float] = []
        for jet in x_jets:
            row.extend(jet)
        for value in self.parameter_values:
            param_jet = zero.copy()
            param_jet[0] = float(value)
            row.extend(param_jet)

        values = self.f_dual_evaluator(sector.dual_shape).evaluate([row])[0]
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

        rows_in = np.asarray(y_values, dtype=float)
        x_jets = sector.map_dual_eval_batch(rows_in, timing)
        n_rows = rows_in.shape[0]
        dual_len = len(sector.dual_shape)
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

        evaluator = self.f_dual_evaluator(sector.dual_shape)
        values = self._timed_evaluate(evaluator, rows, timing)
        return np.asarray(values, dtype=np.complex128)


def build_topology(request: IntegralRequest) -> TopologyDefinition:
    """Construct the U/F topology definition for the requested family."""
    m2 = request.m * request.m
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
            jit_compile_evaluators=request.jit_compile_evaluators,
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
            jit_compile_evaluators=request.jit_compile_evaluators,
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
    """Training weight used for one finite-part sample."""
    return abs(complex(value))


def complex_abs_for_training_array(values: np.ndarray) -> np.ndarray:
    """Vectorized finite-part training weight."""
    return np.abs(np.asarray(values, dtype=np.complex128))


class SectorProcessor:
    """Generic sector application layer.

    This class deliberately knows nothing about triangle or box topology.  All
    topology-specific information is carried by TopologyDefinition and
    SectorDefinition.  The U/F polynomials are accessed only through evaluators.
    """

    def __init__(self, topology: TopologyDefinition, boundary_tol: float = 1.0e-14) -> None:
        """Store topology evaluators and endpoint-detection tolerance."""
        self.topology = topology
        self.boundary_tol = boundary_tol

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

        timing = HotPathTiming()
        if len(sector.singular_axes) == 0:
            coeffs, training = self._finite_contribution_batch(sector, rows, timing)
        elif len(sector.singular_axes) == 1:
            coeffs, training = self._one_axis_subtraction_batch(sector, rows, timing)
        elif len(sector.singular_axes) == 2:
            coeffs, training = self._two_axis_subtraction_batch(sector, rows, timing)
        else:
            raise ValueError(
                f"{sector.name}: only zero, one, and two singular axes are currently supported"
            )
        return coeffs, training, timing

    def _finite_contribution_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a sector with no endpoint monomial extraction."""
        x = sector.map_eval_batch(y_values, timing)
        u = self.topology.u_values(x, timing)
        f = self.topology.f_values(x, timing)
        regular_j = sector.jacobian_eval_batch(y_values, timing).astype(np.complex128)
        value = regular_j * np.power(u, self.topology.u_power_base) * np.power(
            f, -self.topology.f_power_base
        )
        coeffs = np.zeros((y_values.shape[0], 3), dtype=np.complex128)
        coeffs[:, 2] = value
        return coeffs, complex_abs_for_training_array(value)

    def _phi_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        x_values: np.ndarray | None,
        f_values: np.ndarray | None,
        timing: HotPathTiming,
    ) -> np.ndarray:
        """Evaluate the regular residual ``phi = F(X(y))/M(y)``.

        Interior points use the scalar F evaluator and divide by the declared
        monomial.  Endpoint points use dual Taylor coefficients of the black-box
        F evaluator composed with sector-map dual jets.
        """
        axes = sector.singular_axes
        rows = np.asarray(y_values, dtype=float)
        phi = np.empty(rows.shape[0], dtype=np.complex128)
        if not axes:
            if f_values is not None:
                return np.asarray(f_values, dtype=np.complex128)
            if x_values is None:
                x_values = sector.map_eval_batch(rows, timing)
            return self.topology.f_values(x_values, timing)

        axis_values = rows[:, axes]
        interior = np.all(axis_values > self.boundary_tol, axis=1)
        if np.any(interior):
            if f_values is None:
                if x_values is None:
                    x_values = sector.map_eval_batch(rows[interior], timing)
                    f_interior = self.topology.f_values(x_values, timing)
                else:
                    f_interior = self.topology.f_values(x_values[interior], timing)
            else:
                f_interior = np.asarray(f_values, dtype=np.complex128)[interior]
            phi[interior] = f_interior / sector.f_monomial_value_batch(rows[interior])

        boundary = ~interior
        if np.any(boundary):
            boundary_rows = rows[boundary]
            taylor = self.topology.f_taylor_batch(sector, boundary_rows, timing)
            boundary_phi = np.empty(boundary_rows.shape[0], dtype=np.complex128)
            boundary_flags = boundary_rows[:, axes] <= self.boundary_tol
            for pattern in product((False, True), repeat=len(axes)):
                row_mask = np.all(boundary_flags == np.asarray(pattern, dtype=bool), axis=1)
                if not np.any(row_mask):
                    continue
                multi_index: list[int] = []
                denominator = np.ones(int(np.count_nonzero(row_mask)), dtype=float)
                for axis, is_boundary in zip(axes, pattern):
                    power = sector.f_monomial_powers[axis]
                    if is_boundary:
                        multi_index.append(power)
                    else:
                        multi_index.append(0)
                        denominator *= boundary_rows[row_mask, axis] ** power
                boundary_phi[row_mask] = (
                    taylor[row_mask, sector.dual_index(tuple(multi_index))] / denominator
                )
            phi[boundary] = boundary_phi

        return phi

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
        phi = self._phi_batch(sector, y_values, x, f, timing)
        regular_j = sector.jacobian_eval_batch(y_values, timing).astype(np.complex128)
        pref = regular_j * np.power(u, self.topology.u_power_base) * np.power(
            phi, -self.topology.f_power_base
        )
        exponent_log = (
            self.topology.eps_log_u_coeff * feynman_log_array(u)
            + self.topology.eps_log_f_coeff * feynman_log_array(phi)
        )
        coeffs = np.empty((y_values.shape[0], 3), dtype=np.complex128)
        coeffs[:, 0] = pref
        coeffs[:, 1] = pref * exponent_log
        coeffs[:, 2] = 0.5 * pref * exponent_log * exponent_log
        return coeffs

    def _one_axis_subtraction_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the localized one-axis logarithmic endpoint subtraction."""
        self._check_supported_singular_powers(sector)
        axis = sector.singular_axes[0]
        alpha = float(sector.f_monomial_powers[axis])
        coord = y_values[:, axis]

        g_y = self._g_coeffs_batch(sector, y_values, timing)
        y0 = y_values.copy()
        y0[:, axis] = 0.0
        g_0 = self._g_coeffs_batch(sector, y0, timing)

        coeffs = np.zeros((y_values.shape[0], 3), dtype=np.complex128)
        coeffs[:, 1] = -g_0[:, 0] / alpha
        coeffs[:, 2] = -g_0[:, 1] / alpha + (g_y[:, 0] - g_0[:, 0]) / coord
        return coeffs, complex_abs_for_training_array(coeffs[:, 2])

    def _two_axis_subtraction_batch(
        self,
        sector: SectorDefinition,
        y_values: np.ndarray,
        timing: HotPathTiming,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the localized two-axis endpoint subtraction."""
        self._check_supported_singular_powers(sector)
        axis_a, axis_b = sector.singular_axes
        alpha = float(sector.f_monomial_powers[axis_a])
        beta = float(sector.f_monomial_powers[axis_b])
        ya = y_values[:, axis_a]
        yb = y_values[:, axis_b]

        y_a0 = y_values.copy()
        y_a0[:, axis_b] = 0.0
        y_0b = y_values.copy()
        y_0b[:, axis_a] = 0.0
        y_00 = y_0b.copy()
        y_00[:, axis_b] = 0.0

        g_ab = self._g_coeffs_batch(sector, y_values, timing)
        g_a0 = self._g_coeffs_batch(sector, y_a0, timing)
        g_0b = self._g_coeffs_batch(sector, y_0b, timing)
        g_00 = self._g_coeffs_batch(sector, y_00, timing)

        remainder0 = g_ab[:, 0] - g_0b[:, 0] - g_a0[:, 0] + g_00[:, 0]
        edge_b0 = g_0b[:, 0] - g_00[:, 0]
        edge_b1 = g_0b[:, 1] - g_00[:, 1]
        edge_a0 = g_a0[:, 0] - g_00[:, 0]
        edge_a1 = g_a0[:, 1] - g_00[:, 1]

        coeffs = np.empty((y_values.shape[0], 3), dtype=np.complex128)
        coeffs[:, 0] = g_00[:, 0] / (alpha * beta)
        coeffs[:, 1] = (
            g_00[:, 1] / (alpha * beta)
            - edge_b0 / (alpha * yb)
            - edge_a0 / (beta * ya)
        )
        coeffs[:, 2] = (
            g_00[:, 2] / (alpha * beta)
            - (edge_b1 - beta * edge_b0 * np.log(yb)) / (alpha * yb)
            - (edge_a1 - alpha * edge_a0 * np.log(ya)) / (beta * ya)
            + remainder0 / (ya * yb)
        )
        return coeffs, complex_abs_for_training_array(coeffs[:, 2])

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

    def _check_supported_singular_powers(self, sector: SectorDefinition) -> None:
        """Ensure the declared endpoint powers are logarithmic after extraction."""
        for axis in sector.singular_axes:
            base_power = (
                sector.jacobian_monomial_powers[axis]
                - self.topology.f_power_base * sector.f_monomial_powers[axis]
            )
            if base_power != -1:
                raise ValueError(
                    f"{sector.name}: unsupported endpoint power y^{base_power}; "
                    "only logarithmic y^(-1-alpha*eps) factors are implemented"
                )

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
        alpha = float(sector.f_monomial_powers[axis])
        coord = float(y[axis])

        g_y = self._g_coeffs(sector, y)
        y0 = self._with_axis_value(y, axis, 0.0)
        g_0 = self._g_coeffs(sector, y0)

        coeff_m2 = 0.0 + 0.0j
        coeff_m1 = -g_0[0] / alpha
        coeff_0 = -g_0[1] / alpha + (g_y[0] - g_0[0]) / coord
        coeffs = [coeff_m2, coeff_m1, coeff_0]
        return coeffs, complex_abs_for_training(coeff_0)

    def _two_axis_subtraction(
        self, sector: SectorDefinition, y: list[float] | tuple[float, ...]
    ) -> tuple[list[complex], float]:
        """Scalar two-axis subtraction kept for legacy/debug comparisons."""
        self._check_supported_singular_powers(sector)
        axis_a, axis_b = sector.singular_axes
        alpha = float(sector.f_monomial_powers[axis_a])
        beta = float(sector.f_monomial_powers[axis_b])
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

        coeff_m2 = g_00[0] / (alpha * beta)
        coeff_m1 = (
            g_00[1] / (alpha * beta)
            - edge_b0 / (alpha * yb)
            - edge_a0 / (beta * ya)
        )
        coeff_0 = (
            g_00[2] / (alpha * beta)
            - (edge_b1 - beta * edge_b0 * math.log(yb)) / (alpha * yb)
            - (edge_a1 - alpha * edge_a0 * math.log(ya)) / (beta * ya)
            + remainder0 / (ya * yb)
        )

        coeffs = [coeff_m2, coeff_m1, coeff_0]
        return coeffs, complex_abs_for_training(coeff_0)
