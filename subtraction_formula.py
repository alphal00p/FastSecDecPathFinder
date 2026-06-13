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

from itertools import product
from typing import Any

from symbolica import E, Replacement, S


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
    sector: Any,
    signature: tuple[Any, ...],
    formula_class: type,
) -> Any:
    """Build the endpoint-only projector formula for a lower cache signature.

    This formula does not know how the regular function ``g_s`` is obtained.
    Its inputs are the sampled singular coordinates and precomputed
    ``g_{S,alpha,r}`` coefficients for every endpoint projector.  That makes
    the evaluator reusable across sectors that share only endpoint powers,
    Taylor orders, and Laurent range.
    """
    ctx = _EndpointProjectorContext(topology, sector, signature)
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
        taylor_orders=ctx.taylor_orders,
        coefficient_layout=ctx.coefficient_layout,
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

    def __init__(self, topology: Any, sector: Any, signature: tuple[Any, ...]) -> None:
        self.topology = topology
        self.sector = sector
        self.signature = signature
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
        self.coeff_symbols: dict[tuple[tuple[int, ...], tuple[int, ...], int], Any] = {}
        self.coefficient_layout: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
        self._build_coefficient_symbols()

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
            max_orders = [
                self.taylor_orders[position] if position in subset else 0
                for position in range(self.n_axes)
            ]
            mask = _subset_mask(subset)
            for multi_index in _multi_indices(max_orders):
                for regular_order in range(self.topology.coefficient_count):
                    name = f"ep_g_{mask}_{_multi_suffix(multi_index)}_{regular_order}"
                    symbol = S(name)
                    key = (subset, multi_index, regular_order)
                    self.coeff_symbols[key] = symbol
                    self.coefficient_layout.append(key)
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
                    term *= self._regular_eps_series(zero_positions, multi_index)
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
        subset: tuple[int, ...],
        multi_index: tuple[int, ...],
    ) -> Any:
        out = E("0")
        for regular_order in range(self.topology.coefficient_count):
            symbol = self.coeff_symbols[(subset, multi_index, regular_order)]
            if regular_order == 0:
                out += symbol
            else:
                out += symbol * _expr_int_power(self.eps, regular_order)
        return out

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


def _multi_indices(max_orders: list[int]) -> list[tuple[int, ...]]:
    if not max_orders:
        return [()]
    ranges = [range(int(order) + 1) for order in max_orders]
    return [tuple(values) for values in product(*ranges)]


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
