"""pySecDec boundary for DOT-driven FSD topologies.

This is intentionally the only FSD-owned module importing pySecDec.  The rest
of the code receives ordinary ``TopologyDefinition`` and ``SectorDefinition``
objects backed by Symbolica evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import json
import os
import shutil
import subprocess
from typing import Any

import numpy as np
from symbolica import E, S

from definitions import EpsilonExpansion, IntegralRequest, ParametricRepresentation
from dot_parser import ParsedDotGraph
from generation_timing import GenerationProgress, GenerationTimings
from integrand import TopologyDefinition
from kinematics import KinematicsDefinition
from numerator_reducer import reduce_dot_product_numerator
from sectors_generator import SectorDefinition, prepare_sector_evaluators


@dataclass
class DotBuildBundle:
    """Topology, sector, and timing objects created from pySecDec."""

    topology: TopologyDefinition
    sectors: list[SectorDefinition]
    timings: GenerationTimings
    loop_integral: Any
    parsed_graph: ParsedDotGraph
    kinematics: KinematicsDefinition


@dataclass
class PySecDecRunResult:
    """Numerical pySecDec integration result in pySecDec convention."""

    coeffs: list[complex]
    errors: list[complex]
    orders: list[int]
    raw_series: Any
    timings: GenerationTimings


def require_pysecdec() -> dict[str, Any]:
    """Import pySecDec pieces and raise a setup-oriented error if unavailable."""
    try:
        from pySecDec.algebra import Polynomial
        from pySecDec.decomposition import Sector
        from pySecDec.decomposition.geometric import (
            Cheng_Wu,
            geometric_decomposition,
            geometric_decomposition_ku,
        )
        from pySecDec.decomposition.iterative import (
            iterative_decomposition,
            primary_decomposition,
        )
        from pySecDec.integral_interface import IntegralLibrary
        from pySecDec.loop_integral import (
            LoopIntegralFromGraph,
            LoopIntegralFromPropagators,
            loop_package,
        )
    except Exception as exc:  # pragma: no cover - depends on optional external package.
        raise RuntimeError(
            "DOT mode requires pySecDec. Install it in this venv with "
            "'.venv/bin/python -m pip install pySecDec' and ensure FORM/Normaliz "
            "requirements are available for the selected sector method."
        ) from exc
    return {
        "Polynomial": Polynomial,
        "Sector": Sector,
        "Cheng_Wu": Cheng_Wu,
        "geometric_decomposition": geometric_decomposition,
        "geometric_decomposition_ku": geometric_decomposition_ku,
        "iterative_decomposition": iterative_decomposition,
        "primary_decomposition": primary_decomposition,
        "IntegralLibrary": IntegralLibrary,
        "LoopIntegralFromGraph": LoopIntegralFromGraph,
        "LoopIntegralFromPropagators": LoopIntegralFromPropagators,
        "loop_package": loop_package,
    }


def _symbolica_text(text: Any) -> str:
    """Convert a pySecDec/SymPy-style expression string to Symbolica syntax."""
    out = str(text).replace("**", "^")
    out = out.replace("I", "i_")
    return out


def _poly_symbols(poly: Any) -> list[str]:
    """Return polynomial variable names from a pySecDec algebra object."""
    symbols = getattr(poly, "polysymbols", None)
    if symbols is None and hasattr(poly, "factors") and poly.factors:
        symbols = getattr(poly.factors[0], "polysymbols", None)
    if symbols is None:
        return []
    return [str(symbol) for symbol in list(symbols)]


def _polynomial_to_symbolica_text(poly: Any) -> str:
    """Render a pySecDec polynomial-like object as a Symbolica expression."""
    if hasattr(poly, "factors"):
        factors = [_polynomial_to_symbolica_text(factor) for factor in poly.factors]
        return "*".join(f"({factor})" for factor in factors if factor != "1") or "1"
    expolist = getattr(poly, "expolist", None)
    coeffs = getattr(poly, "coeffs", None)
    symbols = _poly_symbols(poly)
    if expolist is None or coeffs is None:
        return _symbolica_text(poly)
    terms: list[str] = []
    for powers, coeff in zip(np.asarray(expolist, dtype=object).tolist(), list(coeffs)):
        factors = [f"({_symbolica_text(coeff)})"]
        for symbol, power in zip(symbols, powers):
            power_int = int(power)
            if power_int == 0:
                continue
            if power_int == 1:
                factors.append(symbol)
            else:
                factors.append(f"{symbol}^{power_int}")
        terms.append("*".join(factors))
    return " + ".join(terms) if terms else "0"


def _polynomial_to_expr(poly: Any) -> Any:
    """Convert a pySecDec polynomial-like object into a Symbolica expression."""
    return E(_polynomial_to_symbolica_text(poly))


def _affine_epsilon(expr: Any) -> EpsilonExpansion:
    """Extract ``base + coeff*eps`` from a pySecDec exponent using Symbolica."""
    text = _symbolica_text(expr)
    evaluator = E(text).evaluator([S("eps")])
    base = float(evaluator.evaluate([[0.0]])[0][0])
    at_one = float(evaluator.evaluate([[1.0]])[0][0])
    return EpsilonExpansion(base=base, eps_coeff=at_one - base)


def _prefactor_series(expr: Any, max_order: int) -> tuple[int, list[complex]]:
    """Laurent-expand a pySecDec global prefactor around ``eps=0``.

    Symbolica's lowercase ``gamma`` is meromorphic and its ``series`` method
    exposes negative powers, so this handles prefactors such as ``gamma(eps)``
    without falling back to numeric finite differences.  The returned list is
    ordered from ``min_order`` through ``max_order``.
    """
    text = _symbolica_text(expr).replace("EulerGamma", "γ")
    series = E(text).series(S("eps"), 0, int(max_order))
    present_orders = [int(order) for order, _coeff in series]
    min_order = min(present_orders) if present_orders else 0
    coeffs: list[complex] = []
    for order in range(min_order, int(max_order) + 1):
        coeff_expr = series[order]
        value = coeff_expr.evaluator([]).evaluate([[]])[0][0]
        coeffs.append(complex(float(value), 0.0))
    return min_order, coeffs


def _mass_parameter_names(parsed: ParsedDotGraph) -> list[str]:
    """Return symbolic nonzero masses appearing on internal lines."""
    names: list[str] = []
    for line in parsed.internal_lines:
        mass = line.mass.strip()
        try:
            float(mass)
            continue
        except ValueError:
            pass
        if mass != "0" and mass not in names:
            names.append(mass)
    return names


def _make_loop_integral(parsed: ParsedDotGraph, kin: KinematicsDefinition, request: IntegralRequest, modules: dict[str, Any]) -> Any:
    """Build pySecDec's loop-integral object for the parsed DOT graph."""
    if parsed.numerator is not None:
        propagators = parsed.graph_attr_list("propagators", separator=";")
        loop_momenta = parsed.graph_attr_list("loop_momenta")
        external_momenta = parsed.graph_attr_list("external_momenta")
        lorentz_indices = parsed.graph_attr_list("lorentz_indices")
        if not propagators or not loop_momenta:
            raise ValueError(
                f"{parsed.path}: graph-level num/numerator requires pySecDec-style "
                "graph attributes 'propagators' separated by ';' and 'loop_momenta'"
            )
        if len(propagators) != len(parsed.internal_lines):
            raise ValueError(
                f"{parsed.path}: numerator propagator routing has {len(propagators)} entries, "
                f"but the graph has {len(parsed.internal_lines)} internal lines"
            )
        return modules["LoopIntegralFromPropagators"](
            propagators,
            loop_momenta=loop_momenta,
            external_momenta=external_momenta,
            Lorentz_indices=lorentz_indices,
            numerator=parsed.numerator,
            replacement_rules=kin.pysecdec_replacement_rules(),
            Feynman_parameters="x",
            regulators=["eps"],
            dimensionality="4-2*eps",
        )

    missing_masses = [name for name in _mass_parameter_names(parsed) if name not in kin.values]
    if missing_masses:
        raise ValueError(
            f"{parsed.path}: missing numeric values for mass symbols: {', '.join(missing_masses)}"
        )
    internal_lines: list[list[object]] = []
    for line in parsed.internal_lines:
        mass = line.mass.strip()
        try:
            float(mass)
            resolved_mass = mass
        except ValueError:
            resolved_mass = kin.pysecdec_value_for_symbol(mass)
        internal_lines.append(
            [
                resolved_mass,
                [parsed.vertex_ids[line.source], parsed.vertex_ids[line.target]],
            ]
        )
    return modules["LoopIntegralFromGraph"](
        internal_lines,
        parsed.pysecdec_external_lines(),
        replacement_rules=kin.pysecdec_replacement_rules(),
        Feynman_parameters="x",
        regulators=["eps"],
        dimensionality="4-2*eps",
    )


def _identity_polynomials(li: Any, modules: dict[str, Any]) -> list[Any]:
    """Create pySecDec polynomials carrying original Feynman-parameter maps."""
    Polynomial = modules["Polynomial"]
    symbols = [str(symbol) for symbol in list(li.integration_variables)]
    out: list[Any] = []
    for index in range(len(symbols)):
        powers = [[0 for _ in symbols]]
        powers[0][index] = 1
        out.append(Polynomial(powers, [1], polysymbols=symbols))
    return out


def _coerce_polynomial_symbols(poly: Any, target_symbols: list[str], modules: dict[str, Any]) -> Any:
    """Return ``poly`` with exactly ``target_symbols`` when extra symbols are zero.

    pySecDec momentum numerators may carry bookkeeping symbols such as ``U``
    and ``F`` in their ``polysymbols`` even when every corresponding exponent
    is zero.  Sector decomposition requires all polynomials in ``other`` to
    have the same variable count, so drop those inert columns here and reject
    genuinely non-parametric numerator dependence for this first implementation.
    """
    symbols = _poly_symbols(poly)
    if symbols == target_symbols:
        return poly
    expolist = getattr(poly, "expolist", None)
    coeffs = getattr(poly, "coeffs", None)
    if expolist is None or coeffs is None:
        raise ValueError(f"cannot coerce non-polynomial numerator {poly!r}")
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    missing = [symbol for symbol in target_symbols if symbol not in symbol_index]
    if missing:
        raise ValueError(f"numerator is missing Feynman parameters: {', '.join(missing)}")
    extra = [index for index, symbol in enumerate(symbols) if symbol not in target_symbols]
    exps = np.asarray(expolist, dtype=object)
    if extra and np.any(exps[:, extra] != 0):
        extra_symbols = ", ".join(symbols[index] for index in extra)
        raise ValueError(
            "momentum numerator depends on non-Feynman polynomial symbols "
            f"({extra_symbols}); this FSD path currently supports numerators "
            "that pySecDec parametrizes as regular x-polynomials"
        )
    keep = [symbol_index[symbol] for symbol in target_symbols]
    return modules["Polynomial"](
        exps[:, keep].tolist(),
        list(coeffs),
        polysymbols=target_symbols,
    )


def _expr_to_pysecdec_polynomial(expr: Any, symbols: list[str], modules: dict[str, Any]) -> Any:
    """Convert a Symbolica x-polynomial to a pySecDec Polynomial."""
    variables = [S(symbol) for symbol in symbols]
    polynomial = expr.expand().to_polynomial(vars=variables)
    expolist: list[list[int]] = []
    coeffs: list[float] = []
    for powers, coefficient in polynomial.coefficient_list(vars=variables):
        coeff_text = str(coefficient)
        try:
            coeff_value = float(Fraction(coeff_text))
        except Exception:
            try:
                coeff_value = float(coeff_text)
            except ValueError as exc:
                raise ValueError(
                    f"custom numerator reducer produced nonnumeric coefficient {coefficient}"
                ) from exc
        try:
            coeff_value = float(coeff_value)
        except ValueError as exc:
            raise ValueError(
                f"custom numerator reducer produced nonnumeric coefficient {coefficient}"
            ) from exc
        if abs(coeff_value) <= 1.0e-15:
            continue
        expolist.append([int(power) for power in powers])
        coeffs.append(coeff_value)
    if not expolist:
        expolist = [[0 for _ in symbols]]
        coeffs = [0.0]
    return modules["Polynomial"](expolist, coeffs, polysymbols=symbols)


def _epsilon_coefficients(expr: Any) -> list[Any]:
    """Split a Symbolica expression into coefficients of ``eps``."""
    expanded = expr.expand()
    eps = S("eps")
    polynomial = expanded.to_polynomial(vars=[eps])
    degree = 0
    entries = list(polynomial.coefficient_list(vars=[eps]))
    for powers, _coefficient in entries:
        degree = max(degree, int(powers[0]))
    coeffs = [E("0") for _ in range(degree + 1)]
    for powers, coefficient in entries:
        coeffs[int(powers[0])] += coefficient if hasattr(coefficient, "expand") else E(str(coefficient))
    return [coeff.expand() for coeff in coeffs]


def _pysecdec_numerator_polynomials(
    li: Any,
    modules: dict[str, Any],
    timings: GenerationTimings,
    progress: GenerationProgress | None = None,
) -> list[Any] | None:
    """Convert pySecDec's preliminary numerator to epsilon coefficient polynomials."""
    numerator = getattr(li, "numerator", None)
    if numerator is None:
        return None
    text = _polynomial_to_symbolica_text(numerator).strip()
    if text in {"", "1", "(1)"}:
        return None
    symbols = [str(symbol) for symbol in list(li.integration_variables)]
    with timings.measure("pySecDec preliminary numerator conversion", progress=progress):
        expr = _polynomial_to_expr(numerator)
        # pySecDec's preliminary numerator may keep U and F as symbolic
        # placeholders.  Substitute the already-built pySecDec Symanzik
        # polynomials here, then split the result in eps so the sector
        # processor sees the same epsilon-polynomial layout as for the
        # FSD-owned reducer.
        expr = expr.replace(S("U"), _polynomial_to_expr(li.U))
        expr = expr.replace(S("F"), _polynomial_to_expr(li.F))
        coeffs = _epsilon_coefficients(expr)
        return [_expr_to_pysecdec_polynomial(coeff, symbols, modules) for coeff in coeffs]


def _custom_numerator_polynomials(
    li: Any,
    parsed: ParsedDotGraph,
    kin: KinematicsDefinition,
    request: IntegralRequest,
    modules: dict[str, Any],
    timings: GenerationTimings,
    progress: GenerationProgress | None = None,
) -> list[Any] | None:
    """Return FSD-reduced numerator epsilon-coefficient polynomials."""
    if parsed.numerator is None or request.numerator_reducer != "symbolica":
        return None
    propagators = parsed.graph_attr_list("propagators", separator=";")
    loop_momenta = parsed.graph_attr_list("loop_momenta")
    external_momenta = parsed.graph_attr_list("external_momenta")
    if not propagators or not loop_momenta:
        raise ValueError(
            f"{parsed.path}: --numerator-reducer symbolica requires graph attributes "
            "'propagators' and 'loop_momenta'"
        )
    with timings.measure("FSD Symbolica numerator reduction", progress=progress):
        reduced = reduce_dot_product_numerator(
            numerator=parsed.numerator,
            loop_momenta=loop_momenta,
            external_momenta=external_momenta,
            li=li,
            kinematics=kin,
        )
    symbols = [str(symbol) for symbol in list(li.integration_variables)]
    with timings.measure(
        "FSD numerator Polynomial conversion",
        f"{len(reduced.eps_coefficients)} eps coefficients, rank {reduced.highest_rank}",
        progress=progress,
    ):
        return [
            _expr_to_pysecdec_polynomial(expr, symbols, modules)
            for expr in reduced.eps_coefficients
        ]


def _numerator_polynomials(
    li: Any,
    parsed: ParsedDotGraph,
    kin: KinematicsDefinition,
    request: IntegralRequest,
    modules: dict[str, Any],
    timings: GenerationTimings,
    progress: GenerationProgress | None = None,
) -> list[Any] | None:
    """Return reduced numerator coefficient polynomials for the selected backend."""
    if parsed.numerator is None:
        return None
    if request.numerator_reducer == "symbolica":
        return _custom_numerator_polynomials(
            li,
            parsed,
            kin,
            request,
            modules,
            timings,
            progress=progress,
        )
    if request.numerator_reducer == "pysecdec":
        return _pysecdec_numerator_polynomials(li, modules, timings, progress=progress)
    raise ValueError(f"unsupported numerator reducer {request.numerator_reducer!r}")


def _sector_other_polynomials(
    li: Any,
    modules: dict[str, Any],
    numerator_polynomials: list[Any] | None = None,
) -> list[Any]:
    """Return pySecDec ``other`` polynomials: maps, then optional numerator.

    pySecDec's generated package carries momentum-space numerators as regular
    ``other_polynomials`` after parametrization.  FSD mirrors that ordering:
    the first entries recover the sector map ``x_i(y)`` and the optional last
    entry is a regular numerator factor evaluated by the sector processor.
    """
    others = _identity_polynomials(li, modules)
    if numerator_polynomials is not None:
        others.extend(numerator_polynomials)
        return others
    numerator = getattr(li, "numerator", None)
    if numerator is not None and _polynomial_to_symbolica_text(numerator).strip() not in {"", "1", "(1)"}:
        symbols = [str(symbol) for symbol in list(li.integration_variables)]
        others.append(_coerce_polynomial_symbols(numerator, symbols, modules))
    return others


def _decompose(
    li: Any,
    request: IntegralRequest,
    timings: GenerationTimings,
    modules: dict[str, Any],
    progress: GenerationProgress | None = None,
    numerator_polynomials: list[Any] | None = None,
) -> list[Any]:
    """Run the selected pySecDec decomposition and return pySecDec sectors."""
    Sector = modules["Sector"]
    initial = Sector(
        [li.U, li.F],
        other=_sector_other_polynomials(li, modules, numerator_polynomials),
    )
    normaliz = request.normaliz_executable
    with timings.measure(
        "pySecDec sector decomposition",
        request.sector_method,
        progress=progress,
    ):
        if request.sector_method == "iterative":
            sectors: list[Any] = []
            for primary in modules["primary_decomposition"](initial):
                sectors.extend(list(modules["iterative_decomposition"](primary)))
            return sectors
        workdir = f"normaliz_tmp_{os.getpid()}"
        shutil.rmtree(workdir, ignore_errors=True)
        if request.sector_method == "geometric_ku":
            # Mirror pySecDec's package writer: ``geometric_ku`` performs the
            # usual loop-integral primary decomposition and then applies the KU
            # geometric strategy to every primary sector.  Feeding a Cheng-Wu
            # sector here gives a much smaller but different sector set.
            try:
                sectors: list[Any] = []
                for primary in modules["primary_decomposition"](initial):
                    sectors.extend(
                        list(
                            modules["geometric_decomposition_ku"](
                                primary, normaliz=normaliz, workdir=workdir
                            )
                        )
                    )
                return sectors
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
        cheng_wu = modules["Cheng_Wu"](initial, index=-1)
        # pySecDec's geometric routines default to the literal directory name
        # ``normaliz_tmp``.  DOT workers may decompose the same graph in
        # parallel today, so use a process-local directory and remove it once
        # the generator has been consumed.
        try:
            return list(
                modules["geometric_decomposition"](
                    cheng_wu, normaliz=normaliz, workdir=workdir
                )
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _split_one_term_monomial(poly: Any, dimension: int) -> tuple[list[int], str]:
    """Split a one-term pySecDec monomial into powers and coefficient text."""
    if poly is None:
        return [0 for _ in range(dimension)], "1"
    expolist = getattr(poly, "expolist", None)
    coeffs = getattr(poly, "coeffs", None)
    if expolist is None or coeffs is None:
        text = _polynomial_to_symbolica_text(poly)
        return [0 for _ in range(dimension)], text
    exps = np.asarray(expolist, dtype=object)
    if exps.shape[0] != 1:
        raise ValueError(f"expected one-term monomial, got {poly}")
    coeff_list = list(coeffs)
    coeff = coeff_list[0] if coeff_list else 1
    return [int(power) for power in exps[0].tolist()], _symbolica_text(coeff)


def _split_cast_product(obj: Any, dimension: int) -> tuple[list[int], Any]:
    """Extract monomial powers and the residual polynomial from a cast product."""
    if hasattr(obj, "factors") and obj.factors:
        powers, _coeff = _split_one_term_monomial(obj.factors[0], dimension)
        residual = obj.factors[-1]
        return powers, residual
    return [0 for _ in range(dimension)], obj


def _sector_variable_names(sec: Any) -> list[str]:
    """Return pySecDec sector integration variable names."""
    for poly in list(getattr(sec, "other", [])) + list(getattr(sec, "cast", [])):
        symbols = _poly_symbols(poly)
        if symbols:
            return symbols
    raise ValueError("could not determine pySecDec sector variables")


def _convert_sector(sec: Any, index: int, topology: TopologyDefinition, request: IntegralRequest) -> SectorDefinition:
    """Convert one pySecDec sector into an FSD declarative sector."""
    variable_names = _sector_variable_names(sec)
    dimension = len(variable_names)
    u_powers, _u_residual = _split_cast_product(sec.cast[0], dimension)
    f_powers, _f_residual = _split_cast_product(sec.cast[1], dimension)
    jacobian_powers, jacobian_coeff = _split_one_term_monomial(sec.Jacobian, dimension)
    other_exprs = [_polynomial_to_expr(poly) for poly in sec.other]
    map_exprs = other_exprs[: len(topology.x_names)]
    numerator_eps_exprs = other_exprs[len(topology.x_names):] or [E("1")]
    numerator_expr = numerator_eps_exprs[0]
    if len(map_exprs) != len(topology.x_names):
        raise ValueError(
            f"pySecDec sector {index}: expected {len(topology.x_names)} recovered maps, got {len(map_exprs)}"
        )
    singular_axes: list[int] = []
    endpoint_taylor_orders = [0 for _ in range(dimension)]
    for axis in range(dimension):
        base = (
            float(jacobian_powers[axis])
            + topology.parametric_representation.u_exponent.base * float(u_powers[axis])
            + topology.parametric_representation.f_exponent.base * float(f_powers[axis])
        )
        eps_coeff = (
            topology.parametric_representation.u_exponent.eps_coeff * float(u_powers[axis])
            + topology.parametric_representation.f_exponent.eps_coeff * float(f_powers[axis])
        )
        if base < -1.0e-12:
            rounded_base = round(base)
            if abs(base - rounded_base) > 1.0e-12:
                raise ValueError(
                    f"pySecDec sector {index}: non-integer endpoint power base {base:g} "
                    f"on {variable_names[axis]}"
                )
            if abs(eps_coeff) <= 1.0e-15:
                raise ValueError(
                    f"pySecDec sector {index}: endpoint power {base:g} on {variable_names[axis]} "
                    "has no epsilon regulator"
                )
            singular_axes.append(axis)
            endpoint_taylor_orders[axis] = int(-rounded_base - 1)
    subtraction = (
        "finite"
        if not singular_axes
        else f"{len(singular_axes)}-axis recursive Taylor endpoint subtraction"
    )
    return SectorDefinition(
        name=f"PSD{index}",
        integration_dim=dimension,
        variable_names=variable_names,
        map_exprs=map_exprs,
        regular_jacobian_expr=E(jacobian_coeff),
        numerator_expr=numerator_expr,
        numerator_eps_exprs=numerator_eps_exprs,
        u_monomial_powers=u_powers,
        f_monomial_powers=f_powers,
        jacobian_monomial_powers=jacobian_powers,
        singular_axes=singular_axes,
        subtraction_type=subtraction,
        description="pySecDec-generated DOT sector",
        jit_compile_evaluators=request.jit_compile_evaluators,
        evaluator_compile_mode=request.evaluator_compile_mode,
        real_evaluator=request.real_evaluator,
        endpoint_taylor_orders=endpoint_taylor_orders,
    )


def build_dot_bundle(
    parsed: ParsedDotGraph,
    kin: KinematicsDefinition,
    request: IntegralRequest,
    progress: GenerationProgress | None = None,
) -> DotBuildBundle:
    """Build topology and sectors for FSD from DOT via pySecDec."""
    modules = require_pysecdec()
    timings = GenerationTimings()
    constructor_label = "pySecDec LoopIntegralFromPropagators" if parsed.numerator is not None else "pySecDec LoopIntegralFromGraph"
    with timings.measure(constructor_label, progress=progress):
        li = _make_loop_integral(parsed, kin, request, modules)
    with timings.measure("U/F extraction", progress=progress):
        u_expr = _polynomial_to_expr(li.U)
        f_expr = _polynomial_to_expr(li.F)
        u_exp = _affine_epsilon(li.exponent_U)
        f_exp = _affine_epsilon(li.exponent_F)
    param_names = kin.parameter_names
    param_values = kin.parameter_values
    with timings.measure("Symbolica scalar evaluator build", progress=progress):
        topology = TopologyDefinition(
            family=f"DOT[{parsed.graph_name}]",
            x_names=[str(symbol) for symbol in list(li.integration_variables)],
            parameter_names=param_names,
            parameter_values=param_values,
            u_expr=u_expr,
            f_expr=f_expr,
            u_power_base=u_exp.base,
            f_power_base=-f_exp.base,
            eps_log_u_coeff=u_exp.eps_coeff,
            eps_log_f_coeff=f_exp.eps_coeff,
            expected_laurent_orders=["eps^0"],
            convention_note="DOT scalar integral in pySecDec/FSD sector convention",
            parametric_representation=ParametricRepresentation(
                loop_count=parsed.loop_count,
                propagator_powers=tuple(1.0 for _ in parsed.internal_lines),
                dimension=EpsilonExpansion(4.0, -2.0),
                gamma_argument=EpsilonExpansion(0.0, 0.0),
                u_exponent=u_exp,
                f_exponent=f_exp,
                parameter_weight_powers=tuple(0.0 for _ in parsed.internal_lines),
                prefactor_description=f"pySecDec Gamma/global factor: {li.Gamma_factor}",
                convention_description="FSD coefficients are before convolution with the pySecDec global prefactor",
            ),
            jit_compile_evaluators=request.jit_compile_evaluators,
            evaluator_compile_mode=request.evaluator_compile_mode,
            real_evaluator=request.real_evaluator,
            dual_evaluator_mode=request.dual_evaluator_mode,
            ibp_reduce_to_log_endpoint=request.ibp_reduce_to_log_endpoint,
            ibp_power_goal=request.ibp_power_goal,
        )
    numerator_polynomials = _numerator_polynomials(
        li,
        parsed,
        kin,
        request,
        modules,
        timings,
        progress=progress,
    )
    pysecdec_sectors = _decompose(
        li,
        request,
        timings,
        modules,
        progress=progress,
        numerator_polynomials=numerator_polynomials,
    )
    with timings.measure(
        "FSD SectorDefinition conversion",
        f"{len(pysecdec_sectors)} sectors",
        progress=progress,
    ):
        sectors = [_convert_sector(sec, i, topology, request) for i, sec in enumerate(pysecdec_sectors)]
    max_sector_depth = max((len(sector.singular_axes) for sector in sectors), default=0)
    universal_depth = 2 * parsed.loop_count
    if max_sector_depth > universal_depth:
        worst = [
            sector.name for sector in sectors if len(sector.singular_axes) == max_sector_depth
        ][:5]
        raise ValueError(
            f"sector endpoint pole depth {max_sector_depth} exceeds the scalar 2L depth "
            f"{universal_depth}; examples: {', '.join(worst)}"
        )
    min_order = -universal_depth
    prefactor_series_max_order = max(0, int(request.max_eps_order) - min_order)
    prefactor_min_order, prefactor_coeffs = _prefactor_series(
        li.Gamma_factor,
        prefactor_series_max_order,
    )
    display_min_order = (
        min_order + int(prefactor_min_order)
        if request.prefactor_convention == "pysecdec"
        else min_order
    )
    if request.max_eps_order < display_min_order:
        raise ValueError(
            f"--max-eps-order must be >= eps^{display_min_order}; got eps^{request.max_eps_order}"
        )
    topology.global_prefactor_min_order = int(prefactor_min_order)
    topology.global_prefactor_coeffs = prefactor_coeffs
    sector_max_order = int(request.max_eps_order)
    if request.prefactor_convention == "pysecdec":
        sector_max_order = int(request.max_eps_order) - int(prefactor_min_order)
    topology.set_laurent_range(min_order, sector_max_order)
    if request.command == "generate":
        topology.chain_rule_metadata_only = True
        if request.output is not None:
            topology.streaming_evaluator_cache_dir = str(
                Path(request.output).expanduser().resolve() / ".stream_evaluator_cache"
            )
    with timings.measure(
        "Symbolica sector evaluator build",
        f"{len(sectors)} sectors",
        progress=progress,
    ):
        prepare_sector_evaluators(sectors, progress=progress, include_dual=False)
    if request.sector_evaluator_backend not in {"two-stage-explicit", "explicit"}:
        with timings.measure(
            "Symbolica Taylor evaluator build",
            request.dual_evaluator_mode,
            progress=progress,
        ):
            topology.prepare_dual_evaluators(sectors, request.dual_evaluator_mode, progress=progress)
    else:
        timings.add(
            "Symbolica Taylor evaluator build",
            0.0,
            detail=(
                "skipped: sector evaluator backend prepares explicit sector "
                "integrand evaluators"
            ),
        )
    return DotBuildBundle(
        topology=topology,
        sectors=sectors,
        timings=timings,
        loop_integral=li,
        parsed_graph=parsed,
        kinematics=kin,
    )


def run_pysecdec_package(
    bundle: DotBuildBundle,
    request: IntegralRequest,
    progress: GenerationProgress | None = None,
) -> PySecDecRunResult:
    """Generate, compile, and run pySecDec's own integrator for comparison."""
    modules = require_pysecdec()
    timings = GenerationTimings()
    workdir = Path(request.pysecdec_workdir).expanduser().resolve()
    name = f"fsd_psd_{bundle.parsed_graph.graph_name}"
    package_dir = workdir / name
    if package_dir.exists() and not request.keep_pysecdec_workdir:
        shutil.rmtree(package_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        os.chdir(workdir)
        with timings.measure("pySecDec package generation", progress=progress):
            modules["loop_package"](
                name,
                bundle.loop_integral,
                requested_orders=[bundle.topology.laurent_max_order],
                real_parameters=bundle.kinematics.parameter_names,
                contour_deformation=False,
                decomposition_method=request.sector_method,
                normaliz_executable=request.normaliz_executable,
                enforce_complex=True,
            )
        with timings.measure("pySecDec package compile", "make pylink", progress=progress):
            subprocess.run(["make", "-C", str(package_dir), "pylink", "-j"], check=True)
        shared = package_dir / f"{name}_pylink.so"
        with timings.measure("pySecDec package load", progress=progress):
            integral = modules["IntegralLibrary"](str(shared))
        with timings.measure("pySecDec integration", progress=progress):
            series = integral(
                real_parameters=bundle.kinematics.parameter_values,
                epsrel=request.pysecdec_epsrel,
                maxeval=request.pysecdec_maxeval,
                format="json",
                verbose=False,
            )
    finally:
        os.chdir(cwd)

    orders, coeffs, errors = _parse_pysecdec_json_series(series)
    return PySecDecRunResult(
        coeffs=coeffs,
        errors=errors,
        orders=orders,
        raw_series=series,
        timings=timings,
    )


def _parse_pysecdec_json_series(series_text: str) -> tuple[list[int], list[complex], list[complex]]:
    """Parse pySecDec JSON series into coefficient/error arrays."""
    if isinstance(series_text, tuple):
        data = series_text[0]
    elif isinstance(series_text, str):
        data = json.loads(series_text)
    else:
        data = series_text
    entries = data.get("sums", data.get("sum", data)) if isinstance(data, dict) else data
    if isinstance(entries, dict) and entries and all(isinstance(key, str) for key in entries):
        # pySecDec dict format nests the named sum one level below "sums".
        first_value = next(iter(entries.values()))
        if isinstance(first_value, dict):
            entries = first_value
    if isinstance(entries, dict) and "orders" in entries:
        raw_terms = entries["orders"]
    elif isinstance(entries, list):
        raw_terms = entries
    else:
        raw_terms = entries
    coeff_by_order: dict[int, complex] = {}
    err_by_order: dict[int, complex] = {}
    if isinstance(raw_terms, dict):
        iterable = raw_terms.items()
    else:
        iterable = enumerate(raw_terms)
    for key, value in iterable:
        try:
            order = int(key[0] if isinstance(key, tuple) else key)
        except Exception:
            order = int(value.get("order", 0))
        if isinstance(value, dict):
            central = value.get("value", value.get("result", 0.0))
            error = value.get("error", value.get("uncertainty", 0.0))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            central, error = value
        else:
            central = value
            error = 0.0
        coeff_by_order[order] = _json_complex(central)
        err_by_order[order] = _json_complex(error)
    if not coeff_by_order:
        return [], [], []
    min_order = min(coeff_by_order)
    max_order = max(coeff_by_order)
    orders = list(range(min_order, max_order + 1))
    coeffs = [coeff_by_order.get(order, 0.0 + 0.0j) for order in range(min_order, max_order + 1)]
    errors = [err_by_order.get(order, 0.0 + 0.0j) for order in range(min_order, max_order + 1)]
    return orders, coeffs, errors


def _json_complex(value: Any) -> complex:
    """Best-effort conversion of pySecDec JSON number formats to complex."""
    if isinstance(value, dict):
        return complex(float(value.get("re", value.get("real", 0.0))), float(value.get("im", value.get("imag", 0.0))))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return complex(float(value[0]), float(value[1]))
    return complex(value)
