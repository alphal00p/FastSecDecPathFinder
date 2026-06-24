"""pySecDec boundary for DOT-driven FSD topologies.

This is intentionally the only FSD-owned module importing pySecDec.  The rest
of the code receives ordinary ``TopologyDefinition`` and ``SectorDefinition``
objects backed by Symbolica evaluators.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import json
import re
import os
import shutil
import subprocess
from types import SimpleNamespace
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
from symbolic_constants import symbolica_euler_gamma_decimal, symbolica_zeta_decimal


@dataclass
class DotBuildBundle:
    """Topology, sector, and timing objects created from pySecDec."""

    topology: TopologyDefinition
    sectors: list[SectorDefinition]
    timings: GenerationTimings
    loop_integral: Any
    parsed_graph: ParsedDotGraph
    kinematics: KinematicsDefinition


@dataclass(frozen=True)
class UFTopologyData:
    """Direct Symanzik-polynomial topology input."""

    family: str
    x_names: list[str]
    parameter_names: list[str]
    parameter_values: list[float]
    u_expr_text: str
    f_expr_text: str
    loop_count: int
    dimension: EpsilonExpansion
    propagator_powers: tuple[float, ...]
    measure_powers: tuple[float, ...]
    u_exponent: EpsilonExpansion
    f_exponent: EpsilonExpansion
    global_prefactor: str


@dataclass
class UFBuildBundle:
    """Topology, sector, and timing objects created from direct U/F input."""

    topology: TopologyDefinition
    sectors: list[SectorDefinition]
    timings: GenerationTimings
    loop_integral: Any
    source: UFTopologyData


@dataclass
class PySecDecRunResult:
    """Numerical pySecDec integration result in pySecDec convention."""

    coeffs: list[complex]
    errors: list[complex]
    orders: list[int]
    raw_series: Any
    timings: GenerationTimings


@dataclass(frozen=True)
class PySecDecPackagePaths:
    """Filesystem locations for a generated pySecDec package."""

    package_dir: Path
    integral_dir: Path
    shared_library: Path


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
        from pySecDec.decomposition.common import squash_symmetry_redundant_sectors_sort
        from pySecDec.decomposition.common import _collision_safe_hash, _sector2array
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
        from pySecDec.matrix_sort import Pak_sort, iterative_sort, light_Pak_sort
        from pySecDec.misc import argsort_ND_array
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
        "squash_symmetry_redundant_sectors_sort": squash_symmetry_redundant_sectors_sort,
        "_collision_safe_hash": _collision_safe_hash,
        "_sector2array": _sector2array,
        "iterative_decomposition": iterative_decomposition,
        "primary_decomposition": primary_decomposition,
        "IntegralLibrary": IntegralLibrary,
        "LoopIntegralFromGraph": LoopIntegralFromGraph,
        "LoopIntegralFromPropagators": LoopIntegralFromPropagators,
        "loop_package": loop_package,
        "Pak_sort": Pak_sort,
        "iterative_sort": iterative_sort,
        "light_Pak_sort": light_Pak_sort,
        "argsort_ND_array": argsort_ND_array,
    }


def pysecdec_package_paths(bundle: DotBuildBundle, request: IntegralRequest) -> PySecDecPackagePaths:
    """Return canonical package paths for a DOT pySecDec generated package."""
    workdir = Path(request.pysecdec_workdir).expanduser().resolve()
    name = f"fsd_psd_{bundle.parsed_graph.graph_name}"
    package_dir = workdir / name
    integral_dir = package_dir / f"{name}_integral"
    return PySecDecPackagePaths(
        package_dir=package_dir,
        integral_dir=integral_dir,
        shared_library=package_dir / f"{name}_pylink.so",
    )


def _pysecdec_generation_log_path(paths: PySecDecPackagePaths) -> Path:
    """Return the log file collecting suppressed pySecDec/FORM/make output."""
    return paths.package_dir.parent / f"{paths.package_dir.name}_generation.log"


@contextlib.contextmanager
def _redirect_process_output_to(log_handle: Any):
    """Redirect Python and child-process stdout/stderr to ``log_handle``.

    ``contextlib.redirect_stdout`` only replaces ``sys.stdout`` and misses
    subprocesses spawned by pySecDec/FORM.  pySecDec generation is noisy exactly
    through those descendants, so temporarily redirect the OS file descriptors
    as well.
    """
    sys_stdout = os.dup(1)
    sys_stderr = os.dup(2)
    try:
        log_handle.flush()
        os.dup2(log_handle.fileno(), 1)
        os.dup2(log_handle.fileno(), 2)
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            yield
    finally:
        os.dup2(sys_stdout, 1)
        os.dup2(sys_stderr, 2)
        os.close(sys_stdout)
        os.close(sys_stderr)


def ensure_pysecdec_package(
    bundle: DotBuildBundle,
    request: IntegralRequest,
    *,
    progress: GenerationProgress | None = None,
) -> tuple[PySecDecPackagePaths, GenerationTimings]:
    """Ensure pySecDec generated C++ package artifacts exist on disk."""
    modules = require_pysecdec()
    timings = GenerationTimings()
    paths = pysecdec_package_paths(bundle, request)
    if paths.package_dir.exists() and not request.keep_pysecdec_workdir:
        shutil.rmtree(paths.package_dir)
    paths.package_dir.parent.mkdir(parents=True, exist_ok=True)

    metadata = paths.integral_dir / "disteval" / f"{paths.integral_dir.name}.json"
    static_library = paths.integral_dir / f"lib{paths.integral_dir.name}.a"
    if (
        request.keep_pysecdec_workdir
        and paths.package_dir.exists()
        and paths.shared_library.is_file()
        and metadata.is_file()
        and static_library.is_file()
    ):
        timings.add(
            "pySecDec package generation",
            0.0,
            detail="reused existing generated package",
        )
        timings.add(
            "pySecDec package compile",
            0.0,
            detail="reused existing compiled package",
        )
        return paths, timings

    cwd = Path.cwd()
    log_path = _pysecdec_generation_log_path(paths)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_detail = "" if request.show_pysecdec_output else f"captured output: {log_path}"
    try:
        os.chdir(paths.package_dir.parent)
        with (
            contextlib.nullcontext()
            if request.show_pysecdec_output
            else log_path.open("w", encoding="utf-8")
        ) as log_handle:
            with timings.measure(
                "pySecDec package generation",
                log_detail,
                progress=progress,
            ):
                with (
                    contextlib.nullcontext()
                    if request.show_pysecdec_output
                    else _redirect_process_output_to(log_handle)
                ):
                    modules["loop_package"](
                        paths.package_dir.name,
                        bundle.loop_integral,
                        requested_orders=[bundle.topology.laurent_max_order],
                        real_parameters=bundle.kinematics.parameter_names,
                        contour_deformation=False,
                        decomposition_method=request.sector_method,
                        normaliz_executable=request.normaliz_executable,
                        enforce_complex=True,
                    )
            compile_detail = (
                "make pylink"
                if request.show_pysecdec_output
                else f"make pylink; captured output: {log_path}"
            )
            with timings.measure("pySecDec package compile", compile_detail, progress=progress):
                subprocess.run(
                    ["make", "-C", str(paths.package_dir), "pylink", "-j"],
                    check=True,
                    stdout=None if request.show_pysecdec_output else log_handle,
                    stderr=None if request.show_pysecdec_output else subprocess.STDOUT,
                )
    finally:
        os.chdir(cwd)
    return paths, timings


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


_SINGLE_GAMMA_RE = re.compile(r"^gamma\((.*)\)$")


def _regular_series_multiply(
    left: list[complex],
    right: list[complex],
    max_order: int,
) -> list[complex]:
    """Multiply two regular epsilon series through ``max_order``."""
    out = [0.0 + 0.0j for _ in range(int(max_order) + 1)]
    for left_order, left_value in enumerate(left[: int(max_order) + 1]):
        for right_order, right_value in enumerate(right[: int(max_order) + 1 - left_order]):
            out[left_order + right_order] += left_value * right_value
    return out


def _regular_series_exp(log_coeffs: list[complex], max_order: int) -> list[complex]:
    """Exponentiate a regular epsilon series with zero constant term."""
    out = [1.0 + 0.0j] + [0.0 + 0.0j for _ in range(int(max_order))]
    power = [1.0 + 0.0j] + [0.0 + 0.0j for _ in range(int(max_order))]
    factorial = 1.0
    for term_order in range(1, int(max_order) + 1):
        factorial *= float(term_order)
        power = _regular_series_multiply(power, log_coeffs, int(max_order))
        for output_order, value in enumerate(power):
            out[output_order] += value / factorial
    return out


def _gamma_one_plus_affine_eps_series(eps_coeff: float, max_order: int) -> list[complex]:
    """Return ``Gamma(1 + eps_coeff*eps)`` through ``eps^max_order``."""
    precision = max(50, 10 * (int(max_order) + 2))
    log_coeffs = [0.0 + 0.0j for _ in range(int(max_order) + 1)]
    if max_order >= 1:
        log_coeffs[1] = -float(symbolica_euler_gamma_decimal(precision)) * float(eps_coeff)
    for order in range(2, int(max_order) + 1):
        log_coeffs[order] = (
            ((-1.0) ** order)
            * float(symbolica_zeta_decimal(order, precision))
            * (float(eps_coeff) ** order)
            / float(order)
        )
    return _regular_series_exp(log_coeffs, int(max_order))


def _single_affine_gamma_series(text: str, max_order: int) -> tuple[int, list[complex]] | None:
    """Analytically evaluate Symbolica's single affine-Gamma series.

    Symbolica builds the correct series structure for ``gamma(2+eps)``, but in
    the current development wheel its numerical evaluator gives an inaccurate
    value for ``polygamma(1,2)``.  This helper only handles a single
    optionally signed/scaled ``gamma(a+b*eps)`` with non-negative integer
    ``a``; other prefactors still use the generic Symbolica series fallback
    below.
    """
    cleaned = text.replace(" ", "")
    scale = 1.0
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    elif cleaned.startswith("-"):
        scale = -1.0
        cleaned = cleaned[1:]
    if not cleaned.startswith("gamma("):
        factor, separator, rest = cleaned.partition("*")
        if not separator or not rest.startswith("gamma("):
            return None
        try:
            scale *= float(factor)
        except ValueError:
            return None
        cleaned = rest
    match = _SINGLE_GAMMA_RE.match(cleaned)
    if match is None:
        return None
    affine = _affine_epsilon(match.group(1))
    base_rounded = round(affine.base)
    if abs(affine.base - base_rounded) > 1.0e-12:
        return None
    if abs(affine.eps_coeff) <= 1.0e-15:
        return None
    base = int(base_rounded)
    if base < 0:
        return None

    min_order = -1 if base == 0 else 0
    regular_max_order = int(max_order) - min_order
    coeffs = _gamma_one_plus_affine_eps_series(affine.eps_coeff, regular_max_order)
    if base == 0:
        return min_order, [scale * value / affine.eps_coeff for value in coeffs]
    for offset in range(1, base):
        coeffs = _regular_series_multiply(
            coeffs,
            [float(offset) + 0.0j, float(affine.eps_coeff) + 0.0j],
            regular_max_order,
        )
    return min_order, [scale * value for value in coeffs]


def _prefactor_series(expr: Any, max_order: int) -> tuple[int, list[complex]]:
    """Laurent-expand a pySecDec global prefactor around ``eps=0``.

    Symbolica's lowercase ``gamma`` is meromorphic and its ``series`` method
    exposes negative powers, so this handles prefactors such as ``gamma(eps)``
    without falling back to numeric finite differences.  The returned list is
    ordered from ``min_order`` through ``max_order``.
    """
    text = _symbolica_text(expr).replace("EulerGamma", "γ")
    analytic_gamma = _single_affine_gamma_series(text, int(max_order))
    if analytic_gamma is not None:
        return analytic_gamma
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
    powerlist = [int(line.power) for line in parsed.internal_lines]
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
            powerlist=powerlist,
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
        powerlist=powerlist,
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
    map_polynomial_count = len(list(li.integration_variables))
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
            return _squash_pysecdec_sector_symmetries(
                sectors,
                request,
                timings,
                modules,
                progress,
                ignored_other_count=map_polynomial_count,
            )
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
                return _squash_pysecdec_sector_symmetries(
                    sectors,
                    request,
                    timings,
                    modules,
                    progress,
                    ignored_other_count=map_polynomial_count,
                )
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
        cheng_wu = modules["Cheng_Wu"](initial, index=-1)
        # pySecDec's geometric routines default to the literal directory name
        # ``normaliz_tmp``.  DOT workers may decompose the same graph in
        # parallel today, so use a process-local directory and remove it once
        # the generator has been consumed.
        try:
            sectors = list(
                modules["geometric_decomposition"](
                    cheng_wu, normaliz=normaliz, workdir=workdir
                )
            )
            return _squash_pysecdec_sector_symmetries(
                sectors,
                request,
                timings,
                modules,
                progress,
                ignored_other_count=map_polynomial_count,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _squash_pysecdec_sector_symmetries(
    sectors: list[Any],
    request: IntegralRequest,
    timings: GenerationTimings,
    modules: dict[str, Any],
    progress: GenerationProgress | None = None,
    ignored_other_count: int = 0,
) -> list[Any]:
    """Apply pySecDec's package-writer sector symmetry reduction.

    pySecDec's generated C++ package does not integrate the raw decomposed
    sectors directly.  It first removes sectors related by integration-variable
    permutations, adding their Jacobian coefficients to preserve the integral.
    FSD must do the same before converting sectors; otherwise QMC comparisons
    use a different sector partition from pySecDec and can have very different
    variance at the same nominal lattice size.
    """
    reduced = list(sectors)
    details: list[str] = [f"raw={len(reduced)}"]
    with timings.measure("pySecDec sector symmetry squashing", progress=progress):
        if getattr(request, "pysecdec_use_iterative_sort", True):
            reduced = _squash_fsd_sector_symmetries_preserving_maps(
                reduced,
                modules["iterative_sort"],
                modules,
                ignored_other_count=ignored_other_count,
            )
            details.append(f"iterative={len(reduced)}")
        if getattr(request, "pysecdec_use_light_pak", True):
            reduced = _squash_fsd_sector_symmetries_preserving_maps(
                reduced,
                modules["light_Pak_sort"],
                modules,
                ignored_other_count=ignored_other_count,
            )
            details.append(f"lightPak={len(reduced)}")
        if getattr(request, "pysecdec_use_pak", True):
            reduced = _squash_fsd_sector_symmetries_preserving_maps(
                reduced,
                modules["Pak_sort"],
                modules,
                ignored_other_count=ignored_other_count,
            )
            details.append(f"Pak={len(reduced)}")
    timings.add("pySecDec sector symmetry summary", 0.0, detail=", ".join(details))
    return reduced


def _sector_for_symmetry(sec: Any, modules: dict[str, Any], ignored_other_count: int) -> Any:
    """Return a pySecDec sector view for symmetry tests.

    FSD inserts identity Feynman-parameter maps in ``Sector.other`` solely to
    recover the final sector map.  These maps are not part of pySecDec's
    generated integrand and would artificially prevent permutation symmetry
    detection.  Optional entries after those maps, such as numerator
    polynomials, are retained because they are genuine integrand data.
    """
    if ignored_other_count <= 0:
        return sec
    return modules["Sector"](
        sec.cast,
        other=list(sec.other)[int(ignored_other_count):],
        Jacobian=sec.Jacobian,
    )


def _squash_fsd_sector_symmetries_preserving_maps(
    sectors: list[Any],
    sort_function: Any,
    modules: dict[str, Any],
    ignored_other_count: int = 0,
) -> list[Any]:
    """pySecDec sort-based symmetry squashing while preserving FSD maps.

    This mirrors ``squash_symmetry_redundant_sectors_sort`` but hashes a view
    of each sector with FSD's bookkeeping map polynomials removed.  The output
    sectors are copies of the original representatives, so their recovered maps
    are still available.  Duplicate representatives are accounted for by adding
    their Jacobian coefficients, as pySecDec does.
    """
    if not sectors:
        return []
    symmetry_sectors = [
        _sector_for_symmetry(sec, modules, ignored_other_count)
        for sec in sectors
    ]
    all_expolists: list[Any] = []
    all_coeffs: list[Any] = []
    for sec in symmetry_sectors:
        expolist, coeffs = modules["_sector2array"](sec)
        all_expolists.append(expolist)
        all_coeffs.append(coeffs)
    all_coeffs_array = np.array(all_coeffs)
    all_coeffs_array = modules["_collision_safe_hash"](
        all_coeffs_array.flatten()
    ).reshape(all_coeffs_array.shape)
    sector_arrays: list[Any] = []
    for expolist, coeffs in zip(all_expolists, all_coeffs_array):
        sector_array = np.hstack((coeffs.reshape(-1, 1), expolist))
        sort_function(sector_array)
        sector_arrays.append(sector_array)
    all_sector_arrays = np.array(sector_arrays)
    sorted_indices = modules["argsort_ND_array"](all_sector_arrays)
    previous_index = int(sorted_indices[0])
    previous_array = all_sector_arrays[previous_index]
    previous_sector = sectors[previous_index].copy()
    output = [previous_sector]
    for sector_index_raw in sorted_indices[1:]:
        sector_index = int(sector_index_raw)
        if np.array_equal(all_sector_arrays[sector_index], previous_array):
            previous_sector.Jacobian.coeffs[0] += sectors[sector_index].Jacobian.coeffs[0]
            continue
        previous_index = sector_index
        previous_array = all_sector_arrays[previous_index]
        previous_sector = sectors[previous_index].copy()
        output.append(previous_sector)
    return output


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


def _coefficient_is_one(text: str) -> bool:
    """Return whether a monomial coefficient is numerically the unit constant."""
    stripped = str(text).strip()
    if stripped in {"1", "+1", "(1)", "+(1)", "1.0", "+1.0"}:
        return True
    try:
        value = E(stripped).evaluator([]).evaluate([[]])[0][0]
        return abs(float(value) - 1.0) <= 1.0e-14
    except Exception:
        return False


def _measure_monomial_powers_from_maps(
    sec: Any,
    topology: TopologyDefinition,
    dimension: int,
) -> list[float]:
    """Return sector-variable powers from original ``x_i^(nu_i-1)`` weights.

    pySecDec exposes the transformed Feynman parameters in ``Sector.other``.
    In this first propagator-power implementation we only support the
    monomial maps produced by the existing decomposition paths.  A nontrivial
    regular residual in ``X_i(y)^(nu_i-1)`` would need a separate regular
    measure factor, so fail clearly instead of silently dropping it.
    """
    parametric = topology.parametric_representation
    if parametric is None:
        return [0.0 for _ in range(dimension)]
    weights = list(parametric.parameter_weight_powers)
    if len(weights) != len(topology.x_names):
        raise ValueError(
            f"{topology.family}: parameter_weight_powers has length {len(weights)}, "
            f"expected {len(topology.x_names)}"
        )
    maps = list(sec.other)[: len(topology.x_names)]
    out = [0.0 for _ in range(dimension)]
    for x_index, (weight, map_poly) in enumerate(zip(weights, maps)):
        if abs(float(weight)) <= 1.0e-15:
            continue
        rounded_weight = round(float(weight))
        if abs(float(weight) - rounded_weight) > 1.0e-12 or rounded_weight < 0:
            raise ValueError(
                f"{topology.family}: parameter weight {weight:g} for x{x_index} "
                "is unsupported; only non-negative integer weights are implemented"
            )
        powers, coeff = _split_one_term_monomial(map_poly, dimension)
        if not _coefficient_is_one(coeff):
            raise ValueError(
                f"{topology.family}: transformed x{x_index} has non-unit regular "
                f"coefficient {coeff!r}; non-unit propagator powers require a "
                "regular measure residual, which is not implemented yet"
            )
        for axis, power in enumerate(powers):
            out[axis] += float(rounded_weight * int(power))
    return out


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
    measure_powers = _measure_monomial_powers_from_maps(sec, topology, dimension)
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
            + float(measure_powers[axis])
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
        measure_monomial_powers=measure_powers,
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
        propagator_powers = tuple(float(line.power) for line in parsed.internal_lines)
        parameter_weight_powers = tuple(float(line.power - 1) for line in parsed.internal_lines)
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
                propagator_powers=propagator_powers,
                dimension=EpsilonExpansion(4.0, -2.0),
                gamma_argument=EpsilonExpansion(0.0, 0.0),
                u_exponent=u_exp,
                f_exponent=f_exp,
                parameter_weight_powers=parameter_weight_powers,
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
        if request.sector_evaluator_backend == "explicit":
            # The fully explicit backend substitutes the sector map and regular
            # Jacobian into one sector-level evaluator during the later
            # explicit-sector build.  Compiling separate map/Jacobian support
            # evaluators is therefore pure generation overhead, and it is
            # especially expensive in ``--compile`` mode because it launches a
            # compiler invocation for every tiny map/Jacobian callback.
            timings.add(
                "Symbolica sector evaluator build skipped",
                0.0,
                detail="explicit backend uses sector-level evaluators directly",
            )
        else:
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


def build_native_pysecdec_dot_bundle(
    parsed: ParsedDotGraph,
    kin: KinematicsDefinition,
    request: IntegralRequest,
    progress: GenerationProgress | None = None,
) -> DotBuildBundle:
    """Build only the pySecDec objects needed for native pySecDec integration.

    This path deliberately skips the FSD-side U/F Symbolica evaluators,
    sector conversion, and endpoint-subtraction formula preparation.  Native
    pySecDec mode hands the loop integral to ``loop_package`` and uses the
    generated pySecDec evaluator as the numerical integrand.
    """
    modules = require_pysecdec()
    timings = GenerationTimings()
    constructor_label = (
        "pySecDec LoopIntegralFromPropagators"
        if parsed.numerator is not None
        else "pySecDec LoopIntegralFromGraph"
    )
    with timings.measure(constructor_label, progress=progress):
        li = _make_loop_integral(parsed, kin, request, modules)
    topology = SimpleNamespace(
        family=f"DOT[{parsed.graph_name}]",
        laurent_max_order=int(request.max_eps_order),
    )
    return DotBuildBundle(
        topology=topology,
        sectors=[],
        timings=timings,
        loop_integral=li,
        parsed_graph=parsed,
        kinematics=kin,
    )


def _symbolica_to_pysecdec_text(text: str) -> str:
    """Convert the small expression subset accepted in run YAML to pySecDec syntax."""
    return str(text).replace("^", "**").replace("i_", "I")


def _make_uf_loop_integral(source: UFTopologyData, modules: dict[str, Any]) -> Any:
    """Build a lightweight loop-integral object from explicit U/F polynomials."""
    Polynomial = modules["Polynomial"]
    u_poly = Polynomial.from_expression(
        _symbolica_to_pysecdec_text(source.u_expr_text),
        source.x_names,
    )
    f_poly = Polynomial.from_expression(
        _symbolica_to_pysecdec_text(source.f_expr_text),
        source.x_names,
    )
    return SimpleNamespace(
        U=u_poly,
        F=f_poly,
        integration_variables=list(getattr(u_poly, "polysymbols", source.x_names)),
        exponent_U=source.u_exponent.as_text(),
        exponent_F=source.f_exponent.as_text(),
        Gamma_factor=source.global_prefactor,
        numerator=None,
    )


def build_uf_bundle(
    source: UFTopologyData,
    request: IntegralRequest,
    progress: GenerationProgress | None = None,
) -> UFBuildBundle:
    """Build topology and sectors for FSD from direct Symanzik U/F input."""
    modules = require_pysecdec()
    timings = GenerationTimings()
    with timings.measure("U/F input polynomial construction", progress=progress):
        li = _make_uf_loop_integral(source, modules)
    with timings.measure("U/F extraction", progress=progress):
        u_expr = E(source.u_expr_text)
        f_expr = E(source.f_expr_text)
    with timings.measure("Symbolica scalar evaluator build", progress=progress):
        topology = TopologyDefinition(
            family=source.family,
            x_names=list(source.x_names),
            parameter_names=list(source.parameter_names),
            parameter_values=list(source.parameter_values),
            u_expr=u_expr,
            f_expr=f_expr,
            u_power_base=source.u_exponent.base,
            f_power_base=-source.f_exponent.base,
            eps_log_u_coeff=source.u_exponent.eps_coeff,
            eps_log_f_coeff=source.f_exponent.eps_coeff,
            expected_laurent_orders=["eps^0"],
            convention_note="direct U/F scalar integral in pySecDec/FSD sector convention",
            parametric_representation=ParametricRepresentation(
                loop_count=source.loop_count,
                propagator_powers=tuple(float(value) for value in source.propagator_powers),
                dimension=source.dimension,
                gamma_argument=EpsilonExpansion(0.0, 0.0),
                u_exponent=source.u_exponent,
                f_exponent=source.f_exponent,
                parameter_weight_powers=tuple(float(value) for value in source.measure_powers),
                prefactor_description=f"direct U/F global factor: {source.global_prefactor}",
                convention_description="FSD coefficients are before convolution with the supplied global prefactor",
            ),
            jit_compile_evaluators=request.jit_compile_evaluators,
            evaluator_compile_mode=request.evaluator_compile_mode,
            real_evaluator=request.real_evaluator,
            dual_evaluator_mode=request.dual_evaluator_mode,
            ibp_reduce_to_log_endpoint=request.ibp_reduce_to_log_endpoint,
            ibp_power_goal=request.ibp_power_goal,
        )
    pysecdec_sectors = _decompose(
        li,
        request,
        timings,
        modules,
        progress=progress,
        numerator_polynomials=None,
    )
    with timings.measure(
        "FSD SectorDefinition conversion",
        f"{len(pysecdec_sectors)} sectors",
        progress=progress,
    ):
        sectors = [_convert_sector(sec, i, topology, request) for i, sec in enumerate(pysecdec_sectors)]
    max_sector_depth = max((len(sector.singular_axes) for sector in sectors), default=0)
    universal_depth = 2 * source.loop_count
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
        source.global_prefactor,
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
        if request.sector_evaluator_backend == "explicit":
            timings.add(
                "Symbolica sector evaluator build skipped",
                0.0,
                detail="explicit backend uses sector-level evaluators directly",
            )
        else:
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
    return UFBuildBundle(
        topology=topology,
        sectors=sectors,
        timings=timings,
        loop_integral=li,
        source=source,
    )


def run_pysecdec_package(
    bundle: DotBuildBundle,
    request: IntegralRequest,
    progress: GenerationProgress | None = None,
) -> PySecDecRunResult:
    """Generate, compile, and run pySecDec's own integrator for comparison."""
    modules = require_pysecdec()
    paths, timings = ensure_pysecdec_package(bundle, request, progress=progress)
    cwd = Path.cwd()
    try:
        with timings.measure("pySecDec package load", progress=progress):
            integral = modules["IntegralLibrary"](str(paths.shared_library))
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
