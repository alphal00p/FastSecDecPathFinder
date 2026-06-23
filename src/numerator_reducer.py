"""Symbolica-backed momentum numerator reduction for DOT topologies.

The reducer intentionally covers the scalar dot-product subset of pySecDec's
tensor numerator syntax: products and sums of factors such as ``k1(mu)`` and
``p2(mu)`` where every Lorentz index is fully contracted.  It avoids pySecDec's
``preliminary_numerator`` SymPy substitution path by doing the tensor Wick
contractions directly and by using Symbolica for all polynomial expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import re
from typing import Any, Iterable

from symbolica import E, S

from kinematics import KinematicsDefinition


MOMENTUM_FACTOR_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*|\d+)\s*\)")


@dataclass(frozen=True)
class MomentumOccurrence:
    """One indexed momentum factor from the numerator input."""

    momentum: str
    index: str
    is_loop: bool
    momentum_index: int


@dataclass(frozen=True)
class NumeratorTerm:
    """One expanded numerator monomial."""

    coefficient: complex
    occurrences: tuple[MomentumOccurrence, ...]

    @property
    def rank(self) -> int:
        """Return the number of loop-momentum factors in this term."""
        return sum(1 for occurrence in self.occurrences if occurrence.is_loop)


@dataclass(frozen=True)
class ReducedNumerator:
    """Feynman-parametric numerator as an epsilon-polynomial."""

    eps_coefficients: list[Any]
    highest_rank: int


def _sym(text: str) -> Any:
    """Return a Symbolica symbol."""
    return S(text)


def _expr(text: str) -> Any:
    """Parse a Symbolica expression after pySecDec syntax normalization."""
    return E(str(text).replace("**", "^"))


def _expr_zero() -> Any:
    return E("0")


def _expr_one() -> Any:
    return E("1")


def _expr_number(value: float | int | complex) -> Any:
    """Create a real Symbolica number expression."""
    cvalue = complex(value)
    if abs(cvalue.imag) > 1.0e-15:
        raise ValueError(f"complex numerator coefficients are not supported yet: {value!r}")
    fraction = Fraction(float(cvalue.real)).limit_denominator(10**12)
    if fraction.denominator == 1:
        return E(str(fraction.numerator))
    return E(f"{fraction.numerator}/{fraction.denominator}")


def _eval_scalar(expr: Any, values: dict[str, float]) -> complex:
    """Evaluate a scalar Symbolica coefficient using numeric YAML values."""
    if not hasattr(expr, "evaluator"):
        expr = E(str(expr))
    names = list(values.keys())
    evaluator = expr.evaluator([_sym(name) for name in names])
    row = [[float(values[name]) for name in names]]
    value = evaluator.evaluate(row)[0][0]
    return complex(float(value), 0.0)


def _sanitize_numerator(
    numerator: str,
    loop_momenta: list[str],
    external_momenta: list[str],
) -> tuple[str, list[MomentumOccurrence]]:
    """Replace ``p(mu)`` factors by temporary Symbolica atoms."""
    occurrences: list[MomentumOccurrence] = []
    loop_index = {name: index for index, name in enumerate(loop_momenta)}
    external_index = {name: index for index, name in enumerate(external_momenta)}

    def repl(match: re.Match[str]) -> str:
        momentum = match.group(1)
        index = match.group(2)
        atom = f"__n{len(occurrences)}"
        if momentum in loop_index:
            occurrences.append(
                MomentumOccurrence(momentum, index, True, loop_index[momentum])
            )
        elif momentum in external_index:
            occurrences.append(
                MomentumOccurrence(momentum, index, False, external_index[momentum])
            )
        else:
            raise ValueError(
                f"numerator uses momentum {momentum!r}, not listed as loop or external momentum"
            )
        return atom

    return MOMENTUM_FACTOR_RE.sub(repl, numerator).replace("**", "^"), occurrences


def parse_dot_product_numerator(
    numerator: str,
    loop_momenta: list[str],
    external_momenta: list[str],
    values: dict[str, float],
) -> list[NumeratorTerm]:
    """Expand a dot-product numerator into scalar tensor monomials."""
    sanitized, occurrences = _sanitize_numerator(numerator, loop_momenta, external_momenta)
    atom_names = [f"__n{i}" for i in range(len(occurrences))]
    atom_symbols = [_sym(name) for name in atom_names]
    expanded = _expr(sanitized).expand()
    polynomial = expanded.to_polynomial(vars=atom_symbols)
    terms: list[NumeratorTerm] = []
    for powers, coefficient in polynomial.coefficient_list(vars=atom_symbols):
        expanded_occurrences: list[MomentumOccurrence] = []
        for atom_index, power in enumerate(powers):
            for _ in range(int(power)):
                expanded_occurrences.append(occurrences[atom_index])
        by_lorentz: dict[str, int] = {}
        for occurrence in expanded_occurrences:
            by_lorentz[occurrence.index] = by_lorentz.get(occurrence.index, 0) + 1
        bad = {index: count for index, count in by_lorentz.items() if count != 2}
        if bad:
            raise ValueError(
                "dot-product numerator terms must have every Lorentz index exactly twice; "
                f"bad counts: {bad}"
            )
        terms.append(
            NumeratorTerm(
                coefficient=_eval_scalar(coefficient, values),
                occurrences=tuple(expanded_occurrences),
            )
        )
    return terms


def _metric_pairings(items: tuple[int, ...]) -> Iterable[tuple[tuple[int, int], ...]]:
    """Yield all pairings of the selected loop-slot indices."""
    if not items:
        yield ()
        return
    first = items[0]
    for offset in range(1, len(items)):
        second = items[offset]
        rest = items[1:offset] + items[offset + 1 :]
        for pairing in _metric_pairings(rest):
            yield ((first, second), *pairing)


def _even_subsets(indices: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    """Yield all even-cardinality subsets."""
    for size in range(0, len(indices) + 1, 2):
        for subset in combinations(indices, size):
            yield tuple(subset)


def _pysecdec_poly_to_expr(poly: Any) -> Any:
    """Convert a pySecDec Polynomial-like object to a Symbolica expression."""
    expolist = getattr(poly, "expolist", None)
    coeffs = getattr(poly, "coeffs", None)
    symbols = [str(symbol) for symbol in list(getattr(poly, "polysymbols", []))]
    if expolist is None or coeffs is None:
        return _expr(str(poly))
    out = _expr_zero()
    for powers, coeff in zip(expolist.tolist(), list(coeffs)):
        term = _expr(str(coeff).replace("**", "^"))
        for symbol, power in zip(symbols, powers):
            if int(power) == 0:
                continue
            term *= _sym(symbol) ** int(power)
        out += term
    return out


def _linear_external_components(
    coeff_expr: Any,
    external_momenta: list[str],
    values: dict[str, float],
) -> list[complex]:
    """Return coefficients of a linear combination of external momenta."""
    symbols = [_sym(name) for name in external_momenta]
    polynomial = coeff_expr.to_polynomial(vars=symbols)
    components = [0.0 + 0.0j for _ in external_momenta]
    for powers, coeff in polynomial.coefficient_list(vars=symbols):
        degree = sum(int(power) for power in powers)
        if degree == 0:
            value = _eval_scalar(coeff, values)
            if abs(value) > 1.0e-14:
                raise ValueError(f"unexpected scalar term in vector coefficient: {coeff}")
            continue
        if degree != 1:
            raise ValueError(f"nonlinear external momentum coefficient: {coeff_expr}")
        index = next(i for i, power in enumerate(powers) if int(power) == 1)
        components[index] += _eval_scalar(coeff, values)
    return components


def _q_external_components(
    q_poly: Any,
    x_names: list[str],
    external_momenta: list[str],
    values: dict[str, float],
) -> list[Any]:
    """Decompose a pySecDec Q polynomial into external-vector components."""
    out = [_expr_zero() for _ in external_momenta]
    expolist = getattr(q_poly, "expolist", None)
    coeffs = getattr(q_poly, "coeffs", None)
    if expolist is None or coeffs is None:
        raise ValueError("expected pySecDec Polynomial for Q")
    for powers, coeff in zip(expolist.tolist(), list(coeffs)):
        coeff_expr = _expr(str(coeff).replace("**", "^"))
        components = _linear_external_components(coeff_expr, external_momenta, values)
        monomial = _expr_one()
        for name, power in zip(x_names, powers):
            if int(power):
                monomial *= _sym(name) ** int(power)
        for index, component in enumerate(components):
            if abs(component) > 1.0e-14:
                out[index] += _expr_number(component) * monomial
    return out


def _external_dot_map(kinematics: KinematicsDefinition) -> dict[tuple[str, str], complex]:
    """Build a symmetric external scalar-product lookup."""
    dot_map: dict[tuple[str, str], complex] = {}
    for left, value in kinematics.replacements:
        text = left.replace(" ", "")
        match = re.fullmatch(r"([A-Za-z_]\w*)\*([A-Za-z_]\w*)", text)
        if match is None:
            match_square = re.fullmatch(r"([A-Za-z_]\w*)\*\*2", text)
            if match_square is None:
                continue
            a = b = match_square.group(1)
        else:
            a, b = match.group(1), match.group(2)
        dot_map[(a, b)] = complex(float(value), 0.0)
        dot_map[(b, a)] = complex(float(value), 0.0)
    return dot_map


def _contract_vectors_and_metrics(
    vector_slots: list[tuple[int, str]],
    metric_edges: list[tuple[str, str]],
    external_momenta: list[str],
    dot_map: dict[tuple[str, str], complex],
    dimension_expr: Any,
) -> Any:
    """Evaluate scalar contractions for a tensor monomial configuration."""
    labels = set()
    for _vector, label in vector_slots:
        labels.add(label)
    for left, right in metric_edges:
        labels.add(left)
        labels.add(right)

    parent = {label: label for label in labels}

    def find(label: str) -> str:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in metric_edges:
        union(left, right)

    vectors_by_root: dict[str, list[int]] = {}
    metric_roots: set[str] = set()
    for vector, label in vector_slots:
        vectors_by_root.setdefault(find(label), []).append(vector)
    for left, _right in metric_edges:
        metric_roots.add(find(left))

    factor = _expr_one()
    for root in set(vectors_by_root) | metric_roots:
        vectors = vectors_by_root.get(root, [])
        if len(vectors) == 2:
            a = external_momenta[vectors[0]]
            b = external_momenta[vectors[1]]
            value = dot_map.get((a, b))
            if value is None:
                raise ValueError(f"missing scalar product replacement for {a}*{b}")
            factor *= _expr_number(value)
        elif len(vectors) == 0:
            factor *= dimension_expr
        else:
            raise ValueError(
                "numerator contraction left an unsupported open tensor component "
                f"with {len(vectors)} external vectors"
            )
    return factor


def _expr_degree_in(expr: Any, symbol_name: str) -> int:
    """Return the degree of a Symbolica expression in one variable."""
    symbol = _sym(symbol_name)
    polynomial = expr.to_polynomial(vars=[symbol])
    degree = 0
    for powers, _coeff in polynomial.coefficient_list(vars=[symbol]):
        degree = max(degree, int(powers[0]))
    return degree


def reduce_dot_product_numerator(
    *,
    numerator: str,
    loop_momenta: list[str],
    external_momenta: list[str],
    li: Any,
    kinematics: KinematicsDefinition,
) -> ReducedNumerator:
    """Reduce a scalar dot-product numerator to Feynman parameters.

    The returned expressions are coefficients of ``eps^k`` after substituting
    explicit U and F polynomials.  They are pure polynomials in the Feynman
    parameters and numeric kinematic values.
    """
    if any(str(power) != "1" for power in getattr(li, "powerlist", [])):
        raise ValueError("FSD-owned numerator reducer currently supports unit propagator powers only")

    x_names = [str(symbol) for symbol in list(li.integration_variables)]
    terms = parse_dot_product_numerator(
        numerator,
        loop_momenta,
        external_momenta,
        kinematics.values,
    )
    highest_rank = max((term.rank for term in terms), default=0)
    if highest_rank == 0:
        return ReducedNumerator([_expr(str(numerator))], 0)

    u_expr = _pysecdec_poly_to_expr(li.U)
    f_expr = _pysecdec_poly_to_expr(li.F)
    aM = [
        [_pysecdec_poly_to_expr(li.aM[i, j]) for j in range(li.L)]
        for i in range(li.L)
    ]
    q_components = [
        _q_external_components(li.Q[i], x_names, external_momenta, kinematics.values)
        for i in range(li.L)
    ]
    p_components: list[list[Any]] = []
    for loop_index in range(li.L):
        components: list[Any] = []
        for external_index in range(len(external_momenta)):
            value = _expr_zero()
            for q_index in range(li.L):
                value += aM[loop_index][q_index] * q_components[q_index][external_index]
            components.append(value)
        p_components.append(components)

    dimension_expr = _expr(str(li.dimensionality))
    n_nu = _expr(str(li.N_nu))
    loop_count = int(li.L)
    dot_map = _external_dot_map(kinematics)

    def scalar_factor(pair_count: int) -> Any:
        r = 2 * pair_count
        factor = (E("-1/2") ** pair_count) * (f_expr ** pair_count)
        for q in range(pair_count + 1, highest_rank // 2 + 1):
            factor *= n_nu - dimension_expr * E(str(loop_count)) / E("2") - E(str(q))
        return factor

    total = _expr_zero()
    for term in terms:
        loop_slots = [
            (slot_index, occurrence)
            for slot_index, occurrence in enumerate(term.occurrences)
            if occurrence.is_loop
        ]
        external_slots = [
            (occurrence.momentum_index, occurrence.index)
            for occurrence in term.occurrences
            if not occurrence.is_loop
        ]
        loop_slot_indices = tuple(slot_index for slot_index, _occurrence in loop_slots)
        occurrence_by_slot = {slot_index: occurrence for slot_index, occurrence in loop_slots}
        rank = len(loop_slots)
        for subset in _even_subsets(loop_slot_indices):
            subset_set = set(subset)
            pair_count = len(subset) // 2
            unpaired = [slot for slot in loop_slot_indices if slot not in subset_set]
            for pairing in _metric_pairings(tuple(subset)):
                base = (
                    _expr_number(term.coefficient)
                    * scalar_factor(pair_count)
                    * (u_expr ** (highest_rank - rank))
                )
                metric_edges: list[tuple[str, str]] = []
                for left_slot, right_slot in pairing:
                    left = occurrence_by_slot[left_slot]
                    right = occurrence_by_slot[right_slot]
                    base *= aM[left.momentum_index][right.momentum_index]
                    metric_edges.append((left.index, right.index))

                choices: list[tuple[Any, list[tuple[int, str]]]] = [(base, list(external_slots))]
                for slot in unpaired:
                    occurrence = occurrence_by_slot[slot]
                    next_choices: list[tuple[Any, list[tuple[int, str]]]] = []
                    for expr_value, vectors in choices:
                        for external_index, p_expr in enumerate(p_components[occurrence.momentum_index]):
                            if str(p_expr) == "0":
                                continue
                            next_choices.append(
                                (
                                    expr_value * p_expr,
                                    [*vectors, (external_index, occurrence.index)],
                                )
                            )
                    choices = next_choices

                for expr_value, vectors in choices:
                    total += expr_value * _contract_vectors_and_metrics(
                        vectors,
                        metric_edges,
                        external_momenta,
                        dot_map,
                        dimension_expr,
                    )

    expanded = total.expand()
    eps_degree = _expr_degree_in(expanded, "eps")
    eps_poly = expanded.to_polynomial(vars=[_sym("eps")])
    coeffs = [_expr_zero() for _ in range(eps_degree + 1)]
    for powers, coeff in eps_poly.coefficient_list(vars=[_sym("eps")]):
        coeff_expr = coeff if hasattr(coeff, "expand") else E(str(coeff))
        coeffs[int(powers[0])] += coeff_expr
    return ReducedNumerator([coeff.expand() for coeff in coeffs], highest_rank)
