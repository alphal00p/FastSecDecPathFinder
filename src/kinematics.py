"""Kinematics YAML loading and Symbolica-backed expression evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml
from symbolica import E, S


def _pysecdec_number_text(value: float) -> str:
    """Render a numeric value without decimal points for FORM/pySecDec codegen."""
    fraction = Fraction(float(value)).limit_denominator(10**9)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


@dataclass(frozen=True)
class KinematicsDefinition:
    """Numeric values and replacement rules used by DOT topologies."""

    path: Path
    values: dict[str, float]
    replacements: list[tuple[str, float]]
    replacement_expressions: list[tuple[str, str]]

    @property
    def parameter_names(self) -> list[str]:
        """Return stable parameter order for Symbolica evaluators."""
        return list(self.values.keys())

    @property
    def parameter_values(self) -> list[float]:
        """Return numeric values in ``parameter_names`` order."""
        return [float(self.values[name]) for name in self.parameter_names]

    def value_for_symbol(self, name: str) -> float:
        """Resolve a named mass/invariant from the YAML values block."""
        if name not in self.values:
            raise ValueError(
                f"symbol {name!r} is used by the DOT topology but is absent from {self.path}"
            )
        return float(self.values[name])

    def pysecdec_replacement_rules(self) -> list[tuple[str, str]]:
        """Return fixed numeric replacement rules in FORM-friendly syntax."""
        return [(left, _pysecdec_number_text(value)) for left, value in self.replacements]

    def pysecdec_value_for_symbol(self, name: str) -> str:
        """Resolve a mass/invariant as a FORM-friendly numeric string."""
        return _pysecdec_number_text(self.value_for_symbol(name))


def _evaluate_symbolica_scalar(expr_text: Any, values: dict[str, float]) -> float:
    """Evaluate a scalar arithmetic expression with Symbolica only."""
    if isinstance(expr_text, (int, float)):
        return float(expr_text)
    text = str(expr_text).strip()
    if not text:
        raise ValueError("empty scalar expression in kinematics YAML")
    names = list(values.keys())
    evaluator = E(text).evaluator([S(name) for name in names])
    result = evaluator.evaluate([[float(values[name]) for name in names]])[0][0]
    return float(result)


def load_kinematics(path: str | Path) -> KinematicsDefinition:
    """Load the supported kinematics YAML schema."""
    file_path = Path(path).expanduser()
    raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{file_path}: expected a YAML mapping")
    values_raw = raw.get("values", {})
    replacements_raw = raw.get("replacements", {})
    if not isinstance(values_raw, dict):
        raise ValueError(f"{file_path}: 'values' must be a mapping")
    if not isinstance(replacements_raw, dict):
        raise ValueError(f"{file_path}: 'replacements' must be a mapping")

    values: dict[str, float] = {}
    # Values can depend on previously declared values, so preserve YAML order.
    for name, expr in values_raw.items():
        values[str(name)] = _evaluate_symbolica_scalar(expr, values)

    replacements: list[tuple[str, float]] = []
    replacement_expressions: list[tuple[str, str]] = []
    for left, expr in replacements_raw.items():
        replacements.append((str(left), _evaluate_symbolica_scalar(expr, values)))
        replacement_expressions.append((str(left), str(expr).strip()))

    return KinematicsDefinition(
        path=file_path,
        values=values,
        replacements=replacements,
        replacement_expressions=replacement_expressions,
    )
