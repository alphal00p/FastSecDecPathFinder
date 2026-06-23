"""Formatting utilities for uncertainty parenthesis notation."""

from __future__ import annotations

from decimal import Decimal, localcontext
import math
from typing import Any


def decimal_with_precision(value: Any, precision_digits: int) -> Decimal:
    """Return a Decimal carrying the requested significant-digit precision.

    Symbolica's ``*_with_prec`` evaluators inspect the Decimal payload they
    receive.  A value such as ``Decimal("1.0")`` therefore carries too little
    input precision even if the evaluator is asked for 1000 digits.  Formatting
    through scientific notation pads finite float inputs to the requested
    significant-digit count while preserving the sampled double value as the
    source of truth.
    """
    digits = max(int(precision_digits), 1)
    with localcontext() as context:
        context.prec = digits
        numeric = float(value)
        if not math.isfinite(numeric):
            return Decimal(str(numeric))
        base = value if isinstance(value, Decimal) else Decimal(str(numeric))
        if base.is_zero():
            if digits == 1:
                return Decimal("0")
            return Decimal("0." + ("0" * (digits - 1)))
        padded = format(base, f".{digits - 1}e")
        return +Decimal(padded)


def decimal_complex_with_precision(value: Any, precision_digits: int) -> tuple[Decimal, Decimal]:
    """Return Symbolica's arbitrary-precision complex input tuple."""
    z = complex(value)
    return (
        decimal_with_precision(z.real, precision_digits),
        decimal_with_precision(z.imag, precision_digits),
    )


def _ndec(value: float, offset: int) -> int:
    """Return decimal places needed by the BNL-style uncertainty formatter."""
    ans = int(offset - math.log10(value))
    thresholds = [0.5, 9.5, 99.5]
    if ans > 0 and value * 10.0**ans >= thresholds[offset]:
        ans -= 1
    return max(ans, 0)


def _normalize_exponent(exponent: str) -> str:
    """Strip redundant zero padding from scientific-notation exponents."""
    return str(int(exponent))


def _format_uncertainty(mean: float, error: float) -> str:
    """Format a real value and one-sigma error with two error digits."""
    value = mean
    delta = abs(error)
    if math.isnan(value) or math.isnan(delta):
        return f"{value:e} +/- {delta:e}"
    if math.isinf(delta):
        return f"{value:e} +/- inf"
    if value == 0.0 and not (1e-4 <= delta < 1e5):
        if delta == 0.0:
            return "0(0)"
        mantissa, exponent = f"{delta:.1e}".split("e")
        return f"0.0({mantissa})e{_normalize_exponent(exponent)}"
    if value == 0.0:
        if delta >= 9.95:
            return f"0({delta:.0f})"
        if delta >= 0.995:
            return f"0.0({delta:.1f})"
        decimals = _ndec(delta, 2)
        return f"{value:.{decimals}f}({delta * 10.0**decimals:.0f})"
    if delta == 0.0:
        mantissa, exponent = f"{value:e}".split("e")
        exponent = _normalize_exponent(exponent)
        return f"{mantissa}(0)e{exponent}" if exponent != "0" else f"{mantissa}(0)"
    if delta > 1e4 * abs(value):
        return f"{value:.1e} +/- {delta:.2e}"
    if abs(value) >= 1e6 or abs(value) < 1e-5:
        exponent = math.floor(math.log10(abs(value)))
        scale = 10.0**exponent
        mantissa = _format_uncertainty(value / scale, delta / scale)
        return f"{mantissa}e{exponent}"
    if delta >= 9.95:
        if abs(value) >= 9.5:
            return f"{value:.0f}({delta:.0f})"
        decimals = _ndec(abs(value), 1)
        return f"{value:.{decimals}f}({delta:.{decimals}f})"
    if delta >= 0.995:
        if abs(value) >= 0.95:
            return f"{value:.1f}({delta:.1f})"
        decimals = _ndec(abs(value), 1)
        return f"{value:.{decimals}f}({delta:.{decimals}f})"
    decimals = max(_ndec(abs(value), 1), _ndec(delta, 2))
    return f"{value:.{decimals}f}({delta * 10.0**decimals:.0f})"


def format_uncertainty(mean: float, error: float, force_sign: bool = False) -> str:
    """Public real-valued uncertainty formatter."""
    formatted = _format_uncertainty(mean, error)
    if force_sign and math.copysign(1.0, mean) >= 0.0:
        return f"+{formatted}"
    return formatted


def format_complex_uncertainty(value: complex, error: complex) -> str:
    """Format real or complex coefficients with component-wise MC errors."""
    z = complex(value)
    e = complex(error)
    if abs(z.imag) < 5.0e-15 and abs(e.imag) < 5.0e-15:
        return format_uncertainty(z.real, abs(e.real))
    real_text = format_uncertainty(z.real, abs(e.real))
    imag_text = format_uncertainty(abs(z.imag), abs(e.imag))
    sign = "+" if z.imag >= 0.0 else "-"
    return f"{real_text}{sign}{imag_text}i"


def format_percent(value: float, digits: int = 2) -> str:
    """Format a finite or non-finite percentage value."""
    if math.isnan(value):
        return "nan%"
    if math.isinf(value):
        return "inf%"
    return f"{value:.{digits}f}%"
