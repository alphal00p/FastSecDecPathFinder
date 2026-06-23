"""Symbolica evaluator construction and runtime dispatch helpers."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any
from uuid import uuid4

import numpy as np
from symbolica import CompiledRealEvaluator, Evaluator, Expression


_FSD_EVALUATOR_MAGIC = "FSD_EVALUATOR_WITH_PRECISION_FALLBACK_V1"


def evaluator_mode_from_jit(jit_compile: bool) -> str:
    """Return the legacy mode name for a boolean jit flag."""
    return "jit" if bool(jit_compile) else "eager"


def jit_flag_for_mode(mode: str) -> bool:
    """Return the Symbolica ``jit_compile`` flag for one evaluator mode."""
    normalized = str(mode).strip().lower()
    if normalized == "jit":
        return True
    if normalized in {"eager", "compile"}:
        return False
    raise ValueError(f"unsupported evaluator compile mode {mode!r}")


def evaluator_build_kwargs(
    *,
    evaluator_compile_mode: str = "jit",
    **kwargs: Any,
) -> dict[str, Any]:
    """Return Symbolica evaluator kwargs with a consistent compilation mode."""
    out = dict(kwargs)
    out["jit_compile"] = jit_flag_for_mode(evaluator_compile_mode)
    return out


@dataclass
class HotEvaluatorWithPrecisionFallback:
    """Hot f64 evaluator paired with an eager evaluator for precision rescue."""

    hot_evaluator: Any
    precision_evaluator: Any
    real_evaluator: bool = True
    kind: str = "jit"

    @property
    def source_evaluator(self) -> Any:
        """Return the eager evaluator to use for dualization/precision fallback."""
        return self.precision_evaluator

    def evaluate(self, inputs: Any) -> Any:
        """Evaluate with the hot f64 backend."""
        return self.hot_evaluator.evaluate(inputs)

    def evaluate_complex(self, inputs: Any) -> Any:
        """Evaluate complex rows, using real hot code for real-valued rows."""
        rows = np.asarray(inputs)
        if self.real_evaluator and (
            not np.iscomplexobj(rows) or np.all(np.imag(rows) == 0.0)
        ):
            return np.asarray(
                self.hot_evaluator.evaluate(np.ascontiguousarray(np.real(rows), dtype=float)),
                dtype=np.complex128,
            )
        return self.hot_evaluator.evaluate_complex(inputs)

    def evaluate_with_prec(self, inputs: Any, decimal_digit_precision: int) -> Any:
        """Use the eager evaluator for multiprecision real evaluation."""
        return self.precision_evaluator.evaluate_with_prec(inputs, decimal_digit_precision)

    def evaluate_complex_with_prec(self, inputs: Any, decimal_digit_precision: int) -> Any:
        """Use the eager evaluator for multiprecision complex evaluation."""
        return self.precision_evaluator.evaluate_complex_with_prec(inputs, decimal_digit_precision)

    def save(self) -> bytes:
        """Serialize the hot evaluator for legacy callers.

        Prepared bundles should call ``serialize_evaluator`` to preserve both
        hot and eager fallback evaluators.  This method keeps older ad hoc
        streaming paths functional.
        """
        return self.hot_evaluator.save()


@dataclass
class CompiledEvaluatorWrapper:
    """F64 compiled evaluator with eager fallback for unsupported APIs.

    Symbolica's compiled evaluator currently exposes only double-precision
    ``evaluate`` and ``load`` methods.  FSD still needs arbitrary precision
    rescue near endpoints, so the original source evaluator is retained for
    precision and complex fallbacks.
    """

    precision_evaluator: Any
    compiled_evaluator: Any
    input_len: int
    output_len: int
    number_type: str
    function_name: str
    source_path: str
    library_path: str

    fsd_compiled_artifact: bool = True

    @property
    def source_evaluator(self) -> Any:
        """Return the eager evaluator to use for dualization/precision fallback."""
        return self.precision_evaluator

    def evaluate(self, inputs: Any) -> Any:
        """Evaluate with the compiled f64 backend."""
        return self.compiled_evaluator.evaluate(inputs)

    def evaluate_complex(self, inputs: Any) -> Any:
        """Evaluate complex rows, using compiled real code for real-valued rows."""
        rows = np.asarray(inputs)
        if self.number_type == "real" and (
            not np.iscomplexobj(rows) or np.all(np.imag(rows) == 0.0)
        ):
            return np.asarray(
                self.compiled_evaluator.evaluate(np.ascontiguousarray(np.real(rows), dtype=float)),
                dtype=np.complex128,
            )
        return self.precision_evaluator.evaluate_complex(inputs)

    def evaluate_with_prec(self, inputs: Any, decimal_digit_precision: int) -> Any:
        """Use the source evaluator for multiprecision real evaluation."""
        return self.precision_evaluator.evaluate_with_prec(inputs, decimal_digit_precision)

    def evaluate_complex_with_prec(self, inputs: Any, decimal_digit_precision: int) -> Any:
        """Use the source evaluator for multiprecision complex evaluation."""
        return self.precision_evaluator.evaluate_complex_with_prec(inputs, decimal_digit_precision)

    def save(self) -> bytes:
        """Serialize the eager fallback for legacy callers."""
        return self.precision_evaluator.save()


def _compile_evaluator(
    precision_evaluator: Any,
    *,
    input_len: int,
    output_len: int,
    real_evaluator: bool,
    name_hint: str,
) -> CompiledEvaluatorWrapper:
    """Compile a Symbolica evaluator and return a wrapper with fallbacks."""
    safe_hint = "".join(ch if ch.isalnum() else "_" for ch in name_hint)[:48] or "evaluator"
    tmpdir = Path(tempfile.mkdtemp(prefix="fsd-symbolica-compiled-"))
    function_name = f"fsd_{safe_hint}_{uuid4().hex[:10]}"
    source_path = tmpdir / f"{function_name}.cpp"
    library_path = tmpdir / f"lib{function_name}.so"
    number_type = "real" if real_evaluator else "complex"
    compiled = precision_evaluator.compile(
        function_name,
        str(source_path),
        str(library_path),
        number_type,
    )
    return CompiledEvaluatorWrapper(
        precision_evaluator=precision_evaluator,
        compiled_evaluator=compiled,
        input_len=int(input_len),
        output_len=int(output_len),
        number_type=number_type,
        function_name=function_name,
        source_path=str(source_path),
        library_path=str(library_path),
    )


def build_evaluator(
    expr: Any,
    params: list[Any],
    *,
    evaluator_compile_mode: str = "jit",
    real_evaluator: bool = True,
    name_hint: str = "evaluator",
    **kwargs: Any,
) -> Any:
    """Build one Symbolica evaluator according to the FSD evaluator mode."""
    mode = str(evaluator_compile_mode).strip().lower()
    if mode == "jit":
        precision_evaluator = expr.evaluator(
            params,
            **evaluator_build_kwargs(evaluator_compile_mode="eager", **kwargs),
        )
        hot_evaluator = expr.evaluator(
            params,
            **evaluator_build_kwargs(evaluator_compile_mode="jit", **kwargs),
        )
        return HotEvaluatorWithPrecisionFallback(
            hot_evaluator=hot_evaluator,
            precision_evaluator=precision_evaluator,
            real_evaluator=real_evaluator,
            kind="jit",
        )
    evaluator = expr.evaluator(
        params,
        **evaluator_build_kwargs(evaluator_compile_mode=mode, **kwargs),
    )
    if mode != "compile":
        return evaluator
    return _compile_evaluator(
        evaluator,
        input_len=len(params),
        output_len=1,
        real_evaluator=real_evaluator,
        name_hint=name_hint,
    )


def build_evaluator_multiple(
    exprs: list[Any],
    params: list[Any],
    *,
    evaluator_compile_mode: str = "jit",
    real_evaluator: bool = True,
    name_hint: str = "evaluator_multiple",
    **kwargs: Any,
) -> Any:
    """Build one multi-output Symbolica evaluator according to FSD mode."""
    mode = str(evaluator_compile_mode).strip().lower()
    if mode == "jit":
        precision_evaluator = Expression.evaluator_multiple(
            exprs,
            params,
            **evaluator_build_kwargs(evaluator_compile_mode="eager", **kwargs),
        )
        hot_evaluator = Expression.evaluator_multiple(
            exprs,
            params,
            **evaluator_build_kwargs(evaluator_compile_mode="jit", **kwargs),
        )
        return HotEvaluatorWithPrecisionFallback(
            hot_evaluator=hot_evaluator,
            precision_evaluator=precision_evaluator,
            real_evaluator=real_evaluator,
            kind="jit",
        )
    evaluator = Expression.evaluator_multiple(
        exprs,
        params,
        **evaluator_build_kwargs(evaluator_compile_mode=mode, **kwargs),
    )
    if mode != "compile":
        return evaluator
    return _compile_evaluator(
        evaluator,
        input_len=len(params),
        output_len=len(exprs),
        real_evaluator=real_evaluator,
        name_hint=name_hint,
    )


def has_nonzero_imaginary_part(rows: Any) -> bool:
    """Return whether a numeric row/batch carries nonzero imaginary data."""
    arr = np.asarray(rows)
    return bool(np.iscomplexobj(arr) and np.any(np.imag(arr) != 0.0))


def evaluate_f64(
    evaluator: Any,
    rows: Any,
    *,
    real_evaluator: bool,
) -> Any:
    """Evaluate f64 rows using real APIs whenever possible."""
    if real_evaluator and not has_nonzero_imaginary_part(rows):
        return evaluator.evaluate(np.ascontiguousarray(np.real(np.asarray(rows)), dtype=float))
    return evaluator.evaluate_complex(np.ascontiguousarray(rows))


def evaluate_precise(
    evaluator: Any,
    row: Any,
    precision_digits: int,
    *,
    real_evaluator: bool,
) -> Any:
    """Evaluate one row with Symbolica multiprecision real/complex APIs."""
    if real_evaluator:
        return evaluator.evaluate_with_prec(row, precision_digits)
    return evaluator.evaluate_complex_with_prec(row, precision_digits)


def time_evaluator_call(callback: Any, timing: Any | None) -> Any:
    """Run ``callback`` and charge its wall time to a HotPathTiming object."""
    start = time.perf_counter()
    out = callback()
    if timing is not None:
        timing.add_eval(time.perf_counter() - start)
    return out


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def serialize_evaluator(evaluator: Any) -> bytes:
    """Serialize a Symbolica evaluator, preserving eager precision fallbacks."""
    if isinstance(evaluator, HotEvaluatorWithPrecisionFallback):
        payload = {
            "magic": _FSD_EVALUATOR_MAGIC,
            "kind": "hot-with-precision-fallback",
            "hot_kind": evaluator.kind,
            "real_evaluator": bool(evaluator.real_evaluator),
            "hot": _b64(evaluator.hot_evaluator.save()),
            "precision": _b64(evaluator.precision_evaluator.save()),
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if isinstance(evaluator, CompiledEvaluatorWrapper):
        payload = {
            "magic": _FSD_EVALUATOR_MAGIC,
            "kind": "compiled-with-precision-fallback",
            "number_type": evaluator.number_type,
            "real_evaluator": evaluator.number_type == "real",
            "function_name": evaluator.function_name,
            "input_len": int(evaluator.input_len),
            "output_len": int(evaluator.output_len),
            "library": _b64(Path(evaluator.library_path).read_bytes()),
            "precision": _b64(evaluator.precision_evaluator.save()),
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return evaluator.save()


def deserialize_evaluator(data: bytes) -> Any:
    """Load a Symbolica evaluator serialized by ``serialize_evaluator``."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        return Evaluator.load(data)
    if not isinstance(payload, dict) or payload.get("magic") != _FSD_EVALUATOR_MAGIC:
        return Evaluator.load(data)
    kind = str(payload.get("kind", ""))
    if kind == "hot-with-precision-fallback":
        hot = Evaluator.load(_unb64(str(payload["hot"])))
        precision = Evaluator.load(_unb64(str(payload["precision"])))
        return HotEvaluatorWithPrecisionFallback(
            hot_evaluator=hot,
            precision_evaluator=precision,
            real_evaluator=bool(payload.get("real_evaluator", True)),
            kind=str(payload.get("hot_kind", "jit")),
        )
    if kind == "compiled-with-precision-fallback":
        if str(payload.get("number_type", "real")) != "real":
            raise RuntimeError("only real compiled evaluator artifacts are supported")
        tmpdir = Path(tempfile.mkdtemp(prefix="fsd-symbolica-loaded-compiled-"))
        library_path = tmpdir / "compiled_evaluator.so"
        library_path.write_bytes(_unb64(str(payload["library"])))
        compiled = CompiledRealEvaluator.load(
            str(library_path),
            str(payload["function_name"]),
            int(payload["input_len"]),
            int(payload["output_len"]),
        )
        precision = Evaluator.load(_unb64(str(payload["precision"])))
        wrapper = CompiledEvaluatorWrapper(
            precision_evaluator=precision,
            compiled_evaluator=compiled,
            input_len=int(payload["input_len"]),
            output_len=int(payload["output_len"]),
            number_type="real",
            function_name=str(payload["function_name"]),
            source_path="",
            library_path=str(library_path),
        )
        wrapper._loaded_tmpdir = tmpdir  # type: ignore[attr-defined]
        return wrapper
    return Evaluator.load(data)
