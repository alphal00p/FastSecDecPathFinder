"""Shared immutable request/result containers and timing accumulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ONELOOP_TO_FEYNMAN = -1.0 / (16.0 * 3.141592653589793238462643383279502884**2)
EULER_GAMMA = 0.577215664901532860606512090082402431


@dataclass(frozen=True)
class EpsilonExpansion:
    """Affine coefficient ``base + eps_coeff * epsilon``."""

    base: float
    eps_coeff: float

    def as_text(self, symbol: str = "eps") -> str:
        """Render the affine expansion compactly for summaries."""
        if self.eps_coeff == 0.0:
            return f"{self.base:g}"
        sign = "+" if self.eps_coeff >= 0.0 else "-"
        return f"{self.base:g} {sign} {abs(self.eps_coeff):g}*{symbol}"


@dataclass(frozen=True)
class ParametricRepresentation:
    """General scalar Feynman-parametric representation metadata.

    For an L-loop scalar integral with propagator powers nu_i and
    A=sum_i nu_i, the standard projective representation has the form

      prefactor * int_delta prod_i dx_i x_i^(nu_i-1)
      U^(A-(L+1)D/2) F^(-(A-LD/2)).

    The sector processor does not apply the global prefactor.  It needs the
    affine U/F exponents and parameter weights so sector declarations can
    expose all endpoint monomial sources explicitly.
    """

    loop_count: int
    propagator_powers: tuple[float, ...]
    dimension: EpsilonExpansion
    gamma_argument: EpsilonExpansion
    u_exponent: EpsilonExpansion
    f_exponent: EpsilonExpansion
    parameter_weight_powers: tuple[float, ...]
    prefactor_description: str
    convention_description: str


@dataclass(frozen=True)
class IntegralRequest:
    """Fully validated CLI configuration passed through the implementation."""

    integral: str
    dot_file: str | None
    kinematics_file: str | None
    graph_name: str | None
    sector_method: str
    normaliz_executable: str | None
    dot_engine: str
    sectors: tuple[int, ...] | None
    pysecdec_workdir: str
    pysecdec_epsrel: float
    pysecdec_maxeval: int
    keep_pysecdec_workdir: bool
    progress_value_order: str
    max_eps_order: int
    target_args: tuple[str, ...] | None
    show_results: str | None
    sort_sector_results: str
    result_path: str
    log_level: str
    log_file: str | None
    mode: str
    s: float | None
    s12: float | None
    s23: float | None
    m: float
    gamma_scheme: str
    prefactor_convention: str
    seed: int
    max_iter: int
    min_iter: int
    samples_per_iter: int
    batch_size: int
    target_rel_accuracy: float | None
    min_error: float
    bins: int
    workers: int
    jit_compile_evaluators: bool
    dual_evaluator_mode: str
    subtraction_backend: str
    stability_threshold: float
    high_precision_stability_threshold: float
    stability_precision: int
    high_precision_stability_precision: int
    show_stats: bool
    no_progress: bool
    quiet_summary: bool
    json: bool
    mu: float | None
    onshell_threshold: float | None


@dataclass(frozen=True)
class BenchmarkResult:
    """OneLOopBridge coefficients in raw normalization plus its prefactor."""

    raw: list[complex]
    factor: complex

    @property
    def feynman(self) -> list[complex]:
        """Return coefficients converted to the Feynman-normalized convention."""
        return [self.factor * value for value in self.raw]


@dataclass(frozen=True)
class TargetDefinition:
    """Reference coefficients in the selected display convention."""

    source: str
    convention: str
    coefficients: list[complex]
    errors: list[complex]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SectorIntegrationResult:
    """Per-sector Monte Carlo coefficients before final convention selection."""

    sector_id: int
    sector_name: str
    samples: int
    raw_sector_coeffs: list[complex]
    raw_sector_errors: list[complex]
    precision_counts: dict[str, int]


@dataclass(frozen=True)
class IntegrationResult:
    """Numerical integration output before final display convention selection."""

    raw_sector_coeffs: list[complex]
    raw_sector_errors: list[complex]
    per_sector: list[SectorIntegrationResult]
    samples: int
    elapsed_seconds: float
    avg_eval_us_per_sample_per_worker: float
    eval_seconds: float
    python_seconds: float
    havana_seconds: float
    python_overhead_fraction: float
    precision_counts: dict[str, int]
    interrupted: bool = False


@dataclass
class HotPathTiming:
    """Additive work-time profile split into evaluator, Python, and Havana time."""

    eval_seconds: float = 0.0
    python_seconds: float = 0.0
    havana_seconds: float = 0.0
    precision_digits: int | None = None
    ordinary_precision_samples: int = 0
    stability_precision_samples: int = 0
    high_precision_samples: int = 0

    def add_eval(self, seconds: float) -> None:
        """Accumulate Symbolica evaluator wall time."""
        self.eval_seconds += max(float(seconds), 0.0)

    def add_python(self, seconds: float) -> None:
        """Accumulate Python and NumPy glue time."""
        self.python_seconds += max(float(seconds), 0.0)

    def add_havana(self, seconds: float) -> None:
        """Accumulate Havana sampling, training, merge, and update time."""
        self.havana_seconds += max(float(seconds), 0.0)

    def absorb(self, other: "HotPathTiming") -> None:
        """Merge timing reported by a worker or nested processor call."""
        self.eval_seconds += other.eval_seconds
        self.python_seconds += other.python_seconds
        self.havana_seconds += other.havana_seconds
        self.ordinary_precision_samples += other.ordinary_precision_samples
        self.stability_precision_samples += other.stability_precision_samples
        self.high_precision_samples += other.high_precision_samples

    def add_precision_samples(
        self,
        *,
        ordinary: int = 0,
        stability: int = 0,
        high: int = 0,
    ) -> None:
        """Accumulate how many rows used each evaluator precision tier."""
        self.ordinary_precision_samples += max(int(ordinary), 0)
        self.stability_precision_samples += max(int(stability), 0)
        self.high_precision_samples += max(int(high), 0)

    @property
    def precision_counts(self) -> dict[str, int]:
        """Return JSON-friendly evaluator precision tier counts."""
        return {
            "ordinary": self.ordinary_precision_samples,
            "stability": self.stability_precision_samples,
            "high_precision": self.high_precision_samples,
        }

    @property
    def total_seconds(self) -> float:
        """Return the summed work-time profile, not the elapsed wall time."""
        return self.eval_seconds + self.python_seconds + self.havana_seconds

    @property
    def python_overhead_fraction(self) -> float:
        """Fraction of profiled work attributed to Python and NumPy glue."""
        total = self.total_seconds
        if total <= 0.0:
            return 0.0
        return self.python_seconds / total

    @property
    def evaluator_fraction(self) -> float:
        """Fraction of profiled work spent inside Symbolica evaluators."""
        total = self.total_seconds
        if total <= 0.0:
            return 0.0
        return self.eval_seconds / total

    @property
    def havana_fraction(self) -> float:
        """Fraction of profiled work spent in Havana sampler/grid operations."""
        total = self.total_seconds
        if total <= 0.0:
            return 0.0
        return self.havana_seconds / total


JsonDict = dict[str, Any]
