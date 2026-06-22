"""Adapter from GammaLoop DOT files to pySecDec-backed FSD objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from definitions import IntegralRequest
from dot_parser import ParsedDotGraph, parse_dot_file
from generation_timing import GenerationProgress, GenerationTimings
from kinematics import KinematicsDefinition, load_kinematics
from pysecdec_bridge import DotBuildBundle, build_dot_bundle

if TYPE_CHECKING:
    from integrand import TopologyDefinition
    from sectors_generator import SectorDefinition


@dataclass(frozen=True)
class DotTopologySource:
    """Raw DOT input plus lightweight file metadata."""

    path: Path
    text: str

    @property
    def name(self) -> str:
        """Return a stable display name for this external topology."""
        return self.path.stem


@dataclass(frozen=True)
class GammaLoopTopologyData:
    """Normalized graph plus kinematics extracted from DOT/YAML input."""

    source: DotTopologySource
    graph: ParsedDotGraph
    kinematics: KinematicsDefinition
    timings: GenerationTimings


@dataclass(frozen=True)
class DotTopologyPrintout:
    """Human-readable DOT topology printout for parser/generation diagnostics.

    The normal pre-integration summary prints concrete U/F expressions and
    concrete sectors after pySecDec generation.  This lightweight printout is
    kept as a schema-oriented fallback for early DOT diagnostics.
    """

    source: DotTopologySource

    def header_rows(self) -> list[tuple[str, str]]:
        """Rows mirroring the normal run-summary header table."""
        return [
            ("input mode", "GammaLoop DOT file"),
            ("source name", self.source.name),
            ("source path", str(self.source.path)),
            ("source bytes", str(len(self.source.text.encode("utf-8")))),
            ("parser status", "implemented through pydot + pySecDec bridge"),
        ]

    def topology_rows(self) -> list[tuple[str, str]]:
        """Rows describing future TopologyDefinition printout fields."""
        return [
            ("family", "DOT graph name from pydot"),
            ("loop count", "graph cycle rank E-V+1"),
            ("propagator powers", "unit powers only in this phase"),
            ("dimension", "4 - 2 eps"),
            ("prefactor", "pySecDec Gamma/global prefactor metadata"),
            ("U polynomial", "retained Symbolica expression from pySecDec"),
            ("F polynomial", "retained Symbolica expression from pySecDec"),
            ("evaluator order", "x_i followed by YAML value symbols"),
            ("U/F exponents", "affine powers in eps from pySecDec"),
        ]

    def sector_schema_rows(self) -> list[tuple[str, str, str]]:
        """Rows describing future declarative SectorDefinition fields."""
        return [
            ("map", "x_i = X_{s,i}(y)", "Symbolica expressions and evaluators"),
            ("regular prefactor", "J_{s,reg}(y)", "Jacobian with monomials removed"),
            ("U monomial", "M_{U,s}(y)", "powers used to build psi_s = U/M_U"),
            ("F monomial", "M_{F,s}(y)", "powers used to build phi_s = F/M_F"),
            ("measure powers", "y_a^r", "from dx, x_i^(nu_i-1), and numerator weights"),
            ("singular axes", "subset of y_a", "axes where localized subtraction applies"),
            ("endpoint powers", "rho_{s,a}(eps)", "assembled from topology and sector metadata"),
            ("subtraction", "strategy id", "recursive endpoint subtraction target"),
        ]

    def validation_rows(self) -> list[tuple[str, str]]:
        """Rows describing the future validation gates for DOT input."""
        return [
            ("file exists", "validated before DOT generation starts"),
            ("DOT parser", "pydot"),
            ("topology construction", "pySecDec LoopIntegralFromGraph"),
            ("sector generation", "pySecDec decomposition APIs"),
            ("benchmark mapping", "pySecDec engine in --dot-engine pysecdec|both"),
        ]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the DOT printout."""
        return {
            "header": self.header_rows(),
            "topology": self.topology_rows(),
            "sector_schema": self.sector_schema_rows(),
            "validation": self.validation_rows(),
        }

    def __str__(self) -> str:
        """Plain string fallback used outside colored PrettyTable output."""
        lines = [f"DOT topology printout: {self.source.path}"]
        lines.extend(f"  {key}: {value}" for key, value in self.topology_rows())
        lines.append("  sector schema:")
        lines.extend(f"    {name}: {symbol} ({purpose})" for name, symbol, purpose in self.sector_schema_rows())
        return "\n".join(lines)


class GammaLoopDotTopologyBuilder:
    """Adapter from GammaLoop DOT files to FSD topology and sector objects."""

    def __init__(self, request: IntegralRequest, source: DotTopologySource) -> None:
        """Store the immutable request and already-loaded DOT source."""
        self.request = request
        self.source = source

    @classmethod
    def from_request(cls, request: IntegralRequest) -> "GammaLoopDotTopologyBuilder":
        """Load the DOT source named by the request and create a builder."""
        if request.dot_file is None:
            raise ValueError("DOT topology construction requires --dot-file")
        path = Path(request.dot_file).expanduser()
        source = DotTopologySource(path=path, text=path.read_text(encoding="utf-8"))
        return cls(request=request, source=source)

    def parse_gammaloop_dot(
        self,
        progress: GenerationProgress | None = None,
    ) -> GammaLoopTopologyData:
        """Parse GammaLoop DOT syntax and load kinematics YAML."""
        if self.request.kinematics_file is None:
            raise ValueError("DOT mode requires --kinematics")
        timings = GenerationTimings()
        with timings.measure("DOT parse", progress=progress):
            graph = parse_dot_file(self.source.path, self.request.graph_name)
        with timings.measure("kinematics load/evaluation", progress=progress):
            kinematics = load_kinematics(self.request.kinematics_file)
        return GammaLoopTopologyData(
            source=self.source,
            graph=graph,
            kinematics=kinematics,
            timings=timings,
        )

    def printout_placeholder(self) -> DotTopologyPrintout:
        """Return the structured DOT summary printout for CLI display."""
        return DotTopologyPrintout(source=self.source)

    def build_topology(self) -> "TopologyDefinition":
        """Build a retained U/F topology definition from parsed DOT data."""
        return _bundle_from_request(self.request).topology

    def issue_sector_definitions(self) -> list["SectorDefinition"]:
        """Issue declarative sector definitions for this DOT topology."""
        return _bundle_from_request(self.request).sectors

    def benchmark_request(self) -> object:
        """Build the future OneLOopBridge benchmark request for this topology."""
        return _bundle_from_request(self.request)


_DOT_BUNDLE_CACHE: dict[tuple[object, ...], DotBuildBundle] = {}


def clear_dot_bundle_cache() -> None:
    """Drop in-process DOT build bundles.

    Normal CLI execution benefits from reusing a bundle between topology and
    sector construction.  Batch cache-warming uses this hook between cases so
    per-case timings are not accidentally hidden by a previous in-memory hit.
    """

    _DOT_BUNDLE_CACHE.clear()


def _request_cache_key(request: IntegralRequest) -> tuple[object, ...]:
    """Return a stable cache key for pySecDec DOT generation."""
    return (
        request.dot_file,
        request.kinematics_file,
        request.graph_name,
        request.sector_method,
        request.normaliz_executable,
        request.prefactor_convention,
        request.numerator_reducer,
        request.jit_compile_evaluators,
        request.dual_evaluator_mode,
        request.subtraction_backend,
        request.sector_evaluator_backend,
        request.ibp_power_goal,
        request.max_eps_order,
    )


def _bundle_from_request(
    request: IntegralRequest,
    progress: GenerationProgress | None = None,
) -> DotBuildBundle:
    """Parse DOT/YAML and generate pySecDec-backed topology/sectors once."""
    key = _request_cache_key(request)
    cached = _DOT_BUNDLE_CACHE.get(key)
    if cached is not None:
        if progress is not None and progress.logger is not None:
            progress.logger.info("generation cache hit: reused DOT bundle for %s", request.dot_file)
        return cached
    builder = GammaLoopDotTopologyBuilder.from_request(request)
    data = builder.parse_gammaloop_dot(progress=progress)
    bundle = build_dot_bundle(data.graph, data.kinematics, request, progress=progress)
    bundle.timings.records = [*data.timings.records, *bundle.timings.records]
    _DOT_BUNDLE_CACHE[key] = bundle
    return bundle


def get_dot_bundle(
    request: IntegralRequest,
    progress: GenerationProgress | None = None,
) -> DotBuildBundle:
    """Return the cached DOT build bundle for logging and pySecDec mode."""
    return _bundle_from_request(request, progress=progress)


def build_topology_from_dot_request(request: IntegralRequest) -> "TopologyDefinition":
    """Construct a topology from a DOT-backed request."""
    return GammaLoopDotTopologyBuilder.from_request(request).build_topology()


def generate_sectors_from_dot_request(request: IntegralRequest) -> list["SectorDefinition"]:
    """Construct declarative sector definitions from a DOT-backed request."""
    return GammaLoopDotTopologyBuilder.from_request(request).issue_sector_definitions()
