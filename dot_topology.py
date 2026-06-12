"""GammaLoop DOT-file topology scaffolding.

This module is intentionally a structural placeholder.  It shows where the
future GammaLoop DOT parser will live, how it will feed the retained
``TopologyDefinition`` object, and how it will issue declarative
``SectorDefinition`` objects for the existing black-box sector processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from definitions import IntegralRequest

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
    """Future normalized graph data extracted from GammaLoop DOT input.

    The intended parser output should contain enough information to construct:

    - loop-momentum routing,
    - propagator masses and powers,
    - external momentum assignments and invariants,
    - the Feynman-parameter row order,
    - the Symanzik U/F expressions retained for display/evaluator creation,
    - topology-level parametric metadata such as loop count, propagator powers,
      dimension, global prefactor convention, and affine U/F exponents,
    - topology-specific sector metadata before evaluator generation, including
      U/F monomial powers and measure or numerator monomial powers.

    The exact field set is deliberately not frozen yet because it should follow
    the GammaLoop DOT convention rather than guesses made in this prototype.
    """

    source: DotTopologySource


@dataclass(frozen=True)
class DotTopologyPrintout:
    """Human-readable placeholder for a not-yet-parsed DOT topology.

    The implemented triangle and box examples can print concrete U/F
    expressions and concrete sectors.  DOT input cannot do that until the
    GammaLoop parser exists, but the CLI can still show the exact shape of the
    information that the parser must eventually provide.
    """

    source: DotTopologySource

    def header_rows(self) -> list[tuple[str, str]]:
        """Rows mirroring the normal run-summary header table."""
        return [
            ("input mode", "GammaLoop DOT file"),
            ("source name", self.source.name),
            ("source path", str(self.source.path)),
            ("source bytes", str(len(self.source.text.encode("utf-8")))),
            ("parser status", "not implemented"),
        ]

    def topology_rows(self) -> list[tuple[str, str]]:
        """Rows describing future TopologyDefinition printout fields."""
        return [
            ("family", "to be inferred from DOT graph metadata"),
            ("loop count", "placeholder: parse graph cycle rank"),
            ("propagator powers", "placeholder: parse GammaLoop edge weights"),
            ("dimension", "placeholder: affine D0 + Deps*eps"),
            ("prefactor", "placeholder: convention-dependent global factor"),
            ("U polynomial", "placeholder: retained Symbolica expression"),
            ("F polynomial", "placeholder: retained Symbolica expression"),
            ("evaluator order", "placeholder: x_i followed by kinematic parameters"),
            ("U/F exponents", "placeholder: affine powers in epsilon"),
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
            ("file exists", "validated before this placeholder is printed"),
            ("DOT parser", "NotImplementedError"),
            ("topology construction", "NotImplementedError"),
            ("sector generation", "NotImplementedError"),
            ("benchmark mapping", "NotImplementedError"),
        ]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the placeholder."""
        return {
            "header": self.header_rows(),
            "topology": self.topology_rows(),
            "sector_schema": self.sector_schema_rows(),
            "validation": self.validation_rows(),
        }

    def __str__(self) -> str:
        """Plain string fallback used outside colored PrettyTable output."""
        lines = [f"DOT topology placeholder: {self.source.path}"]
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

    def parse_gammaloop_dot(self) -> GammaLoopTopologyData:
        """Parse GammaLoop DOT syntax into normalized topology data."""
        raise NotImplementedError(
            "GammaLoop DOT parsing is not implemented yet.  This is the entry "
            "point that should decode nodes, edges, propagators, momentum "
            "routing, masses, and external invariants from the DOT file."
        )

    def printout_placeholder(self) -> DotTopologyPrintout:
        """Return the structured DOT summary placeholder for CLI display."""
        return DotTopologyPrintout(source=self.source)

    def build_topology(self) -> "TopologyDefinition":
        """Build a retained U/F topology definition from parsed DOT data."""
        raise NotImplementedError(
            "Building TopologyDefinition from GammaLoop DOT data is not "
            "implemented yet.  This hook should call parse_gammaloop_dot(), "
            "construct U/F Symbolica expressions, fill ParametricRepresentation "
            "metadata, and avoid adding topology-specific branches to "
            "SectorProcessor."
        )

    def issue_sector_definitions(self) -> list["SectorDefinition"]:
        """Issue declarative sector definitions for this DOT topology."""
        raise NotImplementedError(
            "Sector generation from GammaLoop DOT data is not implemented yet.  "
            "This hook should call parse_gammaloop_dot() and return "
            "declarative SectorDefinition objects with maps, Jacobians, "
            "U/F monomials, measure/numerator monomials, and subtraction axes "
            "already specified."
        )

    def benchmark_request(self) -> object:
        """Build the future OneLOopBridge benchmark request for this topology."""
        raise NotImplementedError(
            "Benchmark mapping for arbitrary GammaLoop DOT topologies is not "
            "implemented yet.  This hook should translate parsed DOT topology "
            "data into the benchmark backend call or declare that no benchmark "
            "is available for the topology."
        )


def build_topology_from_dot_request(request: IntegralRequest) -> "TopologyDefinition":
    """Construct a topology from a DOT-backed request."""
    return GammaLoopDotTopologyBuilder.from_request(request).build_topology()


def generate_sectors_from_dot_request(request: IntegralRequest) -> list["SectorDefinition"]:
    """Construct declarative sector definitions from a DOT-backed request."""
    return GammaLoopDotTopologyBuilder.from_request(request).issue_sector_definitions()
