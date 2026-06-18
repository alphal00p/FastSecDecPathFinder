"""GammaLoop-convention DOT parsing for scalar graph topologies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydot


def _clean(value: Any) -> str | None:
    """Normalize pydot attribute strings."""
    if value is None:
        return None
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text


def _is_invisible_node(node: pydot.Node | None) -> bool:
    """Return whether a DOT node represents an external half-edge stub."""
    if node is None:
        return False
    attrs = node.get_attributes()
    style = (_clean(attrs.get("style")) or "").lower()
    return "invis" in style or "invisible" in style


def _edge_sort_key(edge: pydot.Edge, position: int) -> tuple[int, str]:
    """Sort edges by explicit GammaLoop id when available, then file order."""
    raw_id = _clean(edge.get_attributes().get("id"))
    if raw_id is None:
        return (10**9 + position, "")
    try:
        return (int(raw_id), "")
    except ValueError:
        return (position, raw_id)


@dataclass(frozen=True)
class DotInternalLine:
    """One internal propagator line for pySecDec graph construction."""

    name: str
    source: str
    target: str
    mass: str
    power: int


@dataclass(frozen=True)
class DotExternalLine:
    """One external half-edge with signed momentum flow into the graph."""

    name: str
    vertex: str
    momentum: str
    incoming: bool


@dataclass(frozen=True)
class ParsedDotGraph:
    """Normalized GammaLoop DOT graph."""

    path: Path
    graph_name: str
    graph_attributes: dict[str, str]
    internal_vertices: list[str]
    vertex_ids: dict[str, int]
    internal_lines: list[DotInternalLine]
    external_lines: list[DotExternalLine]

    @property
    def loop_count(self) -> int:
        """Return the graph cycle rank for connected scalar graphs."""
        return len(self.internal_lines) - len(self.internal_vertices) + 1

    def pysecdec_internal_lines(self) -> list[list[object]]:
        """Return pySecDec ``internal_lines`` input."""
        out: list[list[object]] = []
        for line in self.internal_lines:
            out.append(
                [
                    line.mass,
                    [self.vertex_ids[line.source], self.vertex_ids[line.target]],
                ]
            )
        return out

    def pysecdec_external_lines(self) -> list[list[object]]:
        """Return pySecDec ``external_lines`` input."""
        out: list[list[object]] = []
        for line in self.external_lines:
            momentum = line.momentum
            if momentum.startswith("-"):
                momentum = momentum[1:]
            out.append([momentum, self.vertex_ids[line.vertex]])
        return out

    @property
    def numerator(self) -> str | None:
        """Return an optional pySecDec momentum-space numerator string.

        The first DOT numerator support intentionally accepts pySecDec's own
        momentum-dot-product syntax through a graph-level ``num`` attribute.
        Routing inference from GammaLoop graph syntax is separate future work,
        so numerator DOT files must also provide the corresponding propagator
        and momentum lists consumed by ``LoopIntegralFromPropagators``.
        """
        text = self.graph_attributes.get("num") or self.graph_attributes.get("numerator")
        if text is None or text.strip() in {"", "1"}:
            return None
        return text.strip()

    def graph_attr_list(self, key: str, *, separator: str = ",") -> list[str]:
        """Return a split graph attribute list with empty entries removed."""
        value = self.graph_attributes.get(key)
        if value is None:
            return []
        return [entry.strip() for entry in value.split(separator) if entry.strip()]


def parse_dot_file(path: str | Path, graph_name: str | None = None) -> ParsedDotGraph:
    """Parse a GammaLoop-style DOT file into the subset needed by pySecDec."""
    file_path = Path(path).expanduser()
    graphs = pydot.graph_from_dot_file(str(file_path))
    if not graphs:
        raise ValueError(f"{file_path}: no DOT graphs found")
    if graph_name is None:
        if len(graphs) != 1:
            names = ", ".join(_clean(graph.get_name()) or "<unnamed>" for graph in graphs)
            raise ValueError(f"{file_path}: --graph-name is required; available graphs: {names}")
        graph = graphs[0]
    else:
        graph = next((item for item in graphs if _clean(item.get_name()) == graph_name), None)
        if graph is None:
            names = ", ".join(_clean(item.get_name()) or "<unnamed>" for item in graphs)
            raise ValueError(f"{file_path}: graph {graph_name!r} not found; available graphs: {names}")

    graph_label = _clean(graph.get_name()) or file_path.stem
    graph_attributes = {
        str(key): _clean(value) or ""
        for key, value in graph.get_attributes().items()
    }
    for node in graph.get_nodes():
        if (_clean(node.get_name()) or "").lower() == "graph":
            graph_attributes.update(
                {
                    str(key): _clean(value) or ""
                    for key, value in node.get_attributes().items()
                }
            )
    nodes_by_name = {
        _clean(node.get_name()) or "": node
        for node in graph.get_nodes()
        if _clean(node.get_name()) not in (None, "node", "graph", "edge", "")
    }

    edge_records = sorted(
        enumerate(graph.get_edges()),
        key=lambda item: _edge_sort_key(item[1], item[0]),
    )
    internal_vertex_names: list[str] = []
    internal_lines: list[DotInternalLine] = []
    external_lines: list[DotExternalLine] = []
    for fallback_id, edge in edge_records:
        source = _clean(edge.get_source()) or ""
        target = _clean(edge.get_destination()) or ""
        attrs = edge.get_attributes()
        source_node = nodes_by_name.get(source)
        target_node = nodes_by_name.get(target)
        source_external = _is_invisible_node(source_node)
        target_external = _is_invisible_node(target_node)
        edge_name = _clean(attrs.get("name")) or _clean(attrs.get("id")) or f"e{fallback_id}"
        mass = _clean(attrs.get("mass")) or "0"
        power_text = _clean(attrs.get("power")) or _clean(attrs.get("pow")) or "1"
        try:
            power = int(power_text)
        except ValueError as exc:
            raise ValueError(f"{file_path}: edge {edge_name} has non-integer power {power_text!r}") from exc
        if power != 1:
            raise ValueError(f"{file_path}: edge {edge_name} has power {power}; only unit powers are supported")

        if source_external ^ target_external:
            vertex = target if source_external else source
            momentum = _clean(attrs.get("mom")) or _clean(attrs.get("momentum")) or edge_name
            incoming = bool(source_external)
            signed_momentum = momentum if incoming else f"-{momentum}"
            external_lines.append(
                DotExternalLine(
                    name=edge_name,
                    vertex=vertex,
                    momentum=signed_momentum,
                    incoming=incoming,
                )
            )
            if vertex not in internal_vertex_names:
                internal_vertex_names.append(vertex)
            continue

        if source_external and target_external:
            raise ValueError(f"{file_path}: edge {edge_name} connects two invisible external nodes")

        internal_lines.append(
            DotInternalLine(
                name=edge_name,
                source=source,
                target=target,
                mass=mass,
                power=power,
            )
        )
        for vertex in (source, target):
            if vertex not in internal_vertex_names:
                internal_vertex_names.append(vertex)

    if not internal_lines:
        raise ValueError(f"{file_path}: no internal propagator edges found")
    vertex_ids = {name: index + 1 for index, name in enumerate(internal_vertex_names)}
    return ParsedDotGraph(
        path=file_path,
        graph_name=graph_label,
        graph_attributes=graph_attributes,
        internal_vertices=internal_vertex_names,
        vertex_ids=vertex_ids,
        internal_lines=internal_lines,
        external_lines=external_lines,
    )
