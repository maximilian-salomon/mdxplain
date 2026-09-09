# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maeve Branwen Butler
# Created with assistance from GitHub Copilot (Claude Sonnet 5.0).
#
# Copyright (C) 2026 Maximilian Salomon and Maeve Branwen Butler
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Stateless helper turning a pipeline operations log into a dependency graph.

Moved out of ``mdxplain`` into ``spec/`` (see repo memory
spec_module_design.md): only ever needed for Graph->JSON translation, never
at real pipeline runtime.

Consumes only ``log["operations"]`` and each entry's ``depends_on`` (written by
``mdxplain/pipeline/helper/log_helper/log_helper.py``). Deliberately independent of
``global_seq``: that field only exists for logs recorded from sequential Python
calls, while a log built from a GUI configuration has no execution order at all.
Everything here - including node layering - is therefore derived from the
dependency edges alone.
"""

from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx


class LogGraphHelper:
    """Stateless helper functions for building operation dependency graphs."""

    @staticmethod
    def build_graph(log: Dict[str, Any], reduce: bool = False) -> nx.DiGraph:
        """
        Build the operation dependency graph from a pipeline operations log.

        Edges point from a dependency to the operation that depends on it, i.e.
        in the direction the data flows through the pipeline.

        Parameters
        ----------
        log : Dict[str, Any]
            A ``pipeline_data.log`` dict (only ``"operations"`` is read).
        reduce : bool, default=False
            Whether to return the transitive reduction, which drops edges that
            are already implied by a longer path. Reachability is unchanged, so
            this is purely a readability aid for display; correctness-sensitive
            callers can use either graph.

        Returns
        -------
        nx.DiGraph
            Graph whose nodes are operation ids, each carrying the full log
            entry as node attributes.

        Raises
        ------
        ValueError
            If ``reduce`` is requested but the graph contains a cycle.
        """
        operations = log["operations"]

        graph = nx.DiGraph()
        for entry_id, entry in operations.items():
            graph.add_node(entry_id, **entry)
        for entry_id, entry in operations.items():
            for dependency_id in entry["depends_on"]:
                if dependency_id in operations:
                    graph.add_edge(dependency_id, entry_id)

        if not reduce:
            return graph
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                "Operation graph contains a cycle and cannot be reduced - "
                "an operation cannot depend on a later one, so this points to "
                "a corrupted log."
            )

        reduced = nx.transitive_reduction(graph)
        reduced.add_nodes_from(graph.nodes(data=True))
        return reduced

    @staticmethod
    def layers(graph: nx.DiGraph) -> List[List[str]]:
        """
        Group operation ids into dependency layers.

        Parameters
        ----------
        graph : nx.DiGraph
            Graph as returned by ``build_graph``.

        Returns
        -------
        List[List[str]]
            One list of operation ids per layer; operations in a layer depend
            only on operations in earlier layers.
        """
        return [sorted(generation) for generation in nx.topological_generations(graph)]
