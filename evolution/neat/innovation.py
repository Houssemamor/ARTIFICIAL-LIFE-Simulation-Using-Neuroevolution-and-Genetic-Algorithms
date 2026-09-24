"""
Global innovation-number bookkeeping for NEAT (Phase 7, plan step 2).

The rule, as specified: two independent, identical structural
mutations occurring in the same generation must receive the same
innovation number; a structure that has not appeared before receives
the next number from a counter that persists across the whole run.

The tracker holds:
  - `counter`: the next innovation number, monotonic for the run.
  - `_current`: per-generation reuse map, structure signature ->
    innovation number. Cleared by `new_generation()` at each boundary
    and consulted by the mutation operators during the generation.
"""

from __future__ import annotations
from typing import Dict, Tuple

ConnectionKey = Tuple[int, int]
NodeKey = Tuple[int, int]


class InnovationTracker:
    """
    Per-run innovation counter with per-generation reuse.
    """

    def __init__(self) -> None:
        self.counter: int = 1
        self._current_connections: Dict[ConnectionKey, int] = {}
        self._current_nodes: Dict[NodeKey, int] = {}
        self.reuse_hits: int = 0
        self.fresh_assignments: int = 0

    def new_generation(self) -> None:
        """
        Clear the per-generation reuse maps.

        Called at every generation boundary: identical structures may
        reuse a number *within* one generation, never across them.
        """
        self._current_connections = {}
        self._current_nodes = {}

    def get_connection_innovation(self, in_node: int, out_node: int) -> int:
        """
        Innovation number for a connection structure (in_node,
        out_node), reusing the same-generation number when present.

        Returns:
            The innovation number for this structure in this generation.
        """
        key = (in_node, out_node)
        if key in self._current_connections:
            self.reuse_hits += 1
            return self._current_connections[key]
        innovation = self.counter
        self.counter += 1
        self._current_connections[key] = innovation
        self.fresh_assignments += 1
        return innovation

    def get_node_innovation(self, in_node: int, out_node: int) -> int:
        """
        Innovation number for a node created by splitting the connection
        (in_node, out_node). The signature is the split edge, so two
        agents splitting the same edge in one generation agree on the
        node's identity.
        """
        key = (in_node, out_node)
        if key in self._current_nodes:
            self.reuse_hits += 1
            return self._current_nodes[key]
        innovation = self.counter
        self.counter += 1
        self._current_nodes[key] = innovation
        self.fresh_assignments += 1
        return innovation
