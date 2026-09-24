"""
Variable-topology genome for the NEAT extension (Phase 7).

Reference: Stanley & Miikkulainen (2002), "Evolving Topologies through
Complexification" - node genes plus connection genes carrying
innovation numbers.

Identity model (minimal, sufficient for every NEAT operator here):
  - Node genes: inputs and outputs have fixed ids (inputs 0..n_inputs-1,
    outputs n_inputs..n_inputs+n_outputs-1, matching the controller
    contract) and no innovation; hidden nodes carry the innovation
    number of the connection they split.
  - Connection genes: keyed by innovation number, referencing node ids.
    Disabled genes are kept (they can be re-enabled by weight
    mutation) and count toward the compatibility distance, per the
    Appendix C formula.

No bias genes: hidden activations apply with zero bias, consistent with
the paper's "no explicit bias nodes" variant of the sparse encoding.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np


INPUT = "input"
HIDDEN = "hidden"
OUTPUT = "output"


@dataclass
class NodeGene:
    """One node: its type and, for hidden nodes, the split innovation."""
    node_id: int
    node_type: str
    innovation: Optional[int] = None


@dataclass
class ConnectionGene:
    """
    One connection: an innovation-tagged edge between two node ids.

    innovation is the structural identity (Stanley & Miikkulainen):
    the same (in_node, out_node) structural mutation anywhere in the
    population receives the same number, which is what makes
    innovation-aligned crossover possible.
    """
    innovation: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True


@dataclass
class NeatGenome:
    """
    A variable-topology NEAT genome.

    Attributes:
        nodes: node_id -> NodeGene.
        connections: innovation -> ConnectionGene.
    """
    nodes: Dict[int, NodeGene] = field(default_factory=dict)
    connections: Dict[int, ConnectionGene] = field(default_factory=dict)

    def copy(self) -> "NeatGenome":
        """Deep copy (operators mutate genomes in place, so this is the
        only sharing-safe operation)."""
        return NeatGenome(
            nodes={node_id: replace(gene) for node_id, gene in self.nodes.items()},
            connections={innov: replace(gene)
                         for innov, gene in self.connections.items()},
        )

    @property
    def n_inputs(self) -> int:
        return sum(1 for gene in self.nodes.values() if gene.node_type == INPUT)

    @property
    def n_outputs(self) -> int:
        return sum(1 for gene in self.nodes.values() if gene.node_type == OUTPUT)

    @property
    def hidden_nodes(self) -> List[NodeGene]:
        return [gene for gene in self.nodes.values() if gene.node_type == HIDDEN]

    def enabled_connections(self) -> List[ConnectionGene]:
        return [gene for gene in self.connections.values() if gene.enabled]

    def complexity(self) -> int:
        """
        Nodes plus enabled connection genes - the standard NEAT
        complexity measure used for the complexity-over-generations
        plot.
        """
        return len(self.nodes) + len(self.enabled_connections())

    def next_node_id(self) -> int:
        """One past the highest node id (fresh ids for added nodes)."""
        return max(self.nodes) + 1 if self.nodes else 0

    def has_connection(self, in_node: int, out_node: int) -> bool:
        return any(gene.in_node == in_node and gene.out_node == out_node
                   for gene in self.connections.values())

    def output_node_ids(self) -> List[int]:
        """Output node ids in controller order (steer, accel, eat)."""
        return sorted(node_id for node_id, gene in self.nodes.items()
                      if gene.node_type == OUTPUT)

    def node_depths(self) -> Dict[int, int]:
        """
        Longest-path depth of every node from the input layer.

        Inputs are depth 0; each other node sits one below its deepest
        enabled incoming source. Structural operators only ever add
        edges from existing nodes to fresh nodes, so the graph stays a
        DAG; a cycle raises rather than looping forever.

        Returns:
            node_id -> depth.
        """
        depths: Dict[int, int] = {
            node_id: 0 for node_id, gene in self.nodes.items()
            if gene.node_type == INPUT
        }
        remaining = [node_id for node_id, gene in self.nodes.items()
                     if gene.node_type != INPUT]
        incoming: Dict[int, List[int]] = {node_id: [] for node_id in remaining}
        for gene in self.enabled_connections():
            if gene.out_node in incoming:
                incoming[gene.out_node].append(gene.in_node)

        # Iterative relaxation: at most len(remaining) passes suffice for
        # a DAG; more passes than nodes means a cycle.
        max_passes = len(remaining) + 1
        for _ in range(max_passes):
            changed = False
            for node_id in remaining:
                if node_id in depths:
                    continue
                sources = incoming[node_id]
                if not sources:
                    # Source-less node (crossover can drop every edge of
                    # a hidden node): it fires on zero input, matching
                    # the per-agent reference where the activation is
                    # tanh(0) = 0
                    depths[node_id] = 0
                    changed = True
                elif all(source in depths for source in sources):
                    depths[node_id] = 1 + max(depths[source]
                                              for source in sources)
                    changed = True
            if not changed:
                break
        missing = [node_id for node_id in remaining if node_id not in depths]
        if missing:
            raise ValueError(
                f"genome contains a cycle: nodes {missing} have no "
                f"reachable path from the inputs")
        return depths


def minimal_genome(n_inputs: int, n_outputs: int, tracker,
                   weight_min: float = -1.0, weight_max: float = 1.0,
                   rng: Optional[np.random.Generator] = None) -> NeatGenome:
    """
    Minimal topology per plan step 8: direct input->output connections
    only, no hidden nodes.

    Every genome created within the same generation through the same
    tracker shares the connection innovations, because the tracker's
    per-generation reuse map keys the structure (in, out).

    Args:
        n_inputs: Number of input nodes (controller observations).
        n_outputs: Number of output nodes (steer, accel, eat).
        tracker: InnovationTracker.
        weight_min/max: Initial connection weight range.
        rng: Random generator for initial weights.

    Returns:
        The new NeatGenome.
    """
    if rng is None:
        rng = np.random.default_rng()
    genome = NeatGenome()
    for node_id in range(n_inputs):
        genome.nodes[node_id] = NodeGene(node_id, INPUT)
    for offset in range(n_outputs):
        node_id = n_inputs + offset
        genome.nodes[node_id] = NodeGene(node_id, OUTPUT)

    for in_node in range(n_inputs):
        for out_node in range(n_inputs, n_inputs + n_outputs):
            innovation = tracker.get_connection_innovation(in_node, out_node)
            weight = float(rng.uniform(weight_min, weight_max))
            genome.connections[innovation] = ConnectionGene(
                innovation, in_node, out_node, weight)
    return genome
