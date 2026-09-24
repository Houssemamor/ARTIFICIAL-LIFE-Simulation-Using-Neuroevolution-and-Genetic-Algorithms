"""
NEAT mutation operators (Phase 7, plan step 3).

Weight mutation follows Stanley & Miikkulainen (2002): every enabled
connection gene has a small probability of receiving a Gaussian
perturbation; disabled genes may be re-enabled (also Gaussian, from a
near-zero base) so old structure can return.

Structural mutation:
  - add_connection: a fresh enabled edge between a feed-forward-valid
    node pair that has no connection yet, with a random initial weight.
  - add_node: split one enabled connection (in -> out, weight w) into
    in -> new (weight w) and new -> out (weight 1.0). The new node's
    outgoing weight 1.0 is the paper's minimal-disruption choice: the
    old edge's function is exactly preserved in the limit. The old
    gene is kept but disabled, as the paper requires.
"""

from __future__ import annotations
from typing import List

import numpy as np

from evolution.neat.genome import (ConnectionGene, HIDDEN, INPUT, NeatGenome,
                                   NodeGene, OUTPUT)
from evolution.neat.innovation import InnovationTracker


def mutate_weights(
    genome: NeatGenome,
    probability: float,
    sigma: float,
    rng: np.random.Generator,
) -> int:
    """
    Gaussian weight mutation over all connection genes.

    Enabled genes perturb from their current weight. Disabled genes get
    a chance to come back: a gene that is not re-enabled is left
    untouched, and one that is gets a small Gaussian weight from zero.

    Args:
        genome: Mutated in place.
        probability: Per-gene perturbation probability.
        sigma: Gaussian standard deviation.
        rng: Random generator.

    Returns:
        int: Number of genes actually perturbed (or re-enabled).
    """
    changed = 0
    for gene in genome.connections.values():
        if rng.random() >= probability:
            continue
        if gene.enabled:
            gene.weight = float(gene.weight + rng.normal(0.0, sigma))
        else:
            # Re-enable with a small fresh weight drawn from zero
            gene.weight = float(rng.normal(0.0, sigma))
            gene.enabled = True
        changed += 1
    return changed


def add_connection(
    genome: NeatGenome,
    tracker: InnovationTracker,
    rng: np.random.Generator,
    weight_min: float = -1.0,
    weight_max: float = 1.0,
) -> bool:
    """
    Add one enabled connection between a feed-forward-valid pair with
    no existing connection.

    Valid pairs run from inputs or hidden nodes into hidden or output
    nodes, excluding pairs that would close a cycle (the target
    already reaches the source) and self-loops. Because valid pairs are
    (in, out) node-id pairs, two genomes adding the same pair in one
    generation receive the same innovation number from the tracker.

    Returns:
        bool: True if a connection was added.
    """
    input_ids = [node_id for node_id, gene in genome.nodes.items()
                 if gene.node_type == INPUT]
    hidden_ids = [node_id for node_id, gene in genome.nodes.items()
                  if gene.node_type == HIDDEN]
    output_ids = [node_id for node_id, gene in genome.nodes.items()
                  if gene.node_type == OUTPUT]
    sources = input_ids + hidden_ids
    targets = hidden_ids + output_ids
    if not sources or not targets:
        return False

    order = rng.permutation(len(sources))
    for source_index in order:
        in_node = sources[int(source_index)]
        for out_node in targets:
            if in_node == out_node or genome.has_connection(in_node, out_node):
                continue
            if _reaches(out_node, in_node, genome):
                continue  # would close a cycle
            innovation = tracker.get_connection_innovation(in_node, out_node)
            genome.connections[innovation] = ConnectionGene(
                innovation, in_node, out_node,
                float(rng.uniform(weight_min, weight_max)))
            return True
    return False


def add_node(
    genome: NeatGenome,
    tracker: InnovationTracker,
    rng: np.random.Generator,
    weight_min: float = -1.0,
    weight_max: float = 1.0,
) -> bool:
    """
    Split one enabled connection by inserting a new hidden node.

    The split edge (in -> out, weight w) becomes in -> new (weight w)
    and new -> out (weight 1.0). The old gene is disabled, not deleted.
    The new node's outgoing weight of 1.0 is the minimal-disruption
    initialization from the paper.

    Returns:
        bool: True if a node was added.
    """
    enabled = genome.enabled_connections()
    if not enabled:
        return False
    chosen = enabled[int(rng.integers(len(enabled)))]
    new_node_id = genome.next_node_id()
    node_innovation = tracker.get_node_innovation(chosen.in_node, chosen.out_node)
    genome.nodes[new_node_id] = NodeGene(new_node_id, HIDDEN, node_innovation)

    # The old edge is disabled but retained with its original weight
    # (gene history still counts toward compatibility distance)
    original_weight = chosen.weight
    chosen.enabled = False

    in_innovation = tracker.get_connection_innovation(chosen.in_node, new_node_id)
    genome.connections[in_innovation] = ConnectionGene(
        in_innovation, chosen.in_node, new_node_id, original_weight)
    out_innovation = tracker.get_connection_innovation(new_node_id, chosen.out_node)
    genome.connections[out_innovation] = ConnectionGene(
        out_innovation, new_node_id, chosen.out_node, 1.0)
    return True


def mutate_structure(
    genome: NeatGenome,
    tracker: InnovationTracker,
    rng: np.random.Generator,
    add_node_probability: float,
    add_connection_probability: float,
    weight_min: float = -1.0,
    weight_max: float = 1.0,
) -> None:
    """
    Apply one structural mutation attempt (mutated in place).

    Order matches the paper: weight mutation first, then a single
    structural attempt chosen by the configured probabilities.
    """
    roll = rng.random()
    if roll < add_node_probability:
        add_node(genome, tracker, rng, weight_min, weight_max)
    elif roll < add_node_probability + add_connection_probability:
        add_connection(genome, tracker, rng, weight_min, weight_max)


def _reaches(start: int, goal: int, genome: NeatGenome) -> bool:
    """
    True if there is an enabled path start -> ... -> goal. Used to keep
    add_connection feed-forward (never cycle-closing).
    """
    stack = [start]
    seen = set()
    outgoing: dict[int, List[int]] = {}
    for gene in genome.enabled_connections():
        outgoing.setdefault(gene.in_node, []).append(gene.out_node)
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(outgoing.get(node, []))
    return False
