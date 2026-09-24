"""
Innovation-aligned crossover for NEAT (Phase 7, plan step 4).

Rules, as specified:
  - Matching genes (same innovation in both parents): the child keeps
    the topology and takes the weight from either parent with equal
    probability.
  - Disjoint genes (in the first parent only) and excess genes (in the
    second parent only): the child takes them from the fitter parent.
  - Node genes are the union of both parents' nodes, keyed by identity
    (fixed id for inputs/outputs, split innovation for hidden nodes),
    so every inherited connection has both endpoints present.

The caller passes adjusted fitnesses (post-sharing); the fitter parent
is the one with the higher value. Ties go to the first parent.
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from evolution.neat.genome import (ConnectionGene, HIDDEN, INPUT, NeatGenome,
                                   NodeGene, OUTPUT)


def crossover(first: NeatGenome, second: NeatGenome,
              first_fitness: float, second_fitness: float,
              rng: Optional[np.random.Generator] = None) -> NeatGenome:
    """
    Produce one child by innovation-aligned crossover.

    Args:
        first, second: Parent genomes (unmodified).
        first_fitness, second_fitness: Adjusted fitnesses used to pick
            the source of disjoint/excess genes.
        rng: Random generator for matching-gene weight choice.

    Returns:
        The child genome.
    """
    if rng is None:
        rng = np.random.default_rng()
    first_is_fitter = first_fitness >= second_fitness

    child = NeatGenome()
    # Node union keyed by structural identity. Two different hidden
    # nodes (distinct split innovations) can carry the same local
    # node_id in different parents, so every parent gets an explicit
    # parent_id -> child_id endpoint map and connections are rewritten
    # through it.
    key_index = {}
    endpoint_maps = []
    next_id = 0
    for parent in (first, second):
        mapping = {}
        for parent_id, gene in parent.nodes.items():
            key = _node_key(gene)
            if key in key_index:
                child_id = key_index[key]
            else:
                child_id = next_id
                next_id += 1
                child.nodes[child_id] = NodeGene(
                    child_id, gene.node_type, gene.innovation)
                key_index[key] = child_id
            mapping[parent_id] = child_id
        endpoint_maps.append(mapping)
    # Canonical io ids keep the controller contract (inputs 0..n-1,
    # outputs contiguous and ordered)
    _canonicalize_io_ids(child)

    first_map, second_map = endpoint_maps
    for innovation, gene in first.connections.items():
        if innovation in second.connections:
            other = second.connections[innovation]
            weight = float(gene.weight) if rng.random() < 0.5 else float(other.weight)
            child.connections[innovation] = ConnectionGene(
                innovation, first_map[gene.in_node], first_map[gene.out_node],
                weight, gene.enabled)
        elif first_is_fitter:
            # disjoint gene held by the fitter first parent
            child.connections[innovation] = ConnectionGene(
                innovation, first_map[gene.in_node], first_map[gene.out_node],
                gene.weight, gene.enabled)
        # else: disjoint gene of the less-fit first parent -> dropped

    for innovation, gene in second.connections.items():
        if innovation in first.connections:
            continue  # matching gene already handled
        if not first_is_fitter:
            # excess gene held by the fitter second parent
            child.connections[innovation] = ConnectionGene(
                innovation, second_map[gene.in_node], second_map[gene.out_node],
                gene.weight, gene.enabled)
        # else: excess gene of the less-fit second parent -> dropped

    return child


def _node_key(gene: NodeGene) -> tuple:
    """Structural identity of a node: fixed for io nodes, innovation for hidden."""
    if gene.node_type == INPUT:
        return (INPUT, gene.node_id)
    if gene.node_type == OUTPUT:
        return (OUTPUT, gene.node_id)
    return (HIDDEN, gene.innovation)


def _canonicalize_io_ids(genome: NeatGenome) -> None:
    """
    Force input ids to 0..n_inputs-1 and output ids to the contiguous
    output block, so the controller contract (12 inputs, 3 ordered
    outputs) holds for every child regardless of parent id layouts.
    """
    input_genes = sorted((gene for gene in genome.nodes.values()
                          if gene.node_type == INPUT),
                         key=lambda gene: gene.node_id)
    output_genes = sorted((gene for gene in genome.nodes.values()
                           if gene.node_type == OUTPUT),
                          key=lambda gene: gene.node_id)
    n_inputs = len(input_genes)
    for new_id, gene in enumerate(input_genes):
        gene.node_id = new_id
    for offset, gene in enumerate(output_genes):
        gene.node_id = n_inputs + offset
