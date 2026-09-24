"""
Unit tests for the Phase 7 depth-layered padded batch (plan step 7).

The load-bearing property: batched inference must produce the same
actions as the per-agent reference, for both minimal and
structurally-mutated (variable-depth) genomes, regardless of how the
population's topologies are mixed.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from evolution.neat.genome import minimal_genome
from evolution.neat.innovation import InnovationTracker
from evolution.neat.mutation import add_node, add_connection
from neural.sparse_inference import (batched_forward,
                                     batched_forward_by_species,
                                     forward_single)

N_INPUTS = 12


def mixed_population(seed=0, n=8, mutations=3):
    """Population of minimal and structurally-mutated genomes with
    varied depths, all sharing one innovation tracker."""
    rng = np.random.default_rng(seed)
    tracker = InnovationTracker()
    genomes = []
    for index in range(n):
        genome = minimal_genome(N_INPUTS, 3, tracker, rng=rng)
        for _ in range(index % (mutations + 1)):
            add_node(genome, tracker, rng)
            add_connection(genome, tracker, rng)
        genomes.append(genome)
    return genomes


def test_batched_matches_reference_on_minimal_population():
    genomes = mixed_population(seed=1, n=6, mutations=0)
    observations = torch.randn(len(genomes), N_INPUTS)

    batched = batched_forward(observations, genomes)
    for index, genome in enumerate(genomes):
        reference = forward_single(observations[index:index + 1], genome)
        assert torch.allclose(batched[index:index + 1], reference,
                              atol=1e-5), f"mismatch on genome {index}"


def test_batched_matches_reference_on_variable_topologies():
    genomes = mixed_population(seed=2, n=10, mutations=4)
    # Sanity: the population really does mix depths
    depths = {len(genome.node_depths()) for genome in genomes}
    assert len(depths) > 1, "test needs mixed node counts"
    observations = torch.randn(len(genomes), N_INPUTS)

    batched = batched_forward(observations, genomes)
    for index, genome in enumerate(genomes):
        reference = forward_single(observations[index:index + 1], genome)
        assert torch.allclose(batched[index:index + 1], reference,
                              atol=1e-5), f"mismatch on genome {index}"


def test_per_species_batching_matches_population_batching():
    genomes = mixed_population(seed=3, n=9, mutations=3)
    observations = torch.randn(len(genomes), N_INPUTS)
    species_ids = [index % 3 for index in range(len(genomes))]

    population_wide = batched_forward(observations, genomes)
    per_species = batched_forward_by_species(observations, genomes,
                                             species_ids)
    assert torch.allclose(population_wide, per_species, atol=1e-5)


def test_batched_outputs_respect_activation_ranges():
    genomes = mixed_population(seed=4, n=6, mutations=3)
    observations = torch.randn(len(genomes), N_INPUTS) * 5.0

    actions = batched_forward(observations, genomes)
    assert torch.all(actions[:, 0] >= -1.0) and torch.all(actions[:, 0] <= 1.0)
    assert torch.all(actions[:, 1] >= 0.0) and torch.all(actions[:, 1] <= 1.0)
    assert torch.all(actions[:, 2] >= 0.0) and torch.all(actions[:, 2] <= 1.0)


def test_batched_matches_reference_with_orphaned_hidden_node():
    # Crossover can leave a hidden node with no enabled edges at all
    # (both of its unique connections came from the less-fit parent).
    # Such a node fires on zero input in the reference; the batched
    # path must agree.
    tracker = InnovationTracker()
    genome = minimal_genome(N_INPUTS, 3, tracker,
                            rng=np.random.default_rng(20))
    rng = np.random.default_rng(21)
    add_node(genome, tracker, rng)
    hidden_ids = [node_id for node_id, gene in genome.nodes.items()
                  if gene.node_type == 'hidden']
    for gene in genome.connections.values():
        if gene.in_node in hidden_ids or gene.out_node in hidden_ids:
            gene.enabled = False

    observations = torch.randn(1, N_INPUTS)
    batched = batched_forward(observations, [genome])
    reference = forward_single(observations, genome)
    assert torch.allclose(batched, reference, atol=1e-5)


def test_single_genome_batch():
    tracker = InnovationTracker()
    genome = minimal_genome(N_INPUTS, 3, tracker,
                            rng=np.random.default_rng(5))
    observations = torch.randn(1, N_INPUTS)
    batched = batched_forward(observations, [genome])
    reference = forward_single(observations, genome)
    assert torch.allclose(batched, reference, atol=1e-5)
