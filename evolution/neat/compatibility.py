"""
Compatibility distance for NEAT speciation (Phase 7, plan step 5).

The Appendix C formula, implemented exactly:

    delta = (c1 * E + c2 * D) / N + c3 * Wbar

where
    E    = enabled connection-gene count of the first genome
    D    = |total connection genes of g1 - total connection genes of g2|
          (disabled genes count, per the specification's gene-count
          term)
    N    = max(total connection genes of g1, total connection genes of g2)
    Wbar = mean absolute weight difference over innovations present in
          both genomes (0.0 when no innovation is shared)

Baseline constants (Appendix C): c1 = 1.0, c2 = 1.0, c3 = 0.4,
species threshold delta_t = 3.0.
"""

from __future__ import annotations

import numpy as np

from evolution.neat.genome import NeatGenome


def compatibility_distance(
    first: NeatGenome,
    second: NeatGenome,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.4,
) -> float:
    """
    Compatibility distance delta between two genomes per the formula
    above.

    Args:
        first, second: The genomes to compare.
        c1: Coefficient on the enabled-gene count.
        c2: Coefficient on the gene-count difference.
        c3: Coefficient on the mean weight difference.

    Returns:
        float: delta; genomes belong to the same species when
        delta <= compatibility_threshold.
    """
    enabled_count = len(first.enabled_connections())
    total_first = len(first.connections)
    total_second = len(second.connections)

    gene_count_difference = abs(total_first - total_second)
    normalizer = max(total_first, total_second)
    if normalizer == 0:
        structural_term = 0.0
    else:
        structural_term = (c1 * enabled_count + c2 * gene_count_difference) / normalizer

    shared = [
        (first.connections[innovation].weight,
         second.connections[innovation].weight)
        for innovation in first.connections
        if innovation in second.connections
    ]
    if shared:
        weights_first, weights_second = zip(*shared)
        mean_weight_difference = float(np.mean(np.abs(
            np.asarray(weights_first) - np.asarray(weights_second))))
    else:
        # No shared innovations: no evidence of weight similarity or
        # difference, so the weight term contributes nothing.
        mean_weight_difference = 0.0

    return structural_term + c3 * mean_weight_difference
