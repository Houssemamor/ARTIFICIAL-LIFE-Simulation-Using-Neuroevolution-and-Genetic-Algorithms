"""
Mutation module for Artificial Life Neuroevolution Simulation.

Implements Gaussian mutation with clipping to [-1, 1] bounds.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def gaussian_mutation(
    genome: np.ndarray,
    mutation_rate: float = 0.05,
    mutation_strength: float = 0.1,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply Gaussian mutation with clipping.

    Each gene has probability `mutation_rate` of being mutated by adding
    Gaussian noise with standard deviation `mutation_strength`. Results
    are clipped to [clip_min, clip_max].

    Args:
        genome: Input genome array.
        mutation_rate: Probability per gene to mutate.
        mutation_strength: Standard deviation of Gaussian noise.
        clip_min: Minimum value after mutation.
        clip_max: Maximum value after mutation.
        rng: Random generator.

    Returns:
        Mutated genome (clipped to bounds).
    """
    if rng is None:
        rng = np.random.default_rng()

    mutated = genome.copy()
    mask = rng.random(genome.shape) < mutation_rate
    noise = rng.normal(0, mutation_strength, size=genome.shape).astype(genome.dtype)
    mutated[mask] += noise[mask]
    np.clip(mutated, clip_min, clip_max, out=mutated)
    return mutated


def mutate_population(
    population: list[np.ndarray],
    mutation_rate: float = 0.05,
    mutation_strength: float = 0.1,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> list[np.ndarray]:
    """
    Apply mutation to an entire population.

    Args:
        population: List of genome arrays.
        mutation_rate: Probability per gene to mutate.
        mutation_strength: Standard deviation of Gaussian noise.
        clip_min: Minimum value after mutation.
        clip_max: Maximum value after mutation.
        rng: Random generator.

    Returns:
        List of mutated genomes.
    """
    if rng is None:
        rng = np.random.default_rng()

    return [
        gaussian_mutation(
            genome,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            clip_min=clip_min,
            clip_max=clip_max,
            rng=rng,
        )
        for genome in population
    ]