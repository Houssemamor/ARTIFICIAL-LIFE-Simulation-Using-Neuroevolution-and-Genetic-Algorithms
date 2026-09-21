"""
Crossover module for Artificial Life Neuroevolution Simulation.

Implements three crossover methods:
- blend: arithmetic mean of parent genomes (BLX-alpha with alpha=0.5)
- uniform: per-gene random choice from either parent
- none: no crossover, first parent copied (baseline)

The competing-conventions risk: different crossover operators can produce
dramatically different results on the same problem. This module documents
the tradeoffs directly so future users can choose appropriately.

Design doc reference: Section 19.2
"""

from __future__ import annotations
from typing import Literal, Optional
import numpy as np


CrossoverMethod = Literal["blend", "uniform", "none"]


def blend_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Blend crossover (BLX-alpha): child = parent1 + alpha * (parent2 - parent1).

    With alpha=0.5, this is the arithmetic mean: child = 0.5 * (p1 + p2).

    Args:
        parent1: First parent genome.
        parent2: Second parent genome.
        alpha: Blend factor (0.5 = mean, 0 = p1, 1 = p2).
        rng: Random generator (unused, for API consistency).

    Returns:
        Child genome.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Add small noise around the blend to maintain diversity
    child = parent1 + alpha * (parent2 - parent1)
    noise = rng.normal(0, 0.01, size=child.shape).astype(child.dtype)
    return (child + noise).astype(parent1.dtype)


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Uniform crossover: each gene independently chosen from either parent.

    This is more disruptive than blend crossover and can combine
    building blocks from both parents.

    Args:
        parent1: First parent genome.
        parent2: Second parent genome.
        rng: Random generator.

    Returns:
        Child genome.
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = rng.random(parent1.shape) < 0.5
    child = np.where(mask, parent1, parent2)
    return child.astype(parent1.dtype)


def no_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    No crossover: return a copy of the first parent.

    Used as a baseline or when crossover_rate=0.

    Args:
        parent1: First parent genome.
        parent2: Second parent genome (ignored).
        rng: Random generator (unused).

    Returns:
        Copy of parent1.
    """
    return parent1.copy()


def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    method: CrossoverMethod = "blend",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Dispatch to the selected crossover method.

    Args:
        parent1: First parent genome.
        parent2: Second parent genome.
        method: Crossover method ("blend", "uniform", "none").
        rng: Random generator.

    Returns:
        Child genome.

    Raises:
        ValueError: If method is unknown.
    """
    if rng is None:
        rng = np.random.default_rng()

    if method == "blend":
        return blend_crossover(parent1, parent2, rng=rng)
    elif method == "uniform":
        return uniform_crossover(parent1, parent2, rng=rng)
    elif method == "none":
        return no_crossover(parent1, parent2, rng=rng)
    else:
        raise ValueError(f"Unknown crossover method: {method}")


def crossover_population(
    parents: list[np.ndarray],
    fitnesses: np.ndarray,
    offspring_count: int,
    crossover_rate: float = 0.7,
    method: CrossoverMethod = "blend",
    rng: Optional[np.random.Generator] = None,
) -> list[np.ndarray]:
    """
    Produce offspring from parents using crossover.

    Args:
        parents: List of parent genomes (selected via tournament/elitism).
        fitnesses: Fitness values for parents.
        offspring_count: Number of offspring to produce.
        crossover_rate: Probability of crossover vs. cloning.
        method: Crossover method.
        rng: Random generator.

    Returns:
        List of offspring genomes.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(parents) < 2:
        # Fallback: clone the single parent
        return [parents[0].copy() for _ in range(offspring_count)]

    # Ensure fitnesses are valid probabilities
    fitnesses = np.asarray(fitnesses, dtype=np.float64)
    if np.any(np.isnan(fitnesses)) or np.sum(fitnesses) <= 0:
        # Uniform fallback if fitnesses are invalid
        probs = np.ones(len(parents)) / len(parents)
    else:
        probs = fitnesses / np.sum(fitnesses)

    offspring = []
    for _ in range(offspring_count):
        if rng.random() < crossover_rate and len(parents) >= 2:
            # Select two distinct parents weighted by fitness
            # Ensure at least 2 parents have non-zero probability
            valid_parents = np.where(probs > 0)[0]
            if len(valid_parents) >= 2:
                p1_idx, p2_idx = rng.choice(
                    valid_parents,
                    size=2,
                    replace=False,
                    p=probs[valid_parents] / np.sum(probs[valid_parents]),
                )
                child = crossover(
                    parents[p1_idx], parents[p2_idx], method, rng
                )
            else:
                # Not enough valid parents, clone
                p_idx = rng.choice(len(parents), p=probs)
                child = parents[p_idx].copy()
        else:
            # Clone a random parent
            p_idx = rng.choice(len(parents), p=probs)
            child = parents[p_idx].copy()
        offspring.append(child)

    return offspring
