"""
Selection module for Artificial Life Neuroevolution Simulation.

Implements tournament selection with elitism.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def tournament_selection(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    tournament_size: int = 3,
    num_parents: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> list[np.ndarray]:
    """
    Select parents using tournament selection.

    Args:
        population: List of genome arrays.
        fitnesses: Fitness values for each genome (higher is better).
        tournament_size: Number of individuals in each tournament.
        num_parents: Number of parents to select.
        rng: Optional random number generator.

    Returns:
        List of selected parent genomes.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(population)
    if n == 0:
        return []

    parents = []
    for _ in range(num_parents):
        tournament_indices = rng.choice(
            n, size=min(tournament_size, n), replace=False
        )
        tournament_fitnesses = fitnesses[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        parents.append(population[winner_idx].copy())

    return parents


def elitism_selection(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    elite_count: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Select elite individuals and return (elites, remaining_population).

    Args:
        population: List of genome arrays.
        fitnesses: Fitness values for each genome.
        elite_count: Number of top individuals to preserve.

    Returns:
        Tuple of (elite_genomes, remaining_genomes).
    """
    if elite_count <= 0:
        return [], population

    n = len(population)
    elite_count = min(elite_count, n)

    sorted_indices = np.argsort(fitnesses)[::-1]
    elite_indices = sorted_indices[:elite_count]
    remaining_indices = sorted_indices[elite_count:]

    elites = [population[i].copy() for i in elite_indices]
    remaining = [population[i].copy() for i in remaining_indices]

    return elites, remaining


def select_parents(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    num_parents: int,
    tournament_size: int = 3,
    elite_count: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Select parents for the next generation, preserving elites.

    Args:
        population: List of genome arrays.
        fitnesses: Fitness values for each genome.
        num_parents: Total number of parents needed (including elites).
        tournament_size: Tournament size for non-elite selection.
        elite_count: Number of elite individuals to preserve.
        rng: Optional random number generator.

    Returns:
        Tuple of (all_parents, elites).
    """
    if rng is None:
        rng = np.random.default_rng()

    elites, remaining_pop = elitism_selection(
        population, fitnesses, elite_count)

    if len(elites) >= num_parents:
        return elites[:num_parents], elites

    num_tournament_parents = num_parents - len(elites)
    if len(remaining_pop) == 0:
        return elites, elites

    tournament_parents = tournament_selection(
        remaining_pop,
        fitnesses[np.argsort(fitnesses)[::-1][elite_count:]],
        tournament_size=tournament_size,
        num_parents=num_tournament_parents,
        rng=rng,
    )

    return elites + tournament_parents, elites
