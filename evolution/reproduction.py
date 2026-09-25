"""
In-episode reproduction for Artificial Life Neuroevolution Simulation.

Phase 6: eligible live agents birth asexual offspring (clone of the
parent genome plus Gaussian mutation) during an episode. This is the
ecosystem's population dynamics, deliberately separate from the
generation-level GA replacement in genetic_algorithm.run_generation /
run_coevolution_generation.

Eligibility follows the plan's rule:
    age >= min_age and energy >= min_energy and fitness >= min_fitness
Per-role carrying capacities (max_population_prey /
max_population_predator) keep the predator population a small fraction
of the prey population.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from agents.organism import Organism
from simulation.world_config import ReproductionConfig
from simulation.environment import World


def is_eligible(agent: Organism, config: ReproductionConfig) -> bool:
    """
    Check the plan's three reproduction eligibility conditions.

    Args:
        agent: Candidate parent (must be alive).
        config: Reproduction thresholds.

    Returns:
        bool: True if the agent may reproduce this step.
    """
    return (
        agent.is_alive
        and agent.age >= config.min_age
        and agent.energy >= config.min_energy
        and agent.fitness >= config.min_fitness
    )


def create_offspring(parent: Organism, offspring_id: int, world: World,
                     config: ReproductionConfig,
                     rng: Optional[np.random.Generator] = None) -> Organism:
    """
    Create one offspring next to the parent: clone of the parent genome
    plus Gaussian mutation, clipped to the genome's [-1, 1] bounds.

    Args:
        parent: Eligible parent organism.
        offspring_id: Unique id for the newborn.
        world: World providing spawn bounds.
        config: Reproduction parameters (mutation_rate/strength).
        rng: Random generator for mutation and spawn jitter.

    Returns:
        Organism: The newborn, same role as the parent.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Clone + mutate: reuse the GA's Gaussian mutation semantics via a
    # local application so reproduction does not depend on population-
    # level helpers (evolution.mutation works on genome lists).
    genome = parent.genome.copy()
    mutation_mask = rng.random(len(genome)) < config.mutation_rate
    noise = rng.normal(0.0, config.mutation_strength, size=len(genome))
    genome[mutation_mask] += noise[mutation_mask]
    np.clip(genome, -1.0, 1.0, out=genome)

    # Spawn near the parent with small jitter, clamped inside the world
    # margins so newborns are always in-bounds
    offset = rng.normal(0.0, 8.0, size=2)
    margin = world.boundary_margin
    x = float(np.clip(parent.position.x + offset[0],
                      margin, world.width - margin))
    y = float(np.clip(parent.position.y + offset[1],
                      margin, world.height - margin))

    # energy_config is inherited from the parent: a newborn that fell
    # back to the module default would run a different energy equation
    # than its parent and silently change the ecosystem's dynamics.
    offspring = Organism(offspring_id, x, y, initial_energy=100.0,
                         role=parent.role,
                         energy_config=parent.energy_config)
    offspring.genome = genome.astype(np.float32)

    # Parent pays the energy cost after a successful birth. Floored at
    # 0: giving birth at the edge of starvation is fatal (the standard
    # update_energy death check handles it next step).
    parent.energy = max(parent.energy - config.reproduction_cost, 0.0)

    return offspring


def reproduction_pass(agents: List[Organism], world: World,
                      config: ReproductionConfig,
                      next_id: int = 0,
                      rng: Optional[np.random.Generator] = None
                      ) -> tuple[List[Organism], int]:
    """
    Run one reproduction pass over the whole agent list.

    Every eligible live agent births at most one offspring per pass.
    The per-role cap counts all agents of that role (live + dead) in the
    list, so long episodes do not accumulate unbounded populations.

    Args:
        agents: Current agent list (both roles; the caller extends its
            agent list with the returned newborns).
        world: World providing spawn bounds.
        config: Reproduction parameters; enabled=False is a no-op.
        next_id: Id to assign to the first newborn.
        rng: Random generator.

    Returns:
        Tuple of (newborns, next_unused_id). The caller extends its
        agent list with the newborns.
    """
    if not config.enabled:
        return [], next_id
    if rng is None:
        rng = np.random.default_rng()

    # Cap counts LIVE agents only: counting corpses would permanently
    # block births in long ecosystem runs once enough agents had died
    role_counts = {'prey': 0, 'predator': 0}
    for agent in agents:
        if agent.is_alive:
            role_counts[agent.role] = role_counts.get(agent.role, 0) + 1
    role_caps = {'prey': config.max_population_prey,
                 'predator': config.max_population_predator}

    newborns: List[Organism] = []
    for agent in agents:
        if role_counts[agent.role] >= role_caps[agent.role]:
            continue
        if not is_eligible(agent, config):
            continue
        newborns.append(
            create_offspring(agent, next_id, world, config, rng))
        next_id += 1
        role_counts[agent.role] += 1

    return newborns, next_id
