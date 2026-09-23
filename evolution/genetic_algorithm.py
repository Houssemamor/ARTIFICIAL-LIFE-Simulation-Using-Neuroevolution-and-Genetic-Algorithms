"""
Genetic algorithm orchestration for Artificial Life Neuroevolution Simulation.

Implements the run_generation loop: observe/act (Phase 2) + fitness
accumulation + selection + crossover + mutation + replacement.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

from agents.organism import Organism
from agents.sensors import RayCaster
from agents.energy import EnergyConfig, DEFAULT_ENERGY_CONFIG
from neural.batched_inference import batched_forward, stack_population_weights
from simulation.environment import World
from simulation.engine import resolve_collisions
from simulation.world_config import BaselineConfig
from evolution.selection import select_parents
from evolution.crossover import crossover_population
from evolution.mutation import mutate_population


@dataclass
class GenerationMetrics:
    """Metrics collected during a generation."""
    generation: int
    population_size: int
    mean_fitness: float
    max_fitness: float
    min_fitness: float
    mean_survival: float
    mean_food: float
    mean_exploration: float
    mean_collisions: float
    num_alive_end: int


def compute_fitness(
    agent: Organism,
    steps_survived: int,
    food_eaten: int,
    exploration_distance: float,
    collisions: int,
    calibration_scales: dict,
    fitness_weights: dict,
) -> float:
    """
    Compute normalized fitness from raw components.

    Raw components are divided by calibration reference scales, then
    combined with configurable weights.

    Args:
        agent: The organism.
        steps_survived: Number of steps the agent stayed alive.
        food_eaten: Number of food items consumed.
        exploration_distance: Mean distance from start position.
        collisions: Number of collisions.
        calibration_scales: Dict with keys 'survival', 'food',
            'exploration', 'collision'.
        fitness_weights: Dict with same keys, values sum to 1.0.

    Returns:
        Normalized fitness score.
    """
    survival_ref = calibration_scales.get('survival', 1.0)
    food_ref = calibration_scales.get('food', 1.0)
    exploration_ref = calibration_scales.get('exploration', 1.0)
    collision_ref = calibration_scales.get('collision', 1.0)

    survival_norm = steps_survived / max(survival_ref, 1.0)
    food_norm = food_eaten / max(food_ref, 1.0)
    exploration_norm = exploration_distance / max(exploration_ref, 1.0)
    collision_norm = collisions / max(collision_ref, 1.0)

    fitness = (
        fitness_weights.get('survival', 0.0) * survival_norm +
        fitness_weights.get('food', 0.0) * food_norm +
        fitness_weights.get('exploration', 0.0) * exploration_norm -
        fitness_weights.get('collision', 0.0) * collision_norm
    )

    return max(fitness, 0.0)


def run_generation(
    config: BaselineConfig,
    population: list[Organism],
    generation: int,
    calibration_scales: dict,
    fitness_weights: dict,
    energy_config: Optional[EnergyConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[Organism], GenerationMetrics]:
    """
    Run one full generation: evaluate population, compute fitness, evolve.

    Args:
        config: Baseline configuration.
        population: List of Organism instances (will be replaced).
        generation: Generation number.
        calibration_scales: Reference scales for fitness normalization.
        fitness_weights: Weights for fitness components.
        energy_config: Energy parameters.
        rng: Random generator.

    Returns:
        Tuple of (new_population, generation_metrics).
    """
    if rng is None:
        rng = np.random.default_rng()

    if energy_config is None:
        energy_config = DEFAULT_ENERGY_CONFIG

    # Layout seed: fixed food/obstacle placement for this generation's
    # world so evaluation conditions are identical across the population
    if config.world.layout_seed is not None:
        np.random.seed(config.world.layout_seed)

    world = World(config.world.width, config.world.height)
    world.load_from_config(config.model_dump())

    population_size = config.population.size
    max_steps = config.evolution.evaluation_steps

    raycaster = RayCaster()

    # Tracking for fitness computation
    food_eaten = np.zeros(population_size, dtype=int)
    # ponytail: exploration_distance is never written below, so the
    # 'exploration' fitness weight is dead (always normalizes 0). Wire it
    # up (distance from per-agent spawn position) when the dynamics make
    # the exploration component live; see docs/generalization_results.md.
    exploration_distance = np.zeros(population_size, dtype=float)
    collisions_total = np.zeros(population_size, dtype=int)

    # Evaluation phase
    for step in range(max_steps):
        live_agents = [a for a in population if a.is_alive]
        if not live_agents:
            break

        observations = raycaster.observe_population(live_agents, world)

        genomes = np.stack([a.genome for a in live_agents]).astype(np.float32)
        weights = stack_population_weights(genomes)
        action_logits = batched_forward(
            torch.as_tensor(observations, dtype=torch.float32),
            weights,
        )

        for idx, agent in enumerate(live_agents):
            steering = float(action_logits[idx, 0])
            acceleration = float(action_logits[idx, 1])
            eat_signal = float(action_logits[idx, 2])
            agent.apply_action(steering, acceleration, eat_signal, world,
                               dt=1.0)
            if eat_signal > 0.5:
                if agent.consume_food(world):
                    food_eaten[agent.id] += 1

        resolve_collisions(live_agents, world)
        for agent in live_agents:
            collisions_total[agent.id] += agent.collisions
            agent.collisions = 0

        for agent in population:
            agent.update_energy(energy_config)
            agent.increment_age()

    # Compute fitness for each agent. Dead agents are scored on what they
    # achieved before death (age, food, collisions) rather than skipped:
    # skipping them gave every death an identical fitness of 0, erasing the
    # difference between dying at step 1 and step evaluation_steps-1.
    fitnesses = np.zeros(population_size, dtype=float)
    for agent in population:
        agent_id = agent.id
        steps = agent.age
        food = food_eaten[agent_id]
        expl = (exploration_distance[agent_id]
                if agent_id < len(exploration_distance) else 0.0)
        cols = collisions_total[agent_id]
        fitnesses[agent_id] = compute_fitness(
            agent, steps, food, expl, cols, calibration_scales, fitness_weights
        )

    # Evolution: selection, crossover, mutation
    # Get elite genomes (top fitness)
    sorted_indices = np.argsort(fitnesses)[::-1]
    elite_count = config.evolution.elitism_count
    elite_indices = sorted_indices[:elite_count]
    elite_genomes = [population[i].genome.copy() for i in elite_indices]

    # Select parents for crossover (includes elites + tournament winners)
    parents, _ = select_parents(
        [a.genome for a in population],
        fitnesses,
        num_parents=max(elite_count + 2, 2),  # at least 2 for crossover
        tournament_size=3,
        elite_count=elite_count,
        rng=rng,
    )

    # Parent fitnesses aligned with parents list
    # Match each parent genome to its fitness in the original population
    parent_fitnesses = []
    for p in parents:
        matched = False
        for i in range(population_size):
            genome_i = population[i].genome
            if genome_i is not None and np.array_equal(p, genome_i):
                parent_fitnesses.append(fitnesses[i])
                matched = True
                break
        if not matched:
            parent_fitnesses.append(0.0)
    parent_fitnesses = np.array(parent_fitnesses)

    # Generate offspring
    offspring_genomes = crossover_population(
        parents=parents,
        fitnesses=parent_fitnesses,
        offspring_count=population_size - elite_count,
        crossover_rate=config.evolution.crossover_rate,
        method=config.crossover_method,
        rng=rng,
    )

    # Mutate offspring (not elites)
    offspring_genomes = mutate_population(
        offspring_genomes,
        mutation_rate=config.evolution.mutation_rate,
        mutation_strength=0.1,
        clip_min=-1.0,
        clip_max=1.0,
        rng=rng,
    )

    # Create new population
    new_population = []
    for i in range(population_size):
        x = rng.uniform(
            world.boundary_margin, world.width - world.boundary_margin
        )
        y = rng.uniform(
            world.boundary_margin, world.height - world.boundary_margin
        )
        agent = Organism(i, x, y, initial_energy=100.0)
        if i < elite_count:
            agent.genome = elite_genomes[i]
        else:
            agent.genome = offspring_genomes[i - elite_count]
        new_population.append(agent)

    # Compute metrics
    metrics = GenerationMetrics(
        generation=generation,
        population_size=population_size,
        mean_fitness=float(np.mean(fitnesses)),
        max_fitness=float(np.max(fitnesses)),
        min_fitness=float(np.min(fitnesses)),
        mean_survival=float(np.mean([a.age for a in population])),
        mean_food=float(np.mean(food_eaten)),
        mean_exploration=float(np.mean(exploration_distance)),
        mean_collisions=float(np.mean(collisions_total)),
        num_alive_end=sum(1 for a in population if a.is_alive),
    )

    return new_population, metrics
