"""
NEAT evolution loop (Phase 7, plan steps 7-9).

One generation:
  1. Build the benchmark world from config (layout_seed applies, so
     every generation presents the same food/obstacle layout).
  2. Spawn one agent per genome at uniform in-bounds positions (the
     Phase 5 review's fix: no corner-line artifact).
  3. Evaluate the whole population in the shared world with the
     depth-layered padded batch (neural.sparse_inference) - the
     batching strategy chosen for Phase 7. One batched forward per
     step, exactly like the fixed-topology engine, then the same
     physics: eat gate, collisions, energy, aging.
  4. Speciate, share fitness, reproduce: each species gets offspring
     proportional to its members' adjusted fitness, with
     min_offspring_per_species, structural + weight mutation, and
     same-species innovation-aligned crossover at crossover_rate.
  5. The previous champion is carried over unchanged (elitism), since
     NEAT has no other mechanism guaranteeing the best genome's
     survival across reproduction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from agents.organism import Organism
from agents.sensors import RayCaster
from evolution.neat.crossover import crossover
from evolution.neat.genome import NeatGenome, minimal_genome
from evolution.neat.innovation import InnovationTracker
from evolution.neat.mutation import mutate_structure, mutate_weights
from evolution.neat.speciation import Species, assign_species
from neural.sparse_inference import compile_population, run_compiled
from simulation.environment import World
from simulation.engine import resolve_collisions
from simulation.world_config import BaselineConfig, NeatConfig

EAT_SIGNAL_THRESHOLD = 0.5


@dataclass
class NeatGenerationMetrics:
    """Per-generation aggregates for the NEAT benchmark."""
    generation: int
    population_size: int
    species_count: int
    mean_fitness: float
    max_fitness: float
    champion_fitness: float
    mean_complexity: float
    max_complexity: int
    mean_food_eaten: float
    num_alive_end: int


def initialize_population(
    size: int,
    tracker: InnovationTracker,
    neat_config: NeatConfig,
    rng: np.random.Generator,
) -> List[NeatGenome]:
    """Minimal-topology starting population (plan step 8)."""
    return [
        minimal_genome(
            neat_config.n_inputs,
            neat_config.n_outputs,
            tracker,
            weight_min=neat_config.initial_weight_min,
            weight_max=neat_config.initial_weight_max,
            rng=rng,
        )
        for _ in range(size)
    ]


def evaluate_population(
    config: BaselineConfig,
    genomes: List[NeatGenome],
    calibration_scales: dict,
    fitness_weights: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, List[Organism]]:
    """
    One shared-world evaluation episode for the whole population.

    Returns:
        (fitnesses per genome, the evaluation agents - whose counters
        feed the next fitness computation and the complexity report).
    """
    if config.world.layout_seed is not None:
        np.random.seed(config.world.layout_seed)
    world = World(config.world.width, config.world.height)
    world.load_from_config(config.model_dump())
    raycaster = RayCaster()

    population_size = len(genomes)
    agents: List[Organism] = []
    margin = world.boundary_margin
    for index in range(population_size):
        x = rng.uniform(margin, world.width - margin)
        y = rng.uniform(margin, world.height - margin)
        agent = Organism(index, x, y, initial_energy=100.0)
        agents.append(agent)

    # Topologies are fixed for the whole episode: compile the padded
    # depth-layer blocks once, then run them per step over the live rows
    compiled = compile_population(genomes)

    for _ in range(config.evolution.evaluation_steps):
        live_indices = [i for i, agent in enumerate(agents) if agent.is_alive]
        if not live_indices:
            break
        live_agents = [agents[i] for i in live_indices]

        observations = torch.as_tensor(
            raycaster.observe_population(live_agents, world),
            dtype=torch.float32)
        action_logits = run_compiled(compiled, observations, live_indices)

        for row, agent in enumerate(live_agents):
            steering = float(action_logits[row, 0])
            acceleration = float(action_logits[row, 1])
            eat_signal = float(action_logits[row, 2])
            agent.apply_action(steering, acceleration, eat_signal, world, dt=1.0)
            if eat_signal > EAT_SIGNAL_THRESHOLD:
                agent.consume_food(world)

        resolve_collisions(live_agents, world)
        for agent in agents:
            agent.update_energy()
            agent.increment_age()

    from evolution.genetic_algorithm import compute_fitness
    fitnesses = np.zeros(population_size, dtype=float)
    for index, agent in enumerate(agents):
        agent.fitness = compute_fitness(
            agent, agent.age, agent.food_eaten, 0.0, agent.collisions,
            calibration_scales, fitness_weights)
        fitnesses[index] = agent.fitness
    return fitnesses, agents


def reproduce(
    genomes: List[NeatGenome],
    speciation_result,
    generation: int,
    tracker: InnovationTracker,
    neat_config: NeatConfig,
    population_size: int,
    champion: NeatGenome,
    rng: np.random.Generator,
) -> List[NeatGenome]:
    """
    Build the next generation.

    Species receive offspring proportional to their total adjusted
    fitness (the paper's species reproduction), with
    min_offspring_per_species guaranteed for species that improved, one
    slot reserved for the carried-over champion, and each offspring
    either an innovation-aligned crossover child or a mutated copy.
    """
    species = speciation_result.species
    adjusted = speciation_result.adjusted_fitness

    # Proportional allocation by species adjusted-fitness mass
    masses = []
    for sp in species:
        mass = sum(adjusted[member] for member in sp.members)
        masses.append(mass)
    total_mass = sum(masses)

    next_generation: List[NeatGenome] = [champion.copy()]
    offspring_budget = population_size - 1

    counts = [0] * len(species)
    if total_mass > 0:
        for index, mass in enumerate(masses):
            counts[index] = int(round(offspring_budget * mass / total_mass))
    # Guarantee at least min_offspring_per_species for species that
    # improved this generation, then fix the sum by trimming the
    # largest allocations
    for index, sp in enumerate(species):
        improved = sp.generation_improved == generation
        if improved and counts[index] < neat_config.min_offspring_per_species:
            counts[index] = neat_config.min_offspring_per_species

    total_counted = sum(counts)
    if total_counted > offspring_budget:
        # Trim proportionally from the largest counts
        while total_counted > offspring_budget:
            largest = max(range(len(counts)), key=lambda i: counts[i])
            if counts[largest] == 0:
                break
            counts[largest] -= 1
            total_counted -= 1
    elif total_counted < offspring_budget:
        # Give the remainder to the strongest species
        strongest = max(range(len(species)),
                        key=lambda i: sum(adjusted[m] for m in species[i].members))
        counts[strongest] += offspring_budget - total_counted

    for species_index, count in enumerate(counts):
        members = species[species_index].members
        for _ in range(count):
            parent_index = members[int(rng.integers(len(members)))]
            parent = genomes[parent_index]
            if (len(members) > 1 and rng.random() < neat_config.crossover_rate):
                other_index = members[int(rng.integers(len(members)))]
                while other_index == parent_index and len(members) > 1:
                    other_index = members[int(rng.integers(len(members)))]
                child = crossover(parent, genomes[other_index],
                                  float(adjusted[parent_index]),
                                  float(adjusted[other_index]), rng)
            else:
                child = parent.copy()
            mutate_weights(child, neat_config.weight_mutation_probability,
                           neat_config.weight_mutation_sigma, rng)
            mutate_structure(
                child, tracker, rng,
                neat_config.add_node_probability,
                neat_config.add_connection_probability,
                neat_config.initial_weight_min,
                neat_config.initial_weight_max,
            )
            next_generation.append(child)

    # Any shortfall from integer rounding: duplicate mutated copies of
    # the champion so the population size is exact
    while len(next_generation) < population_size:
        child = champion.copy()
        mutate_structure(
            child, tracker, rng,
            neat_config.add_node_probability,
            neat_config.add_connection_probability,
            neat_config.initial_weight_min,
            neat_config.initial_weight_max,
        )
        next_generation.append(child)
    return next_generation[:population_size]


def run_generation(
    config: BaselineConfig,
    genomes: List[NeatGenome],
    generation: int,
    tracker: InnovationTracker,
    previous_species: Optional[List[Species]],
    champion: Optional[NeatGenome],
    champion_fitness: float,
    calibration_scales: dict,
    rng: np.random.Generator,
) -> tuple[List[NeatGenome], List[Species], NeatGenome, float,
           NeatGenerationMetrics]:
    """
    One full NEAT generation: evaluate, speciate, reproduce.

    Returns:
        (next_genomes, next_species, champion, champion_fitness,
        metrics).
    """
    if config.neat is None:
        raise ValueError("run_generation requires a config with a neat section")
    neat_config = config.neat
    tracker.new_generation()

    fitnesses, agents = evaluate_population(
        config, genomes, calibration_scales, config.fitness_weights, rng)

    speciation_result = assign_species(
        genomes, fitnesses, generation,
        c1=neat_config.c1_enabled_gene_coefficient,
        c2=neat_config.c2_gene_count_coefficient,
        c3=neat_config.c3_weight_coefficient,
        threshold=neat_config.compatibility_threshold,
        stagnation_generations=neat_config.stagnation_generations,
        min_species_size=neat_config.min_species_size,
        previous_species=previous_species,
    )

    champion_index = int(np.argmax(fitnesses))
    if fitnesses[champion_index] > champion_fitness:
        current_champion = genomes[champion_index].copy()
        current_champion_fitness = float(fitnesses[champion_index])
    else:
        current_champion = (champion if champion is not None
                            else genomes[champion_index].copy())
        current_champion_fitness = champion_fitness

    next_genomes = reproduce(
        genomes, speciation_result, generation, tracker, neat_config,
        len(genomes), current_champion, rng)

    complexities = np.array([genome.complexity() for genome in genomes],
                            dtype=float)
    metrics = NeatGenerationMetrics(
        generation=generation,
        population_size=len(genomes),
        species_count=len(speciation_result.species),
        mean_fitness=float(np.mean(fitnesses)),
        max_fitness=float(np.max(fitnesses)),
        champion_fitness=float(fitnesses[champion_index]),
        mean_complexity=float(np.mean(complexities)),
        max_complexity=int(np.max(complexities)),
        mean_food_eaten=float(np.mean([agent.food_eaten for agent in agents])),
        num_alive_end=sum(1 for agent in agents if agent.is_alive),
    )
    return (next_genomes, speciation_result.species, current_champion,
            current_champion_fitness, metrics)
