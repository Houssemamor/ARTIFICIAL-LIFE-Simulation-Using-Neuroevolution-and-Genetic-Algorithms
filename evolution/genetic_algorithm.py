"""
Genetic algorithm orchestration for Artificial Life Neuroevolution Simulation.

Implements the run_generation loop: observe/act (Phase 2) + fitness
accumulation + selection + crossover + mutation + replacement.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch

from agents.organism import Organism
from agents.sensors import RayCaster
from agents.energy import EnergyConfig, energy_config_from_settings
from neural.batched_inference import batched_forward, stack_population_weights
from simulation.environment import World
from simulation.engine import resolve_collisions, step_simulation, resolve_captures
from simulation.geometry import AGENT_SPAWN_CLEARANCE
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


def evolve_genomes(genomes: List[np.ndarray], fitnesses: np.ndarray,
                   config: BaselineConfig,
                   rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """
    One evolution step over a single role's genome list: elitism,
    parent selection, crossover, mutation.

    Extracted from run_generation so the Phase 6 co-evolution loop
    reuses the exact same evolution machinery per role instead of
    duplicating it.

    Args:
        genomes: Genome arrays for one role's population.
        fitnesses: Fitness per genome, aligned with genomes.
        config: Baseline config (elitism/crossover/mutation params).
        rng: Random generator.

    Returns:
        List of next-generation genomes, length len(genomes): the
        elite copies first, then mutated offspring.
    """
    if rng is None:
        rng = np.random.default_rng()

    population_size = len(genomes)

    # Get elite genomes (top fitness)
    sorted_indices = np.argsort(fitnesses)[::-1]
    elite_count = min(config.evolution.elitism_count, population_size)
    elite_indices = sorted_indices[:elite_count]
    elite_genomes = [genomes[i].copy() for i in elite_indices]

    # Select parents for crossover (includes elites + tournament winners)
    parents, _ = select_parents(
        genomes,
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
            if genomes[i] is not None and np.array_equal(p, genomes[i]):
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

    return elite_genomes + offspring_genomes


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
        energy_config = energy_config_from_settings(config.energy)

    # Layout seed: fixed food/obstacle placement for this generation's
    # world so evaluation conditions are identical across the population
    if config.world.layout_seed is not None:
        np.random.seed(config.world.layout_seed)

    world = World(config.world.width, config.world.height)
    world.load_from_config(config.model_dump())

    # Obstacles are solid, so a caller-supplied population may contain
    # agents whose pre-built spawn overlaps one (make_population draws
    # before the world exists). Relocate them before the exploration
    # origins are recorded, otherwise the episode starts inside a wall.
    for agent in population:
        agent.position.x, agent.position.y = world.clear_position(
            agent.position.x, agent.position.y,
            AGENT_SPAWN_CLEARANCE, rng)

    population_size = config.population.size
    max_steps = config.evolution.evaluation_steps

    raycaster = RayCaster()

    # Tracking for fitness computation
    food_eaten = np.zeros(population_size, dtype=int)
    # Exploration = final displacement from the agent's own spawn
    # position (the units configs/calibration.json measures)
    spawn_positions = {agent.id: (agent.position.x, agent.position.y)
                       for agent in population}
    collisions_total = np.zeros(population_size, dtype=int)

    # Evaluation phase
    for step in range(max_steps):
        live_agents = [a for a in population if a.is_alive]
        if not live_agents:
            break

        observations = raycaster.observe_population(live_agents, world)

        genomes = np.stack([a.genome for a in live_agents]).astype(np.float32)
        weights = stack_population_weights(genomes, device=config.device)
        action_logits = batched_forward(
            torch.as_tensor(observations, dtype=torch.float32),
            weights,
        )
        # The agent loop reads every action as a Python float. Copying the
        # small (N,3) result back once avoids one CUDA sync per agent per step.
        actions = action_logits.detach().cpu()

        for idx, agent in enumerate(live_agents):
            steering = float(actions[idx, 0])
            acceleration = float(actions[idx, 1])
            eat_signal = float(actions[idx, 2])
            agent.apply_action(steering, acceleration, eat_signal, world,
                               dt=1.0)
            if eat_signal > 0.5:
                if agent.consume_food(world):
                    food_eaten[agent.id] += 1

        resolve_collisions(live_agents, world)
        for agent in live_agents:
            collisions_total[agent.id] += agent.collisions
            agent.collisions = 0

        # Food regrowth keeps the energy-limited landscape replenished;
        # step_simulation() does this for GUI/ecosystem paths, but the
        # hand-rolled evaluation loop here owns its own stepping.
        world.regrow_food(getattr(world, 'food_regrowth_per_step', 0.0))

        for agent in population:
            agent.update_energy(energy_config)
            agent.increment_age()

    # Compute fitness for each agent. Dead agents are scored on what they
    # achieved before death (age, food, collisions) rather than skipped:
    # skipping them gave every death an identical fitness of 0, erasing the
    # difference between dying at step 1 and step evaluation_steps-1.
    fitnesses = np.zeros(population_size, dtype=float)
    exploration_distance = np.zeros(population_size, dtype=float)
    for agent in population:
        agent_id = agent.id
        steps = agent.age
        food = food_eaten[agent_id]
        spawn_x, spawn_y = spawn_positions[agent_id]
        expl = float(np.hypot(agent.position.x - spawn_x,
                              agent.position.y - spawn_y))
        exploration_distance[agent_id] = expl
        cols = collisions_total[agent_id]
        fitnesses[agent_id] = compute_fitness(
            agent, steps, food, expl, cols, calibration_scales, fitness_weights
        )

    # Evolution: selection, crossover, mutation (shared helper, also used
    # by the Phase 6 co-evolution loop)
    next_genomes = evolve_genomes(
        [a.genome for a in population], fitnesses, config, rng)

    # Create new population
    new_population = []
    for i in range(population_size):
        x, y = world.random_free_position(
            rng, clearance=AGENT_SPAWN_CLEARANCE)
        agent = Organism(i, x, y, initial_energy=100.0,
                         energy_config=energy_config)
        agent.randomize_heading(rng)
        agent.genome = next_genomes[i]
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

@dataclass
class CoevolutionMetrics:
    """Metrics collected during one co-evolution generation (Phase 6)."""
    generation: int
    prey_fitness_mean: float
    prey_fitness_max: float
    predator_fitness_mean: float
    predator_fitness_max: float
    captures: int
    prey_alive_end: int
    predator_alive_end: int


def _spawn_role_agents(genomes: List[np.ndarray], world: World, role: str,
                       start_id: int,
                       rng: np.random.Generator,
                       energy_config: Optional[EnergyConfig] = None
                       ) -> tuple[list, int]:
    """
    Build fresh evaluation agents for one role, one per genome, at
    uniformly random in-bounds positions clear of solid obstacles (the
    fix from the Phase 5 review: uniform spawns, no corner-line
    artifact).

    Returns the spawned agents and the next unused agent id.
    """
    spawned = []
    next_id = start_id
    for genome in genomes:
        x, y = world.random_free_position(
            rng, clearance=AGENT_SPAWN_CLEARANCE)
        agent = Organism(next_id, x, y, initial_energy=100.0, role=role,
                         energy_config=energy_config)
        agent.randomize_heading(rng)
        agent.genome = genome.copy()
        spawned.append(agent)
        next_id += 1
    return spawned, next_id


def run_coevolution_generation(
    config: BaselineConfig,
    predators: list,
    prey: list,
    generation: int,
    calibration_scales: dict,
    fitness_weights: dict,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list, list, CoevolutionMetrics]:
    """
    Run one co-evolution generation: both roles evaluated together in one
    shared world, then each role evolved independently with the standard
    GA machinery (elitism, tournament selection, crossover, mutation).

    Fitness semantics per role (unchanged compute_fitness):
      prey     - survival + plant food eaten + collision penalty
      predator - survival + captures (scored through the same 'food'
                 component via Organism.food_eaten) + collision penalty
    The exploration component stays 0 (dead, see the ponytail note in
    run_generation).

    Args:
        config: Baseline config; config.predation must be present with
            predator_count > 0.
        predators: Predator Organisms of this generation (genomes used).
        prey: Prey Organisms of this generation (genomes used).
        generation: Generation number (for metrics).
        calibration_scales: Reference scales for fitness normalization.
        fitness_weights: Fitness component weights (same dict for both
            roles; the 'food' component means captures for predators).
        rng: Random generator.

    Returns:
        Tuple of (new_predators, new_prey, metrics).
    """
    if rng is None:
        rng = np.random.default_rng()

    if config.predation is None or config.predation.predator_count <= 0:
        raise ValueError(
            "run_coevolution_generation requires config.predation with "
            "predator_count > 0")
    if not predators or not prey:
        raise ValueError(
            f"co-evolution needs both roles non-empty, got "
            f"{len(predators)} predators and {len(prey)} prey")

    # Same layout handling as run_generation: fixed placement so both
    # roles face identical conditions every generation
    if config.world.layout_seed is not None:
        np.random.seed(config.world.layout_seed)

    world = World(config.world.width, config.world.height)
    world.load_from_config(config.model_dump())
    raycaster = RayCaster()

    energy_config = energy_config_from_settings(config.energy)
    prey_agents, next_id = _spawn_role_agents(
        [a.genome for a in prey], world, 'prey', 0, rng, energy_config)
    predator_agents, _ = _spawn_role_agents(
        [a.genome for a in predators], world, 'predator', next_id, rng,
        energy_config)
    agents = prey_agents + predator_agents
    spawn_positions = {agent.id: (agent.position.x, agent.position.y)
                       for agent in agents}

    captures_total = 0
    # Shaping accumulators: prey rewarded for maintaining distance from
    # nearest predator (fear/avoidance), predators rewarded for closing
    # distance to nearest prey and mildly penalized for persistent turning
    # (anti-circling pursuit bias).
    max_world_dist = float(np.hypot(world.width, world.height))
    threat_dist_sum = {a.id: 0.0 for a in prey_agents}
    threat_steps = {a.id: 0 for a in prey_agents}
    pursuit_close_sum = {a.id: 0.0 for a in predator_agents}
    pursuit_steps = {a.id: 0 for a in predator_agents}
    steer_abs_sum = {a.id: 0.0 for a in predator_agents}
    steer_steps = {a.id: 0 for a in predator_agents}

    for _ in range(config.evolution.evaluation_steps):
        if not any(agent.is_alive for agent in agents):
            break

        live_prey = [a for a in prey_agents if a.is_alive]
        live_predators = [a for a in predator_agents if a.is_alive]

        # Pre-step distances for shaping signals.
        prey_to_pred = {}
        for prey_agent in live_prey:
            if live_predators:
                prey_to_pred[prey_agent.id] = min(
                    prey_agent.position.distance_to(pred.position)
                    for pred in live_predators)
        pred_to_prey = {}
        for pred_agent in live_predators:
            if live_prey:
                pred_to_prey[pred_agent.id] = min(
                    pred_agent.position.distance_to(prey.position)
                    for prey in live_prey)

        step_simulation(world, agents, raycaster)
        captures_total += resolve_captures(
            agents,
            capture_radius=config.predation.capture_radius,
            energy_transfer=config.predation.capture_energy_transfer,
        )

        # Post-step shaping accumulation using current live states.
        live_prey_after = [a for a in prey_agents if a.is_alive]
        live_predators_after = [a for a in predator_agents if a.is_alive]
        for prey_agent in live_prey:
            # Reward keeping distance from threats while alive; if no
            # predators remain, no additional threat pressure this step.
            if live_predators_after:
                dist = min(
                    prey_agent.position.distance_to(pred.position)
                    for pred in live_predators_after)
            elif prey_agent.id in prey_to_pred:
                dist = prey_to_pred[prey_agent.id]
            else:
                dist = 0.0
            threat_dist_sum[prey_agent.id] += dist
            threat_steps[prey_agent.id] += 1

        for pred_agent in live_predators:
            steer_abs_sum[pred_agent.id] += abs(
                float(getattr(pred_agent, 'last_steering', 0.0)))
            steer_steps[pred_agent.id] += 1
            if live_prey_after and pred_agent.id in pred_to_prey:
                dist_after = min(
                    pred_agent.position.distance_to(prey.position)
                    for prey in live_prey_after)
                pursuit_close_sum[pred_agent.id] += (
                    pred_to_prey[pred_agent.id] - dist_after)
                pursuit_steps[pred_agent.id] += 1

    # Fitness per agent; dead agents keep their pre-death achievements,
    # matching run_generation's dead-agent scoring policy. Exploration
    # is the final displacement from the agent's spawn.
    for agent in agents:
        spawn_x, spawn_y = spawn_positions[agent.id]
        exploration = float(np.hypot(agent.position.x - spawn_x,
                                     agent.position.y - spawn_y))
        role = getattr(agent, 'role', 'prey')
        if role == 'predator':
            role_weights = config.fitness_weights_predator or fitness_weights
        else:
            role_weights = config.fitness_weights_prey or fitness_weights
        base_fitness = compute_fitness(
            agent, agent.age, agent.food_eaten, exploration, agent.collisions,
            calibration_scales, role_weights)

        shaping_bonus = 0.0
        if role == 'prey':
            steps = threat_steps.get(agent.id, 0)
            if steps > 0 and max_world_dist > 0.0:
                mean_threat_dist = threat_dist_sum[agent.id] / steps
                # Dense avoidance incentive: higher mean distance from
                # predators while alive increases fitness.
                shaping_bonus = 0.30 * (mean_threat_dist / max_world_dist)
        else:
            p_steps = pursuit_steps.get(agent.id, 0)
            if p_steps > 0:
                mean_close = pursuit_close_sum[agent.id] / p_steps
                # Reward net closing progress toward prey (max speed ~5 px/step)
                # and mildly penalize persistent turning/circling bias.
                shaping_bonus += 0.35 * float(np.clip(mean_close / 5.0, -1.0, 1.0))
            s_steps = steer_steps.get(agent.id, 0)
            if s_steps > 0:
                mean_abs_steer = steer_abs_sum[agent.id] / s_steps
                shaping_bonus -= 0.05 * mean_abs_steer

        agent.fitness = max(base_fitness + shaping_bonus, 0.0)

    prey_fitnesses = np.array([a.fitness for a in prey_agents])
    predator_fitnesses = np.array([a.fitness for a in predator_agents])

    # Evolve each role independently against the shared-world results
    new_prey_genomes = evolve_genomes(
        [a.genome for a in prey_agents], prey_fitnesses, config, rng)
    new_predator_genomes = evolve_genomes(
        [a.genome for a in predator_agents], predator_fitnesses, config, rng)

    new_predators, next_id = _spawn_role_agents(
        new_predator_genomes, world, 'predator', 0, rng, energy_config)
    new_prey, _ = _spawn_role_agents(
        new_prey_genomes, world, 'prey', next_id, rng, energy_config)

    metrics = CoevolutionMetrics(
        generation=generation,
        prey_fitness_mean=float(np.mean(prey_fitnesses)),
        prey_fitness_max=float(np.max(prey_fitnesses)),
        predator_fitness_mean=float(np.mean(predator_fitnesses)),
        predator_fitness_max=float(np.max(predator_fitnesses)),
        captures=captures_total,
        prey_alive_end=sum(1 for a in prey_agents if a.is_alive),
        predator_alive_end=sum(1 for a in predator_agents if a.is_alive),
    )

    return new_predators, new_prey, metrics
