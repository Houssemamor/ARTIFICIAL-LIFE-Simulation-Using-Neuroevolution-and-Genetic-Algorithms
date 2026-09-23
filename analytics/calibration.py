"""
Calibration pipeline for Artificial Life Neuroevolution Simulation.

Runs random-policy and stand-still calibration episodes to establish
reference scales for fitness normalization, producing configs/calibration.json.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

from agents.organism import Organism
from agents.sensors import RayCaster
from agents.energy import EnergyConfig, DEFAULT_ENERGY_CONFIG
from neural.batched_inference import batched_forward, stack_population_weights
from neural.genome import genome_size
from simulation.environment import World
from simulation.engine import step_simulation
from simulation.world_config import load_config, BaselineConfig
from simulation.physics import Vector2D


@dataclass
class CalibrationResult:
    """
    Reference scales for fitness normalization.

    Each field is the mean value of a raw fitness component measured
    over calibration episodes. These are used as divisors to normalize
    raw fitness components to ~1.0 scale.
    """
    survival: float
    food: float
    exploration: float
    collision: float
    episodes_run: int
    steps_per_episode: int
    policy: str  # "random" or "stand_still"
    timestamp: float


def run_calibration_episode(
    config: BaselineConfig,
    seed: int,
    policy: str,
    max_steps: int,
    energy_config: Optional[EnergyConfig] = None,
) -> tuple[float, float, float, float]:
    """
    Run a single calibration episode and return raw fitness components.

    Args:
        config: Baseline configuration.
        seed: Random seed for reproducibility.
        policy: "random" or "stand_still".
        max_steps: Maximum steps per episode.
        energy_config: Energy parameters.

    Returns:
        Tuple of (survival_time, food_eaten, exploration, collisions).
    """
    if energy_config is None:
        energy_config = DEFAULT_ENERGY_CONFIG

    np.random.seed(seed)
    torch.manual_seed(seed)

    world = World(config.world.width, config.world.height)
    world.load_from_config(config.model_dump())

    population_size = config.population.size
    agents = []
    for i in range(population_size):
        x = np.random.uniform(world.boundary_margin, world.width - world.boundary_margin)
        y = np.random.uniform(world.boundary_margin, world.height - world.boundary_margin)
        agent = Organism(i, x, y, initial_energy=100.0)
        agent.initialize_genome(genome_size=genome_size())
        agents.append(agent)

    raycaster = RayCaster()

    live_agents = agents
    initial_positions = {a.id: (a.position.x, a.position.y) for a in agents}
    food_by_agent = {a.id: 0 for a in agents}

    for step in range(max_steps):
        if not live_agents:
            break

        observations = raycaster.observe_population(live_agents, world)

        if policy == "random":
            genomes = np.stack([a.genome for a in live_agents]).astype(np.float32)
            weights = stack_population_weights(genomes)
            action_logits = batched_forward(
                torch.as_tensor(observations, dtype=torch.float32),
                weights,
            )
        elif policy == "stand_still":
            n = len(live_agents)
            action_logits = np.zeros((n, 3), dtype=np.float32)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        for idx, agent in enumerate(live_agents):
            steering = float(action_logits[idx, 0])
            acceleration = float(action_logits[idx, 1])
            eat_signal = float(action_logits[idx, 2])
            agent.apply_action(steering, acceleration, eat_signal, world, dt=1.0)
            if eat_signal > 0.5:
                if agent.consume_food(world):
                    food_by_agent[agent.id] += 1

        from simulation.engine import resolve_collisions
        resolve_collisions(live_agents, world)

        for agent in agents:
            agent.update_energy()
            agent.increment_age()

        live_agents = [a for a in agents if a.is_alive]

    # Per-agent component scales in the SAME units compute_fitness
    # consumes: survival = steps survived, food = items eaten,
    # exploration = distance from start, collision = contact events
    # accumulated over the episode. Averaged over the population.
    survivals = [float(a.age) for a in agents]
    foods = [float(food_by_agent[a.id]) for a in agents]
    explorations = []
    for a in agents:
        x0, y0 = initial_positions[a.id]
        explorations.append(float(np.hypot(a.position.x - x0,
                                            a.position.y - y0)))
    collisions = [float(a.collisions) for a in agents]

    return (float(np.mean(survivals)), float(np.mean(foods)),
            float(np.mean(explorations)), float(np.mean(collisions)))


def run_calibration(
    config_path: str,
    episodes: int = 10,
    steps_per_episode: int = 500,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> CalibrationResult:
    """
    Run calibration episodes and write reference scales to JSON.

    Args:
        config_path: Path to baseline config JSON.
        episodes: Number of episodes per policy.
        steps_per_episode: Steps per calibration episode.
        seed: Base random seed.
        output_path: Output JSON path (default: configs/calibration.json).

    Returns:
        CalibrationResult with mean reference scales.
    """
    config = load_config(config_path)

    if output_path is None:
        output_path = "configs/calibration.json"

    random_results = []
    stand_still_results = []

    for ep in range(episodes):
        ep_seed = seed + ep
        random_results.append(run_calibration_episode(
            config, ep_seed, "random", steps_per_episode))
        stand_still_results.append(run_calibration_episode(
            config, ep_seed + episodes, "stand_still", steps_per_episode))

    random_means = np.mean(random_results, axis=0)
    stand_still_means = np.mean(stand_still_results, axis=0)

    # Use random policy as the primary reference (more representative of actual behavior)
    survival_ref = max(random_means[0], 1.0)
    food_ref = max(random_means[1], 1.0)
    exploration_ref = max(random_means[2], 1.0)
    collision_ref = max(random_means[3], 1.0)

    result = CalibrationResult(
        survival=survival_ref,
        food=food_ref,
        exploration=exploration_ref,
        collision=collision_ref,
        episodes_run=episodes,
        steps_per_episode=steps_per_episode,
        policy="random",
        timestamp=time.time(),
    )

    with open(output_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    return result


def load_calibration(calibration_path: str = "configs/calibration.json") -> CalibrationResult:
    """
    Load calibration reference scales from JSON.

    Args:
        calibration_path: Path to calibration JSON.

    Returns:
        CalibrationResult with loaded reference scales.
    """
    with open(calibration_path, 'r') as f:
        data = json.load(f)
    return CalibrationResult(**data)