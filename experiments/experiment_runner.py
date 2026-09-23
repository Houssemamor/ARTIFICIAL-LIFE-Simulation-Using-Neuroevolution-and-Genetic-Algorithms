"""
Reusable experiment-runner primitives for Phase 5+ experiments.

Shared by experiments/run_generalization.py and experiments/run_experiment_d.py
so both use identical training/evaluation code paths. Evaluation of a frozen
genome reuses run_generation with a population of clones: since every genome
is identical, the returned mean fitness is that genome's score on the layout
(one generation of wasted crossover work is negligible next to evaluation).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from agents.organism import Organism
from agents.energy import DEFAULT_ENERGY_CONFIG
from evolution.genetic_algorithm import run_generation
from neural.genome import genome_size
from simulation.world_config import BaselineConfig

# Fixed reference scales for fitness normalization. Identical across all
# conditions/layouts so scores are comparable; only relative comparisons
# are interpreted, never absolute values.
CALIBRATION_SCALES = {"survival": 100.0, "food": 10.0,
                      "exploration": 50.0, "collision": 5.0}

# Balanced evaluation weights used to score frozen genomes in EVERY
# condition. Experiment D trains under different weightings, but evolved
# genomes must be compared on one common metric or the comparison is
# meaningless (each condition would optimize a different objective).
BALANCED_WEIGHTS = {"survival": 0.4, "food": 0.3,
                    "exploration": 0.2, "collision": 0.1}

# Spawn margin matching World.boundary_margin so positions drawn here are
# always inside the configured world, whatever the population size.
SPAWN_MARGIN = 10.0


def load_layouts(path: str) -> List[Tuple[str, BaselineConfig]]:
    """
    Load a base_config + layouts file (generalization_train/test.json).

    Each layout entry merges its layout_seed into base_config.world and the
    result is validated through BaselineConfig.

    Returns:
        List of (layout_name, validated_config) in file order.
    """
    with open(path, "r") as f:
        data = json.load(f)
    base = data["base_config"]
    layouts = []
    for layout in data["layouts"]:
        merged = dict(base)
        merged["world"] = {**base["world"], "layout_seed": layout["layout_seed"]}
        layouts.append((layout["name"], BaselineConfig(**merged)))
    return layouts


def _spawn_positions(config: BaselineConfig, count: int) -> List[Tuple[float, float]]:
    """
    Draw spawn positions from the global RNG (already seeded by the
    caller) uniformly inside the world bounds, so positions stay valid
    for any population/world-size combination instead of scaling off the
    top-left corner.
    """
    margin = SPAWN_MARGIN
    return [(float(np.random.uniform(margin, config.world.width - margin)),
             float(np.random.uniform(margin, config.world.height - margin)))
            for _ in range(count)]


def make_population(config: BaselineConfig, run_seed: int) -> List[Organism]:
    """
    Create a fresh random population for the given config.

    Global np.random is seeded with run_seed first so spawn positions and
    genome initialization (Organism.initialize_genome uses the global
    RNG) are reproducible.
    """
    np.random.seed(run_seed)
    positions = _spawn_positions(config, config.population.size)
    population = []
    for i, (x, y) in enumerate(positions):
        agent = Organism(i, x, y, initial_energy=100.0)
        agent.initialize_genome(genome_size=genome_size())
        population.append(agent)
    return population


def make_clone_population(config: BaselineConfig, genome: np.ndarray,
                          run_seed: int) -> List[Organism]:
    """Create a population of identical-genome agents for frozen evaluation."""
    np.random.seed(run_seed)
    positions = _spawn_positions(config, config.population.size)
    return [_clone_agent(i, x, y, genome)
            for i, (x, y) in enumerate(positions)]


def _clone_agent(agent_id: int, x: float, y: float,
                 genome: np.ndarray) -> Organism:
    agent = Organism(agent_id, x, y, initial_energy=100.0)
    agent.genome = genome.copy()
    return agent


def train_best_genome(config: BaselineConfig, run_seed: int) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run a full GA training run and return (best_genome, per-generation metrics).

    The best genome is the top elite of the final population (run_generation
    assigns elite genomes in fitness order, so index 0 is the best of the
    last evaluated generation).

    Raises:
        ValueError: If elitism_count < 1 - without elites, population[0]
            is an arbitrary offspring and "best genome" is undefined.
    """
    if config.evolution.elitism_count < 1:
        raise ValueError("train_best_genome requires elitism_count >= 1")
    population = make_population(config, run_seed)
    rng = np.random.default_rng(run_seed)
    history: List[Dict] = []
    for gen in range(config.evolution.generations):
        population, metrics = run_generation(
            config=config,
            population=population,
            generation=gen,
            calibration_scales=CALIBRATION_SCALES,
            fitness_weights=config.fitness_weights,
            energy_config=DEFAULT_ENERGY_CONFIG,
            rng=rng,
        )
        history.append(vars(metrics))
    return population[0].genome.copy(), history


def evaluate_genome(config: BaselineConfig, genome: np.ndarray,
                     run_seed: int) -> Dict:
    """
    Evaluate one frozen genome on the config's layout with the balanced
    evaluation weights.

    Returns:
        Dict with mean_fitness, mean_food, mean_survival, mean_collisions,
        max_fitness over the clone population.
    """
    clones = make_clone_population(config, genome, run_seed)
    _, metrics = run_generation(
        config=config,
        population=clones,
        generation=0,
        calibration_scales=CALIBRATION_SCALES,
        fitness_weights=BALANCED_WEIGHTS,
        energy_config=DEFAULT_ENERGY_CONFIG,
        rng=np.random.default_rng(run_seed),
    )
    return {
        "mean_fitness": metrics.mean_fitness,
        "max_fitness": metrics.max_fitness,
        "mean_food": metrics.mean_food,
        "mean_survival": metrics.mean_survival,
        "mean_collisions": metrics.mean_collisions,
    }


def write_stats_summary(output_path: Path, summary: Dict) -> None:
    """Write a stats summary dict to JSON, creating parent dirs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)