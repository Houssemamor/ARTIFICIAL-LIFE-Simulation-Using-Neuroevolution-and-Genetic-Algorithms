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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from agents.organism import Organism
from agents.energy import EnergyConfig, energy_config_from_settings
from evolution.genetic_algorithm import run_generation
from experiments.run_state import (append_history_csv, load_run_state,
                                   restore_population, restore_rng,
                                   save_run_state, verify_config)
from neural.genome import genome_size
from simulation.world_config import BaselineConfig

# Balanced evaluation weights used to score frozen genomes in EVERY
# condition. Experiment D trains under different weightings, but evolved
# genomes must be compared on one common metric or the comparison is
# meaningless (each condition would optimize a different objective).
BALANCED_WEIGHTS = {"survival": 0.4, "food": 0.3,
                    "exploration": 0.2, "collision": 0.1}

# Spawn margin matching World.boundary_margin so positions drawn here are
# always inside the configured world, whatever the population size.
SPAWN_MARGIN = 10.0

CALIBRATION_PATH = "configs/calibration.json"


def load_calibration_scales(path: str = CALIBRATION_PATH) -> Dict[str, float]:
    """
    Load measured fitness reference scales from the calibration output.

    Phase 3's design: fitness normalization uses the calibration pipeline's
    measured component scales - never hand-picked constants. The scales
    must be regenerated (analytics/calibration.py) whenever the evaluation
    regime (population size, step count, energy parameters) changes.

    Raises:
        FileNotFoundError: If the calibration file is missing - fail loud
            rather than silently falling back to invented constants.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return {"survival": float(data["survival"]),
            "food": float(data["food"]),
            "exploration": float(data["exploration"]),
            "collision": float(data["collision"])}


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
    energy_config = energy_config_from_settings(config.energy)
    population = []
    for i, (x, y) in enumerate(positions):
        agent = Organism(i, x, y, initial_energy=100.0,
                         energy_config=energy_config)
        # Draw heading from the seeded global stream before genome init so
        # the random-state sequence remains explicit and reproducible.
        agent.randomize_heading(np.random)
        agent.initialize_genome(genome_size=genome_size())
        population.append(agent)
    return population


def make_clone_population(config: BaselineConfig, genome: np.ndarray,
                          run_seed: int) -> List[Organism]:
    """Create a population of identical-genome agents for frozen evaluation."""
    np.random.seed(run_seed)
    positions = _spawn_positions(config, config.population.size)
    energy_config = energy_config_from_settings(config.energy)
    return [_clone_agent(i, x, y, genome, energy_config, heading_rng=np.random)
            for i, (x, y) in enumerate(positions)]


def _clone_agent(agent_id: int, x: float, y: float,
                 genome: np.ndarray,
                 energy_config: Optional[EnergyConfig] = None,
                 heading_rng=None) -> Organism:
    agent = Organism(agent_id, x, y, initial_energy=100.0,
                     energy_config=energy_config)
    if heading_rng is not None:
        agent.randomize_heading(heading_rng)
    agent.genome = genome.copy()
    return agent


def train_best_genome(config: BaselineConfig, run_seed: int,
                      calibration_scales: Optional[Dict[str, float]] = None,
                      *,
                      checkpoint_path: Optional[str] = None,
                      resume: bool = False,
                      save_interval: Optional[int] = None,
                      history_csv: Optional[str] = None,
                      progress_callback: Optional[Callable[[int, Dict], None]] = None,
                      ) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run a GA training run, optionally resuming from a full-state checkpoint.

    The best genome is the top elite of the final population (run_generation
    assigns elite genomes in fitness order, so index 0 is the best of the
    last evaluated generation).

    Args:
        config: Validated experiment configuration.
        run_seed: Seed for population init and GA operations.
        calibration_scales: Fitness reference scales. Defaults to the
            measured scales from configs/calibration.json (Phase 3 rule:
            never hand-picked constants). Pass explicitly only for
            isolated tests.
        checkpoint_path: Where to persist full run state.
        resume: Restore population, RNGs and history from checkpoint_path.
        save_interval: Generations between checkpoint and CSV flushes.
        history_csv: Append-only per-generation CSV for long runs.
        progress_callback: Called with (completed_generations, metrics).

    Raises:
        ValueError: If elitism_count < 1 - without elites, population[0]
            is an arbitrary offspring and "best genome" is undefined.
        ValueError: If resume is requested without a checkpoint, or the
            checkpoint was produced under a different config.
    """
    if config.evolution.elitism_count < 1:
        raise ValueError("train_best_genome requires elitism_count >= 1")
    if calibration_scales is None:
        calibration_scales = load_calibration_scales()

    interval = save_interval or config.experiment.save_interval
    if resume:
        if checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_path")
        state = load_run_state(checkpoint_path)
        verify_config(state, config)
        population = restore_population(state, config)
        rng = np.random.default_rng(run_seed)
        restore_rng(state, rng)
        history: List[Dict] = list(state["history"])
        start_generation = state["generation"]
    else:
        population = make_population(config, run_seed)
        rng = np.random.default_rng(run_seed)
        history = []
        start_generation = 0

    pending_csv_rows: List[Dict] = []
    last_generation = config.evolution.generations
    for gen in range(start_generation, last_generation):
        population, metrics = run_generation(
            config=config,
            population=population,
            generation=gen,
            calibration_scales=calibration_scales,
            fitness_weights=config.fitness_weights,
            rng=rng,
        )
        row = vars(metrics)
        history.append(row)
        pending_csv_rows.append(row)
        completed = gen + 1
        is_last = completed == last_generation

        # Flush in chunks so a Colab disconnect costs at most one interval
        # of work, and Drive sees few small writes instead of one huge file.
        if history_csv is not None and (completed % interval == 0 or is_last):
            append_history_csv(history_csv, pending_csv_rows)
            pending_csv_rows = []
        if checkpoint_path is not None and (completed % interval == 0 or is_last):
            save_run_state(checkpoint_path, completed, population, rng,
                           history, config, run_seed)
        if progress_callback is not None:
            progress_callback(completed, row)

    return population[0].genome.copy(), history


def evaluate_genome(config: BaselineConfig, genome: np.ndarray,
                    run_seed: int,
                    calibration_scales: Optional[Dict[str, float]] = None
                    ) -> Dict:
    """
    Evaluate one frozen genome on the config's layout with the balanced
    evaluation weights.

    Args:
        config: Validated experiment configuration.
        genome: Frozen genome to evaluate.
        run_seed: Seed for clone spawn positions.
        calibration_scales: Fitness reference scales (defaults to
            configs/calibration.json, same rule as train_best_genome).

    Returns:
        Dict with mean_fitness, mean_food, mean_survival, mean_collisions,
        max_fitness over the clone population.
    """
    if calibration_scales is None:
        calibration_scales = load_calibration_scales()
    clones = make_clone_population(config, genome, run_seed)
    _, metrics = run_generation(
        config=config,
        population=clones,
        generation=0,
        calibration_scales=calibration_scales,
        fitness_weights=BALANCED_WEIGHTS,
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