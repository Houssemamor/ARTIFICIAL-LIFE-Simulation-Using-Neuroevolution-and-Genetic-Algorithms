"""
Integration test for CPU-deterministic reproducibility (Phase 4).

Runs the same seeded config twice and verifies byte-for-bit identical
generation_metrics.csv output.
"""

from __future__ import annotations
import filecmp
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from simulation.world_config import load_config
from simulation.determinism import set_deterministic_seeds, DeterminismConfig
from agents.organism import Organism
from agents.sensors import RayCaster
from agents.energy import DEFAULT_ENERGY_CONFIG
from evolution.genetic_algorithm import run_generation
from neural.genome import genome_size
from analytics.experiment_logger import ExperimentLogger


@pytest.fixture
def temp_experiment_dir():
    """Create a temporary directory for experiment outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _run_experiment(
    temp_dir: Path,
    exp_name: str,
    config_path: str = "configs/baseline.json",
) -> Path:
    """
    Run a minimal experiment and return the experiment directory.
    """
    config = load_config(config_path)
    config.population.size = 5  # tiny population for speed
    config.evolution.generations = 3
    config.evolution.elitism_count = 1

    # Deterministic seeds
    seeds = {"torch": 42, "numpy": 123, "random": 456}

    # Apply determinism config
    det_config = DeterminismConfig(
        torch_seed=seeds["torch"],
        numpy_seed=seeds["numpy"],
        random_seed=seeds["random"],
        device="cpu",
        reproducibility_tier="cpu-deterministic",
        num_threads=1,
    )
    set_deterministic_seeds(det_config)

    # Deterministic time provider (counter-based)
    time_counter = [0.0]
    def deterministic_time():
        time_counter[0] += 1.0
        return time_counter[0]

    # Create experiment logger using context manager
    with ExperimentLogger(
        config=config,
        seeds=seeds,
        device="cpu",
        reproducibility_tier="cpu-deterministic",
        base_dir=str(temp_dir),
        time_provider=deterministic_time,
    ) as logger:
        # Create initial population
        population = []
        for i in range(config.population.size):
            agent = Organism(i, 100.0 + i * 10, 100.0 + i * 10, initial_energy=100.0)
            agent.initialize_genome(genome_size=genome_size())
            population.append(agent)

        calibration_scales = {"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        fitness_weights = config.fitness_weights

        # Run generations
        for gen in range(config.evolution.generations):
            population, metrics = run_generation(
                config=config,
                population=population,
                generation=gen,
                calibration_scales=calibration_scales,
                fitness_weights=fitness_weights,
                energy_config=DEFAULT_ENERGY_CONFIG,
                rng=np.random.default_rng(42 + gen),
            )
            logger.log_generation({
                "generation": metrics.generation,
                "population_size": metrics.population_size,
                "mean_fitness": metrics.mean_fitness,
                "max_fitness": metrics.max_fitness,
                "min_fitness": metrics.min_fitness,
                "mean_survival": metrics.mean_survival,
                "mean_food": metrics.mean_food,
                "mean_exploration": metrics.mean_exploration,
                "mean_collisions": metrics.mean_collisions,
                "num_alive_end": metrics.num_alive_end,
            })
            for agent in population:
                logger.log_agent(agent, gen)

        return logger.exp_dir


class TestSeededRerun:
    """Test that identical seeds produce byte-for-bit identical outputs."""

    def test_identical_generation_metrics(self, temp_experiment_dir: Path) -> None:
        """Two runs with identical seeds must produce identical generation_metrics.csv."""
        exp1_dir = _run_experiment(temp_experiment_dir, "exp1")
        exp2_dir = _run_experiment(temp_experiment_dir, "exp2")

        csv1 = exp1_dir / "generation_metrics.csv"
        csv2 = exp2_dir / "generation_metrics.csv"

        assert csv1.exists() and csv2.exists()
        # Byte-for-byte comparison
        assert filecmp.cmp(csv1, csv2, shallow=False), \
            "generation_metrics.csv must be byte-for-byte identical across runs"

    def test_identical_agent_metrics(self, temp_experiment_dir: Path) -> None:
        """Two runs with identical seeds must produce identical agent_metrics.csv."""
        exp1_dir = _run_experiment(temp_experiment_dir, "exp1")
        exp2_dir = _run_experiment(temp_experiment_dir, "exp2")

        csv1 = exp1_dir / "agent_metrics.csv"
        csv2 = exp2_dir / "agent_metrics.csv"

        assert csv1.exists() and csv2.exists()
        assert filecmp.cmp(csv1, csv2, shallow=False), \
            "agent_metrics.csv must be byte-for-byte identical across runs"

    def test_identical_config_and_metadata(self, temp_experiment_dir: Path) -> None:
        """Config and metadata must be identical (except timestamp/git)."""
        exp1_dir = _run_experiment(temp_experiment_dir, "exp1")
        exp2_dir = _run_experiment(temp_experiment_dir, "exp2")

        # Config should be identical
        with open(exp1_dir / "config.json") as f1, open(exp2_dir / "config.json") as f2:
            assert json.load(f1) == json.load(f2)

        # Metadata: seeds, device, tier should match; timestamp/git may differ
        with open(exp1_dir / "metadata.json") as f1, open(exp2_dir / "metadata.json") as f2:
            m1 = json.load(f1)
            m2 = json.load(f2)
            assert m1["seeds"] == m2["seeds"]
            assert m1["device"] == m2["device"]
            assert m1["reproducibility_tier"] == m2["reproducibility_tier"]
            assert m1["config"] == m2["config"]

    def test_different_seeds_produce_different_outputs(self, temp_experiment_dir: Path) -> None:
        """Different seeds should produce different outputs."""
        config = load_config("configs/baseline.json")
        config.population.size = 5
        config.evolution.generations = 2
        config.evolution.elitism_count = 1

        # Run with seed set A
        det_a = DeterminismConfig(
            torch_seed=42, numpy_seed=123, random_seed=456,
            device="cpu", reproducibility_tier="cpu-deterministic", num_threads=1
        )
        set_deterministic_seeds(det_a)

        logger_a = ExperimentLogger(
            config=config,
            seeds={"torch": 42, "numpy": 123, "random": 456},
            device="cpu",
            reproducibility_tier="cpu-deterministic",
            base_dir=str(temp_experiment_dir),
        )

        pop_a = []
        for i in range(config.population.size):
            agent = Organism(i, 100.0 + i * 10, 100.0 + i * 10, initial_energy=100.0)
            agent.initialize_genome(genome_size=genome_size())
            pop_a.append(agent)

        pop_a, _ = run_generation(
            config=config,
            population=pop_a,
            generation=0,
            calibration_scales={"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0},
            fitness_weights=config.fitness_weights,
            energy_config=DEFAULT_ENERGY_CONFIG,
            rng=np.random.default_rng(42),
        )
        logger_a.log_generation({"generation": 0, "population_size": 5, "mean_fitness": 1.0,
                                 "max_fitness": 1.0, "min_fitness": 0.5, "mean_survival": 100.0,
                                 "mean_food": 0.0, "mean_exploration": 0.0, "mean_collisions": 0.0,
                                 "num_alive_end": 5})
        logger_a.finalize()

        # Run with seed set B
        det_b = DeterminismConfig(
            torch_seed=999, numpy_seed=888, random_seed=777,
            device="cpu", reproducibility_tier="cpu-deterministic", num_threads=1
        )
        set_deterministic_seeds(det_b)

        logger_b = ExperimentLogger(
            config=config,
            seeds={"torch": 999, "numpy": 888, "random": 777},
            device="cpu",
            reproducibility_tier="cpu-deterministic",
            base_dir=str(temp_experiment_dir),
        )

        pop_b = []
        for i in range(config.population.size):
            agent = Organism(i, 100.0 + i * 10, 100.0 + i * 10, initial_energy=100.0)
            agent.initialize_genome(genome_size=genome_size())
            pop_b.append(agent)

        pop_b, _ = run_generation(
            config=config,
            population=pop_b,
            generation=0,
            calibration_scales={"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0},
            fitness_weights=config.fitness_weights,
            energy_config=DEFAULT_ENERGY_CONFIG,
            rng=np.random.default_rng(999),
        )
        logger_b.log_generation({"generation": 0, "population_size": 5, "mean_fitness": 1.0,
                                 "max_fitness": 1.0, "min_fitness": 0.5, "mean_survival": 100.0,
                                 "mean_food": 0.0, "mean_exploration": 0.0, "mean_collisions": 0.0,
                                 "num_alive_end": 5})
        logger_b.finalize()

        # Results should differ
        csv_a = logger_a.exp_dir / "generation_metrics.csv"
        csv_b = logger_b.exp_dir / "generation_metrics.csv"
        assert not filecmp.cmp(csv_a, csv_b, shallow=False), \
            "Different seeds must produce different outputs"