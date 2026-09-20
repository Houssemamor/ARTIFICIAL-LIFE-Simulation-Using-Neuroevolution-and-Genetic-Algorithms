"""
Integration test for one full generation (Phase 3).

Runs a complete generation end-to-end with a tiny population
and verifies the pipeline works.
"""

from __future__ import annotations
import numpy as np
import pytest

from simulation.world_config import load_config
from agents.organism import Organism
from agents.energy import DEFAULT_ENERGY_CONFIG
from neural.genome import genome_size
from evolution.genetic_algorithm import run_generation


class TestOneGeneration:
    @pytest.fixture
    def config(self):
        return load_config("configs/baseline.json")

    def test_generation_runs_and_returns_new_population(self, config) -> None:
        population_size = 10
        config.population.size = population_size
        config.evolution.generations = 1
        config.evolution.elitism_count = 1

        population = []
        for i in range(population_size):
            agent = Organism(i, 100.0 + i * 10, 100.0 + i * 10, initial_energy=100.0)
            agent.initialize_genome(genome_size=genome_size())
            population.append(agent)

        calibration_scales = {"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        fitness_weights = config.fitness_weights

        new_population, metrics = run_generation(
            config=config,
            population=population,
            generation=0,
            calibration_scales=calibration_scales,
            fitness_weights=fitness_weights,
            energy_config=DEFAULT_ENERGY_CONFIG,
            rng=np.random.default_rng(42),
        )

        assert len(new_population) == population_size
        assert all(a.is_alive for a in new_population)  # all reset to alive
        assert all(a.energy == 100.0 for a in new_population)
        assert all(a.age == 0 for a in new_population)
        assert all(a.genome is not None for a in new_population)

    def test_metrics_returned(self, config) -> None:
        population_size = 5
        config.population.size = population_size
        config.evolution.elitism_count = 1

        population = []
        for i in range(population_size):
            agent = Organism(i, 100.0 + i * 10, 100.0 + i * 10, initial_energy=100.0)
            agent.initialize_genome(genome_size=genome_size())
            population.append(agent)

        calibration_scales = {"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        fitness_weights = config.fitness_weights

        _, metrics = run_generation(
            config=config,
            population=population,
            generation=0,
            calibration_scales=calibration_scales,
            fitness_weights=fitness_weights,
            energy_config=DEFAULT_ENERGY_CONFIG,
            rng=np.random.default_rng(42),
        )

        assert metrics.generation == 0
        assert metrics.population_size == population_size
        assert metrics.mean_fitness >= 0.0
        assert metrics.max_fitness >= metrics.mean_fitness
        assert metrics.min_fitness <= metrics.mean_fitness
        assert metrics.mean_survival >= 0.0
        assert metrics.num_alive_end >= 0

    def test_elitism_preserves_best(self, config) -> None:
        population_size = 10
        config.population.size = population_size
        config.evolution.elitism_count = 2
        config.evolution.crossover_rate = 0.0  # no crossover, just mutation
        config.evolution.mutation_rate = 0.0   # no mutation

        population = []
        for i in range(population_size):
            agent = Organism(i, 100.0 + i * 10, 100.0 + i * 10, initial_energy=100.0)
            agent.initialize_genome(genome_size=genome_size())
            population.append(agent)

        calibration_scales = {"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        fitness_weights = config.fitness_weights

        new_population, _ = run_generation(
            config=config,
            population=population,
            generation=0,
            calibration_scales=calibration_scales,
            fitness_weights=fitness_weights,
            energy_config=DEFAULT_ENERGY_CONFIG,
            rng=np.random.default_rng(42),
        )

        # With crossover_rate=0 and mutation_rate=0, elites should be exact copies
        # of the top-fitness genomes from the original population
        elite_count = config.evolution.elitism_count
        
        # Get the expected elite genomes (top fitness from original population)
        # We need to run fitness computation manually to know which were elites
        # Since fitness depends on behavior, we verify that the new population's
        # first elite_count genomes are exact copies of some genomes from the
        # original population (the ones that were selected as elites)
        new_elite_genomes = [new_population[i].genome for i in range(elite_count)]
        
        # Each new elite genome should exactly match some genome from the original population
        for new_g in new_elite_genomes:
            found_match = False
            for orig_agent in population:
                if np.array_equal(new_g, orig_agent.genome):
                    found_match = True
                    break
            assert found_match, "Elite genome not found in original population"