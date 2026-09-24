"""Integration tests for the Phase 6 co-evolution GA generation loop."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from simulation.world_config import load_config
from evolution.genetic_algorithm import run_coevolution_generation
from analytics.hall_of_fame import HallOfFame
from neural.genome import genome_size

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "configs", "predator_prey.json")


def make_population(count, role):
    """Random-genome organisms for one role."""
    rng = np.random.default_rng(1)
    population = []
    size = genome_size()
    for i in range(count):
        genome = rng.uniform(-1, 1, size).astype(np.float32)
        agent = type("Agent", (), {})()
        agent.genome = genome
        population.append(agent)
    return population


def small_config():
    """Real config, short horizon so generations stay fast."""
    config = load_config(CONFIG_PATH)
    return config.model_copy(update={
        "evolution": config.evolution.model_copy(
            update={"evaluation_steps": 40}),
        "population": config.population.model_copy(update={"size": 6}),
        "predation": config.predation.model_copy(update={"predator_count": 3}),
    })


def test_coevolution_generation_sizes_and_roles():
    config = small_config()
    prey = make_population(6, 'prey')
    predators = make_population(3, 'predator')

    new_predators, new_prey, metrics = run_coevolution_generation(
        config, predators, prey, 1,
        {"survival": 149.85, "food": 1.0, "exploration": 92.08,
         "collision": 3.09},
        config.fitness_weights, rng=np.random.default_rng(5))

    assert len(new_predators) == 3
    assert len(new_prey) == 6
    assert all(a.role == 'predator' for a in new_predators)
    assert all(a.role == 'prey' for a in new_prey)
    assert metrics.generation == 1
    assert 0 <= metrics.prey_alive_end <= 6
    assert 0 <= metrics.predator_alive_end <= 3


def test_coevolution_requires_predation_config():
    config = load_config(CONFIG_PATH)
    no_predation = config.model_copy(update={"predation": None})
    prey = make_population(6, 'prey')
    predators = make_population(3, 'predator')

    try:
        run_coevolution_generation(
            no_predation, predators, prey, 1,
            {"survival": 1.0, "food": 1.0, "exploration": 1.0,
             "collision": 1.0},
            config.fitness_weights)
        assert False, "missing predation must raise"
    except ValueError:
        pass


def test_coevolution_requires_both_roles_nonempty():
    config = small_config()
    prey = make_population(6, 'prey')

    try:
        run_coevolution_generation(
            config, [], prey, 1,
            {"survival": 1.0, "food": 1.0, "exploration": 1.0,
             "collision": 1.0},
            config.fitness_weights)
        assert False, "empty predator role must raise"
    except ValueError:
        pass


def test_hof_snapshot_flow_across_generations():
    # Two tiny generations: HOF records at the snapshot interval and the
    # duel evaluation runs once the opposing archive is non-empty
    config = small_config()
    config = config.model_copy(update={
        "experiment": config.experiment.model_copy(
            update={"save_interval": 1})})
    prey = make_population(6, 'prey')
    predators = make_population(3, 'predator')
    hof = HallOfFame(snapshot_interval=1)
    rng = np.random.default_rng(5)

    for generation in (1, 2):
        predators, prey, metrics = run_coevolution_generation(
            config, predators, prey, generation,
            {"survival": 149.85, "food": 1.0, "exploration": 92.08,
             "collision": 3.09},
            config.fitness_weights, rng=rng)
        # evolve_genomes orders elites first: population[0] is the best
        hof.maybe_snapshot(generation, 'prey', prey[0].genome,
                           metrics.prey_fitness_max)
        hof.maybe_snapshot(generation, 'predator', predators[0].genome,
                           metrics.predator_fitness_max)

    assert len(hof.entries('prey')) == 2
    assert len(hof.entries('predator')) == 2
