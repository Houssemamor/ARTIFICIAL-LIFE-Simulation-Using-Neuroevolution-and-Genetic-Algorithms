"""Unit tests for Phase 6 in-episode reproduction."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agents.organism import Organism
from simulation.environment import World
from simulation.world_config import ReproductionConfig
from evolution.reproduction import (is_eligible, create_offspring,
                                     reproduction_pass)


def make_agent(role='prey', age=100, energy=80.0, fitness=0.3):
    agent = Organism(0, 100.0, 100.0, role=role)
    agent.age = age
    agent.energy = energy
    agent.fitness = fitness
    agent.genome = np.zeros(10, dtype=np.float32)
    return agent


def test_eligibility_requires_all_three_conditions():
    config = ReproductionConfig(min_age=40, min_energy=60.0, min_fitness=0.15)

    assert is_eligible(make_agent(), config)

    too_young = make_agent(age=39)
    assert not is_eligible(too_young, config)

    too_hungry = make_agent(energy=59.9)
    assert not is_eligible(too_hungry, config)

    too_unfit = make_agent(fitness=0.14)
    assert not is_eligible(too_unfit, config)

    dead = make_agent()
    dead.is_alive = False
    assert not is_eligible(dead, config)


def test_offspring_inherits_role_and_mutated_genome():
    world = World(400, 300)
    config = ReproductionConfig(mutation_rate=0.0)
    parent = make_agent(role='predator')

    offspring = create_offspring(parent, 7, world, config,
                                 np.random.default_rng(1))

    assert offspring.role == 'predator'
    assert offspring.id == 7
    # Zero mutation rate: exact clone
    assert np.array_equal(offspring.genome, parent.genome)


def test_offspring_genome_respects_clip_bounds():
    world = World(400, 300)
    config = ReproductionConfig(mutation_rate=1.0, mutation_strength=8.0)
    parent = make_agent()

    offspring = create_offspring(parent, 1, world, config,
                                 np.random.default_rng(2))

    assert np.all(offspring.genome >= -1.0)
    assert np.all(offspring.genome <= 1.0)


def test_birth_costs_parent_energy():
    world = World(400, 300)
    config = ReproductionConfig(reproduction_cost=50.0)
    parent = make_agent(energy=90.0)

    create_offspring(parent, 1, world, config, np.random.default_rng(3))

    assert parent.energy == 40.0


def test_reproduction_pass_disabled_is_noop():
    world = World(400, 300)
    config = ReproductionConfig(enabled=False)
    agents = [make_agent()]

    newborns, next_id = reproduction_pass(agents, world, config, 100)

    assert newborns == []
    assert next_id == 100


def test_cap_counts_live_agents_only():
    # Regression test: the cap once counted corpses, permanently blocking
    # births in long ecosystem runs once enough agents had died
    world = World(400, 300)
    config = ReproductionConfig(enabled=True, max_population_prey=4,
                                max_population_predator=2)
    agents = []
    for i in range(4):
        agent = make_agent()
        agent.id = i
        agents.append(agent)
    # One dead corpse must not count toward the cap
    agents[3].is_alive = False

    newborns, next_id = reproduction_pass(agents, world, config, 50,
                                          np.random.default_rng(4))

    # 3 live prey < cap 4, so the eligible ones birth
    assert len(newborns) >= 1
    assert all(n.role == 'prey' for n in newborns)


def test_cap_blocks_births_at_live_capacity():
    world = World(400, 300)
    config = ReproductionConfig(enabled=True, max_population_prey=4,
                                max_population_predator=2)
    agents = []
    for i in range(4):
        agent = make_agent()
        agent.id = i
        agents.append(agent)

    newborns, _ = reproduction_pass(agents, world, config, 50,
                                    np.random.default_rng(5))

    assert newborns == []


def test_config_validation_rejects_negative_thresholds():
    import pytest
    with pytest.raises(Exception):
        ReproductionConfig(min_energy=-1.0)
    with pytest.raises(Exception):
        ReproductionConfig(min_age=-5)
