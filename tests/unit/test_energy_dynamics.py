"""
Unit tests for the energy-limited dynamics (Phase 9).

The energy equation's per-config overrides and the live exploration
component are the two mechanics that replaced the survival-saturated
landscape; both are covered here, including a regression that the
experiment pipeline actually applies them (it once passed
DEFAULT_ENERGY_CONFIG explicitly, silently overriding config.energy).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agents.energy import (DEFAULT_ENERGY_CONFIG, EnergyConfig,
                           energy_config_from_settings)
from simulation.world_config import EnergySettings, load_config


def test_settings_none_yields_defaults():
    assert energy_config_from_settings(None) is DEFAULT_ENERGY_CONFIG


def test_partial_override_changes_only_that_field():
    settings = EnergySettings(metabolic_base_cost=0.7)
    resolved = energy_config_from_settings(settings)
    assert resolved.metabolic_base_cost == 0.7
    assert resolved.k_accel == DEFAULT_ENERGY_CONFIG.k_accel
    assert resolved.eaten_energy_value == DEFAULT_ENERGY_CONFIG.eaten_energy_value


def test_override_does_not_mutate_the_default():
    before = DEFAULT_ENERGY_CONFIG.metabolic_base_cost
    energy_config_from_settings(EnergySettings(metabolic_base_cost=5.0))
    assert DEFAULT_ENERGY_CONFIG.metabolic_base_cost == before


def test_energy_settings_validation():
    with pytest.raises(Exception):
        EnergySettings(metabolic_base_cost=-1.0)
    with pytest.raises(Exception):
        EnergySettings(max_energy=0.0)


def test_experiment_configs_carry_the_energy_limited_dynamics():
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    flat = load_config(os.path.join(root, "configs",
                                    "neat_food_seeking.json"))
    assert flat.energy is not None
    assert flat.energy.metabolic_base_cost == 0.7
    assert flat.world.food_regrowth_per_step > 0


def test_organism_uses_the_supplied_energy_config():
    from agents.organism import Organism
    custom = EnergyConfig(metabolic_base_cost=0.7)
    agent = Organism(0, 10.0, 10.0, energy_config=custom)
    assert agent._energy_config is custom


def test_higher_metabolic_cost_actually_starsves_agents():
    from agents.organism import Organism
    from simulation.environment import World
    hungry = EnergyConfig(metabolic_base_cost=0.7)
    agent = Organism(0, 50.0, 50.0, energy_config=hungry)
    world = World(200, 200)
    world.load_from_config({"world": {"width": 200, "height": 200,
                                      "food_count": 0,
                                      "obstacle_count": 0}})
    for _ in range(150):
        agent.update_energy()
        if not agent.is_alive:
            break
    assert not agent.is_alive, "0.7/step must starve within 150 steps"


def test_exploration_component_is_live_in_run_generation():
    from simulation.world_config import BaselineConfig
    from agents.organism import Organism
    from evolution.genetic_algorithm import run_generation
    from neural.genome import genome_size
    config = BaselineConfig(
        world={"width": 300, "height": 300, "food_count": 10,
               "obstacle_count": 0},
        population={"size": 6, "agent": {"sensors": 7,
                    "brain": {"architecture": [12, 3]}}},
        evolution={"evaluation_steps": 40}, experiment={})
    agents = []
    for index in range(6):
        agent = Organism(index, 50.0 + index, 50.0)
        agent.genome = np.random.default_rng(index).uniform(
            -1, 1, genome_size()).astype(np.float32)
        agents.append(agent)
    _, metrics = run_generation(
        config, agents, generation=0,
        calibration_scales={"survival": 100.0, "food": 1.0,
                            "exploration": 50.0, "collision": 2.0},
        fitness_weights=config.fitness_weights,
        rng=np.random.default_rng(1))
    # Exploration was dead (always 0) until the Phase 9 fix; random
    # controllers moving for 40 steps must register non-zero displacement
    assert metrics.mean_exploration > 0.0


def test_run_generation_regrows_food_each_step(monkeypatch):
    """Regression: the hand-rolled solo evaluation loop owns its own
    stepping, so it must call world.regrow_food() explicitly - the
    configured regrowth was silently inert without it."""
    from simulation.world_config import BaselineConfig
    from simulation.environment import World
    from agents.organism import Organism
    from evolution.genetic_algorithm import run_generation
    from neural.genome import genome_size
    calls = []
    original_regrow = World.regrow_food

    def counting_regrow(self, rate):
        calls.append(1)
        original_regrow(self, rate)

    monkeypatch.setattr(World, "regrow_food", counting_regrow)
    config = BaselineConfig(
        world={"width": 300, "height": 300, "food_count": 2,
               "obstacle_count": 0, "food_regrowth_per_step": 0.3},
        population={"size": 4, "agent": {"sensors": 7,
                    "brain": {"architecture": [12, 3]}}},
        evolution={"evaluation_steps": 30}, experiment={})
    agents = []
    for index in range(4):
        # Spread spawns: overlapping agents collide every step and the
        # collision penalty kills them before the full horizon.
        agent = Organism(index, 20.0 + index * 50, 20.0 + index * 50)
        agent.genome = np.zeros(genome_size(), dtype=np.float32)
        agents.append(agent)
    run_generation(
        config, agents, generation=0,
        calibration_scales={"survival": 100.0, "food": 1.0,
                            "exploration": 50.0, "collision": 2.0},
        fitness_weights=config.fitness_weights,
        rng=np.random.default_rng(1))
    assert len(calls) == 30, "one regrow call per evaluation step"


def test_newborns_inherit_the_parent_energy_config():
    from agents.organism import Organism
    from evolution.reproduction import create_offspring
    from simulation.environment import World
    from simulation.world_config import ReproductionConfig
    parent = Organism(0, 50.0, 50.0, role='predator',
                      energy_config=EnergyConfig(metabolic_base_cost=0.7))
    from neural.genome import genome_size
    parent.initialize_genome(genome_size())
    world = World(200, 200)
    world.load_from_config({"world": {"width": 200, "height": 200,
                                      "food_count": 0,
                                      "obstacle_count": 0}})
    offspring = create_offspring(parent, 1, world,
                                 ReproductionConfig(enabled=True),
                                 rng=np.random.default_rng(0))
    assert offspring.energy_config.metabolic_base_cost == 0.7
    assert offspring.role == 'predator'


def test_offspring_energy_config_accessor_is_read_only():
    from agents.organism import Organism
    agent = Organism(0, 10.0, 10.0)
    with pytest.raises(AttributeError):
        agent.energy_config = EnergyConfig()
