"""
Unit tests for configuration loading via simulation.world_config.
"""

import os
import sys

# Add the project root to the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from simulation.world_config import load_config, BaselineConfig

# Resolve configs/baseline.json relative to this file so the test is
# independent of the working directory pytest is invoked from
BASELINE_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'baseline.json')


def test_load_baseline_config():
    """Deliverable 4: baseline.json validates end-to-end via load_config."""
    config = load_config(BASELINE_CONFIG_PATH)

    assert config.experiment_name == 'baseline'
    assert config.mode in ('gui', 'headless')
    assert config.device in ('cpu', 'cuda', 'auto')
    assert config.reproducibility_tier == 'cpu-deterministic'

    # World section
    assert config.world.width == 1200
    assert config.world.height == 700
    assert config.world.food_count == 20
    assert config.world.obstacle_count == 10

    # Population section
    assert config.population.size == 250
    assert config.population.agent.sensors == 7
    assert config.population.agent.brain.architecture == [12, 32, 16, 3]

    # Evolution section
    assert config.evolution.generations == 1000
    assert config.evolution.mutation_rate == 0.05
    assert config.evolution.crossover_rate == 0.7
    assert config.evolution.elitism_count == 2

    # Experiment section
    assert config.experiment.seeds == 10
    assert config.experiment.save_interval == 50

    # Fitness weights present and sum to 1.0 (validated by the model)
    assert abs(sum(config.fitness_weights.values()) - 1.0) < 1e-6


def test_load_config_into_world():
    """The loaded config drives World construction end-to-end."""
    from simulation.environment import World

    config = load_config(BASELINE_CONFIG_PATH)
    world = World()
    world.load_from_config(config.model_dump())

    assert world.width == config.world.width
    assert world.height == config.world.height
    assert len(world.food) == config.world.food_count
    assert len(world.obstacles) == config.world.obstacle_count


def test_load_config_rejects_invalid_weights():
    """Negative test: fitness weights that do not sum to 1.0 are rejected."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BaselineConfig(
            world={'width': 100, 'height': 100, 'food_count': 1, 'obstacle_count': 1},
            population={'size': 10, 'agent': {'sensors': 7, 'brain': {'architecture': [12, 32, 16, 3]}}},
            evolution={'generations': 10, 'mutation_rate': 0.05, 'crossover_rate': 0.7, 'elitism_count': 2},
            experiment={'seeds': 1, 'save_interval': 1},
            fitness_weights={'survival': 0.5, 'food': 0.3, 'exploration': 0.2, 'collision': 0.5}
        )


def test_load_config_missing_file():
    """Negative test: loading a missing config file raises FileNotFoundError."""
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config(os.path.join(PROJECT_ROOT, 'configs', 'does_not_exist.json'))
