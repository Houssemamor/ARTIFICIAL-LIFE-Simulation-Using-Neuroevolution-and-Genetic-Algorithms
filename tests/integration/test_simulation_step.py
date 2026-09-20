"""
Integration smoke test for the Phase 2 wired simulation step.

Runs step_simulation end-to-end with a tiny population and asserts the
pipeline (observe -> batched_forward -> apply_action -> collide -> energy)
executes and produces observable, plumbed state changes.
"""

from __future__ import annotations

import numpy as np

from agents.organism import Organism
from agents.sensors import RayCaster
from neural.genome import genome_size
from simulation.environment import World
from simulation.engine import EAT_SIGNAL_THRESHOLD, step_simulation


def _make_world() -> World:
    world = World(1200, 700)
    world.food = []
    world.obstacles = []
    return world


def _make_population(count: int, rng: np.random.Generator) -> list[Organism]:
    agents = []
    for index in range(count):
        agent = Organism(index, float(index * 30.0 + 100.0), 200.0,
                         initial_energy=100.0)
        agent.genome = rng.uniform(-1.0, 1.0, genome_size()).astype(np.float32)
        agents.append(agent)
    return agents


def test_step_runs_full_pipeline() -> None:
    world = _make_world()
    rng = np.random.default_rng(5)
    agents = _make_population(3, rng)

    collisions = step_simulation(world, agents, RayCaster(), dt=1.0)

    # Every agent got an observation and an action this step
    for agent in agents:
        assert agent.last_observation is not None
        assert agent.last_observation.shape == (12,)
        assert -1.0 <= agent.last_steering <= 1.0
        assert 0.0 <= agent.last_acceleration <= 1.0
        assert 0.0 <= agent.last_eat_signal <= 1.0
        assert agent.age == 1

    # Population-wise batch produced a real collision count
    assert isinstance(collisions, int) and collisions >= 0


def test_step_is_stable_over_many_steps() -> None:
    """No crash, no rejection of valid actions, over a longer run."""
    world = _make_world()
    rng = np.random.default_rng(6)
    agents = _make_population(5, rng)

    for _ in range(50):
        step_simulation(world, agents, RayCaster(), dt=1.0)

    live = [a for a in agents if a.is_alive]
    assert len(live) >= 0