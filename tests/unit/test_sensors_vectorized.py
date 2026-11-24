"""
Equivalence gate for the vectorised sensor path.

The scalar RayCaster.observe_agent is kept as the oracle: randomized worlds
must produce the same 12-dim observations through the batched
observe_population path, for both roles, dead observers and edge geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents.organism import Organism
from agents.sensors import RayCaster
from simulation.environment import Food, Obstacle, World
from simulation.physics import Vector2D


def _population(rng: np.random.Generator, count: int) -> list[Organism]:
    """Build agents with mixed roles, headings, velocities and alive flags."""
    agents = []
    for index in range(count):
        role = "predator" if index % 3 == 0 else "prey"
        agent = Organism(index,
                         float(rng.uniform(20.0, 1180.0)),
                         float(rng.uniform(20.0, 680.0)),
                         initial_energy=float(rng.uniform(10.0, 200.0)),
                         role=role)
        agent.heading = float(rng.uniform(-math.pi, math.pi))
        speed = float(rng.uniform(0.0, 8.0))
        agent.velocity = Vector2D(math.cos(agent.heading) * speed,
                                  math.sin(agent.heading) * speed)
        if index % 7 == 0:
            agent.is_alive = False
        agents.append(agent)
    return agents


def _world(rng: np.random.Generator) -> World:
    """Random food and obstacles, all inside the arena."""
    world = World(1200, 700)
    world.food = [Food(float(x), float(y)) for x, y in
                  rng.uniform(20.0, 1180.0, (20, 2))]
    world.obstacles = [Obstacle(float(x), float(y)) for x, y in
                       rng.uniform(40.0, 1160.0, (10, 2))]
    return world


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_vectorized_matches_scalar_oracle(seed: int) -> None:
    """Randomized differential test against the unchanged scalar path."""
    rng = np.random.default_rng(seed)
    agents = _population(rng, 25)
    world = _world(rng)
    live = [agent for agent in agents if agent.is_alive]

    expected = np.stack(
        [RayCaster().observe_agent(agent, world, live) for agent in agents])
    actual = RayCaster().observe_population(agents, world)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_edge_geometry_matches_scalar_oracle() -> None:
    """Parallel rays, obstacle entry at zero and overlapping targets."""
    world = World(1200, 700)
    world.food = [Food(200.0, 150.0), Food(200.0, 150.0)]
    world.obstacles = [Obstacle(180.0, 150.0), Obstacle(400.0, 150.0)]
    agents = [
        Organism(0, 100.0, 150.0, role="prey"),
        Organism(1, 180.0, 150.0, role="predator"),
        Organism(2, 100.0, 150.0, role="prey"),
    ]
    for agent in agents:
        agent.heading = 0.0
    agents[2].is_alive = False
    live = [agent for agent in agents if agent.is_alive]

    expected = np.stack(
        [RayCaster().observe_agent(agent, world, live) for agent in agents])
    actual = RayCaster().observe_population(agents, world)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_empty_world_and_population() -> None:
    """Degenerate inputs must keep shape and saturated distances."""
    world = World(1200, 700)
    world.food = []
    world.obstacles = []
    assert RayCaster().observe_population([], world).shape == (0, 12)

    agents = [Organism(0, 100.0, 100.0)]
    observations = RayCaster().observe_population(agents, world)
    assert observations.shape == (1, 12)
    assert np.all(observations[0, 0:7] == 1.0)
    assert np.all(observations[0, 7:10] == 0.0)
