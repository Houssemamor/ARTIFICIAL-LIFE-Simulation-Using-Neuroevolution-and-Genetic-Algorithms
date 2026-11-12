"""
Unit tests for solid obstacle collision resolution.

The bug these tests lock down: the sensors treated an obstacle as a
small circle while the collision engine treated it as a square and only
charged energy for contact, so agents could see a wall and still drive
through it. These tests assert that the body the sensors see is the body
the physics engine stops at.
"""

from __future__ import annotations

import math

from agents.organism import Organism
from simulation.engine import resolve_collisions
from simulation.environment import Obstacle, World
from simulation.geometry import (AGENT_HALF_SIZE, OBSTACLE_HALF_SIZE,
                                 OBSTACLE_SIZE)


def _world_with_obstacle() -> World:
    world = World(400, 300)
    world.food = []
    world.obstacles = [Obstacle(200.0, 150.0)]
    return world


def _expanded_reach() -> float:
    """Centre distance at which the agent body touches the obstacle."""
    return OBSTACLE_HALF_SIZE + AGENT_HALF_SIZE


def test_obstacle_body_is_the_shared_square():
    """One constant drives sensors, physics, renderer and replay."""
    assert OBSTACLE_SIZE == 40.0
    assert OBSTACLE_HALF_SIZE == OBSTACLE_SIZE / 2.0


def test_agent_inside_obstacle_is_pushed_out():
    world = _world_with_obstacle()
    agent = Organism(0, 200.0, 150.0)

    resolve_collisions([agent], world)

    dx = abs(agent.position.x - 200.0)
    dy = abs(agent.position.y - 150.0)
    assert max(dx, dy) >= _expanded_reach()
    assert agent.collisions >= 1


def test_agent_cannot_drive_through_obstacle():
    """Repeated forward thrust must not tunnel through the square."""
    world = _world_with_obstacle()
    agent = Organism(0, 200.0, 100.0)
    agent.heading = math.pi / 2.0  # straight down toward the obstacle

    for _ in range(30):
        # Full throttle, no steering: the only thing that can stop the
        # agent is the solid body itself.
        agent.apply_action(0.0, 1.0, 0.0, world, dt=1.0)
        resolve_collisions([agent], world)

    # The obstacle's top face is at y = 150 - 20 = 130; the agent body
    # half is 4, so the deepest legal centre is 126.
    assert agent.position.y <= 150.0 - _expanded_reach() + 1e-6
    assert agent.collisions >= 1


def test_agent_clear_of_obstacle_is_not_moved():
    world = _world_with_obstacle()
    agent = Organism(0, 50.0, 50.0)
    before = (agent.position.x, agent.position.y)

    resolve_collisions([agent], world)

    assert (agent.position.x, agent.position.y) == before
    assert agent.collisions == 0


def test_obstacle_contact_keeps_energy_penalty():
    """Solid resolution must not remove the configured contact cost."""
    world = _world_with_obstacle()
    agent = Organism(0, 200.0, 150.0)

    resolve_collisions([agent], world)

    assert agent._collision_penalty_this_step > 0.0
