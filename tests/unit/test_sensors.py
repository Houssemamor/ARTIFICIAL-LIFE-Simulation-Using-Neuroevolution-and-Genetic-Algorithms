"""
Unit tests for the fixed observation encoding and ray casting (Phase 2).

Verifies the geometry (ray hit distances) and the 12-dim vector layout
(distance/ bearing/state sections) against hand-constructed scenarios, so a
silent encoding drift breaks a test rather than a later experiment.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents.organism import Organism
from agents.sensors import (
    INITIAL_ENERGY,
    MAX_RANGE,
    OBSERVATION_DIM,
    RAY_ANGLES_DEG,
    RayCaster,
)
from simulation.environment import Food, Obstacle, World
from simulation.physics import Vector2D


def _agent(x: float = 100.0, y: float = 100.0, heading: float = 0.0,
           energy: float = INITIAL_ENERGY, speed: float = 0.0) -> Organism:
    """Build a live agent at a known pose, outside any obstacle the test
    places, with a deterministic identity for sensor ordering."""
    agent = Organism(0, x, y, initial_energy=energy)
    agent.heading = heading
    if speed > 0:
        agent.velocity = Vector2D(math.cos(heading) * speed,
                                  math.sin(heading) * speed)
    return agent


@pytest.fixture
def empty_world() -> World:
    """A world with food and obstacles lists we can populate per test."""
    world = World(1200, 700)
    world.food = []
    world.obstacles = []
    return world


@pytest.fixture
def raycaster() -> RayCaster:
    return RayCaster()


class TestObservationLayout:
    def test_dimension_matches_architecture_input(self) -> None:
        """The 12-dim sensor layout must feed the fixed 12-input controller."""
        from neural.network import input_size
        assert OBSERVATION_DIM == 12
        assert input_size() == OBSERVATION_DIM

    def test_ray_count_is_seven(self) -> None:
        """Baseline ray count is exactly seven per design doc Section 11.2."""
        assert len(RAY_ANGLES_DEG) == 7

    def test_default_observation_length(self, empty_world: World,
                                        raycaster: RayCaster) -> None:
        """No objects anywhere -> all distances saturated to 1.0 and zero state."""
        agent = _agent()
        obs = raycaster.observe_agent(agent, empty_world, [])
        assert obs.shape == (OBSERVATION_DIM,)
        assert np.allclose(obs[:7], 1.0)  # nothing in range
        assert np.allclose(obs[7:10], 0.0)  # no nearest entities -> 0 bearing
        assert np.isclose(obs[10], 1.0)  # full energy
        assert np.isclose(obs[11], 0.0)  # at rest

    def test_energy_and_speed_normalization(self, empty_world: World,
                                            raycaster: RayCaster) -> None:
        agent = _agent(energy=50.0, speed=2.5)
        obs = raycaster.observe_agent(agent, empty_world, [])
        assert np.isclose(obs[10], 0.5)  # 50 / 100
        assert np.isclose(obs[11], 0.5)  # 2.5 / 5.0


class TestRayCasting:
    def test_food_along_forward_ray(self, empty_world: World,
                                    raycaster: RayCaster) -> None:
        """A food dead ahead saturates the 0-degree ray distance only."""
        agent = _agent(heading=0.0)
        empty_world.food = [Food(250.0, 100.0)]  # 150 px ahead
        obs = raycaster.observe_agent(agent, empty_world, [])
        # Forward ray is index 3 (0 deg); indices 0..6 are -90..90 deg
        assert np.isclose(obs[3], 150.0 / MAX_RANGE)
        # Side rays see nothing (rays only; don't include bearing/state)
        assert np.allclose(np.concatenate([obs[:3], obs[4:7]]), 1.0)

    def test_food_right_of_heading_updates_bearing(self, empty_world: World,
                                                   raycaster: RayCaster) -> None:
        """Food at +45 deg relative to heading -> bearing ~ +0.25."""
        agent = _agent(heading=0.0)
        # From (100,100), a food 100 px right and 100 px below => 45 deg
        empty_world.food = [Food(200.0, 200.0)]
        obs = raycaster.observe_agent(agent, empty_world, [])
        assert np.isclose(obs[7], 0.25, atol=0.02)

    def test_food_left_of_heading_negative_bearing(self, empty_world: World,
                                                   raycaster: RayCaster) -> None:
        agent = _agent(heading=0.0)
        # 100 px right and 100 px above => +x, -y = -45 deg relative
        empty_world.food = [Food(200.0, 0.0)]
        obs = raycaster.observe_agent(agent, empty_world, [])
        assert np.isclose(obs[7], -0.25, atol=0.02)

    def test_threat_bearing_uses_other_agent(self, empty_world: World,
                                             raycaster: RayCaster) -> None:
        """A second live agent registers in the threat bearing channel."""
        observer = _agent()
        threat = _agent(x=200.0, y=100.0)  # dead ahead (0 deg) of observer
        obs = raycaster.observe_agent(observer, empty_world, [threat])
        assert np.isclose(obs[8], 0.0, atol=0.01)

    def test_nearest_object_of_multiple(self, empty_world: World,
                                        raycaster: RayCaster) -> None:
        """Closest object in a ray wins over farther ones."""
        agent = _agent(heading=0.0)
        empty_world.food = [Food(300.0, 100.0)]  # 200 px ahead
        empty_world.obstacles = [Obstacle(150.0, 100.0)]  # 50 px ahead
        obs = raycaster.observe_agent(agent, empty_world, [])
        assert np.isclose(obs[3], 50.0 / MAX_RANGE)

    def test_obstacle_beyond_back_ray(self, empty_world: World,
                                      raycaster: RayCaster) -> None:
        """Obstacle behind the agent does not saturate the forward ray."""
        agent = _agent(heading=0.0)
        empty_world.obstacles = [Obstacle(100.0, 300.0)]
        # That obstacle is straight down from the agent: 200 px on the +90
        # (down) ray, so the -90 (up) and 0 (right) rays see nothing.
        obs = raycaster.observe_agent(agent, empty_world, [])
        assert np.isclose(obs[0], 1.0)  # forward ray clear
        assert np.isclose(obs[6], 200.0 / MAX_RANGE)  # down ray hit


class TestPopulationObservation:
    def test_matrix_shape_and_row_alignment(self, empty_world: World,
                                            raycaster: RayCaster) -> None:
        """Population matrix rows align with agent order; forward threat hit."""
        agents = [_agent(), _agent(x=150.0, y=100.0)]  # agent 1 is dead ahead
        obs = raycaster.observe_population(agents, empty_world)
        assert obs.shape == (2, OBSERVATION_DIM)
        # Agent 0's forward ray (index 3, 0 deg) is saturated by the
        # threat 50 px ahead
        assert np.isclose(obs[0, 3], 50.0 / MAX_RANGE)
        # Agent 0's side rays are clear
        assert np.allclose(np.concatenate([obs[0, :3], obs[0, 4:7]]), 1.0)
        # Agent 0's threat bearing points dead ahead (0 deg relative)
        assert np.isclose(obs[0, 8], 0.0, atol=0.01)