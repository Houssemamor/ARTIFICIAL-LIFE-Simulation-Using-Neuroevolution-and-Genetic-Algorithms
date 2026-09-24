"""
Unit tests for environment module.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import World, Food, Obstacle


def test_world_creation():
    """Test world creation with default parameters."""
    world = World()
    assert world.width == 1200
    assert world.height == 700
    assert len(world.food) == 0
    assert len(world.obstacles) == 0
    assert world.boundary_margin == 10.0


def test_world_load_from_config():
    """Test loading world configuration from a dictionary."""
    world = World()
    config = {
        'world': {
            'width': 800,
            'height': 600,
            'food_count': 5,
            'obstacle_count': 2
        }
    }
    world.load_from_config(config)
    assert world.width == 800
    assert world.height == 600
    assert len(world.food) == 5
    assert len(world.obstacles) == 2

    # Check that food and obstacles are within bounds
    margin = world.boundary_margin
    for food in world.food:
        assert margin <= food.x <= world.width - margin
        assert margin <= food.y <= world.height - margin
    for obstacle in world.obstacles:
        assert margin <= obstacle.x <= world.width - margin
        assert margin <= obstacle.y <= world.height - margin


def test_spawn_food_and_obstacles():
    """Test spawning food and obstacles."""
    world = World(width=200, height=200)  # Small world for easier testing
    world.spawn_food(10)
    world.spawn_obstacles(5)
    assert len(world.food) == 10
    assert len(world.obstacles) == 5

    # Check positions are within bounds
    margin = world.boundary_margin
    for food in world.food:
        assert margin <= food.x <= world.width - margin
        assert margin <= food.y <= world.height - margin
    for obstacle in world.obstacles:
        assert margin <= obstacle.x <= world.width - margin
        assert margin <= obstacle.y <= world.height - margin


def test_is_inside_boundaries():
    """Test boundary checking."""
    world = World(width=100, height=100)
    assert world.is_inside_boundaries(50, 50) == True
    assert world.is_inside_boundaries(0, 50) == False  # Outside margin
    assert world.is_inside_boundaries(100, 50) == False
    assert world.is_inside_boundaries(50, 0) == False
    assert world.is_inside_boundaries(50, 100) == False

    # Test with margin
    margin = world.boundary_margin
    assert world.is_inside_boundaries(margin, margin) == True
    assert world.is_inside_boundaries(world.width - margin, world.height - margin) == True
    assert world.is_inside_boundaries(margin - 1, margin) == False
    assert world.is_inside_boundaries(world.width - margin + 1, world.height - margin) == False


def test_distance_to_nearest_food():
    """Test distance to nearest food."""
    world = World()
    # No food
    assert world.distance_to_nearest_food(0, 0) == float('inf')

    # Add one food
    world.food = [Food(10.0, 10.0)]
    assert world.distance_to_nearest_food(0, 0) == 10.0 * 2**0.5  # sqrt(200)
    assert world.distance_to_nearest_food(10, 10) == 0.0

    # Add multiple foods
    world.food.append(Food(20.0, 20.0))
    assert world.distance_to_nearest_food(0, 0) == 10.0 * 2**0.5  # Still the first is closer
    assert world.distance_to_nearest_food(15, 15) == 5.0 * 2**0.5  # Distance to (10,10) is sqrt(50)~7.07, to (20,20) is sqrt(50)~7.07 -> actually both same? Let's compute: (15-10)^2+(15-10)^2=50, sqrt=7.07; (15-20)^2+(15-20)^2=50, sqrt=7.07. So equal.
    # We'll just test that it's not infinity and is a reasonable value.
    assert world.distance_to_nearest_food(15, 15) > 0


def test_distance_to_nearest_obstacle():
    """Test distance to nearest obstacle."""
    world = World()
    # No obstacles
    assert world.distance_to_nearest_obstacle(0, 0) == float('inf')

    # Add one obstacle
    world.obstacles = [Obstacle(10.0, 10.0)]
    assert world.distance_to_nearest_obstacle(0, 0) == 10.0 * 2**0.5
    assert world.distance_to_nearest_obstacle(10, 10) == 0.0

    # Add multiple obstacles
    world.obstacles.append(Obstacle(20.0, 20.0))
    assert world.distance_to_nearest_obstacle(0, 0) == 10.0 * 2**0.5
    assert world.distance_to_nearest_obstacle(15, 15) > 0


def test_world_state_serialization():
    """Test getting and loading world state."""
    world = World(width=100, height=100)
    world.spawn_food(2)
    world.spawn_obstacles(2)

    state = world.get_state()
    assert state['width'] == 100
    assert state['height'] == 100
    assert len(state['food']) == 2
    assert len(state['obstacles']) == 2
    assert state['boundary_margin'] == 10.0

    # Create a new world and load the state
    new_world = World()
    new_world.load_state(state)
    assert new_world.width == 100
    assert new_world.height == 100
    assert len(new_world.food) == 2
    assert len(new_world.obstacles) == 2
    assert new_world.boundary_margin == 10.0

    # Check that the positions match
    for i in range(2):
        assert new_world.food[i].x == world.food[i].x
        assert new_world.food[i].y == world.food[i].y
        assert new_world.obstacles[i].x == world.obstacles[i].x
        assert new_world.obstacles[i].y == world.obstacles[i].y