"""
Integration test for environment reset.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import World
from simulation.world_config import load_config


def test_reset_environment():
    """Reset the environment N times and verify entity counts match config."""
    # Load a configuration
    config = load_config('configs/baseline.json')
    world_config = config.world

    world = World()
    reset_count = 10

    for i in range(reset_count):
        # Reset the world by loading the config again
        world.load_from_config(config.dict())

        # Verify world dimensions
        assert world.width == world_config.width
        assert world.height == world_config.height

        # Verify food count
        assert len(world.food) == world_config.food_count, \
            f"Reset {i}: expected {world_config.food_count} food, got {len(world.food)}"

        # Verify obstacle count
        assert len(world.obstacles) == world_config.obstacle_count, \
            f"Reset {i}: expected {world_config.obstacle_count} obstacles, got {len(world.obstacles)}"

        # Optional: verify that positions are within bounds
        margin = world.boundary_margin
        for food in world.food:
            assert margin <= food.x <= world.width - margin, \
                f"Food out of bounds: ({food.x}, {food.y})"
            assert margin <= food.y <= world.height - margin, \
                f"Food out of bounds: ({food.x}, {food.y})"
        for obstacle in world.obstacles:
            assert margin <= obstacle.x <= world.width - margin, \
                f"Obstacle out of bounds: ({obstacle.x}, {obstacle.y})"
            assert margin <= obstacle.y <= world.height - margin, \
                f"Obstacle out of bounds: ({obstacle.x}, {obstacle.y})"


def test_reset_with_different_configs():
    """Test reset with different configurations."""
    # Test with a small config
    small_config = {
        'world': {
            'width': 100,
            'height': 100,
            'food_count': 1,
            'obstacle_count': 1
        }
    }
    world = World()
    for _ in range(5):
        world.load_from_config(small_config)
        assert len(world.food) == 1
        assert len(world.obstacles) == 1
        assert world.width == 100
        assert world.height == 100

    # Test with a larger config
    large_config = {
        'world': {
            'width': 2000,
            'height': 1500,
            'food_count': 50,
            'obstacle_count': 20
        }
    }
    for _ in range(5):
        world.load_from_config(large_config)
        assert len(world.food) == 50
        assert len(world.obstacles) == 20
        assert world.width == 2000
        assert world.height == 1500