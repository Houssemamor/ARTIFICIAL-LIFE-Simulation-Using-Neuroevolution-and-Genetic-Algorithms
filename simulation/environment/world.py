"""
World module for Artificial Life Neuroevolution Simulation.

Contains the World class that manages the 2D continuous space,
food, obstacles, boundaries, and agent spawning/resetting.
"""

import numpy as np
from typing import List, Tuple, Optional
import json


class World:
    """
    Represents the 2D continuous world where agents live and interact.

    Attributes:
        width (int): Width of the world in pixels
        height (int): Height of the world in pixels
        food_positions (List[Tuple[float, float]]): List of food positions
        obstacle_positions (List[Tuple[float, float]]): List of obstacle positions
        boundary_margin (float): Margin from edges where agents cannot go
    """

    def __init__(self, width: int = 1200, height: int = 700):
        """
        Initialize the world with given dimensions.

        Args:
            width (int): Width of the world in pixels (default: 1200)
            height (int): Height of the world in pixels (default: 700)
        """
        self.width = width
        self.height = height
        self.food_positions: List[Tuple[float, float]] = []
        self.obstacle_positions: List[Tuple[float, float]] = []
        self.boundary_margin = 10.0  # pixels from edge that agents cannot cross

    def load_from_config(self, config: dict) -> None:
        """
        Load world configuration from a dictionary.

        Args:
            config (dict): Configuration dictionary containing world settings
        """
        world_config = config.get('world', {})
        self.width = world_config.get('width', self.width)
        self.height = world_config.get('height', self.height)

        # Initialize food and obstacles based on config
        food_count = world_config.get('food_count', 20)
        obstacle_count = world_config.get('obstacle_count', 10)

        self.spawn_food(food_count)
        self.spawn_obstacles(obstacle_count)

    def spawn_food(self, count: int) -> None:
        """
        Spawn food at random positions in the world.

        Args:
            count (int): Number of food items to spawn
        """
        self.food_positions = []
        for _ in range(count):
            x = np.random.uniform(self.boundary_margin,
                                self.width - self.boundary_margin)
            y = np.random.uniform(self.boundary_margin,
                                self.height - self.boundary_margin)
            self.food_positions.append((float(x), float(y)))

    def spawn_obstacles(self, count: int) -> None:
        """
        Spawn obstacles at random positions in the world.

        Args:
            count (int): Number of obstacles to spawn
        """
        self.obstacle_positions = []
        for _ in range(count):
            x = np.random.uniform(self.boundary_margin,
                                self.width - self.boundary_margin)
            y = np.random.uniform(self.boundary_margin,
                                self.height - self.boundary_margin)
            self.obstacle_positions.append((float(x), float(y)))

    def is_inside_boundaries(self, x: float, y: float) -> bool:
        """
        Check if a position is inside the world boundaries (with margin).

        Args:
            x (float): X coordinate
            y (float): Y coordinate

        Returns:
            bool: True if position is inside boundaries, False otherwise
        """
        return (self.boundary_margin <= x <= self.width - self.boundary_margin and
                self.boundary_margin <= y <= self.height - self.boundary_margin)

    def distance_to_nearest_food(self, x: float, y: float) -> float:
        """
        Calculate distance to the nearest food item.

        Args:
            x (float): X coordinate
            y (float): Y coordinate

        Returns:
            float: Distance to nearest food, or infinity if no food exists
        """
        if not self.food_positions:
            return float('inf')

        min_dist = float('inf')
        for food_x, food_y in self.food_positions:
            dist = np.sqrt((x - food_x)**2 + (y - food_y)**2)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def distance_to_nearest_obstacle(self, x: float, y: float) -> float:
        """
        Calculate distance to the nearest obstacle.

        Args:
            x (float): X coordinate
            y (float): Y coordinate

        Returns:
            float: Distance to nearest obstacle, or infinity if no obstacles exist
        """
        if not self.obstacle_positions:
            return float('inf')

        min_dist = float('inf')
        for obs_x, obs_y in self.obstacle_positions:
            dist = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def get_state(self) -> dict:
        """
        Get the current state of the world for serialization.

        Returns:
            dict: Dictionary containing world state
        """
        return {
            'width': self.width,
            'height': self.height,
            'food_positions': self.food_positions,
            'obstacle_positions': self.obstacle_positions,
            'boundary_margin': self.boundary_margin
        }

    def load_state(self, state: dict) -> None:
        """
        Load world state from a dictionary.

        Args:
            state (dict): Dictionary containing world state
        """
        self.width = state['width']
        self.height = state['height']
        self.food_positions = state['food_positions']
        self.obstacle_positions = state['obstacle_positions']
        self.boundary_margin = state['boundary_margin']