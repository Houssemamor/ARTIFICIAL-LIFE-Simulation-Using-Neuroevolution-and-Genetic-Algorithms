"""
Environment module for Artificial Life Neuroevolution Simulation.

Contains the World, Food, and Obstacle classes.
"""

import numpy as np
from typing import List, Optional


class Food:
    """Represents a food item in the world."""

    def __init__(self, x: float, y: float):
        """
        Initialize a food item.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
        """
        self.x = float(x)
        self.y = float(y)


class Obstacle:
    """Represents an obstacle in the world."""

    def __init__(self, x: float, y: float):
        """
        Initialize an obstacle.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
        """
        self.x = float(x)
        self.y = float(y)


class World:
    """
    Represents the 2D continuous world where agents live and interact.

    Attributes:
        width (int): Width of the world in pixels
        height (int): Height of the world in pixels
        food (List[Food]): List of food items
        obstacles (List[Obstacle]): List of obstacles
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
        self.food: List[Food] = []
        self.obstacles: List[Obstacle] = []
        self.boundary_margin = 10.0  # pixels from edge that agents cannot cross

        # Food regrowth state (Phase 6); target stays None until
        # load_from_config sets it, so directly-constructed worlds
        # keep the legacy finite-food behavior
        self.food_target: Optional[int] = None
        self.food_regrowth_per_step: float = 0.0
        self._food_regrowth_pending: float = 0.0

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

        # Regrowth target: regrow_food() replenishes consumed items
        # back toward this count (Phase 6 ecosystem mode)
        self.food_target = food_count
        self.food_regrowth_per_step = world_config.get('food_regrowth_per_step', 0.0)
        self._food_regrowth_pending = 0.0

        self.spawn_food(food_count)
        self.spawn_obstacles(obstacle_count)

    def regrow_food(self, rate: float) -> int:
        """
        Replenish consumed food items up to the configured target.

        Called once per simulation step by the engine; `rate` is items
        per step, accumulated fractionally so slow regrowth (e.g. 0.3)
        still produces steady growth. New items spawn at uniformly
        random positions, like the initial layout.

        Args:
            rate (float): Food items regrown per step.

        Returns:
            int: Number of items actually spawned this step.
        """
        # Legacy worlds (constructed directly, not via load_from_config)
        # have no target and never regrow
        target = getattr(self, 'food_target', None)
        if target is None or rate <= 0.0:
            return 0

        self._food_regrowth_pending += rate
        spawned = 0
        while (self._food_regrowth_pending >= 1.0
               and len(self.food) < target):
            self._food_regrowth_pending -= 1.0
            x = np.random.uniform(self.boundary_margin,
                                  self.width - self.boundary_margin)
            y = np.random.uniform(self.boundary_margin,
                                  self.height - self.boundary_margin)
            self.food.append(Food(float(x), float(y)))
            spawned += 1
        # Do not bank excess accumulation once the target is reached
        if len(self.food) >= target:
            self._food_regrowth_pending = 0.0
        return spawned

    def spawn_food(self, count: int) -> None:
        """
        Spawn food at random positions in the world.

        Args:
            count (int): Number of food items to spawn
        """
        self.food = []
        for _ in range(count):
            x = np.random.uniform(self.boundary_margin,
                                  self.width - self.boundary_margin)
            y = np.random.uniform(self.boundary_margin,
                                  self.height - self.boundary_margin)
            self.food.append(Food(float(x), float(y)))

    def spawn_obstacles(self, count: int) -> None:
        """
        Spawn obstacles at random positions in the world.

        Args:
            count (int): Number of obstacles to spawn
        """
        self.obstacles = []
        for _ in range(count):
            x = np.random.uniform(self.boundary_margin,
                                  self.width - self.boundary_margin)
            y = np.random.uniform(self.boundary_margin,
                                  self.height - self.boundary_margin)
            self.obstacles.append(Obstacle(float(x), float(y)))

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
        if not self.food:
            return float('inf')

        min_dist = float('inf')
        for food in self.food:
            dist = np.sqrt((x - food.x)**2 + (y - food.y)**2)
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
        if not self.obstacles:
            return float('inf')

        min_dist = float('inf')
        for obstacle in self.obstacles:
            dist = np.sqrt((x - obstacle.x)**2 + (y - obstacle.y)**2)
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
            'food': [(f.x, f.y) for f in self.food],
            'obstacles': [(o.x, o.y) for o in self.obstacles],
            'boundary_margin': self.boundary_margin,
            'food_target': self.food_target,
            'food_regrowth_per_step': self.food_regrowth_per_step
        }

    def load_state(self, state: dict) -> None:
        """
        Load world state from a dictionary.

        Args:
            state (dict): Dictionary containing world state
        """
        self.width = state['width']
        self.height = state['height']
        self.food = [Food(x, y) for x, y in state['food']]
        self.obstacles = [Obstacle(x, y) for x, y in state['obstacles']]
        self.boundary_margin = state['boundary_margin']
        # Regrowth state round-trips too; older serialized states without
        # these keys fall back to the legacy no-regrowth behavior
        self.food_target = state.get('food_target')
        self.food_regrowth_per_step = state.get('food_regrowth_per_step', 0.0)
        self._food_regrowth_pending = 0.0