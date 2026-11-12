"""
Environment module for Artificial Life Neuroevolution Simulation.

Contains the World, Food, and Obstacle classes.
"""

import numpy as np
from typing import List, Optional

from simulation.geometry import (
    FOOD_SPAWN_CLEARANCE,
    OBSTACLE_HALF_SIZE,
    OBSTACLE_SIZE,
)

# Edge-to-edge gap kept between spawned obstacles. Without it the larger
# solid bodies can touch and leave no corridor for an 8 px agent.
OBSTACLE_MIN_GAP: float = 12.0

# Rejection-sampling budget for obstacle placement. The configured
# worlds use a few percent of their area, so this is generous; the
# fallback keeps a requested count exact even in an over-packed world.
OBSTACLE_SPAWN_ATTEMPTS: int = 200

# Rejection-sampling budget for free positions (food and agents).
FREE_POSITION_ATTEMPTS: int = 200


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

        # Obstacles first: food placement rejects positions inside them,
        # so the solids must exist before the food pass runs.
        self.spawn_obstacles(obstacle_count)
        self.spawn_food(food_count)

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
            x, y = self.random_free_position(
                clearance=FOOD_SPAWN_CLEARANCE)
            self.food.append(Food(x, y))
            spawned += 1
        # Do not bank excess accumulation once the target is reached
        if len(self.food) >= target:
            self._food_regrowth_pending = 0.0
        return spawned

    def spawn_food(self, count: int) -> None:
        """
        Spawn food at random positions clear of every obstacle.

        Food inside a solid obstacle would be permanently unreachable and
        would consume a slot in the regrowth target.

        Args:
            count (int): Number of food items to spawn
        """
        self.food = []
        for _ in range(count):
            x, y = self.random_free_position(
                clearance=FOOD_SPAWN_CLEARANCE)
            self.food.append(Food(x, y))

    def spawn_obstacles(self, count: int) -> None:
        """
        Spawn obstacles at random positions in the world.

        Each 40 px square is kept fully inside the boundary margin and
        separated from the others by OBSTACLE_MIN_GAP so agents always
        have a corridor between two solids. The count is exact: if the
        world is too packed to satisfy the gap after the sampling
        budget, the last candidate is used anyway.

        Args:
            count (int): Number of obstacles to spawn
        """
        self.obstacles = []
        half = OBSTACLE_HALF_SIZE
        margin = self.boundary_margin
        # Centre range that keeps the whole square inside the margin.
        # Degenerate worlds smaller than the obstacle collapse to the
        # centre instead of raising from uniform(low > high).
        low_x = min(margin + half, self.width / 2.0)
        high_x = max(self.width - margin - half, self.width / 2.0)
        low_y = min(margin + half, self.height / 2.0)
        high_y = max(self.height - margin - half, self.height / 2.0)

        for _ in range(count):
            x = float(np.random.uniform(low_x, high_x))
            y = float(np.random.uniform(low_y, high_y))
            for _ in range(OBSTACLE_SPAWN_ATTEMPTS):
                if not self._overlaps_obstacle(x, y):
                    break
                x = float(np.random.uniform(low_x, high_x))
                y = float(np.random.uniform(low_y, high_y))
            self.obstacles.append(Obstacle(x, y))

    def _overlaps_obstacle(self, x: float, y: float,
                           gap: float = OBSTACLE_MIN_GAP) -> bool:
        """True if a square centred at (x, y) would violate `gap`."""
        # Two squares are closer than the gap when their centre
        # separation is below one full side plus the gap on both axes.
        threshold = OBSTACLE_SIZE + gap
        for obstacle in self.obstacles:
            if (abs(x - obstacle.x) < threshold
                    and abs(y - obstacle.y) < threshold):
                return True
        return False

    def is_position_clear(self, x: float, y: float,
                          clearance: float = 0.0) -> bool:
        """
        Check whether a point is in bounds and outside every obstacle.

        Args:
            x, y: Point to test.
            clearance: Extra distance required from the obstacle faces.
                Used to keep an agent body or a food sprite from
                overlapping a solid.

        Returns:
            bool: True when the position is a legal spawn point.
        """
        if not self.is_inside_boundaries(x, y):
            return False
        reach = OBSTACLE_HALF_SIZE + clearance
        for obstacle in self.obstacles:
            if abs(x - obstacle.x) <= reach and abs(y - obstacle.y) <= reach:
                return False
        return True

    def random_free_position(self, rng=None, clearance: float = 0.0,
                             max_attempts: int = FREE_POSITION_ATTEMPTS
                             ) -> tuple[float, float]:
        """
        Rejection-sample an in-bounds position outside every obstacle.

        Args:
            rng: numpy Generator or the numpy.random module. None uses
                the global stream, preserving layout_seed determinism.
            clearance: Extra distance required from obstacle faces.
            max_attempts: Sampling budget before falling back to the
                last candidate.

        Returns:
            (x, y) tuple. The fallback can overlap an obstacle only when
            the world is too packed for the requested clearance.
        """
        generator = rng if rng is not None else np.random
        margin = self.boundary_margin
        x = float(generator.uniform(margin, self.width - margin))
        y = float(generator.uniform(margin, self.height - margin))
        for _ in range(max_attempts):
            if self.is_position_clear(x, y, clearance):
                break
            x = float(generator.uniform(margin, self.width - margin))
            y = float(generator.uniform(margin, self.height - margin))
        return x, y

    def clear_position(self, x: float, y: float, clearance: float = 0.0,
                       rng=None) -> tuple[float, float]:
        """
        Return (x, y) unchanged when it is a legal spawn, otherwise a
        free replacement. Used to relocate pre-built agents before an
        evaluation starts.
        """
        if self.is_position_clear(x, y, clearance):
            return float(x), float(y)
        return self.random_free_position(rng, clearance)

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