"""
Organism module for Artificial Life Neuroevolution Simulation.

Contains the Organism class that represents individual agents with
state, lifecycle, and basic motion.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from simulation.physics import Vector2D
from simulation.environment import World
from agents.energy import EnergyConfig, update_energy, DEFAULT_ENERGY_CONFIG


class Organism:
    """
    Represents an individual agent in the simulation.

    Attributes:
        id (int): Unique identifier for the organism
        position (Vector2D): Current position in the world
        velocity (Vector2D): Current velocity
        heading (float): Current heading in radians (0 = right, pi/2 = down)
        energy (float): Current energy level
        health (float): Current health level (0-100)
        age (int): Current age in simulation steps
        fitness (float): Current fitness score
        genome (Optional[np.ndarray]): Genetic material encoding neural network weights
        is_alive (bool): Whether the organism is currently alive
    """

    def __init__(self, organism_id: int, x: float, y: float,
                 initial_energy: float = 100.0):
        """
        Initialize an organism.

        Args:
            organism_id (int): Unique identifier for this organism
            x (float): Initial x position
            y (float): Initial y position
            initial_energy (float): Starting energy level (default: 100.0)
        """
        self.id = organism_id
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(0.0, 0.0)
        self.heading = 0.0  # radians, 0 = right
        self.energy = float(initial_energy)
        self.health = 100.0
        self.age = 0
        self.fitness = 0.0
        self.genome: Optional[np.ndarray] = None
        self.is_alive = True

        # Contact counter, incremented by collision resolution; feeds the
        # 'collision' fitness weight (configs/baseline.json) in later phases
        self.collisions = 0

        # Accumulated collision penalty this step, applied in update_energy
        self._collision_penalty_this_step = 0.0

        # Food consumed this step (for energy gain)
        self._food_consumed_this_step = False

        # Energy configuration (allows per-agent override for testing)
        self._energy_config = DEFAULT_ENERGY_CONFIG

        # Neural network controller wiring (Phase 2)
        self.brain = None
        self.sensors = None

        # Last observation vector fed to the controller (12-dim); None until
        # the first engine step. Kept for debugging and the Phase 2 dashboards.
        self.last_observation = None

        # Motion state set by apply_action; consumed by update_energy to
        # compute the action energy cost in Phase 3
        self.last_steering = 0.0  # -1 to 1, from network output
        self.last_acceleration = 0.0  # 0 to 1, from network output
        self.last_eat_signal = 0.0  # 0 to 1, from network output

    def initialize_genome(self, genome_size: int) -> None:
        """
        Initialize the organism's genome with random values.

        Args:
            genome_size (int): Size of the genome array
        """
        self.genome = np.random.uniform(-1, 1, genome_size).astype(np.float32)

    def apply_action(self, steering: float, acceleration: float,
                     eat_signal: float, world: 'World', dt: float = 1.0) -> None:
        """
        Apply one decoded network action to the agent's physics.

        Replaces the Phase 1 random-motion stub. The network output already
        uses the design doc's activation space, so the only work here is
        scaling to sim units and integrating velocity into position.

        Args:
            steering (float): heading change in [-1, 1] (controller tanh).
            acceleration (float): throttle in [0, 1] (controller sigmoid).
            eat_signal (float): eat gate in [0, 1] (controller sigmoid).
            world (World): The world instance for boundary checking.
            dt (float): Time step (default: 1.0).
        """
        if not self.is_alive:
            return

        self.last_steering = float(steering)
        self.last_acceleration = float(acceleration)
        self.last_eat_signal = float(eat_signal)

        # Max heading change per step: 0.1 radians (matches Phase 1
        # placeholder tuning so behavior is comparable at dt=1)
        heading_change = steering * 0.1
        # Max speed: 5.0 pixels per step (same cap as Phase 1)
        speed = acceleration * 5.0

        self.heading += heading_change
        self.heading = self.heading % (2 * np.pi)

        self.velocity.x = np.cos(self.heading) * speed
        self.velocity.y = np.sin(self.heading) * speed

        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt

        # Enforce world boundaries (with margin)
        margin = getattr(world, 'boundary_margin', 10.0)
        if self.position.x < margin:
            self.position.x = margin
            self.velocity.x = 0
        elif self.position.x > world.width - margin:
            self.position.x = world.width - margin
            self.velocity.x = 0

        if self.position.y < margin:
            self.position.y = margin
            self.velocity.y = 0
        elif self.position.y > world.height - margin:
            self.position.y = world.height - margin
            self.velocity.y = 0

    def consume_food(self, world: 'World') -> bool:
        """
        Attempt to consume food at current position.

        Args:
            world (World): The world instance

        Returns:
            bool: True if food was consumed, False otherwise
        """
        if not self.is_alive:
            return False

        # Check if there's food nearby (simplified consumption radius)
        consumption_radius = 5.0  # pixels

        for i, food in enumerate(world.food):
            distance = self.position.distance_to(Vector2D(food.x, food.y))
            if distance < consumption_radius:
                # Consume the food
                world.food.pop(i)
                self._food_consumed_this_step = True
                return True

        return False

    def update_energy(self, energy_config: Optional[EnergyConfig] = None) -> None:
        """
        Update energy level using the Phase 3 energy balance equation.

        Args:
            energy_config: Optional EnergyConfig override. Uses the agent's
                stored config if not provided.
        """
        if not self.is_alive:
            return

        config = energy_config or self._energy_config

        new_energy, new_health, is_dead = update_energy(
            energy=self.energy,
            health=self.health,
            acceleration=self.last_acceleration,
            steering=self.last_steering,
            eat_event=self._food_consumed_this_step,
            collision_penalty_applied=self._collision_penalty_this_step,
            config=config,
        )

        self.energy = new_energy
        self.health = new_health
        self._collision_penalty_this_step = 0.0
        self._food_consumed_this_step = False

        if is_dead:
            self.is_alive = False
            self.energy = 0.0

    def increment_age(self) -> None:
        """Increment the organism's age by one simulation step."""
        if self.is_alive:
            self.age += 1

    def add_collision_penalty(self, penalty: float) -> None:
        """
        Accumulate collision penalty for this step.

        The penalty is applied in update_energy.

        Args:
            penalty: Energy cost to add.
        """
        if self.is_alive:
            self._collision_penalty_this_step += penalty

    def get_state(self) -> dict:
        """
        Get the current state of the organism for serialization.

        Returns:
            dict: Dictionary containing organism state
        """
        return {
            'id': self.id,
            'position': (self.position.x, self.position.y),
            'velocity': (self.velocity.x, self.velocity.y),
            'heading': self.heading,
            'energy': self.energy,
            'health': self.health,
            'age': self.age,
            'fitness': self.fitness,
            'genome': self.genome.tolist() if self.genome is not None else None,
            'is_alive': self.is_alive,
            'collisions': self.collisions
        }

    def load_state(self, state: dict) -> None:
        """
        Load organism state from a dictionary.

        Args:
            state (dict): Dictionary containing organism state
        """
        self.id = state['id']
        self.position = Vector2D(state['position'][0], state['position'][1])
        self.velocity = Vector2D(state['velocity'][0], state['velocity'][1])
        self.heading = state['heading']
        self.energy = state['energy']
        self.health = state['health']
        self.age = state['age']
        self.fitness = state['fitness']
        if state['genome'] is not None:
            self.genome = np.array(state['genome'], dtype=np.float32)
        else:
            self.genome = None
        self.is_alive = state['is_alive']
        self.collisions = state['collisions']