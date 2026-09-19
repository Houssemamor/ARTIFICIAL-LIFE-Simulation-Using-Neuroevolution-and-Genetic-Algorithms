"""
Organism module for Artificial Life Neuroevolution Simulation.

Contains the Organism class that represents individual agents with
neural network controllers, genomes, and life cycle management.
"""

import numpy as np
from typing import List, Tuple, Optional
import json
import sys
import os

# Add the project root to the Python path (three levels up:
# organism.py -> agents/organism -> agents -> project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation.environment import World


class Organism:
    """
    Represents an individual agent in the simulation.

    Attributes:
        id (int): Unique identifier for the organism
        x, y (float): Current position in the world
        energy (float): Current energy level
        age (int): Current age in simulation steps
        genome (np.ndarray): Genetic material encoding neural network weights
        brain: Neural network controller (to be implemented)
        sensors: Sensor system for observing environment (to be implemented)
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
        self.x = float(x)
        self.y = float(y)
        self.energy = float(initial_energy)
        self.age = 0
        self.genome: Optional[np.ndarray] = None
        self.brain = None  # To be implemented in neural.network module
        self.sensors = None  # To be implemented in agents.sensors module
        self.is_alive = True
        self.last_action = np.zeros(3)  # [steering, acceleration, eat] - to match brain output

    def initialize_genome(self, genome_size: int) -> None:
        """
        Initialize the organism's genome with random values.

        Args:
            genome_size (int): Size of the genome array
        """
        self.genome = np.random.uniform(-1, 1, genome_size).astype(np.float32)

    def initialize_brain_and_sensors(self, brain_architecture: List[int],
                                   num_sensors: int) -> None:
        """
        Initialize the organism's brain and sensory systems.
        This is a placeholder - actual implementation will be in neural.network
        and agents.sensors modules.

        Args:
            brain_architecture (List[int]): Layer sizes for the neural network
            num_sensors (int): Number of sensory inputs
        """
        # Placeholder - actual implementation will import from neural.network
        # and agents.sensors modules once they are implemented
        self.brain_architecture = brain_architecture
        self.num_sensors = num_sensors
        self.is_alive = True

    def update_sensors(self, world: World) -> np.ndarray:
        """
        Update sensor readings based on current world state.
        This is a placeholder - actual implementation will be in agents.sensors module.

        Args:
            world (World): The current world state

        Returns:
            np.ndarray: Normalized sensor readings
        """
        # Placeholder implementation - returns dummy sensor data
        # Actual implementation will compute ray-based sensor readings
        # as described in the plan: 7 ray-based sensors, normalized observations
        sensor_input = np.random.uniform(-1, 1, self.num_sensors).astype(np.float32)
        return sensor_input

    def think(self, sensor_input: np.ndarray) -> np.ndarray:
        """
        Process sensor input through the neural network to get actions.
        This is a placeholder - actual implementation will be in neural.network module.

        Args:
            sensor_input (np.ndarray): Normalized sensor readings

        Returns:
            np.ndarray: Action vector [steering, acceleration, eat]
        """
        # Placeholder implementation - returns random actions
        # Actual implementation will forward-pass through neural network
        if self.brain_architecture is not None:
            # For now, just return random actions in correct format
            action = np.random.uniform(-1, 1, 3).astype(np.float32)
            # Clip to reasonable ranges
            action[0] = np.clip(action[0], -1, 1)  # steering: -1 (left) to 1 (right)
            action[1] = np.clip(action[1], 0, 1)   # acceleration: 0 (none) to 1 (full)
            action[2] = np.clip(action[2], 0, 1)   # eat: 0 (none) to 1 (full attempt)
            return action
        else:
            return np.zeros(3, dtype=np.float32)

    def update_position(self, steering: float, acceleration: float,
                       world: World, dt: float = 1.0) -> None:
        """
        Update the organism's position based on steering and acceleration.

        Args:
            steering (float): Steering input (-1 to 1)
            acceleration (float): Acceleration input (0 to 1)
            world (World): The world instance for boundary checking
            dt (float): Time step (default: 1.0)
        """
        if not self.is_alive:
            return

        # Simple physics model - to be refined in simulation.physics module
        # Convert steering to angle change, acceleration to velocity change
        speed = acceleration * 5.0  # Max speed of 5 pixels per step
        angle_change = steering * 0.1  # Max angle change of 0.1 radians per step

        # Update position (simplified - proper implementation in simulation.physics)
        # For now, just store the action for physics engine to process
        self.last_action = np.array([steering, acceleration, self.last_action[2]])

        # Simple movement - to be replaced with proper physics
        # self.x += speed * np.cos(angle_change) * dt
        # self.y += speed * np.sin(angle_change) * dt

        # Apply boundary constraints
        # self.x = max(world.boundary_margin,
        #              min(self.width - world.boundary_margin, self.x))
        # self.y = max(world.boundary_margin,
        #              min(self.height - world.boundary_margin, self.y))

    def consume_food(self, world: World) -> bool:
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
            distance = np.sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
            if distance < consumption_radius:
                # Consume the food
                world.food.pop(i)
                self.energy += 50.0  # Gain energy from food
                return True

        return False

    def update_energy(self, metabolism_rate: float = 0.1) -> None:
        """
        Update energy level based on metabolism and actions.

        Args:
            metabolism_rate (float): Rate of energy consumption per step (default: 0.1)
        """
        if not self.is_alive:
            return

        # Basic metabolism
        self.energy -= metabolism_rate

        # Additional energy cost for actions
        action_cost = np.abs(self.last_action[0]) * 0.01 + self.last_action[1] * 0.05
        self.energy -= action_cost

        # Check if organism has died from lack of energy
        if self.energy <= 0:
            self.is_alive = False
            self.energy = 0.0

    def increment_age(self) -> None:
        """Increment the organism's age by one simulation step."""
        if self.is_alive:
            self.age += 1

    def get_state(self) -> dict:
        """
        Get the current state of the organism for serialization.

        Returns:
            dict: Dictionary containing organism state
        """
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'energy': self.energy,
            'age': self.age,
            'genome': self.genome.tolist() if self.genome is not None else None,
            'is_alive': self.is_alive,
            'last_action': self.last_action.tolist()
        }

    def load_state(self, state: dict) -> None:
        """
        Load organism state from a dictionary.

        Args:
            state (dict): Dictionary containing organism state
        """
        self.id = state['id']
        self.x = state['x']
        self.y = state['y']
        self.energy = state['energy']
        self.age = state['age']
        if state['genome'] is not None:
            self.genome = np.array(state['genome'], dtype=np.float32)
        else:
            self.genome = None
        self.is_alive = state['is_alive']
        self.last_action = np.array(state['last_action'], dtype=np.float32)