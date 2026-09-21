#!/usr/bin/env python3
"""
Artificial Life Neuroevolution Simulation - GUI Mode Entry Point

Supports two agent modes:
- neural: agents use the fixed 12-32-16-3 neural controller (Phase 2+)
- placeholder: agents use random-walk motion (Phase 1 baseline)
"""

import argparse
import json
import sys
import os
import numpy as np
import pygame
from typing import List

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.world_config import load_config
from simulation.environment import World
from simulation.engine import step_simulation
from agents.organism import Organism
from agents.sensors import RayCaster
from neural.genome import genome_size
from visualization.renderer import Renderer


def step_placeholder(world, agents, dt: float = 1.0) -> int:
    """
    Phase 1 placeholder motion: random steering, constant low acceleration.

    This is the pre-neural random walk used to validate physics/render loop
    before the neural controller was wired in Phase 2.
    """
    for agent in agents:
        if not agent.is_alive:
            continue
        # Random steering in [-1, 1], constant low acceleration
        steering = np.random.uniform(-1.0, 1.0)
        acceleration = 0.3
        eat_signal = 0.0
        agent.apply_action(steering, acceleration, eat_signal, world, dt=dt)

    from simulation.engine import resolve_collisions
    collisions = resolve_collisions(agents, world)

    for agent in agents:
        agent.update_energy()
        agent.increment_age()

    return collisions


def main():
    """Main entry point for GUI mode."""
    parser = argparse.ArgumentParser(description='Run Artificial Life Neuroevolution Simulation (GUI)')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--mode', type=str, choices=['gui', 'headless'], default='gui', help='Execution mode')
    parser.add_argument('--agent-mode', type=str, choices=['neural', 'placeholder'], default='neural',
                        help='Agent behavior: neural (Phase 2+ NN controller) or placeholder (Phase 1 random walk)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print(f"Starting Artificial Life Neuroevolution Simulation")
    print(f"Configuration: {args.config}")
    print(f"Mode: {args.mode}")
    print(f"Agent mode: {args.agent_mode}")
    print(f"Experiment: {config.experiment_name}")

    # Create world from config
    world = World()
    world.load_from_config(config.model_dump())

    # Create initial population
    population_size = config.population.size
    agents: List[Organism] = []
    for i in range(population_size):
        x = np.random.uniform(world.boundary_margin, world.width - world.boundary_margin)
        y = np.random.uniform(world.boundary_margin, world.height - world.boundary_margin)
        agent = Organism(i, x, y, initial_energy=100.0)
        if args.agent_mode == 'neural':
            agent.initialize_genome(genome_size=genome_size())
        else:
            # Placeholder agents don't need a genome
            agent.genome = np.zeros(genome_size(), dtype=np.float32)
        agents.append(agent)

    # Sensor system (only used in neural mode)
    agent_raycaster = RayCaster()

    # Set up renderer
    renderer = Renderer(world.width, world.height)

    # Simulation state
    simulation_paused = False
    simulation_speed = 1.0
    physics_dt = 1.0 / 60.0
    physics_accumulator = 0.0
    render_dt = 1.0 / 30.0

    import time
    last_time = time.time()

    # Main loop
    running = True
    while running:
        current_time = time.time()
        elapsed = current_time - last_time
        last_time = current_time

        running = renderer.handle_events()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            simulation_paused = not simulation_paused
            pygame.time.wait(200)

        if not simulation_paused:
            physics_accumulator += elapsed * simulation_speed

            while physics_accumulator >= physics_dt:
                if args.agent_mode == 'neural':
                    step_simulation(world, agents, agent_raycaster, dt=physics_dt)
                else:
                    step_placeholder(world, agents, dt=physics_dt)
                physics_accumulator -= physics_dt

        renderer.draw_world(world, agents)
        renderer.clock.tick(60)

    renderer.quit()
    print("Simulation ended.")


if __name__ == "__main__":
    main()