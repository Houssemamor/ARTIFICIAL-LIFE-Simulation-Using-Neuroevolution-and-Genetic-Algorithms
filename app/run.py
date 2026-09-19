#!/usr/bin/env python3
"""
Artificial Life Neuroevolution Simulation - GUI Mode Entry Point
"""

import argparse
import json
import sys
import os
from typing import List

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.world_config import load_config
from simulation.environment import World
from agents.organism import Organism
from visualization.renderer import Renderer


def main():
    """Main entry point for GUI mode."""
    parser = argparse.ArgumentParser(description='Run Artificial Life Neuroevolution Simulation (GUI)')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--mode', type=str, choices=['gui', 'headless'], default='gui', help='Execution mode')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print(f"Starting Artificial Life Neuroevolution Simulation")
    print(f"Configuration: {args.config}")
    print(f"Mode: {args.mode}")
    print(f"Experiment: {config.experiment_name}")

    # Create world from config
    world = World()
    world.load_from_config(config.dict())

    # Create initial population
    population_size = config.population.size
    agents: List[Organism] = []
    for i in range(population_size):
        # Spawn agents at random positions within the world
        x = np.random.uniform(world.boundary_margin, world.width - world.boundary_margin)
        y = np.random.uniform(world.boundary_margin, world.height - world.boundary_margin)
        agent = Organism(i, x, y, initial_energy=100.0)
        agent.initialize_genome(genome_size=100)  # Placeholder genome size
        agents.append(agent)

    # Set up renderer
    renderer = Renderer(world.width, world.height)

    # Simulation state
    simulation_paused = False
    simulation_speed = 1.0  # 1.0 = real-time, 10.0 = 10x faster
    physics_dt = 1.0 / 60.0  # Fixed physics timestep (60 Hz)
    physics_accumulator = 0.0
    last_render_time = 0.0
    render_dt = 1.0 / 30.0  # Target render FPS (30 Hz) - can be lower than physics

    # Clock for timing
    import time
    last_time = time.time()

    # Main loop
    running = True
    while running:
        # Calculate elapsed time
        current_time = time.time()
        elapsed = current_time - last_time
        last_time = current_time

        # Handle events
        running = renderer.handle_events()
        # Check for pause (simplify: we'll use spacebar to pause)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            simulation_paused = not simulation_paused
            # Debounce: wait a bit to avoid multiple toggles
            pygame.time.wait(200)

        # Update simulation if not paused
        if not simulation_paused:
            # Accumulate time for physics steps
            physics_accumulator += elapsed * simulation_speed

            # Perform physics steps while we have enough accumulated time
            while physics_accumulator >= physics_dt:
                # Update each organism with placeholder motion
                for agent in agents:
                    agent.update_placeholder_motion(world, dt=physics_dt)
                    agent.update_energy()
                    agent.increment_age()
                physics_accumulator -= physics_dt

        # Render at most at the target render FPS
        # We'll render every frame for simplicity, but we could limit it
        renderer.draw_world(world, agents)

        # Cap the render loop to prevent excessive CPU usage
        renderer.clock.tick(60)  # Limit to 60 FPS

    # Clean up
    renderer.quit()
    print("Simulation ended.")


if __name__ == "__main__":
    main()