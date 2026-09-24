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
import time
import numpy as np
import pygame
from pathlib import Path
from typing import List

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.world_config import load_config
from simulation.determinism import DeterminismConfig, set_deterministic_seeds
from simulation.environment import World
from simulation.engine import step_simulation, resolve_captures
from agents.organism import Organism
from agents.sensors import RayCaster
from neural.genome import genome_size
from visualization.renderer import Renderer
from visualization.dashboard_advanced import (format_ecosystem_lines,
                                              format_hof_lines, draw_panel)
from evolution.reproduction import reproduction_pass


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

    # Placeholder motion never fires the eat gate, so a predation
    # config in placeholder mode produces no captures and no regrowth
    # (regrowth lives inside step_simulation). Warn rather than refuse:
    # placeholder mode exists for renderer/physics debugging.
    if config.predation and config.predation.predator_count > 0 \
            and args.agent_mode == 'placeholder':
        print("Warning: placeholder agents cannot capture or regrow food; "
              "predation mechanics need --agent-mode neural")

    # Enforce the cpu-deterministic tier before any simulation state is
    # created. The config's three seed fields are used when set; when they
    # are null (interactive runs), a time-derived base is used and printed
    # so an interesting run can in principle be reproduced.
    base_seed = config.seed.get("numpy")
    if base_seed is None:
        base_seed = int(time.time())
        print(f"Config seeds are null; using time-derived base seed {base_seed}")
    torch_seed = config.seed.get("torch")
    if torch_seed is None:
        torch_seed = base_seed + 1
    random_seed = config.seed.get("random")
    if random_seed is None:
        random_seed = base_seed + 2
    set_deterministic_seeds(DeterminismConfig(
        torch_seed=torch_seed,
        numpy_seed=base_seed,
        random_seed=random_seed,
        device="cpu",
        reproducibility_tier="cpu-deterministic",
        num_threads=1,
    ))

    print(f"Starting Artificial Life Neuroevolution Simulation")
    print(f"Configuration: {args.config}")
    print(f"Mode: {args.mode}")
    print(f"Agent mode: {args.agent_mode}")
    print(f"Experiment: {config.experiment_name}")

    # Create world from config
    world = World()
    world.load_from_config(config.model_dump())

    # Create initial population. With a Phase 6 predation config, the
    # first predator_count agents are predators and the rest are prey;
    # otherwise every agent is prey (legacy behavior).
    predation = config.predation
    predator_count = predation.predator_count if predation else 0
    population_size = config.population.size
    agents: List[Organism] = []
    for i in range(population_size + predator_count):
        x = np.random.uniform(world.boundary_margin, world.width - world.boundary_margin)
        y = np.random.uniform(world.boundary_margin, world.height - world.boundary_margin)
        role = 'predator' if i < predator_count else 'prey'
        agent = Organism(i, x, y, initial_energy=100.0, role=role)
        if args.agent_mode == 'neural':
            agent.initialize_genome(genome_size=genome_size())
        else:
            # Placeholder agents don't need a genome
            agent.genome = np.zeros(genome_size(), dtype=np.float32)
        agents.append(agent)

    # Phase 6 ecosystem state: reproduction and capture bookkeeping for
    # the live dashboard panel
    repro = config.reproduction
    next_agent_id = len(agents)
    capture_total = 0
    birth_total = 0
    show_panel = False

    # Hall-of-fame panel data: the latest co-evolution run's frozen-
    # opponent evaluations, if one exists on disk
    hof_evaluations = None
    hof_path = Path("experiments/EXP-COEV/coevolution_summary.json")
    if hof_path.is_file():
        try:
            hof_evaluations = json.loads(hof_path.read_text()).get(
                "hof_evaluations")
        except (OSError, ValueError):
            hof_evaluations = None

    # Reproduction gates on agent.fitness, which the GA sets at
    # generation end. The GUI has no GA, so - like ecosystem mode -
    # fitness is refreshed every evaluation_steps physics steps from
    # the same measured calibration scales. Without this the births
    # counter on the panel is structurally always zero.
    calibration_scales = None
    if repro and repro.enabled:
        from evolution.genetic_algorithm import compute_fitness
        from experiments.experiment_runner import load_calibration_scales
        calibration_scales = load_calibration_scales()
    physics_step_count = 0

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
        # F9 (consumed inside renderer.handle_events) toggles the Phase 6
        # advanced dashboard panel
        if renderer.panel_toggled:
            renderer.panel_toggled = False
            show_panel = not show_panel

        if not simulation_paused:
            physics_accumulator += elapsed * simulation_speed

            while physics_accumulator >= physics_dt:
                if args.agent_mode == 'neural':
                    step_simulation(world, agents, agent_raycaster, dt=physics_dt)
                    if predation:
                        # Ecosystem mechanics: predators capture prey,
                        # eligible agents reproduce (Phase 6)
                        capture_total += resolve_captures(
                            agents,
                            capture_radius=predation.capture_radius,
                            energy_transfer=predation.capture_energy_transfer)
                        if repro and repro.enabled:
                            physics_step_count += 1
                            if physics_step_count % config.evolution.evaluation_steps == 0:
                                # Day-boundary fitness refresh (matches
                                # ecosystem mode) so the reproduction
                                # gate sees real achievement
                                for agent in agents:
                                    agent.fitness = compute_fitness(
                                        agent, agent.age, agent.food_eaten,
                                        0.0, agent.collisions,
                                        calibration_scales,
                                        config.fitness_weights)
                            newborns, next_agent_id = reproduction_pass(
                                agents, world, repro, next_agent_id)
                            agents.extend(newborns)
                            birth_total += len(newborns)
                else:
                    step_placeholder(world, agents, dt=physics_dt)
                physics_accumulator -= physics_dt

        renderer.draw_world(world, agents)
        if show_panel:
            prey_alive = sum(1 for a in agents if a.is_alive and a.role == 'prey')
            predator_alive = sum(1 for a in agents
                                 if a.is_alive and a.role == 'predator')
            draw_panel(renderer,
                       format_ecosystem_lines(prey_alive, predator_alive,
                                              capture_total, birth_total,
                                              len(agents)) +
                       format_hof_lines(hof_evaluations))
        renderer.clock.tick(60)

    renderer.quit()
    print("Simulation ended.")


if __name__ == "__main__":
    main()