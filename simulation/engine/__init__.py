"""
Simulation engine for Artificial Life Neuroevolution Simulation.

Owns the physics-step logic shared by the GUI and headless entrypoints
(design doc tree: simulation/engine.py - main loop, GUI/headless orchestration).
"""

import numpy as np
import torch

from agents.sensors import RayCaster
from neural.batched_inference import batched_forward, stack_population_weights
from neural.genome import genome_size
from simulation.physics import AABB, SpatialHash, Vector2D

# Collision body sizes in pixels, matching the renderer's drawn sizes
# (renderer.py draws agents as 8 px triangles and obstacles as 10 px squares)
AGENT_BODY_SIZE = 8.0
OBSTACLE_BODY_SIZE = 10.0

# Eat gate: an eat event fires only when the network output exceeds this
# threshold, per design doc Section 11.4.
EAT_SIGNAL_THRESHOLD = 0.5


def stack_genomes(agents) -> np.ndarray:
    """
    Assemble the population's genomes into a single (N, genome_size) array.

    Args:
        agents: list of Organisms, each with a genome attribute.

    Returns:
        float32 (N, genome_size) array in population order.
    """
    return np.stack([agent.genome for agent in agents]).astype(np.float32)


def step_simulation(world, agents, agent_raycaster: RayCaster,
                    dt: float = 1.0) -> int:
    """
    Run one batched simulation step for the whole population.

    Implements the observe -> batched_forward -> act -> physics pipeline from
    PLAN.md Phase 2 step 4, replacing the Phase 1 random-motion placeholder.

    Args:
        world: World instance.
        agents: list of Organism instances.
        agent_raycaster (RayCaster): The sensor system (stateless).
        dt (float): Physics timestep (default: 1.0).

    Returns:
        int: Total collision contact events this step (see resolve_collisions).
    """
    live_agents = [agent for agent in agents if agent.is_alive]
    if not live_agents:
        return 0

    # 1. Observe: sensor matrix over live agents, row i <=> live_agents[i]
    observations = agent_raycaster.observe_population(live_agents, world)

    # 2. Batched forward: one pass across all live agents (N, 3)
    genomes = stack_genomes(live_agents)
    weights = stack_population_weights(genomes)
    action_logits = batched_forward(
        torch.as_tensor(observations, dtype=torch.float32),
        weights,
    )

    # 3. Apply: decode tanh/sigmoid outputs (already activated) into physics.
    #    The eat gate fires an eat event when the signal clears its threshold.
    for index, agent in enumerate(live_agents):
        steering = float(action_logits[index, 0])
        acceleration = float(action_logits[index, 1])
        eat_signal = float(action_logits[index, 2])
        agent.last_observation = observations[index]
        agent.apply_action(steering, acceleration, eat_signal, world, dt=dt)
        if eat_signal > EAT_SIGNAL_THRESHOLD:
            agent.consume_food(world)

    # 4. Physics: move all agents first so collisions see final positions
    collisions = resolve_collisions(agents, world)

    # 5. Metabolism and aging after motion and collision resolution
    for agent in agents:
        agent.update_energy()
        agent.increment_age()

    return collisions


def resolve_collisions(agents, world, penalty: float = 2.0) -> int:
    """
    Detect contacts between live agents and obstacles/other agents, and apply
    the plan's collision penalty: an energy cost per contact with no velocity
    reflection (the recommended baseline - avoids the instability of velocity
    reflection; see PLAN.md Phase 1 step 3).

    Rebuilds a SpatialHash over all bodies each step instead of updating
    positions incrementally: with ~260 objects at 60 Hz the rebuild is cheap,
    and it avoids the fragile incremental-update path in SpatialHash.

    Args:
        agents: List of Organism instances
        world: World instance providing obstacle positions
        penalty (float): Energy cost per contact (default: 2.0)

    Returns:
        int: Total contact events this step (a mutual agent-agent contact is
        counted once per participant, so it is counted twice)
    """
    # Dead agents neither collide nor accrue penalties
    live_agents = [agent for agent in agents if agent.is_alive]
    if not live_agents:
        return 0

    spatial_hash = SpatialHash(cell_size=16.0)
    # Map spatial-hash ids back to the sim objects they belong to, so query
    # hits can be resolved to a specific agent or obstacle
    id_map = {}

    for obstacle in world.obstacles:
        body = AABB.from_center_size(obstacle.x, obstacle.y,
                                     OBSTACLE_BODY_SIZE, OBSTACLE_BODY_SIZE)
        obj_id = spatial_hash.add_object(Vector2D(obstacle.x, obstacle.y), body)
        id_map[obj_id] = obstacle

    for agent in live_agents:
        body = AABB.from_center_size(agent.position.x, agent.position.y,
                                     AGENT_BODY_SIZE, AGENT_BODY_SIZE)
        obj_id = spatial_hash.add_object(agent.position, body)
        id_map[obj_id] = agent

    total_contacts = 0
    for agent in live_agents:
        body = AABB.from_center_size(agent.position.x, agent.position.y,
                                     AGENT_BODY_SIZE, AGENT_BODY_SIZE)
        for hit_id in spatial_hash.query_rect(body):
            hit = id_map[hit_id]
            if hit is agent:
                # A body always intersects itself; that is not a collision
                continue
            # Penalty on contact: energy cost, tracked per agent to feed the
            # 'collision' fitness weight (configs/baseline.json) in later phases
            agent.energy -= penalty
            agent.collisions += 1
            total_contacts += 1

        # Mirror update_energy's death check so standalone calls (e.g. tests)
        # never leave a negative-energy agent alive; idempotent in the main
        # loop because update_energy runs after this and skips dead agents
        if agent.energy <= 0:
            agent.is_alive = False
            agent.energy = 0.0

    return total_contacts
