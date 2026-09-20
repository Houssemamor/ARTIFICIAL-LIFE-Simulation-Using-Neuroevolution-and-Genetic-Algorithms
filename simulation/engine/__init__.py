"""
Simulation engine for Artificial Life Neuroevolution Simulation.

Owns the physics-step logic shared by the GUI and headless entrypoints
(design doc tree: simulation/engine.py - main loop, GUI/headless orchestration).
"""

from simulation.physics import AABB, SpatialHash, Vector2D

# Collision body sizes in pixels, matching the renderer's drawn sizes
# (renderer.py draws agents as 8 px triangles and obstacles as 10 px squares)
AGENT_BODY_SIZE = 8.0
OBSTACLE_BODY_SIZE = 10.0


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
