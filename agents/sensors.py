"""
Ray-based sensor system and fixed observation encoding (Phase 2).

Implements the design doc Section 11.2-11.3 sensor scheme: seven rays cast at
fixed world-angles relative to the agent heading, plus a fixed 12-dim
observation vector consumed by the fixed controller.

Observation layout (fixed and versioned; do not reorder -- genome
compatibility depends on a stable ordering):

    [0:7]   ray distances       [0,1], 0 = object at the agent, 1 = beyond range
    [7]     nearest-food bearing [-1,1], relative to heading (right positive)
    [8]     nearest-threat bearing [-1,1] (threat = other live agents)
    [9]     nearest-obstacle bearing [-1,1]
    [10]    energy               [0,1]
    [11]    speed                [0,1]

This gives 12 inputs for the 12-32-16-3 controller. The three bearings encode
direction to the nearest food/threat/obstacle resolved over the full 360 deg,
while the per-ray distances encode local directional proximity.
"""

from __future__ import annotations

import math

import numpy as np

# Seven rays at fixed offsets relative to heading (design doc Section 11.2).
# +x axis is heading 0; screen coordinates put +y downward, so "up" is -90 deg.
RAY_ANGLES_DEG: list[float] = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]

# Maximum sensing range in pixels; beyond this a ray reports saturated 1.0.
# Chosen as a documented constant well inside the 1200x700 world so agents
# must navigate to find anything.
MAX_RANGE: float = 200.0

# Hit radii treating food as a point, obstacles as squares of 10px (circle
# inscribed radius ~5), agents as 8px triangles (inscribed radius ~4).
FOOD_RADIUS: float = 2.0
OBSTACLE_RADIUS: float = 5.5
AGENT_RADIUS: float = 4.0

# Max speed used to normalize the speed feature; matches the Phase 1
# placeholder's 5.0 px/step cap so the encoding stays comparable.
MAX_SPEED: float = 5.0

# Full-observation dimension; must equal neural.network.input_size().
OBSERVATION_DIM: int = 12

# Energy normalization reference: the initial agent energy.
INITIAL_ENERGY: float = 100.0


def _clamp_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _ray_circle_hit(ray_dir_x: float, ray_dir_y: float,
                    center_x: float, center_y: float,
                    hit_radius: float) -> float:
    """
    Distance from the ray origin to an object's center, or infinity.

    The ray "hits" when it passes within `hit_radius` of the object's center
    (hit_radius acts as the ray's angular tolerance for point-like targets).
    The reported distance is the object's center projected onto the ray --
    the distance to the object's position, not to its surface, matching the
    encoding where 0 is "at the agent" and 1 is "beyond MAX_RANGE".

    Args:
        ray_dir_x, ray_dir_y: unit ray direction.
        center_x, center_y: object center offset from the ray origin.
        hit_radius: object hit radius / tolerance.

    Returns:
        Distance to the object's center in pixels, or math.inf if the ray
        passes farther than hit_radius from the center or behind the origin.
    """
    dx, dy = center_x, center_y
    dot = dx * ray_dir_x + dy * ray_dir_y
    # Objects behind the origin are not visible along this ray
    if dot <= 0.0:
        return math.inf
    # Squared perpendicular distance from the ray line to the center
    closest_sq = dx * dx + dy * dy - dot * dot
    if closest_sq > hit_radius * hit_radius:
        return math.inf
    return dot


def _raycast_ray(agent_x: float, agent_y: float,
                 ray_dir_x: float, ray_dir_y: float,
                 world, live_agents: list) -> float:
    """
    Cast one ray against food, obstacles, and other agents.

    Args:
        agent_x, agent_y: ray origin (the observing agent).
        ray_dir_x, ray_dir_y: unit direction vector of the ray.
        world: World providing food and obstacles.
        live_agents: list of live agents (threats), excluding the observer.

    Returns:
        Normalized distance [0,1] to the nearest object hit along this ray,
        with 1.0 meaning nothing within MAX_RANGE.
    """
    nearest_raw = MAX_RANGE
    closest = MAX_RANGE

    for food in world.food:
        dx = food.x - agent_x
        dy = food.y - agent_y
        dist = _ray_circle_hit(ray_dir_x, ray_dir_y, dx, dy, FOOD_RADIUS)
        # A hit at the origin (t=0) is behind the ray; require t > 0
        if 0.0 < dist < nearest_raw:
            nearest_raw = dist

    for obstacle in world.obstacles:
        dx = obstacle.x - agent_x
        dy = obstacle.y - agent_y
        dist = _ray_circle_hit(ray_dir_x, ray_dir_y, dx, dy, OBSTACLE_RADIUS)
        if 0.0 < dist < nearest_raw:
            nearest_raw = dist

    for threat in live_agents:
        dx = threat.position.x - agent_x
        dy = threat.position.y - agent_y
        dist = _ray_circle_hit(ray_dir_x, ray_dir_y, dx, dy, AGENT_RADIUS)
        if 0.0 < dist < nearest_raw:
            nearest_raw = dist

    if nearest_raw < closest:
        return nearest_raw / MAX_RANGE
    return 1.0


def _bearing_to_nearest(agent_x: float, agent_y: float, heading: float,
                        candidates: list) -> float | None:
    """
    Signed, normalized bearing to the nearest candidate object.

    Normalized as angle_delta / pi in [-1, 1]; positive when the object is to
    the agent's right. Returns None when no candidates exist.

    Args:
        agent_x, agent_y: observing agent position.
        heading: agent heading in radians (0 = +x).
        candidates: objects with .x/.y or .position.x/.position.y.

    Returns:
        Bearing in [-1, 1], or None if the candidate list is empty.
    """
    if not candidates:
        return None

    nearest_angle = 0.0
    nearest_sq = math.inf
    for candidate in candidates:
        # agents expose .position; food/obstacles expose .x/.y directly
        pos = getattr(candidate, "position", None)
        cx = getattr(candidate, "x", pos.x if pos is not None else 0.0)
        cy = getattr(candidate, "y", pos.y if pos is not None else 0.0)
        dx = cx - agent_x
        dy = cy - agent_y
        dist_sq = dx * dx + dy * dy
        if dist_sq < nearest_sq:
            nearest_sq = dist_sq
            nearest_angle = math.atan2(dy, dx)

    delta = _clamp_angle(nearest_angle - heading)
    return delta / math.pi


class RayCaster:
    """
    Computes the fixed 12-dim observation for a population of agents.

    No state of its own; exists so appended pipeline concerns (e.g. cached
    per-ray bookkeeping for the dashboard in later phases) have a stable home.
    """

    def observe_agent(self, agent, world, live_agents: list) -> np.ndarray:
        """
        Return one agent's 12-dim observation vector.

        Args:
            agent: the observing Organism.
            world: World instance.
            live_agents: all live agents used for the threat rays/bearing
                (the observer is excluded for rays and bearings).

        Returns:
            float32 array of length OBSERVATION_DIM.
        """
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        ax = agent.position.x
        ay = agent.position.y
        others = [t for t in live_agents if t is not agent]

        # Per-ray distances (indices 0..6)
        ray_dists: list[float] = []
        for offset_deg in RAY_ANGLES_DEG:
            ray_angle = agent.heading + math.radians(offset_deg)
            dir_x = math.cos(ray_angle)
            dir_y = math.sin(ray_angle)
            ray_dists.append(_raycast_ray(ax, ay, dir_x, dir_y, world, others))
        obs[0:7] = ray_dists

        # Global nearest bearings (indices 7..9)
        food_bearing = _bearing_to_nearest(ax, ay, agent.heading, world.food)
        threat_bearing = _bearing_to_nearest(ax, ay, agent.heading, others)
        obstacle_bearing = _bearing_to_nearest(ax, ay, agent.heading, world.obstacles)
        obs[7] = food_bearing if food_bearing is not None else 0.0
        obs[8] = threat_bearing if threat_bearing is not None else 0.0
        obs[9] = obstacle_bearing if obstacle_bearing is not None else 0.0

        # Internal state (indices 10..11)
        obs[10] = max(0.0, min(1.0, agent.energy / INITIAL_ENERGY))
        speed = agent.velocity.magnitude()
        obs[11] = max(0.0, min(1.0, speed / MAX_SPEED))

        return obs

    def observe_population(self, agents: list, world) -> np.ndarray:
        """
        Return the full observation matrix for the population.

        Args:
            agents: all agents in the population.
            world: World instance.

        Returns:
            float32 (N, OBSERVATION_DIM) matrix, one row per agent in the
            same order as `agents`.
        """
        live_agents = [agent for agent in agents if agent.is_alive]
        observations = np.stack([
            self.observe_agent(agent, world, live_agents)
            for agent in agents
        ]).astype(np.float32)
        return observations