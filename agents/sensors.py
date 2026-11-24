"""
Ray-based sensor system and fixed observation encoding (Phase 2).

Implements the design doc Section 11.2-11.3 sensor scheme: seven rays cast at
fixed world-angles relative to the agent heading, plus a fixed 12-dim
observation vector consumed by the fixed controller.

Observation layout (fixed and versioned; do not reorder -- genome
compatibility depends on a stable ordering). The meaning of the food and
threat channels depends on the observer's role, because a predator cannot
eat plants and must not be told that plants are food:

    [0:7]   ray distances       [0,1], 0 = object at the agent, 1 = beyond range
    [7]     prey: nearest-food bearing; predator: nearest-prey bearing [-1,1]
    [8]     prey: nearest live agent; predator: nearest rival predator [-1,1]
    [9]     nearest-obstacle bearing [-1,1]
    [10]    energy               [0,1]
    [11]    speed                [0,1]

Bearings are relative to heading (right positive). Predator rays ignore
plant food entirely, so a predator perceives only prey, rival predators
and obstacles. Prey observations are byte-identical to the original
encoding, so the prey-only experiments remain reproducible.
"""

from __future__ import annotations

import builtins
import math

import numpy as np

from simulation.geometry import OBSTACLE_HALF_SIZE

# Seven rays at fixed offsets relative to heading (design doc Section 11.2).
# +x axis is heading 0; screen coordinates put +y downward, so "up" is -90 deg.
RAY_ANGLES_DEG: list[float] = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
# Precomputed with the same math.radians the scalar oracle uses, so the
# vectorised path does not drift in the last floating-point bits.
RAY_ANGLES_RAD: np.ndarray = np.array(
    [math.radians(angle) for angle in RAY_ANGLES_DEG], dtype=np.float64)

# Maximum sensing range in pixels; beyond this a ray reports saturated 1.0.
# Chosen as a documented constant well inside the 1200x700 world so agents
# must navigate to find anything.
MAX_RANGE: float = 200.0

# Hit radii treating food as a point and agents as 8px triangles (inscribed
# radius ~4). Obstacles are not approximated by a circle: their rays use
# the exact square body from simulation.geometry, the same body the
# collision engine treats as solid.
FOOD_RADIUS: float = 2.0
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


def _ray_aabb_hit(ray_dir_x: float, ray_dir_y: float,
                  center_x: float, center_y: float,
                  half_size: float) -> float:
    """
    Distance along a unit ray to an axis-aligned square's near face.

    Slab method: intersect the ray with the x and y slabs of the square
    and keep the overlapping interval. This is the exact geometry of the
    solid collision body, so a ray can never report an obstacle the
    physics engine would let an agent pass through (the bug the old
    5.5 px circle approximation caused).

    Args:
        ray_dir_x, ray_dir_y: unit ray direction.
        center_x, center_y: square centre offset from the ray origin.
        half_size: half the square's side length.

    Returns:
        Distance to the entry point, 0.0 when the origin is inside the
        square, or math.inf when the ray misses or points away.
    """
    t_min = -math.inf
    t_max = math.inf
    for direction, center in ((ray_dir_x, center_x),
                              (ray_dir_y, center_y)):
        low = center - half_size
        high = center + half_size
        if abs(direction) < 1e-12:
            # Parallel to this slab: it only intersects when the ray
            # line already lies inside the slab.
            if not (low <= 0.0 <= high):
                return math.inf
            continue
        t1 = low / direction
        t2 = high / direction
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return math.inf
    if t_max < 0.0:
        return math.inf
    if t_min > 0.0:
        return t_min
    return 0.0


def _raycast_ray(agent_x: float, agent_y: float,
                 ray_dir_x: float, ray_dir_y: float,
                 food_targets: list, obstacles: list,
                 live_agents: list) -> float:
    """
    Cast one ray against the given food targets, obstacles, and agents.

    Args:
        agent_x, agent_y: ray origin (the observing agent).
        ray_dir_x, ray_dir_y: unit direction vector of the ray.
        food_targets: food items this observer can perceive (empty for a
            predator, which cannot eat plants).
        obstacles: world obstacles.
        live_agents: list of live agents (excluding the observer).

    Returns:
        Normalized distance [0,1] to the nearest object hit along this ray,
        with 1.0 meaning nothing within MAX_RANGE. Food and agents report
        their centre distance; obstacles report the entry point of their
        square body.
    """
    nearest_raw = MAX_RANGE

    for food in food_targets:
        dx = food.x - agent_x
        dy = food.y - agent_y
        dist = _ray_circle_hit(ray_dir_x, ray_dir_y, dx, dy, FOOD_RADIUS)
        # A hit at the origin (t=0) is behind the ray; require t > 0
        if 0.0 < dist < nearest_raw:
            nearest_raw = dist

    for obstacle in obstacles:
        dx = obstacle.x - agent_x
        dy = obstacle.y - agent_y
        dist = _ray_aabb_hit(ray_dir_x, ray_dir_y, dx, dy,
                             OBSTACLE_HALF_SIZE)
        # 0.0 is valid here: the origin is inside the solid body.
        if 0.0 <= dist < nearest_raw:
            nearest_raw = dist

    for threat in live_agents:
        dx = threat.position.x - agent_x
        dy = threat.position.y - agent_y
        dist = _ray_circle_hit(ray_dir_x, ray_dir_y, dx, dy, AGENT_RADIUS)
        if 0.0 < dist < nearest_raw:
            nearest_raw = dist

    if nearest_raw < MAX_RANGE:
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


# Rows processed per vectorised batch. Keeps the (rows, rays, objects)
# temporaries at a few MB even for large populations.
_VECTOR_CHUNK_ROWS = 32


def _nearest_bearings(observer_xy: np.ndarray, headings: np.ndarray,
                      candidate_xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Vectorised equivalent of _bearing_to_nearest for every observer at once.

    np.argmin returns the first minimum, matching the strict '<' comparison
    in the scalar reference when two candidates are equidistant.
    """
    count = observer_xy.shape[0]
    candidate_count = candidate_xy.shape[0]
    if candidate_count == 0:
        return np.zeros(count, dtype=np.float64)

    delta_x = candidate_xy[None, :, 0] - observer_xy[:, None, 0]
    delta_y = candidate_xy[None, :, 1] - observer_xy[:, None, 1]
    distance_sq = np.where(valid, delta_x * delta_x + delta_y * delta_y, np.inf)
    indices = np.argmin(distance_sq, axis=1)
    rows = np.arange(count)
    found = np.isfinite(distance_sq[rows, indices])
    angle = np.arctan2(delta_y[rows, indices], delta_x[rows, indices])
    # Same wrap as _clamp_angle: (angle + pi) % (2pi) - pi.
    wrapped = np.mod(angle - headings + math.pi, 2.0 * math.pi) - math.pi
    return np.where(found, wrapped / math.pi, 0.0)


def _circle_ray_distances(observer_xy: np.ndarray, directions: np.ndarray,
                          centers: np.ndarray, radius: float,
                          valid: np.ndarray) -> np.ndarray:
    """
    Nearest circle-hit distance per observer and ray; MAX_RANGE when none.

    Mirrors _ray_circle_hit exactly: dot > 0 and squared perpendicular
    distance <= radius^2, reporting the centre's projection onto the ray.
    """
    count = observer_xy.shape[0]
    best = np.full((count, len(RAY_ANGLES_DEG)), MAX_RANGE, dtype=np.float64)
    if centers.shape[0] == 0:
        return best

    for start in range(0, count, _VECTOR_CHUNK_ROWS):
        end = min(start + _VECTOR_CHUNK_ROWS, count)
        delta_x = (centers[None, None, :, 0]
                   - observer_xy[start:end, None, None, 0])
        delta_y = (centers[None, None, :, 1]
                   - observer_xy[start:end, None, None, 1])
        ray = directions[start:end]
        dot = delta_x * ray[:, :, None, 0] + delta_y * ray[:, :, None, 1]
        closest_sq = delta_x * delta_x + delta_y * delta_y - dot * dot
        hit = ((dot > 0.0) & (closest_sq <= radius * radius)
               & valid[start:end, None, :])
        distances = np.where(hit, dot, np.inf)
        # Minimum over hits equals the scalar loop's strict '<' update: when
        # two hits tie the reported distance is the same either way.
        best[start:end] = np.minimum(best[start:end], distances.min(axis=2))
    return best


def _aabb_ray_distances(observer_xy: np.ndarray, directions: np.ndarray,
                        centers: np.ndarray, half_size: float,
                        valid: np.ndarray) -> np.ndarray:
    """
    Nearest square-body hit distance per observer and ray; MAX_RANGE if none.

    Slab method with the same ordering and boundary rules as _ray_aabb_hit,
    including a ray origin already inside the square (distance 0).
    """
    count = observer_xy.shape[0]
    best = np.full((count, len(RAY_ANGLES_DEG)), MAX_RANGE, dtype=np.float64)
    if centers.shape[0] == 0:
        return best

    for start in range(0, count, _VECTOR_CHUNK_ROWS):
        end = min(start + _VECTOR_CHUNK_ROWS, count)
        delta_x = (centers[None, None, :, 0]
                   - observer_xy[start:end, None, None, 0])
        delta_y = (centers[None, None, :, 1]
                   - observer_xy[start:end, None, None, 1])
        ray = directions[start:end]
        shape = (end - start, len(RAY_ANGLES_DEG), centers.shape[0])
        t_min = np.full(shape, -np.inf)
        t_max = np.full(shape, np.inf)
        missed = np.zeros(shape, dtype=bool)

        for axis in (0, 1):
            component = ray[:, :, axis][:, :, None]
            center = delta_x if axis == 0 else delta_y
            low = center - half_size
            high = center + half_size
            parallel = np.abs(component) < 1e-12
            inside = (low <= 0.0) & (0.0 <= high)
            missed |= parallel & ~inside
            # Substituting a non-zero denominator for parallel rays is safe:
            # those entries are masked out of t_min/t_max immediately below.
            safe = np.where(parallel, 1.0, component)
            first = low / safe
            second = high / safe
            swapped = first > second
            entry = np.where(swapped, second, first)
            exit_ = np.where(swapped, first, second)
            t_min = np.where(parallel, t_min, np.maximum(t_min, entry))
            t_max = np.where(parallel, t_max, np.minimum(t_max, exit_))
            missed |= t_min > t_max

        distances = np.where(t_max < 0.0, np.inf,
                             np.where(t_min > 0.0, t_min, 0.0))
        distances = np.where(missed | ~valid[start:end, None, :],
                             np.inf, distances)
        best[start:end] = np.minimum(best[start:end], distances.min(axis=2))
    return best


def _observe_population_vectorized(agents: list, world) -> np.ndarray:
    """
    Population-wide observation, equivalent to observe_agent for every agent.

    The scalar path is retained as the test oracle. Cost is O(rows x rays x
    objects) NumPy work per step instead of O(rows x rays x objects) Python
    loop iterations, which is the dominant cost of a GA generation.
    """
    count = len(agents)
    observations = np.zeros((count, OBSERVATION_DIM), dtype=np.float32)
    if count == 0:
        return observations

    live = [agent for agent in agents if agent.is_alive]
    live_count = len(live)
    observer_xy = np.array([[a.position.x, a.position.y] for a in agents],
                           dtype=np.float64).reshape(count, 2)
    headings = np.array([a.heading for a in agents], dtype=np.float64)
    velocities = np.array([[a.velocity.x, a.velocity.y] for a in agents],
                          dtype=np.float64).reshape(count, 2)
    roles = [getattr(a, "role", "prey") for a in agents]
    is_predator = np.array([role == "predator" for role in roles], dtype=bool)

    live_xy = np.array([[a.position.x, a.position.y] for a in live],
                       dtype=np.float64).reshape(live_count, 2)
    live_roles = [getattr(a, "role", "prey") for a in live]
    live_is_prey = np.array([r == "prey" for r in live_roles], dtype=bool)
    live_is_predator = ~live_is_prey

    food_xy = np.array([[f.x, f.y] for f in world.food],
                       dtype=np.float64).reshape(len(world.food), 2)
    obstacle_xy = np.array([[o.x, o.y] for o in world.obstacles],
                           dtype=np.float64).reshape(len(world.obstacles), 2)

    # Identity-based self exclusion matches "t is not agent" in the scalar
    # path, including the case where the observer itself is not in live.
    live_positions = {builtins.id(a): index for index, a in enumerate(live)}
    visible = np.ones((count, live_count), dtype=bool)
    for index, agent in enumerate(agents):
        own = live_positions.get(builtins.id(agent))
        if own is not None:
            visible[index, own] = False

    all_food_valid = np.ones((count, food_xy.shape[0]), dtype=bool)
    all_obstacle_valid = np.ones((count, obstacle_xy.shape[0]), dtype=bool)

    prey_food_bearing = _nearest_bearings(
        observer_xy, headings, food_xy, all_food_valid)
    predator_food_bearing = _nearest_bearings(
        observer_xy, headings, live_xy, visible & live_is_prey[None, :])
    prey_threat_bearing = _nearest_bearings(
        observer_xy, headings, live_xy, visible)
    predator_threat_bearing = _nearest_bearings(
        observer_xy, headings, live_xy, visible & live_is_predator[None, :])
    obstacle_bearing = _nearest_bearings(
        observer_xy, headings, obstacle_xy, all_obstacle_valid)

    bearings = np.column_stack([
        np.where(is_predator, predator_food_bearing, prey_food_bearing),
        np.where(is_predator, predator_threat_bearing, prey_threat_bearing),
        obstacle_bearing,
    ])
    observations[:, 7:10] = bearings.astype(np.float32)

    ray_angles = headings[:, None] + RAY_ANGLES_RAD[None, :]
    ray_directions = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=-1)

    nearest = np.full((count, len(RAY_ANGLES_DEG)), MAX_RANGE, dtype=np.float64)
    food_distance = _circle_ray_distances(
        observer_xy, ray_directions, food_xy, FOOD_RADIUS, all_food_valid)
    # A predator cannot perceive plants, so its food rays never see them.
    nearest = np.where(is_predator[:, None], nearest,
                       np.minimum(nearest, food_distance))
    obstacle_distance = _aabb_ray_distances(
        observer_xy, ray_directions, obstacle_xy, OBSTACLE_HALF_SIZE,
        all_obstacle_valid)
    nearest = np.minimum(nearest, obstacle_distance)
    agent_distance = _circle_ray_distances(
        observer_xy, ray_directions, live_xy, AGENT_RADIUS, visible)
    nearest = np.minimum(nearest, agent_distance)

    observations[:, 0:7] = np.where(
        nearest < MAX_RANGE, nearest / MAX_RANGE, 1.0).astype(np.float32)
    energies = np.array([a.energy for a in agents], dtype=np.float64)
    observations[:, 10] = np.clip(
        energies / INITIAL_ENERGY, 0.0, 1.0).astype(np.float32)
    speed = np.sqrt(velocities[:, 0] * velocities[:, 0]
                    + velocities[:, 1] * velocities[:, 1])
    observations[:, 11] = np.clip(speed / MAX_SPEED, 0.0, 1.0).astype(np.float32)
    return observations


class RayCaster:
    """
    Computes the fixed 12-dim observation for a population of agents.

    No state of its own; exists so appended pipeline concerns (e.g. cached
    per-ray bookkeeping for the dashboard in later phases) have a stable home.
    """

    def observe_agent(self, agent, world, live_agents: list) -> np.ndarray:
        """
        Return one agent's 12-dim observation vector.

        The observation is role-aware: prey see plants as food and every
        other live agent as a threat; predators see prey as food, rival
        predators as threats, and never see plants at all.

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

        # Role-aware perception: a predator cannot consume plants
        # (step_simulation blocks that), so plants are removed from its rays
        # and its food channel instead points at live prey. The threat
        # channel points at rival predators. Prey keep the original encoding
        # exactly, preserving the prey-only experiment artifacts.
        role = getattr(agent, "role", "prey")
        if role == "predator":
            food_targets: list = []
            food_candidates = [t for t in others
                               if getattr(t, "role", "prey") == "prey"]
            threat_candidates = [t for t in others
                                 if getattr(t, "role", "prey") == "predator"]
        else:
            food_targets = world.food
            food_candidates = world.food
            threat_candidates = others

        # Per-ray distances (indices 0..6)
        ray_dists: list[float] = []
        for offset_deg in RAY_ANGLES_DEG:
            ray_angle = agent.heading + math.radians(offset_deg)
            dir_x = math.cos(ray_angle)
            dir_y = math.sin(ray_angle)
            ray_dists.append(_raycast_ray(ax, ay, dir_x, dir_y, food_targets,
                                          world.obstacles, others))
        obs[0:7] = ray_dists

        # Global nearest bearings (indices 7..9)
        food_bearing = _bearing_to_nearest(ax, ay, agent.heading, food_candidates)
        threat_bearing = _bearing_to_nearest(ax, ay, agent.heading, threat_candidates)
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
        # Batched path is the production implementation; observe_agent stays
        # as the scalar oracle used by the differential equivalence tests.
        return _observe_population_vectorized(agents, world)
