"""
Replay recorder for the showcase dashboard.

Runs the production simulation step (`step_simulation` plus
`resolve_captures`) unchanged and snapshots every step into plain
JSON-serialisable data, together with the full layer-by-layer activations
of chosen "featured" agents. The browser only plays the recording back;
it never re-implements physics, sensing or the controller, so what the
viewer sees is exactly what the Python simulation did.

Observation of frame t is the sensor reading taken at the START of step t,
so each frame stores the pre-step state together with the activations
computed from it (rays drawn from the agent line up with its position).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from agents.organism import Organism
from agents.sensors import MAX_RANGE, RAY_ANGLES_DEG, RayCaster
from neural.genome import unpack_genome
from neural.network import ARCHITECTURE
from simulation.engine import resolve_captures, step_simulation
from simulation.environment import World

# Labels for the fixed 12-value observation (order is part of the genome
# compatibility contract, see agents/sensors.py). The food and threat
# channels are role-aware: prey read plants and any other agent, while
# predators read live prey and rival predators and never see plants.
INPUT_LABELS: List[str] = (
    [f"ray {int(angle)}" for angle in RAY_ANGLES_DEG]
    + ["food bearing", "threat bearing", "obstacle bearing", "energy", "speed"]
)
PREDATOR_INPUT_LABELS: List[str] = (
    [f"ray {int(angle)}" for angle in RAY_ANGLES_DEG]
    + ["prey bearing", "rival bearing", "obstacle bearing", "energy", "speed"]
)
INPUT_LABELS_BY_ROLE: Dict[str, List[str]] = {
    "prey": INPUT_LABELS,
    "predator": PREDATOR_INPUT_LABELS,
}
OUTPUT_LABELS: List[str] = ["steer", "accelerate", "eat / capture"]

# Obstacles are drawn as squares of this side length (see the renderer).
OBSTACLE_SIZE = 10


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def forward_trace(genome: np.ndarray, observation: Sequence[float]) -> Dict[str, List[float]]:
    """
    Layer-by-layer forward pass in numpy for ONE genome.

    Mirrors neural.batched_inference.batched_forward (ReLU hidden layers,
    tanh steering, sigmoid acceleration and eat gate); a unit test checks
    the two agree. Returned values are rounded so the recording stays small.
    """
    weights, biases = unpack_genome(genome)
    inputs = np.asarray(observation, dtype=np.float32)
    hidden1 = np.maximum(inputs @ weights[0] + biases[0], 0.0)
    hidden2 = np.maximum(hidden1 @ weights[1] + biases[1], 0.0)
    raw = hidden2 @ weights[2] + biases[2]
    outputs = np.array([np.tanh(raw[0]), _sigmoid(raw[1]), _sigmoid(raw[2])],
                       dtype=np.float32)
    return {
        "in": [round(float(v), 3) for v in inputs],
        "h1": [round(float(v), 2) for v in hidden1],
        "h2": [round(float(v), 2) for v in hidden2],
        "out": [round(float(v), 3) for v in outputs],
    }


def network_snapshot(genome: np.ndarray) -> Dict[str, List]:
    """Weights and biases for the network map, rounded for size."""
    weights, biases = unpack_genome(genome)
    return {
        "weights": [np.round(w, 3).tolist() for w in weights],
        "biases": [np.round(b, 3).tolist() for b in biases],
    }


def _snapshot_agents(agents: Sequence[Organism]) -> List[List[float]]:
    """[x, y, heading, energy, alive, eaten] per agent, list order."""
    return [[round(float(a.position.x), 1), round(float(a.position.y), 1),
             round(float(a.heading), 2), int(round(a.energy)),
             1 if a.is_alive else 0, int(a.food_eaten)]
            for a in agents]


def _snapshot_food(world: World) -> List[int]:
    """Flat [x0, y0, x1, y1, ...] with integer pixels."""
    flat: List[int] = []
    for food in world.food:
        flat.append(int(round(food.x)))
        flat.append(int(round(food.y)))
    return flat


def world_description(world: World) -> Dict:
    """Static arena geometry shared by every frame of a scenario."""
    return {
        "width": world.width,
        "height": world.height,
        "obstacles": [[int(round(o.x)), int(round(o.y))]
                      for o in world.obstacles],
        "obstacle_size": OBSTACLE_SIZE,
    }


def record_replay(world: World, agents: List[Organism],
                  featured: Dict[str, Organism], steps: int,
                  capture_radius: Optional[float] = None,
                  capture_energy_transfer: Optional[float] = None,
                  on_step: Optional[Callable[[int], None]] = None) -> Dict:
    """
    Run `steps` simulation steps and record every one.

    Args:
        world: Built world (food and obstacles placed).
        agents: All agents, in the order the viewer will index them.
        featured: role name -> the agent whose activations are recorded.
        steps: Maximum number of steps (stops early if everyone dies).
        capture_radius: Enables predator capture resolution when set.
        capture_energy_transfer: Energy a predator gains per capture.
        on_step: Optional progress callback.

    Returns:
        {"frames", "networks", "summary"} - plain JSON-serialisable data.
    """
    raycaster = RayCaster()
    frames: List[Dict] = []
    previous_food: Optional[List[int]] = None
    captures_total = 0
    steps_run = 0

    for step in range(steps):
        if not any(agent.is_alive for agent in agents):
            break
        frame: Dict = {"a": _snapshot_agents(agents)}
        food = _snapshot_food(world)
        if food != previous_food:
            # Food is only written when it changed; the viewer keeps the
            # last list otherwise (regrowth and eating are infrequent).
            frame["f"] = food
            previous_food = food
        alive_before = {role: agent.is_alive for role, agent in featured.items()}

        step_simulation(world, agents, raycaster)
        if capture_radius is not None:
            captures_total += resolve_captures(
                agents, capture_radius=capture_radius,
                energy_transfer=capture_energy_transfer)

        traces = {}
        for role, agent in featured.items():
            # Only agents alive at the start of the step were observed.
            if alive_before[role] and agent.last_observation is not None:
                traces[role] = forward_trace(agent.genome, agent.last_observation)
        if traces:
            frame["n"] = traces
        frames.append(frame)
        steps_run += 1
        if on_step is not None:
            on_step(step)

    # Final state, no activations (nothing was observed after the last step).
    final_frame: Dict = {"a": _snapshot_agents(agents)}
    final_food = _snapshot_food(world)
    if final_food != previous_food:
        final_frame["f"] = final_food
    frames.append(final_frame)

    prey = [a for a in agents if a.role == "prey"]
    predators = [a for a in agents if a.role == "predator"]
    summary = {
        "steps": steps_run,
        "prey_alive_end": sum(1 for a in prey if a.is_alive),
        "predator_alive_end": sum(1 for a in predators if a.is_alive),
        "prey_total": len(prey),
        "predator_total": len(predators),
        "captures": captures_total,
        "prey_food_eaten": sum(a.food_eaten for a in prey),
        "featured_alive_end": {role: bool(agent.is_alive)
                               for role, agent in featured.items()},
        "featured_eaten": {role: int(agent.food_eaten)
                           for role, agent in featured.items()},
    }
    return {
        "frames": frames,
        "networks": {role: network_snapshot(agent.genome)
                     for role, agent in featured.items()},
        "summary": summary,
    }


def pick_featured(agents: Sequence[Organism]) -> Dict[str, int]:
    """
    Index of the best performer per role after a finished run.

    Rule (fixed up front, not tuned per recording): most food eaten
    (captures for predators), then alive at the end, then lowest index.
    """
    best: Dict[str, int] = {}
    for index, agent in enumerate(agents):
        current = best.get(agent.role)
        if current is None:
            best[agent.role] = index
            continue
        incumbent = agents[current]
        if ((agent.food_eaten, agent.is_alive)
                > (incumbent.food_eaten, incumbent.is_alive)):
            best[agent.role] = index
    return best


def record_best_featured(prepare: Callable[[], tuple], steps: int,
                         capture_radius: Optional[float] = None,
                         capture_energy_transfer: Optional[float] = None
                         ) -> Dict:
    """
    Replay twice from identical state: a scouting pass picks the featured
    agent per role (see pick_featured), the second pass records it.

    Args:
        prepare: Zero-argument callable returning a freshly built
            (world, agents) pair with ALL random state re-pinned, so both
            passes are identical (replays are deterministic).
    """
    world, agents = prepare()
    record_replay(world, agents, {}, steps, capture_radius,
                  capture_energy_transfer)
    chosen = pick_featured(agents)

    world, agents = prepare()
    featured = {role: agents[index] for role, index in chosen.items()}
    run = record_replay(world, agents, featured, steps, capture_radius,
                        capture_energy_transfer)
    run["featured"] = chosen
    run["world"] = world_description(world)
    run["agents"] = [{"id": a.id, "role": a.role} for a in agents]
    return run


def recording_constants() -> Dict:
    """Constants the viewer needs to draw rays and label the network."""
    return {
        "architecture": list(ARCHITECTURE),
        "ray_angles_deg": list(RAY_ANGLES_DEG),
        "max_range": MAX_RANGE,
        "input_labels": INPUT_LABELS,
        "input_labels_by_role": INPUT_LABELS_BY_ROLE,
        "output_labels": OUTPUT_LABELS,
    }
