"""
Tensor-native predator/prey ecosystem for high-generation experiments.

Unlike simulation.engine, this module keeps the whole population, food and
obstacles as torch tensors on one device. A generation compiles stacked
controller weights once and runs every step as batched tensor algebra, which
removes the per-step Python loops that capped the legacy engine. It is a new
experiment instrument, not a bit-exact replacement for simulation.engine:
its own configs and fitness scales govern its results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from evolution.genetic_algorithm import evolve_genomes
from neural.batched_inference import batched_forward, stack_population_weights
from neural.genome import genome_size
from simulation.geometry import OBSTACLE_HALF_SIZE
from simulation.world_config import BaselineConfig


@dataclass(frozen=True)
class EcosystemConfig:
    """All tensor-engine knobs; defaults mirror configs/predator_prey.json."""

    width: float = 600.0
    height: float = 400.0
    food_count: int = 80
    obstacle_count: int = 8
    layout_seed: int = 42
    food_regrowth: float = 0.25
    prey_size: int = 24
    predator_size: int = 6
    generations: int = 1000
    evaluation_steps: int = 150
    capture_radius: float = 30.0
    capture_transfer: float = 60.0
    consumption_radius: float = 12.0
    max_range: float = 200.0
    metabolic_cost: float = 0.25
    k_accel: float = 0.05
    k_steer: float = 0.02
    eaten_value: float = 50.0
    collision_penalty: float = 2.0
    max_energy: float = 100.0
    repro_enabled: bool = True
    repro_min_age: int = 160
    repro_min_energy: float = 30.0
    repro_cost: float = 20.0
    repro_interval: int = 20
    repro_sigma: float = 0.1
    max_prey: int = 40
    max_predator: int = 6
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7
    elitism_count: int = 2
    weights_prey: Optional[Dict[str, float]] = None
    weights_predator: Optional[Dict[str, float]] = None
    seed: int = 1

    @classmethod
    def from_config(cls, config: BaselineConfig) -> "EcosystemConfig":
        """Build tensor settings from a validated BaselineConfig."""
        predation = config.predation
        reproduction = config.reproduction
        energy = config.energy
        seed = config.seed.get("numpy") or cls.seed
        return cls(
            width=float(config.world.width),
            height=float(config.world.height),
            food_count=int(config.world.food_count),
            obstacle_count=int(config.world.obstacle_count),
            layout_seed=int(config.world.layout_seed or cls.layout_seed),
            food_regrowth=float(config.world.food_regrowth_per_step),
            prey_size=int(config.population.size),
            predator_size=int(predation.predator_count if predation else 0),
            generations=int(config.evolution.generations),
            evaluation_steps=int(config.evolution.evaluation_steps),
            capture_radius=float(predation.capture_radius if predation else 16.0),
            capture_transfer=float(
                predation.capture_energy_transfer if predation else 60.0),
            metabolic_cost=float(energy.metabolic_base_cost)
            if energy and energy.metabolic_base_cost else cls.metabolic_cost,
            repro_enabled=bool(reproduction.enabled) if reproduction else False,
            repro_min_age=int(reproduction.min_age) if reproduction else 160,
            repro_min_energy=float(reproduction.min_energy) if reproduction else 30.0,
            repro_cost=float(reproduction.reproduction_cost) if reproduction else 20.0,
            repro_sigma=float(reproduction.mutation_strength) if reproduction else 0.1,
            max_prey=int(reproduction.max_population_prey) if reproduction else 40,
            max_predator=int(reproduction.max_population_predator)
            if reproduction else 6,
            mutation_rate=float(config.evolution.mutation_rate),
            crossover_rate=float(config.evolution.crossover_rate),
            elitism_count=int(config.evolution.elitism_count),
            weights_prey=config.fitness_weights_prey,
            weights_predator=config.fitness_weights_predator,
            seed=int(seed),
        )


class TensorWorld:
    """Food and obstacle tensors plus per-step food regrowth."""

    def __init__(self, settings: EcosystemConfig, device: torch.device):
        generator = torch.Generator(device="cpu").manual_seed(settings.layout_seed)
        self.food = torch.rand(
            (settings.food_count, 2), generator=generator, dtype=torch.float32
        ) * torch.tensor([settings.width, settings.height]) + 10.0
        self.obstacles = torch.rand(
            (settings.obstacle_count, 2), generator=generator, dtype=torch.float32
        ) * torch.tensor([settings.width - 80.0, settings.height - 80.0]) + 40.0
        self.food_alive = torch.ones(settings.food_count, dtype=torch.bool,
                                     device=device)
        self.food = self.food.to(device)
        self.obstacles = self.obstacles.to(device)
        self.settings = settings

    def respawn_food(self) -> None:
        """Respawn consumed food at random legal positions."""
        missing = (~self.food_alive).nonzero(as_tuple=False).flatten()
        if missing.numel() == 0:
            return
        positions = torch.rand(
            (missing.numel(), 2), device=self.food.device, dtype=torch.float32
        ) * torch.tensor([self.settings.width - 20.0,
                          self.settings.height - 20.0],
                         device=self.food.device) + 10.0
        self.food[missing] = positions
        self.food_alive[missing] = True


class TensorAgents:
    """Fixed-size tensor population with dynamic alive mask."""

    def __init__(self, genomes: np.ndarray, roles: np.ndarray,
                 settings: EcosystemConfig, device: torch.device):
        count = len(genomes)
        generator = torch.Generator(device="cpu").manual_seed(settings.seed)
        positions = torch.rand((count, 2), generator=generator,
                               dtype=torch.float32) * torch.tensor(
                                   [settings.width - 40.0,
                                    settings.height - 40.0]) + 20.0
        headings = torch.rand(count, generator=generator,
                              dtype=torch.float32) * (2.0 * math.pi)
        self.genomes = torch.as_tensor(genomes, dtype=torch.float32,
                                       device=device)
        self.role = torch.as_tensor(roles, dtype=torch.int64, device=device)
        self.position = positions.to(device)
        self.spawn = self.position.clone()
        self.heading = headings.to(device)
        self.velocity = torch.zeros((count, 2), dtype=torch.float32,
                                    device=device)
        self.energy = torch.full((count,), settings.max_energy,
                                 dtype=torch.float32, device=device)
        self.alive = torch.ones(count, dtype=torch.bool, device=device)
        self.age = torch.zeros(count, dtype=torch.int64, device=device)
        self.food_eaten = torch.zeros(count, dtype=torch.int64, device=device)
        self.collisions = torch.zeros(count, dtype=torch.int64, device=device)
        self.last_eat = torch.zeros(count, dtype=torch.bool, device=device)
        self.last_steer = torch.zeros(count, dtype=torch.float32, device=device)
        self.last_accel = torch.zeros(count, dtype=torch.float32, device=device)

    @property
    def count(self) -> int:
        return int(self.role.numel())


def _ray_directions(heading: torch.Tensor,
                    settings: EcosystemConfig) -> torch.Tensor:
    """(N,7,2) unit vectors for the fixed ray fan."""
    offsets = torch.tensor([-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0],
                           dtype=torch.float32, device=heading.device)
    angles = heading[:, None] + torch.deg2rad(offsets)[None, :]
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def _circle_distances(origin: torch.Tensor, directions: torch.Tensor,
                      centers: torch.Tensor, radius: float,
                      valid: torch.Tensor) -> torch.Tensor:
    """Nearest circle-hit projection per observer and ray."""
    delta_x = centers[None, None, :, 0] - origin[:, None, None, 0]
    delta_y = centers[None, None, :, 1] - origin[:, None, None, 1]
    dot = delta_x * directions[:, :, None, 0] + delta_y * directions[:, :, None, 1]
    perpendicular_sq = delta_x * delta_x + delta_y * delta_y - dot * dot
    hit = ((dot > 0.0) & (perpendicular_sq <= radius * radius)
           & valid[:, None, :])
    distances = torch.where(hit, dot, torch.full_like(dot, math.inf))
    return distances.amin(dim=2)


def _aabb_distances(origin: torch.Tensor, directions: torch.Tensor,
                    centers: torch.Tensor, half_size: float,
                    valid: torch.Tensor) -> torch.Tensor:
    """Nearest square-body hit per observer and ray (slab method)."""
    delta_x = centers[None, None, :, 0] - origin[:, None, None, 0]
    delta_y = centers[None, None, :, 1] - origin[:, None, None, 1]
    shape = (origin.shape[0], directions.shape[1], centers.shape[0])
    t_min = torch.full(shape, -math.inf, device=origin.device)
    t_max = torch.full(shape, math.inf, device=origin.device)
    missed = torch.zeros(shape, dtype=torch.bool, device=origin.device)
    for axis in (0, 1):
        component = directions[:, :, axis][:, :, None]
        center = delta_x if axis == 0 else delta_y
        low = center - half_size
        high = center + half_size
        parallel = component.abs() < 1e-12
        missed |= parallel & ~((low <= 0.0) & (0.0 <= high))
        safe = torch.where(parallel, torch.ones_like(component), component)
        first = low / safe
        second = high / safe
        swapped = first > second
        entry = torch.where(swapped, second, first)
        exit_ = torch.where(swapped, first, second)
        t_min = torch.where(parallel, t_min, torch.maximum(t_min, entry))
        t_max = torch.where(parallel, t_max, torch.minimum(t_max, exit_))
        missed |= t_min > t_max
    distances = torch.where(t_max < 0.0, torch.full_like(t_max, math.inf),
                            torch.where(t_min > 0.0, t_min,
                                        torch.zeros_like(t_min)))
    distances = torch.where(missed | ~valid[:, None, :],
                            torch.full_like(distances, math.inf), distances)
    return distances.amin(dim=2)


def _nearest_bearing(origin: torch.Tensor, heading: torch.Tensor,
                     candidates: torch.Tensor,
                     valid: torch.Tensor) -> torch.Tensor:
    """Normalized signed bearing to the nearest valid candidate."""
    if candidates.shape[0] == 0:
        return torch.zeros(origin.shape[0], dtype=torch.float32,
                           device=origin.device)
    delta = candidates[None, :, :] - origin[:, None, :]
    distance_sq = torch.where(valid, (delta * delta).sum(dim=2),
                              torch.full_like(delta[:, :, 0], math.inf))
    indices = distance_sq.argmin(dim=1)
    rows = torch.arange(origin.shape[0], device=origin.device)
    selected = delta[rows, indices]
    angle = torch.atan2(selected[:, 1], selected[:, 0])
    wrapped = (angle - heading + math.pi) % (2.0 * math.pi) - math.pi
    found = torch.isfinite(distance_sq[rows, indices])
    return torch.where(found, wrapped / math.pi, torch.zeros_like(wrapped))


def observe(agents: TensorAgents, world: TensorWorld,
            settings: EcosystemConfig) -> torch.Tensor:
    """(N,12) role-aware observations, analogous to RayCaster."""
    visible = agents.alive[None, :].expand(agents.count, -1).clone()
    visible.fill_diagonal_(False)
    is_predator = agents.role == 1
    prey_role = agents.role == 0

    food_visible = torch.ones_like(world.food_alive)[None, :].expand(
        agents.count, -1)
    prey_food = _nearest_bearing(agents.position, agents.heading,
                                 world.food[world.food_alive],
                                 food_visible[:, world.food_alive])
    predator_food = _nearest_bearing(
        agents.position, agents.heading, agents.position,
        visible & prey_role[None, :])
    threat_prey = _nearest_bearing(agents.position, agents.heading,
                                   agents.position, visible)
    threat_predator = _nearest_bearing(
        agents.position, agents.heading, agents.position,
        visible & is_predator[None, :])
    obstacle_bearing = _nearest_bearing(
        agents.position, agents.heading, world.obstacles,
        torch.ones((agents.count, world.obstacles.shape[0]), dtype=torch.bool,
                   device=agents.position.device))

    directions = _ray_directions(agents.heading, settings)
    nearest = torch.full((agents.count, 7), settings.max_range,
                         dtype=torch.float32, device=agents.position.device)
    food_distance = _circle_distances(
        agents.position, directions, world.food[world.food_alive],
        2.0,
        food_visible[:, world.food_alive])
    nearest = torch.where(is_predator[:, None], nearest,
                          torch.minimum(nearest, food_distance))
    obstacle_distance = _aabb_distances(
        agents.position, directions, world.obstacles, OBSTACLE_HALF_SIZE,
        torch.ones((agents.count, world.obstacles.shape[0]), dtype=torch.bool,
                   device=agents.position.device))
    nearest = torch.minimum(nearest, obstacle_distance)
    agent_distance = _circle_distances(
        agents.position, directions, agents.position, 4.0, visible)
    nearest = torch.minimum(nearest, agent_distance)
    ray = torch.where(nearest < settings.max_range,
                      nearest / settings.max_range,
                      torch.ones_like(nearest))

    speed = agents.velocity.norm(dim=1)
    return torch.stack([
        *[ray[:, index] for index in range(7)],
        torch.where(is_predator, predator_food, prey_food),
        torch.where(is_predator, threat_predator, threat_prey),
        obstacle_bearing,
        agents.energy / settings.max_energy,
        speed / 5.0,
    ], dim=1).clamp(-1.0, 1.0)


def apply_actions(agents: TensorAgents, world: TensorWorld,
                  actions: torch.Tensor, settings: EcosystemConfig) -> None:
    """Batched motion, obstacle separation and contact penalties."""
    steer = actions[:, 0]
    accel = agents.alive.float() * actions[:, 1]
    agents.last_steer = steer
    agents.last_accel = accel
    agents.last_eat = agents.alive & (actions[:, 2] > 0.5)

    agents.heading = (agents.heading + steer * 0.1) % (2.0 * math.pi)
    agents.velocity = torch.stack([torch.cos(agents.heading),
                                   torch.sin(agents.heading)], dim=1) * (
                                       accel * 5.0)[:, None]
    agents.position = agents.position + agents.velocity
    margin = 10.0
    low = torch.tensor([margin, margin], device=agents.position.device)
    high = torch.tensor([settings.width - margin, settings.height - margin],
                        device=agents.position.device)
    outside_low = agents.position < low
    outside_high = agents.position > high
    agents.position = agents.position.clamp(min=low, max=high)
    agents.velocity = torch.where(outside_low | outside_high,
                                  torch.zeros_like(agents.velocity),
                                  agents.velocity)

    # Three separation passes, same least-penetration rule as the legacy engine.
    for _ in range(3):
        for obstacle in world.obstacles:
            delta = agents.position - obstacle
            overlap_x = (OBSTACLE_HALF_SIZE + 4.0) - delta[:, 0].abs()
            overlap_y = (OBSTACLE_HALF_SIZE + 4.0) - delta[:, 1].abs()
            along_x = (overlap_x <= overlap_y) & (overlap_x > 0.0) & (overlap_y > 0.0)
            along_y = (~along_x) & (overlap_y > 0.0) & (overlap_x > 0.0)
            direction_x = torch.where(delta[:, 0] >= 0.0, 1.0, -1.0)
            direction_y = torch.where(delta[:, 1] >= 0.0, 1.0, -1.0)
            agents.position[:, 0] += torch.where(along_x,
                                                 direction_x * overlap_x, 0.0)
            agents.position[:, 1] += torch.where(along_y,
                                                 direction_y * overlap_y, 0.0)
            agents.velocity[:, 0] = torch.where(along_x, 0.0,
                                                agents.velocity[:, 0])
            agents.velocity[:, 1] = torch.where(along_y, 0.0,
                                                agents.velocity[:, 1])
    agents.position = agents.position.clamp(min=low, max=high)

    # Contact count: obstacle overlap plus neighbour proximity.
    obstacle_delta = agents.position[:, None, :] - world.obstacles[None, :, :]
    overlap = ((obstacle_delta.abs() < (OBSTACLE_HALF_SIZE + 4.0)).all(dim=2))
    neighbour_delta = (agents.position[:, None, :]
                       - agents.position[None, :, :])
    # AABB overlap: both axes must overlap the 8 px agent body.
    neighbours = (neighbour_delta.abs() < 8.0).all(dim=2)
    # Obstacle contacts and neighbour contacts have different axis sizes;
    # sum each axis independently before adding the per-agent counts.
    contact_count = overlap.sum(dim=1) + neighbours.sum(dim=1)
    agents.collisions += contact_count
    agents.energy -= contact_count.float() * settings.collision_penalty


def consume_food(agents: TensorAgents, world: TensorWorld,
                 settings: EcosystemConfig) -> None:
    """First eligible prey eats each food item within consumption radius."""
    prey_index = (agents.alive & (agents.role == 0) & agents.last_eat
                  ).nonzero(as_tuple=False).flatten()
    if prey_index.numel() == 0 or world.food_alive.sum() == 0:
        return
    food_index = world.food_alive.nonzero(as_tuple=False).flatten()
    delta = (world.food[food_index][None, :, :]
             - agents.position[prey_index][:, None, :])
    distance_sq = (delta * delta).sum(dim=2)
    within = distance_sq < settings.consumption_radius ** 2
    # argmax on 0/1 gives the first eligible prey (lowest row wins ties).
    chosen_food = within.to(torch.int64).argmax(dim=0)
    row = torch.arange(food_index.numel(), device=distance_sq.device)
    hit = within[chosen_food, row]
    chosen_prey = prey_index[chosen_food]
    world.food_alive[food_index[row[hit]]] = False
    agents.food_eaten.index_add_(0, chosen_prey[hit],
                                 torch.ones(int(hit.sum()), dtype=torch.int64,
                                            device=hit.device))


def resolve_captures(agents: TensorAgents,
                     settings: EcosystemConfig) -> None:
    """Nearest live prey per attempting predator; first predator wins ties."""
    predator_index = (agents.alive & (agents.role == 1) & agents.last_eat
                      ).nonzero(as_tuple=False).flatten()
    prey_index = (agents.alive & (agents.role == 0)
                  ).nonzero(as_tuple=False).flatten()
    if predator_index.numel() == 0 or prey_index.numel() == 0:
        return
    delta = (agents.position[predator_index][:, None, :]
             - agents.position[prey_index][None, :, :])
    distance_sq = (delta * delta).sum(dim=2)
    within = distance_sq <= settings.capture_radius ** 2
    # First predator in row order wins when two predators tie on distance.
    chosen_prey = within.to(torch.int64).argmax(dim=1)
    row = torch.arange(predator_index.numel(), device=distance_sq.device)
    hit = within[row, chosen_prey]
    predator_row = predator_index[row[hit]]
    prey_row = prey_index[chosen_prey[hit]]
    agents.alive[prey_row] = False
    agents.energy[prey_row] = 0.0
    agents.energy[predator_row] = torch.minimum(
        agents.energy[predator_row] + settings.capture_transfer,
        torch.full_like(agents.energy[predator_row], settings.max_energy))
    agents.food_eaten[predator_row] += 1


def update_energy(agents: TensorAgents, settings: EcosystemConfig) -> None:
    """Vectorized energy equation and death check."""
    # Plant food only: predators gain energy through capture_transfer instead.
    eaten = (agents.last_eat & (agents.role == 0)).float() * settings.eaten_value
    cost = (settings.metabolic_cost
            + settings.k_accel * agents.last_accel ** 2
            + settings.k_steer * agents.last_steer.abs())
    agents.energy = (agents.energy - cost + eaten).clamp(0.0, settings.max_energy)
    agents.alive &= agents.energy > 0.0
    agents.age += agents.alive.long()


def reproduce(agents: TensorAgents, settings: EcosystemConfig) -> None:
    """In-episode asexual reproduction with per-role carrying capacity."""
    if not settings.repro_enabled:
        return
    for role, cap in ((0, settings.max_prey), (1, settings.max_predator)):
        candidates = ((agents.alive) & (agents.role == role)
                      & (agents.age >= settings.repro_min_age)
                      & (agents.age % settings.repro_interval == 0)
                      & (agents.energy >= settings.repro_min_energy)
                      ).nonzero(as_tuple=False).flatten()
        current = int(((agents.role == role) & agents.alive).sum())
        room = cap - current
        if room <= 0 or candidates.numel() == 0:
            continue
        parents = candidates[:room]
        agents.energy[parents] -= settings.repro_cost
        agents.alive[parents] &= agents.energy[parents] > 0.0
        survivors = parents[agents.alive[parents]]
        if survivors.numel() == 0:
            continue
        child_genomes = (agents.genomes[survivors]
                         + torch.randn_like(agents.genomes[survivors])
                         * settings.repro_sigma).clamp(-1.0, 1.0)
        # Spare slots reuse dead agents so tensor shapes stay fixed.
        dead = ((agents.role == role) & ~agents.alive
                ).nonzero(as_tuple=False).flatten()
        free = dead[:survivors.numel()]
        if free.numel() < survivors.numel():
            break
        agents.genomes[free] = child_genomes[:free.numel()]
        agents.position[free] = agents.position[survivors[:free.numel()]] + 1.0
        agents.spawn[free] = agents.position[free]
        agents.heading[free] = agents.heading[survivors[:free.numel()]]
        agents.velocity[free] = 0.0
        agents.energy[free] = settings.repro_min_energy
        agents.alive[free] = True
        agents.age[free] = 0
        agents.food_eaten[free] = 0
        agents.collisions[free] = 0
        agents.last_eat[free] = False


def _fitness(agents: TensorAgents, settings: EcosystemConfig,
             scales: Dict[str, float]) -> np.ndarray:
    """Per-agent normalized fitness with role weights and shaping bonuses."""
    role = agents.role.cpu().numpy()
    alive = agents.alive.cpu().numpy()
    age = agents.age.cpu().numpy()
    food = agents.food_eaten.cpu().numpy()
    collision = agents.collisions.cpu().numpy()
    position = agents.position.cpu().numpy()
    spawn = agents.spawn.cpu().numpy()
    exploration = np.linalg.norm(position - spawn, axis=1)
    distance = np.linalg.norm(
        position[:, None, :] - position[None, :, :], axis=2)
    max_distance = float(np.hypot(settings.width, settings.height))
    fitness = np.zeros(len(role), dtype=np.float64)
    for index in range(len(role)):
        weights = (settings.weights_predator if role[index] == 1
                   else settings.weights_prey)
        weights = weights or {"survival": 0.4, "food": 0.3,
                              "exploration": 0.2, "collision": 0.1}
        value = (weights["survival"] * age[index] / max(scales["survival"], 1.0)
                 + weights["food"] * food[index] / max(scales["food"], 1.0)
                 + weights["exploration"] * exploration[index]
                 / max(scales["exploration"], 1.0)
                 - weights["collision"] * collision[index]
                 / max(scales["collision"], 1.0))
        if role[index] == 0:
            others = distance[index][(role == 1) & alive]
            value += 0.30 * (others.mean() / max_distance) if others.size else 0.0
        else:
            prey = distance[index][(role == 0) & alive]
            value += 0.35 * (1.0 - prey.min() / 200.0) if prey.size else 0.0
        fitness[index] = max(value, 0.0)
    return fitness


class TensorEcosystem:
    """Runs tensor-native generations and reports population dynamics."""

    def __init__(self, settings: EcosystemConfig, device: torch.device,
                 scales: Dict[str, float]):
        self.settings = settings
        self.device = device
        self.scales = scales
        self.generator = np.random.default_rng(settings.seed)

    def run_generation(self, prey_genomes: np.ndarray,
                       predator_genomes: np.ndarray, generation: int
                       ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        settings = self.settings
        genomes = np.concatenate([prey_genomes, predator_genomes], axis=0)
        roles = np.concatenate([
            np.zeros(len(prey_genomes), dtype=np.int64),
            np.ones(len(predator_genomes), dtype=np.int64)])
        agents = TensorAgents(genomes, roles, settings, self.device)
        world = TensorWorld(settings, self.device)
        weights = stack_population_weights(agents.genomes,
                                           device=self.device.type)

        weights_stale = False
        prey_steps = 0
        predator_steps = 0
        executed_steps = 0
        for step in range(settings.evaluation_steps):
            if not bool(agents.alive.any()):
                break
            executed_steps += 1
            prey_steps += int(((agents.alive) & (agents.role == 0)).sum())
            predator_steps += int(((agents.alive) & (agents.role == 1)).sum())
            if weights_stale:
                # Reproduced children inherit new genomes; rebuild stacked
                # weights only on steps where the population actually changed.
                weights = stack_population_weights(agents.genomes,
                                                   device=self.device.type)
                weights_stale = False
            observations = observe(agents, world, settings)
            actions = batched_forward(observations, weights)
            actions = actions.detach()
            apply_actions(agents, world, actions, settings)
            consume_food(agents, world, settings)
            resolve_captures(agents, settings)
            update_energy(agents, settings)
            before_reproduction = agents.alive.clone()
            reproduce(agents, settings)
            if not torch.equal(before_reproduction, agents.alive):
                weights_stale = True
            if settings.food_regrowth > 0.0 and self.generator.random() < settings.food_regrowth:
                world.respawn_food()

        fitness = _fitness(agents, settings, self.scales)
        prey_alive = int(((agents.role == 0) & agents.alive).sum())
        predator_alive = int(((agents.role == 1) & agents.alive).sum())
        metrics = {
            "generation": generation,
            "prey_alive": prey_alive,
            "predator_alive": predator_alive,
            "prey_alive_step_mean": round(prey_steps / max(executed_steps, 1), 2),
            "predator_alive_step_mean": round(
                predator_steps / max(executed_steps, 1), 2),
            "prey_fitness_mean": float(
                fitness[agents.role.cpu().numpy() == 0].mean())
            if prey_genomes.size else 0.0,
            "predator_fitness_mean": float(
                fitness[agents.role.cpu().numpy() == 1].mean())
            if predator_genomes.size else 0.0,
            "prey_food": int(agents.food_eaten[agents.role == 0].sum()),
            "captures": int(agents.food_eaten[agents.role == 1].sum()),
        }
        prey_fitness = fitness[:len(prey_genomes)]
        predator_fitness = fitness[len(prey_genomes):]

        settings_for_ga = self._ga_settings(len(prey_genomes))
        next_prey = evolve_genomes(list(prey_genomes), prey_fitness,
                                   settings_for_ga, self.generator)
        next_predator = evolve_genomes(list(predator_genomes), predator_fitness,
                                       settings_for_ga, self.generator)
        return (np.stack(next_prey), np.stack(next_predator), metrics)

    def _ga_settings(self, population_size: int) -> BaselineConfig:
        """Build the config shape evolve_genomes expects."""
        data = {
            "world": {"width": int(self.settings.width),
                      "height": int(self.settings.height)},
            "population": {"size": population_size, "agent": {
                "sensors": 7, "brain": {"architecture": [12, 32, 16, 3]}}},
            "evolution": {"generations": self.settings.generations,
                          "mutation_rate": self.settings.mutation_rate,
                          "crossover_rate": self.settings.crossover_rate,
                          "elitism_count": self.settings.elitism_count},
            "experiment": {"seeds": 1, "save_interval": 50},
        }
        return BaselineConfig(**data)

    def train(self, generations: Optional[int] = None
              ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Train both roles and return final genomes plus generation metrics."""
        settings = self.settings
        total = generations or settings.generations
        size = genome_size()
        prey = self.generator.uniform(-1.0, 1.0,
                                      (settings.prey_size, size)).astype(np.float32)
        predators = self.generator.uniform(
            -1.0, 1.0, (settings.predator_size, size)).astype(np.float32)
        history: List[Dict] = []
        for generation in range(total):
            prey, predators, metrics = self.run_generation(
                prey, predators, generation)
            history.append(metrics)
        return prey, predators, history
