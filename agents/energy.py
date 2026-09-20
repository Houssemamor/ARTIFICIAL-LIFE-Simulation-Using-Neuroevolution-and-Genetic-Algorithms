"""
Energy balance module for Artificial Life Neuroevolution Simulation.

Implements the energy equation from PLAN.md Phase 3:
    energy[t+1] = energy[t] - metabolic_base_cost
                  - k_accel * accel^2
                  - k_steer * |steering|
                  + eaten_energy_value * eat_event

Death condition: energy <= 0 or health <= 0
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EnergyConfig:
    """
    Energy balance parameters.

    All costs are per simulation step (dt=1.0). The defaults match the
    design document's Phase 3 calibration targets.
    """
    metabolic_base_cost: float = 0.1
    k_accel: float = 0.05
    k_steer: float = 0.02
    eaten_energy_value: float = 50.0
    collision_penalty: float = 2.0
    death_energy_threshold: float = 0.0
    death_health_threshold: float = 0.0
    max_energy: float = 100.0


DEFAULT_ENERGY_CONFIG = EnergyConfig()


def update_energy(
    energy: float,
    health: float,
    acceleration: float,
    steering: float,
    eat_event: bool,
    collision_penalty_applied: float = 0.0,
    config: Optional[EnergyConfig] = None,
) -> tuple[float, float, bool]:
    """
    Compute next-step energy, health, and death state.

    Args:
        energy: Current energy level.
        health: Current health level (0-100).
        acceleration: Network acceleration output in [0, 1].
        steering: Network steering output in [-1, 1].
        eat_event: Whether the agent consumed food this step.
        collision_penalty_applied: Energy penalty from collisions this step.
        config: EnergyConfig with tunable parameters.

    Returns:
        Tuple of (new_energy, new_health, is_dead).
    """
    if config is None:
        config = DEFAULT_ENERGY_CONFIG

    new_energy = energy
    new_health = health

    new_energy -= config.metabolic_base_cost
    new_energy -= config.k_accel * (acceleration ** 2)
    new_energy -= config.k_steer * abs(steering)

    if eat_event:
        new_energy += config.eaten_energy_value

    new_energy -= collision_penalty_applied

    new_energy = max(new_energy, 0.0)
    new_energy = min(new_energy, config.max_energy)

    is_dead = (
        new_energy <= config.death_energy_threshold
        or new_health <= config.death_health_threshold
    )

    return new_energy, new_health, is_dead


def compute_energy_cost(
    acceleration: float,
    steering: float,
    config: Optional[EnergyConfig] = None,
) -> float:
    """
    Compute the action energy cost for a single step.

    Used for fitness calibration to measure baseline action costs.

    Args:
        acceleration: Network acceleration output in [0, 1].
        steering: Network steering output in [-1, 1].
        config: EnergyConfig with tunable parameters.

    Returns:
        Total action cost (excluding base metabolism and collision penalty).
    """
    if config is None:
        config = DEFAULT_ENERGY_CONFIG

    return (
        config.k_accel * (acceleration ** 2)
        + config.k_steer * abs(steering)
    )