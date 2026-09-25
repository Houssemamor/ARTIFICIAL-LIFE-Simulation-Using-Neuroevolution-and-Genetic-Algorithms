"""
Energy balance module for Artificial Life Neuroevolution Simulation.

Implements the energy equation from PLAN.md Phase 3:
    energy[t+1] = energy[t] - metabolic_base_cost
                  - k_accel * accel^2
                  - k_steer * |steering|
                  + eaten_energy_value * eat_event
                  - collision_penalty

Death condition: energy <= 0 or health <= 0

Documented deviation from the plan equation: energy is clamped to
[0, max_energy] with max_energy = 100 (the initial energy). Without the
upper clamp, an agent eating at full energy could bank unbounded reserves
and the death condition would never bind again; the clamp keeps food
valuable only when hungry. If the design doc is revised to allow energy
banking, change max_energy here and the tests in
tests/unit/test_energy_balance.py.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
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


def energy_config_from_settings(settings) -> EnergyConfig:
    """
    Build an EnergyConfig from a config's optional energy overrides.

    None means "keep the default", so a config that only sets
    metabolic_base_cost gets exactly that change and nothing else.

    Args:
        settings: An EnergySettings model (or None).

    Returns:
        The resolved EnergyConfig.
    """
    if settings is None:
        return DEFAULT_ENERGY_CONFIG
    # exclude_none keeps the "None means keep the default" contract in
    # one expression; model_dump avoids the deprecated instance access
    # of model_fields.
    overrides = settings.model_dump(exclude_none=True)
    return replace(DEFAULT_ENERGY_CONFIG, **overrides)


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