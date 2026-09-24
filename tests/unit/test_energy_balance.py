"""
Unit tests for the energy balance model (Phase 3).

Verifies the energy equation:
    energy[t+1] = energy[t] - metabolic_base_cost
                  - k_accel * accel^2
                  - k_steer * |steering|
                  + eaten_energy_value * eat_event
                  - collision_penalty
"""

from __future__ import annotations

from agents.energy import EnergyConfig, update_energy, compute_energy_cost, DEFAULT_ENERGY_CONFIG


class TestEnergyConfig:
    def test_defaults_match_design_doc(self) -> None:
        cfg = EnergyConfig()
        assert cfg.metabolic_base_cost == 0.1
        assert cfg.k_accel == 0.05
        assert cfg.k_steer == 0.02
        assert cfg.eaten_energy_value == 50.0
        assert cfg.collision_penalty == 2.0


class TestEnergyCost:
    def test_zero_actions_zero_cost(self) -> None:
        cost = compute_energy_cost(acceleration=0.0, steering=0.0)
        assert cost == 0.0

    def test_acceleration_cost_quadratic(self) -> None:
        cost_low = compute_energy_cost(acceleration=0.5, steering=0.0)
        cost_high = compute_energy_cost(acceleration=1.0, steering=0.0)
        assert cost_high == 4 * cost_low  # (1.0/0.5)^2 = 4

    def test_steering_cost_linear(self) -> None:
        cost_low = compute_energy_cost(acceleration=0.0, steering=0.5)
        cost_high = compute_energy_cost(acceleration=0.0, steering=1.0)
        assert cost_high == 2 * cost_low

    def test_both_actions_sum(self) -> None:
        cost_accel = compute_energy_cost(acceleration=1.0, steering=0.0)
        cost_steer = compute_energy_cost(acceleration=0.0, steering=1.0)
        cost_both = compute_energy_cost(acceleration=1.0, steering=1.0)
        assert abs(cost_both - (cost_accel + cost_steer)) < 1e-6


class TestUpdateEnergy:
    def test_base_metabolism_only(self) -> None:
        new_e, new_h, dead = update_energy(
            energy=100.0, health=100.0,
            acceleration=0.0, steering=0.0,
            eat_event=False, collision_penalty_applied=0.0,
        )
        expected = 100.0 - DEFAULT_ENERGY_CONFIG.metabolic_base_cost
        assert abs(new_e - expected) < 1e-6
        assert new_h == 100.0
        assert not dead

    def test_acceleration_cost(self) -> None:
        new_e, _, _ = update_energy(
            energy=100.0, health=100.0,
            acceleration=1.0, steering=0.0,
            eat_event=False, collision_penalty_applied=0.0,
        )
        expected = 100.0 - 0.1 - 0.05 * 1.0**2
        assert abs(new_e - expected) < 1e-6

    def test_steering_cost(self) -> None:
        new_e, _, _ = update_energy(
            energy=100.0, health=100.0,
            acceleration=0.0, steering=1.0,
            eat_event=False, collision_penalty_applied=0.0,
        )
        expected = 100.0 - 0.1 - 0.02 * 1.0
        assert abs(new_e - expected) < 1e-6

    def test_eat_event_gain(self) -> None:
        new_e, _, _ = update_energy(
            energy=50.0, health=100.0,
            acceleration=0.0, steering=0.0,
            eat_event=True, collision_penalty_applied=0.0,
        )
        expected = 50.0 - 0.1 + 50.0
        assert abs(new_e - expected) < 1e-6

    def test_collision_penalty(self) -> None:
        new_e, _, _ = update_energy(
            energy=100.0, health=100.0,
            acceleration=0.0, steering=0.0,
            eat_event=False, collision_penalty_applied=2.0,
        )
        expected = 100.0 - 0.1 - 2.0
        assert abs(new_e - expected) < 1e-6

    def test_death_at_zero_energy(self) -> None:
        new_e, _, dead = update_energy(
            energy=0.1, health=100.0,
            acceleration=0.0, steering=0.0,
            eat_event=False, collision_penalty_applied=0.0,
        )
        assert dead
        assert new_e == 0.0

    def test_death_at_negative_energy(self) -> None:
        new_e, _, dead = update_energy(
            energy=1.0, health=100.0,
            acceleration=0.0, steering=0.0,
            eat_event=False, collision_penalty_applied=5.0,
        )
        assert dead
        assert new_e == 0.0

    def test_energy_clamped_to_max(self) -> None:
        new_e, _, _ = update_energy(
            energy=90.0, health=100.0,
            acceleration=0.0, steering=0.0,
            eat_event=True, collision_penalty_applied=0.0,
        )
        assert new_e == 100.0  # max_energy

    def test_health_threshold_death(self) -> None:
        new_e, new_h, dead = update_energy(
            energy=100.0, health=0.0,
            acceleration=0.0, steering=0.0,
            eat_event=False, collision_penalty_applied=0.0,
        )
        assert dead

    def test_custom_config(self) -> None:
        cfg = EnergyConfig(metabolic_base_cost=0.5, k_accel=0.1, k_steer=0.1)
        new_e, _, _ = update_energy(
            energy=100.0, health=100.0,
            acceleration=1.0, steering=1.0,
            eat_event=False, collision_penalty_applied=0.0,
            config=cfg,
        )
        expected = 100.0 - 0.5 - 0.1 - 0.1
        assert abs(new_e - expected) < 1e-6