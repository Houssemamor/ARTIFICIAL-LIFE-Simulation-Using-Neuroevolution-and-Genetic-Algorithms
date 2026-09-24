"""
Unit tests for fitness normalization (Phase 3).

Verifies that raw fitness components are correctly normalized by
calibration reference scales and combined with configurable weights.
"""

from __future__ import annotations

from evolution.genetic_algorithm import compute_fitness
from agents.organism import Organism


class TestFitnessNormalization:
    def test_basic_normalization(self) -> None:
        calibration = {"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        weights = {"survival": 0.4, "food": 0.3, "exploration": 0.2, "collision": 0.1}

        agent = Organism(0, 0.0, 0.0)
        fitness = compute_fitness(agent, steps_survived=100, food_eaten=10,
                                  exploration_distance=50.0, collisions=5,
                                  calibration_scales=calibration, fitness_weights=weights)
        # survival: 100/100=1.0, food: 10/10=1.0, expl: 50/50=1.0, coll: 5/5=1.0
        # fitness = 0.4*1 + 0.3*1 + 0.2*1 - 0.1*1 = 0.8
        assert abs(fitness - 0.8) < 1e-6

    def test_zero_raw_scores(self) -> None:
        calibration = {"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        weights = {"survival": 0.4, "food": 0.3, "exploration": 0.2, "collision": 0.1}

        agent = Organism(0, 0.0, 0.0)
        fitness = compute_fitness(agent, steps_survived=0, food_eaten=0,
                                  exploration_distance=0.0, collisions=0,
                                  calibration_scales=calibration, fitness_weights=weights)
        assert fitness == 0.0

    def test_food_heavy_weights(self) -> None:
        calibration = {"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        weights = {"survival": 0.1, "food": 0.8, "exploration": 0.05, "collision": 0.05}

        agent = Organism(0, 0.0, 0.0)
        fitness = compute_fitness(agent, steps_survived=50, food_eaten=10,
                                  exploration_distance=25.0, collisions=2,
                                  calibration_scales=calibration, fitness_weights=weights)
        # survival: 0.5, food: 1.0, expl: 0.5, coll: 0.4
        expected = 0.1*0.5 + 0.8*1.0 + 0.05*0.5 - 0.05*0.4
        assert abs(fitness - expected) < 1e-6

    def test_collision_penalty_subtracted(self) -> None:
        calibration = {"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        weights = {"survival": 0.4, "food": 0.3, "exploration": 0.2, "collision": 0.1}

        agent = Organism(0, 0.0, 0.0)
        fitness_no_coll = compute_fitness(agent, steps_survived=100, food_eaten=10,
                                          exploration_distance=50.0, collisions=0,
                                          calibration_scales=calibration, fitness_weights=weights)
        fitness_with_coll = compute_fitness(agent, steps_survived=100, food_eaten=10,
                                            exploration_distance=50.0, collisions=10,
                                            calibration_scales=calibration, fitness_weights=weights)
        assert fitness_with_coll < fitness_no_coll
        assert fitness_with_coll >= 0.0  # floor at zero

    def test_fitness_floor_at_zero(self) -> None:
        calibration = {"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        weights = {"survival": 0.1, "food": 0.1, "exploration": 0.1, "collision": 0.7}

        agent = Organism(0, 0.0, 0.0)
        fitness = compute_fitness(agent, steps_survived=10, food_eaten=1,
                                  exploration_distance=10.0, collisions=20,
                                  calibration_scales=calibration, fitness_weights=weights)
        assert fitness == 0.0

    def test_calibration_scale_prevents_division_by_zero(self) -> None:
        calibration = {"survival": 0.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        weights = {"survival": 0.4, "food": 0.3, "exploration": 0.2, "collision": 0.1}

        agent = Organism(0, 0.0, 0.0)
        fitness = compute_fitness(agent, steps_survived=100, food_eaten=10,
                                  exploration_distance=50.0, collisions=5,
                                  calibration_scales=calibration, fitness_weights=weights)
        # survival ref is 1.0 (max with 1.0 in compute_fitness)
        assert fitness >= 0.0

    def test_weights_sum_to_one_not_enforced_but_works(self) -> None:
        calibration = {"survival": 100.0, "food": 10.0, "exploration": 50.0, "collision": 5.0}
        weights = {"survival": 0.5, "food": 0.5, "exploration": 0.5, "collision": 0.5}

        agent = Organism(0, 0.0, 0.0)
        fitness = compute_fitness(agent, steps_survived=100, food_eaten=10,
                                  exploration_distance=50.0, collisions=5,
                                  calibration_scales=calibration, fitness_weights=weights)
        expected = 0.5*(1+1+1-1)  # = 1.0
        assert abs(fitness - expected) < 1e-6