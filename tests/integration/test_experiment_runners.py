"""
Integration test for the Phase 5 experiment runners (smoke, tiny params).

Exercises the reusable training/frozen-evaluation primitives and the full
run_generalization pipeline (configs -> training -> freeze -> evaluate ->
statistics) end-to-end in a few seconds.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from simulation.world_config import BaselineConfig
from experiments.experiment_runner import (evaluate_genome,
                                           make_clone_population,
                                           make_population,
                                           train_best_genome)
from experiments.run_generalization import run_generalization


def _tiny_config(layout_seed: int, generations: int = 2,
                 population: int = 6, steps: int = 20) -> BaselineConfig:
    return BaselineConfig(
        experiment_name="tiny",
        mode="headless",
        device="cpu",
        reproducibility_tier="cpu-deterministic",
        seed={"torch": None, "numpy": None, "random": None},
        world={"width": 300, "height": 300, "food_count": 5,
               "obstacle_count": 3, "layout_seed": layout_seed},
        population={"size": population,
                    "agent": {"sensors": 7,
                              "brain": {"architecture": [12, 32, 16, 3]}}},
        evolution={"generations": generations, "mutation_rate": 0.05,
                   "crossover_rate": 0.7, "elitism_count": 1,
                   "evaluation_steps": steps},
        experiment={"seeds": 1, "save_interval": 1},
        crossover_method="blend",
        fitness_weights={"survival": 0.4, "food": 0.3,
                         "exploration": 0.2, "collision": 0.1},
    )


class TestExperimentPrimitives:
    def test_train_best_genome_returns_genome_and_history(self) -> None:
        config = _tiny_config(layout_seed=1)
        genome, history = train_best_genome(config, run_seed=0)
        assert genome.ndim == 1
        assert genome.size == 995
        assert len(history) == config.evolution.generations
        assert history[0]["generation"] == 0

    def test_training_is_reproducible(self) -> None:
        config = _tiny_config(layout_seed=1)
        genome_a, _ = train_best_genome(config, run_seed=7)
        genome_b, _ = train_best_genome(config, run_seed=7)
        assert np.array_equal(genome_a, genome_b)

    def test_evaluate_genome_returns_metrics(self) -> None:
        config = _tiny_config(layout_seed=1)
        genome, _ = train_best_genome(config, run_seed=0)
        result = evaluate_genome(config, genome, run_seed=99)
        assert "mean_fitness" in result
        assert "mean_food" in result
        assert result["mean_fitness"] >= 0.0

    def test_clone_population_is_identical(self) -> None:
        config = _tiny_config(layout_seed=1)
        genome = np.zeros(995, dtype=np.float32)
        clones = make_clone_population(config, genome, run_seed=1)
        assert len(clones) == config.population.size
        assert all(np.array_equal(c.genome, genome) for c in clones)

    def test_layout_seed_changes_world_contents(self) -> None:
        # Two layout seeds must place food differently: build the worlds
        # exactly as run_generation does (seed global RNG, then spawn) and
        # compare actual food coordinates.
        from simulation.environment import World
        import numpy as np

        def _food_positions(layout_seed: int):
            np.random.seed(layout_seed)
            world = World(300, 300)
            world.load_from_config(_tiny_config(layout_seed).model_dump())
            return [(f.x, f.y) for f in world.food]

        pos_a = _food_positions(1)
        pos_b = _food_positions(2)
        assert len(pos_a) == len(pos_b) == 5
        assert pos_a != pos_b, "different layout seeds must place food differently"

    def test_spawn_positions_within_world_bounds(self) -> None:
        # A large population must still spawn inside the world (positions
        # are drawn from the RNG bounded by world size, not scaled by
        # population index)
        config = _tiny_config(layout_seed=1, population=60)
        population = make_population(config, run_seed=3)
        assert len(population) == 60
        for agent in population:
            assert 0.0 < agent.position.x < config.world.width
            assert 0.0 < agent.position.y < config.world.height

    def test_train_requires_elitism(self) -> None:
        config = _tiny_config(layout_seed=1)
        config.evolution.elitism_count = 0
        with pytest.raises(ValueError):
            train_best_genome(config, run_seed=0)


class TestRunGeneralizationPipeline:
    def test_full_pipeline_smoke(self, tmp_path: Path, monkeypatch) -> None:
        """End-to-end: configs -> train -> freeze -> evaluate -> stats."""
        # One train layout + one test layout, one seed, tiny params
        train_file = tmp_path / "train.json"
        test_file = tmp_path / "test.json"
        base = json.loads(_tiny_config(layout_seed=0).model_dump_json())
        for path, seed, name in [(train_file, 11, "A1"), (test_file, 71, "B1")]:
            cfg = dict(base)
            cfg["world"] = {**base["world"], "layout_seed": seed}
            path.write_text(json.dumps({
                "base_config": cfg,
                "layouts": [{"name": name, "layout_seed": seed}],
            }))
        monkeypatch.setattr("experiments.run_generalization.TRAIN_CONFIG",
                            str(train_file))
        monkeypatch.setattr("experiments.run_generalization.TEST_CONFIG",
                            str(test_file))

        results = run_generalization(seeds=2)
        assert results["n_seeds"] == 2
        assert len(results["per_seed"]) == 2
        assert results["train_layouts"] == ["A1"]
        assert results["test_layouts"] == ["B1"]
        # One train score and one unseen score must exist and be finite
        seed_result = results["per_seed"][0]
        assert np.isfinite(seed_result["train_score"])
        assert np.isfinite(seed_result["unseen_score"])
        assert np.isfinite(results["gap_mean"])
        assert 0.0 <= results["p_value"] <= 1.0