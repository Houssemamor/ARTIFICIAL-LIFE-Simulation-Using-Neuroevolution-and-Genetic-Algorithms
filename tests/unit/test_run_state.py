"""
Checkpoint/resume gate for headless GA runs.

A split run must be bit-identical to the equivalent uninterrupted run, and
the incremental CSV must hold exactly one row per generation without gaps or
duplicates.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from experiments.experiment_runner import train_best_genome
from experiments.run_state import load_run_state, verify_config
from simulation.world_config import load_config


def _config(tmp_path, generations: int):
    """Small deterministic config; calibration scales stay project-default."""
    base = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    base["experiment_name"] = "resume-test"
    base["evolution"]["generations"] = generations
    base["evolution"]["evaluation_steps"] = 20
    base["population"]["size"] = 8
    base["seed"] = {"torch": 11, "numpy": 12, "random": 13}
    path = tmp_path / f"config-{generations}.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return load_config(str(path))


@pytest.fixture
def base_config_json(tmp_path):
    """Store a baseline copy for the per-test config builder."""
    source = '{"experiment_name":"resume-test","mode":"headless","device":"cpu",' \
             '"reproducibility_tier":"cpu-deterministic",' \
             '"seed":{"torch":11,"numpy":12,"random":13},' \
             '"world":{"width":1200,"height":700,"food_count":20,' \
             '"obstacle_count":10},"population":{"size":8,"agent":' \
             '{"sensors":7,"brain":{"architecture":[12,32,16,3]}}},' \
             '"evolution":{"generations":2,"mutation_rate":0.05,' \
             '"crossover_rate":0.7,"elitism_count":2,"evaluation_steps":20},' \
             '"experiment":{"seeds":1,"save_interval":2},' \
             '"crossover_method":"blend","fitness_weights":' \
             '{"survival":0.4,"food":0.3,"exploration":0.2,"collision":0.1}}'
    path = tmp_path / "base.json"
    path.write_text(source, encoding="utf-8")
    return path


def test_resume_matches_uninterrupted_run(tmp_path, base_config_json) -> None:
    """Two generations plus resumed two equals four uninterrupted ones."""
    checkpoint = tmp_path / "Drive" / "run.npz"
    history_csv = tmp_path / "Drive" / "history.csv"
    two = _config(tmp_path, 2)
    four = _config(tmp_path, 4)

    full_best, full_history = train_best_genome(four, run_seed=100)
    train_best_genome(two, run_seed=100, checkpoint_path=str(checkpoint),
                      history_csv=str(history_csv), save_interval=2)
    resumed_best, resumed_history = train_best_genome(
        four, run_seed=100, checkpoint_path=str(checkpoint), resume=True,
        history_csv=str(history_csv), save_interval=2)

    assert resumed_history == full_history
    np.testing.assert_array_equal(resumed_best, full_best)
    with history_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert [int(row["generation"]) for row in rows] == [0, 1, 2, 3]


def test_checkpoint_records_generation_and_rng(tmp_path, base_config_json) -> None:
    """Stored generation is the next index, and config drift is rejected."""
    checkpoint = tmp_path / "run.npz"
    config = _config(tmp_path, 2)
    train_best_genome(config, run_seed=100, checkpoint_path=str(checkpoint),
                      save_interval=2)
    state = load_run_state(checkpoint)
    assert state["generation"] == 2
    assert len(state["history"]) == 2
    assert state["genomes"].shape[0] == config.population.size

    changed = config.model_copy(deep=True)
    changed.population.size = 10
    with pytest.raises(ValueError, match="does not match"):
        verify_config(state, changed)
