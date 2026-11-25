"""
The tensor runner must persist the trained controller genomes.

Without saved weights a 1000-generation run leaves only metrics behind, and
the dashboard cannot replay the brain it just evolved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments import run_tensor_ecosystem as runner
from neural.checkpoint import load_checkpoint
from neural.genome import genome_size


def test_runner_saves_prey_and_predator_genomes(tmp_path, monkeypatch) -> None:
    """Both role elites are written in the standard checkpoint format."""
    settings = json.loads(Path("configs/tensor_balance.json").read_text(
        encoding="utf-8"))
    settings["device"] = "cpu"
    settings["evolution"]["generations"] = 1
    settings["evolution"]["evaluation_steps"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(settings), encoding="utf-8")

    history_row = {
        "generation": 0, "prey_alive": 3, "predator_alive": 2,
        "prey_alive_step_mean": 3.0, "predator_alive_step_mean": 2.0,
        "prey_fitness_mean": 0.7, "predator_fitness_mean": 0.5,
        "prey_food": 4, "captures": 1,
    }
    size = genome_size()
    single = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    prey = np.repeat(single[None, :], 3, axis=0)
    predators = np.repeat(single[None, :], 2, axis=0)
    monkeypatch.setattr(runner.TensorEcosystem, "train",
                        lambda self: (prey, predators, [history_row]))

    output = tmp_path / "summary.json"
    weights = tmp_path / "weights"
    monkeypatch.setattr(sys, "argv", [
        "run_tensor_ecosystem", "--config", str(config_path),
        "--device", "cpu", "--output", str(output),
        "--weights-dir", str(weights),
    ])
    runner.main()

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["weights"]["prey"] == str(weights / "prey.npz")
    assert summary["weights"]["predator"] == str(weights / "predator.npz")
    for role in ("prey", "predator"):
        genome, architecture, metadata = load_checkpoint(
            str(weights / f"{role}.npz"))
        assert genome.shape == (size,)
        assert architecture == (12, 32, 16, 3)
        assert metadata["role"] == role
        assert metadata["generation"] == 1
