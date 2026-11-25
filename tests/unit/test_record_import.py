"""
Imported-genome replay gate for the Colab-to-dashboard path.

A genome exported by a Colab run must be usable as the evolved solo brain
without retraining locally, and the recorded generation count must come from
the checkpoint metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import record_showcase
from neural.checkpoint import save_checkpoint
from neural.genome import genome_size
from simulation.world_config import load_config


def test_imported_genome_replays_without_local_training(tmp_path: Path,
                                                        monkeypatch) -> None:
    """solo_genome_path bypasses train_best_genome and keeps its metadata."""
    config_dict = json.loads(Path("configs/baseline.json").read_text(encoding="utf-8"))
    config_dict["evolution"]["evaluation_steps"] = 5
    config_path = tmp_path / "colab.json"
    config_path.write_text(json.dumps(config_dict), encoding="utf-8")
    load_config(str(config_path))  # validate before using it in the replay

    genome_path = tmp_path / "best.npz"
    save_checkpoint(str(genome_path),
                    np.zeros(genome_size(), dtype=np.float32),
                    metadata={"generation": 400})

    def _forbidden(*args, **kwargs):
        raise AssertionError("local training must be skipped for an imported genome")

    monkeypatch.setattr(record_showcase, "train_best_genome", _forbidden)

    recording = record_showcase.build_recording(
        seed=3, steps=5, solo_agents=2, solo_only=True,
        solo_genome_path=genome_path, solo_config_path=config_path)

    assert len(recording["scenarios"]) == 1
    scenario = recording["scenarios"][0]
    assert scenario["id"] == "solo"
    assert scenario["generations"] == 400
    assert set(scenario["runs"]) == {"untrained", "evolved"}
    assert "solo_import_seconds" in recording["environment"]["training_seconds"]


def test_coevolution_replay_uses_imported_elites(tmp_path: Path,
                                                 monkeypatch) -> None:
    """Both co-evolution roles can come from saved elite genomes."""
    config_path = Path("configs/tensor_balance.json")
    prey_path = tmp_path / "prey.npz"
    predator_path = tmp_path / "predator.npz"
    genome = np.zeros(genome_size(), dtype=np.float32)
    save_checkpoint(str(prey_path), genome, metadata={"generation": 1000})
    save_checkpoint(str(predator_path), genome, metadata={"generation": 1000})

    def _forbidden(*args, **kwargs):
        raise AssertionError("local co-evolution training must be skipped")

    monkeypatch.setattr(record_showcase, "train_coevolution", _forbidden)

    recording = record_showcase.build_recording(
        seed=3, steps=5, solo_agents=2,
        solo_genome_path=prey_path, solo_config_path=config_path,
        coev_prey_genome_path=prey_path,
        coev_predator_genome_path=predator_path)

    by_id = {scenario["id"]: scenario for scenario in recording["scenarios"]}
    assert set(by_id) == {"coev", "solo"}
    assert by_id["coev"]["generations"] == 1000
    assert "elite" in by_id["coev"]["description"]
    timings = recording["environment"]["training_seconds"]
    assert "coev_import_seconds" in timings

