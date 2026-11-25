"""
Tensor-native ecosystem gate.

Two kinds of evidence: ray geometry must match the established scalar
oracle, and the three tuned configs must produce three distinct ecological
regimes (prey dominance, predator dominance, coexistence).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.sensors import (_ray_aabb_hit, _ray_circle_hit)
from experiments.experiment_runner import load_calibration_scales
from simulation.geometry import OBSTACLE_HALF_SIZE
from simulation.world_config import load_config
from tensor_ecosystem import (EcosystemConfig, TensorEcosystem,
                              _aabb_distances, _circle_distances)


def test_circle_ray_matches_scalar_oracle() -> None:
    """Batched circle hits must equal the established scalar geometry."""
    generator = np.random.default_rng(4)
    origins = torch.as_tensor(generator.uniform(0, 600, (4, 2)),
                              dtype=torch.float32)
    angles = torch.as_tensor(generator.uniform(-math.pi, math.pi, (4, 7)),
                             dtype=torch.float32)
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    centers = torch.as_tensor(generator.uniform(0, 600, (5, 2)),
                              dtype=torch.float32)
    valid = torch.ones((4, 5), dtype=torch.bool)
    actual = _circle_distances(origins, directions, centers, 2.0, valid).numpy()
    for row in range(4):
        for ray in range(7):
            expected = min(
                _ray_circle_hit(float(directions[row, ray, 0]),
                                float(directions[row, ray, 1]),
                                float(centers[index, 0] - origins[row, 0]),
                                float(centers[index, 1] - origins[row, 1]),
                                2.0)
                for index in range(5))
            assert actual[row, ray] == pytest.approx(
                expected if expected < math.inf else math.inf, rel=1e-4)


def test_aabb_ray_matches_scalar_oracle() -> None:
    """Slab-method batching must keep the exact solid-body geometry."""
    origins = torch.tensor([[100.0, 100.0], [200.0, 200.0]])
    angles = torch.tensor([[0.0, math.pi / 2, -math.pi / 2, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    centers = torch.tensor([[150.0, 100.0], [200.0, 250.0]])
    valid = torch.ones((2, 2), dtype=torch.bool)
    actual = _aabb_distances(origins, directions, centers,
                             OBSTACLE_HALF_SIZE, valid).numpy()
    for row in range(2):
        for ray in range(7):
            expected = min(
                _ray_aabb_hit(float(directions[row, ray, 0]),
                              float(directions[row, ray, 1]),
                              float(centers[index, 0] - origins[row, 0]),
                              float(centers[index, 1] - origins[row, 1]),
                              OBSTACLE_HALF_SIZE)
                for index in range(2))
            assert actual[row, ray] == pytest.approx(
                expected if expected < math.inf else math.inf, rel=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("config_name,expected", [
    ("tensor_prey_wins", "prey"),
    ("tensor_predator_wins", "predator"),
    ("tensor_balance", "balance"),
])
def test_ecosystem_regimes(config_name: str, expected: str) -> None:
    """Short runs must already separate into the three target regimes."""
    config = load_config(str(Path("configs") / f"{config_name}.json"))
    config.evolution.generations = 12
    settings = EcosystemConfig.from_config(config)
    ecosystem = TensorEcosystem(settings, torch.device("cuda"),
                                load_calibration_scales())
    _, _, history = ecosystem.train()
    window = history[-5:]
    prey = sum(row["prey_alive_step_mean"] for row in window) / len(window)
    predator = sum(row["predator_alive_step_mean"] for row in window) / len(window)
    ratio = prey / predator if predator else math.inf

    if expected == "prey":
        assert prey > predator * 3.0
    elif expected == "predator":
        assert predator > prey
    else:
        assert prey > 0.0 and predator > 0.0
        assert 0.25 <= ratio <= 4.0
