"""
Tests for the showcase replay recorder (analytics/replay.py).

The key property: the recording must be a faithful, deterministic
transcript of the production simulation - activations agree with the
project's own batched forward pass, and the same inputs record the same
bytes.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agents.organism import Organism
from analytics.replay import (forward_trace, network_snapshot, pick_featured,
                              record_best_featured, record_replay)
from neural.batched_inference import batched_forward, stack_population_weights
from neural.genome import genome_size
from simulation.environment import World


def _genome(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, genome_size()).astype(np.float32)


def _prepare(num_prey=6, num_predators=1, seed=3):
    """Deterministic small arena; re-callable so both passes are identical."""
    def prepare():
        np.random.seed(seed)
        world = World(300, 200)
        world.spawn_food(15)
        world.spawn_obstacles(3)
        world.food_regrowth_per_step = 0.2
        agents = []
        rng = np.random.default_rng(seed)
        for i in range(num_prey + num_predators):
            role = 'prey' if i < num_prey else 'predator'
            agent = Organism(i, float(rng.uniform(20, 280)),
                             float(rng.uniform(20, 180)), role=role)
            agent.genome = _genome(100 + i)
            agents.append(agent)
        np.random.seed(seed)
        return world, agents
    return prepare


def test_forward_trace_matches_batched_forward():
    # The recording's activations must equal what the simulation computed.
    rng = np.random.default_rng(0)
    for seed in range(5):
        genome = _genome(seed)
        observation = rng.uniform(0, 1, 12).astype(np.float32)
        weights = stack_population_weights(genome[None, :])
        expected = batched_forward(torch.as_tensor(observation[None, :]), weights)[0].numpy()
        trace = forward_trace(genome, observation)
        assert np.allclose(trace["out"], expected, atol=2e-3)
        assert len(trace["in"]) == 12 and len(trace["h1"]) == 32
        assert len(trace["h2"]) == 16 and len(trace["out"]) == 3


def test_network_snapshot_shapes():
    net = network_snapshot(_genome(1))
    assert [np.array(w).shape for w in net["weights"]] == [(12, 32), (32, 16), (16, 3)]
    assert [len(b) for b in net["biases"]] == [32, 16, 3]


def test_replay_is_deterministic_and_complete():
    prepare = _prepare()
    first = record_best_featured(prepare, steps=25, capture_radius=30.0,
                                 capture_energy_transfer=60.0)
    second = record_best_featured(prepare, steps=25, capture_radius=30.0,
                                  capture_energy_transfer=60.0)
    assert json.dumps(first) == json.dumps(second)
    # One frame per step plus the final state.
    assert len(first["frames"]) == first["summary"]["steps"] + 1
    # Every agent row is [x, y, heading, energy, alive, eaten].
    assert all(len(row) == 6 for row in first["frames"][0]["a"])
    assert len(first["agents"]) == len(first["frames"][0]["a"])
    # The first frame always carries the food list (later frames only on change).
    assert "f" in first["frames"][0]


def test_activations_recorded_only_while_featured_agent_alive():
    world, agents = _prepare()()
    # No food and near-zero energy: the agent dies on the first step, so
    # the only way it survives is the bug this test guards against.
    world.food = []
    world.food_regrowth_per_step = 0.0
    agents[0].energy = 0.05
    run = record_replay(world, agents, {'prey': agents[0]}, steps=40)
    with_trace = [("n" in frame and "prey" in frame["n"]) for frame in run["frames"]]
    assert with_trace[0] is True
    died_at = next(i for i, frame in enumerate(run["frames"]) if frame["a"][0][4] == 0)
    # No activations after the agent's death, and none on the final frame.
    assert not any(with_trace[died_at:])
    assert with_trace[-1] is False


def test_rays_use_the_observation_the_agent_acted_on():
    world, agents = _prepare()()
    run = record_replay(world, agents, {'prey': agents[0]}, steps=3)
    trace = run["frames"][0]["n"]["prey"]
    # Normalised observation: ray distances and energy all in [0, 1].
    assert all(0.0 <= v <= 1.0 for v in trace["in"][:7])
    assert all(-1.0 <= v <= 1.0 for v in trace["in"][7:10])


def test_pick_featured_rule_is_most_food_then_alive_then_first():
    a, b, c = (Organism(i, 0, 0, role='prey') for i in range(3))
    b.food_eaten = 2
    c.food_eaten = 2
    c.is_alive = False
    assert pick_featured([a, b, c]) == {'prey': 1}
    d = Organism(3, 0, 0, role='predator')
    assert pick_featured([a, b, c, d]) == {'prey': 1, 'predator': 3}
    # Ties keep the earliest agent (no arbitrary reordering).
    e, f = Organism(0, 0, 0, role='prey'), Organism(1, 0, 0, role='prey')
    assert pick_featured([e, f]) == {'prey': 0}


def test_replay_stops_early_when_everyone_is_dead():
    world, agents = _prepare(num_prey=2, num_predators=0)()
    world.food = []
    world.food_regrowth_per_step = 0.0
    for agent in agents:
        agent.energy = 0.05         # below the metabolic cost: dies in one step
    run = record_replay(world, agents, {}, steps=200)
    assert run["summary"]["steps"] == 1
    assert run["summary"]["prey_alive_end"] == 0
    # One recorded step plus the final state frame.
    assert len(run["frames"]) == 2


@pytest.mark.parametrize("bad_genome_length", [10, genome_size() + 1])
def test_forward_trace_rejects_wrong_genome_size(bad_genome_length):
    with pytest.raises(ValueError):
        forward_trace(np.zeros(bad_genome_length, dtype=np.float32), np.zeros(12))
