"""
Phase 2 equivalence gate: batched inference == independent per-agent network.

The correctness gate that validates Phase 1.5's speed optimization did not
change behavior once genomes and sensors feed the real pipeline: for a random
sample of genomes + observations, the batched population forward pass must
match an independent per-agent nn.Module reference within float tolerance.
"""

from __future__ import annotations

import numpy as np
import torch

from agents.organism import Organism
from agents.sensors import RayCaster, OBSERVATION_DIM
from neural import genome
from neural.batched_inference import batched_forward, stack_population_weights
from neural.network import ARCHITECTURE, FixedController
from simulation.environment import Food, World


def _build_population(agent_count: int, rng: np.random.Generator) -> tuple[list[Organism], np.ndarray]:
    """
    Create agents with random genomes and return them plus the genome array.

    Args:
        agent_count (int): Population size.
        rng: seeded generator for reproducible genomes and initial poses.

    Returns:
        (agents, genomes): agents list and (N, genome_size) float32 array.
    """
    agents = []
    genomes = []
    for index in range(agent_count):
        agent = Organism(index, 100.0 + index * 40.0, 300.0, initial_energy=100.0)
        g = rng.uniform(-1.0, 1.0, genome.genome_size()).astype(np.float32)
        agent.genome = g
        agent.heading = rng.uniform(0.0, 2 * np.pi)
        agents.append(agent)
        genomes.append(g)
    return agents, np.stack(genomes)


class _ReferencePopulation:
    """Independent per-agent reference: one FixedController per agent loaded
    from the genome layout, run one observation at a time."""

    def __init__(self, genomes: np.ndarray) -> None:
        self.controllers = []
        for g in genomes:
            controller = FixedController()
            weights, biases = genome.unpack_genome(g)
            param_iter = list(controller.parameters())
            for index, param in enumerate(param_iter):
                if index % 2 == 0:
                    # genome stores (in, out); nn.Linear weight is (out, in)
                    param.data.copy_(torch.as_tensor(weights[index // 2].T,
                                                     dtype=torch.float32))
                else:
                    param.data.copy_(torch.as_tensor(biases[index // 2], dtype=torch.float32))
            self.controllers.append(controller)

    def forward_all(self, observations: np.ndarray) -> torch.Tensor:
        """Run each controller on its own observation row and stack results."""
        outputs = []
        for index, controller in enumerate(self.controllers):
            obs = torch.as_tensor(observations[index], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                outputs.append(controller(obs))
        return torch.cat(outputs, dim=0)


def _assert_batched_matches_reference(agent_count: int, seed: int,
                                      include_entities: bool) -> None:
    rng = np.random.default_rng(seed)
    agents, genomes = _build_population(agent_count, rng)
    world = World(1200, 700)
    world.food = []
    world.obstacles = []
    if include_entities:
        for _ in range(20):
            world.food.append(Food(float(rng.uniform(0, 1200)),
                                   float(rng.uniform(0, 700))))

    # Real sensor observations (so the equivalence gate covers sensor output)
    observations = RayCaster().observe_population(agents, world)

    weights = stack_population_weights(genomes)
    batched_out = batched_forward(
        torch.as_tensor(observations, dtype=torch.float32), weights)

    reference = _ReferencePopulation(genomes)
    reference_out = reference.forward_all(observations)

    assert batched_out.shape == reference_out.shape == (agent_count, ARCHITECTURE[-1])
    torch.testing.assert_close(batched_out, reference_out, atol=1e-5, rtol=1e-5)


def test_batched_matches_reference_no_entities() -> None:
    """Empty world: all rays saturated, internal state only."""
    _assert_batched_matches_reference(agent_count=10, seed=11, include_entities=False)


def test_batched_matches_reference_with_food() -> None:
    """Food present: sensor distances/bearings vary, output must still match."""
    _assert_batched_matches_reference(agent_count=10, seed=12, include_entities=True)


def test_observation_dimension_feeds_controller() -> None:
    """Sensor rows must be exactly the controller's input size."""
    from neural.network import input_size
    assert OBSERVATION_DIM == input_size()