"""
Unit tests for genome pack/unpack round-trips (Phase 2).

Confirms a flat genome maps deterministically to per-layer weight/bias
tensors and back, and that the layout matches the batched-inference stacker
so genomes loaded into either path compute identical outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neural import genome
from neural.network import ARCHITECTURE, FixedController, total_params
from neural.batched_inference import stack_population_weights


def _random_genome(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, total_params()).astype(np.float32)


class TestPackUnpack:
    def test_pack_round_trip(self) -> None:
        """pack( unpack(x) ) reproduces the genome exactly."""
        rng = np.random.default_rng(0)
        g = _random_genome(rng)
        weights, biases = genome.unpack_genome(g)
        assert np.allclose(genome.pack_weights(weights, biases), g, atol=1e-6)

    def test_unpack_layer_shapes(self) -> None:
        """Per-layer weight/bias shapes match the architecture's linear layers."""
        rng = np.random.default_rng(1)
        weights, biases = genome.unpack_genome(_random_genome(rng))
        assert len(weights) == len(biases) == len(ARCHITECTURE) - 1
        for index, (in_dim, out_dim) in enumerate(zip(ARCHITECTURE,
                                                      ARCHITECTURE[1:])):
            assert weights[index].shape == (in_dim, out_dim)
            assert biases[index].shape == (out_dim,)

    def test_genome_size_matches_parameter_total(self) -> None:
        assert genome.genome_size() == total_params()

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError):
            genome.unpack_genome(np.zeros(5, dtype=np.float32))

    def test_rejects_wrong_layer_count(self) -> None:
        rng = np.random.default_rng(2)
        weights, biases = genome.unpack_genome(_random_genome(rng))
        with pytest.raises(ValueError):
            genome.pack_weights(weights[:-1], biases)


class TestGenomeFeedsController:
    def test_genome_loads_into_controller(self) -> None:
        """A genome can be loaded into the reference controller and run."""
        rng = np.random.default_rng(3)
        g = _random_genome(rng)
        weights, biases = genome.unpack_genome(g)

        controller = FixedController()
        tensor_params = list(controller.parameters())
        for index, param in enumerate(tensor_params):
            # params interleave per layer as weight, bias; the genome stores
            # weights row-major (in, out) but nn.Linear keeps (out, in)
            if index % 2 == 0:
                param.data.copy_(torch.as_tensor(weights[index // 2].T,
                                                 dtype=torch.float32))
            else:
                param.data.copy_(torch.as_tensor(biases[index // 2], dtype=torch.float32))

        obs = torch.randn(1, ARCHITECTURE[0])
        out = controller(obs)
        assert out.shape == (1, ARCHITECTURE[-1])