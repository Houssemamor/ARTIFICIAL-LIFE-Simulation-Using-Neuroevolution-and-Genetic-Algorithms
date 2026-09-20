"""
Naive per-agent inference baseline -- QUARANTINED (Phase 1.5).

Exists purely so the Phase 1.5 benchmark has a comparison point: a plain
nn.Module.forward() called once per agent in a loop. It must never be
imported from simulation/agents or any production path -- it is the exact
per-N pattern batched_inference exists to replace. The leading underscore
keeps it private; the benchmark in tests/performance/ is its only consumer.
"""

from __future__ import annotations

import numpy as np
import torch

from neural.network import FixedController, param_counts


def forward_population_naive(genomes: np.ndarray,
                             observations: torch.Tensor) -> torch.Tensor:
    """
    Run per-agent inference with a fresh controller loaded per agent.

    This is the intentionally slow reference path for benchmarking only.

    Args:
        genomes: float32 array of shape (N, total_params).
        observations: tensor of shape (N, 12).

    Returns:
        Tensor of shape (N, 3), activation semantics identical to the
        batched path (steering tanh, accel/eat sigmoid).
    """
    if genomes.ndim != 2:
        raise ValueError(f"genomes must be 2D (N, total_params), got shape {genomes.shape}")
    if genomes.shape[1] != sum(w + b for w, b in param_counts()):
        raise ValueError("genome length does not match architecture")

    results = []
    for agent_index in range(genomes.shape[0]):
        controller = FixedController()
        offset = 0
        # Flat genome layout is the same as neural.batched_inference: per
        # layer, weights then biases. Loaded into the module so forward() uses
        # identical math (bmm expansion of a linear layer = same result).
        for layer_index, layer in enumerate(controller.layers):
            in_dim = layer.in_features
            out_dim = layer.out_features
            weight_count = in_dim * out_dim
            with torch.no_grad():
                # Genome is row-major (in, out) while nn.Linear stores
                # weight as (out, in) -- transpose on load.
                layer.weight.copy_(
                    torch.as_tensor(genomes[agent_index,
                                            offset:offset + weight_count],
                                    dtype=torch.float32)
                    .reshape(in_dim, out_dim)
                    .t()
                    .contiguous()
                )
                layer.bias.copy_(
                    torch.as_tensor(genomes[agent_index,
                                            offset + weight_count:offset + weight_count + out_dim],
                                    dtype=torch.float32)
                )
            offset += weight_count + out_dim
        with torch.no_grad():
            results.append(controller(observations[agent_index:agent_index + 1]))

    return torch.cat(results, dim=0)