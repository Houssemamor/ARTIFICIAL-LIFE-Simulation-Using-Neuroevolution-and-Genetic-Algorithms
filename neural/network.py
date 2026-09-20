"""
Fixed neural controller network definition (Phase 1.5).

Defines the 12-32-16-3 architecture as pure tensor-shape metadata plus a
drop-in reference module. The network itself is NOT wired to real sensors
yet -- this file exists to make the batched-inference layer correct and
measurable in isolation, per PLAN.md Phase 1.5.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Fixed topology: 12 sensor inputs -> 32 hidden -> 16 hidden -> 3 outputs.
# Outputs: (steering, acceleration, eat_signal). Matches configs/baseline.json
# population.agent.brain.architecture.
ARCHITECTURE: list[int] = [12, 32, 16, 3]

# Split of the 3-dim output: index 0 uses tanh (bounded -1..1), indices 1-2
# use sigmoid (bounded 0..1), exactly the design doc Section 12.3 semantics.
TANH_OUTPUTS: list[int] = [0]
SIGMOID_OUTPUTS: list[int] = [1, 2]


def input_size() -> int:
    """Return the number of network inputs (first architecture layer)."""
    return ARCHITECTURE[0]


def output_size() -> int:
    """Return the number of network outputs (last architecture layer)."""
    return ARCHITECTURE[-1]


def param_counts() -> list[tuple[int, int]]:
    """
    Return (weights, biases) element counts per linear layer.

    Provides the layout needed to unpack a flat genome into per-layer
    weight/bias tensors. Derived from the architecture rather than hardcoded
    so the topology stays manageable in one place.
    """
    counts = []
    for index in range(len(ARCHITECTURE) - 1):
        in_dim = ARCHITECTURE[index]
        out_dim = ARCHITECTURE[index + 1]
        counts.append((in_dim * out_dim, out_dim))
    return counts


def total_params() -> int:
    """Return the total flat-array length of a genome for this architecture."""
    return sum(w + b for w, b in param_counts())


class FixedController(nn.Module):
    """
    Reference per-agent controller (12-32-16-3) with ReLU hidden layers.

    Used only as the benchmark's naive comparison baseline in Phase 1.5.
    Production inference goes through neural.batched_inference instead; see
    the quarantine note in _naive_inference.py.

    Attributes:
        is_quarantined (bool): True -- blocks accidental production use.
    """

    is_quarantined = True

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(len(ARCHITECTURE) - 1):
            in_dim = ARCHITECTURE[index]
            out_dim = ARCHITECTURE[index + 1]
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass; hidden layers use ReLU, output uses the
        tanh/sigmoid split defined by TANH_OUTPUTS/SIGMOID_OUTPUTS."""
        hidden = x
        for layer in self.layers[:-1]:
            hidden = torch.relu(layer(hidden))
        output = self.layers[-1](hidden)
        for dim in TANH_OUTPUTS:
            output[:, dim] = torch.tanh(output[:, dim])
        for dim in SIGMOID_OUTPUTS:
            output[:, dim] = torch.sigmoid(output[:, dim])
        return output