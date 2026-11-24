"""
Batched population inference via stacked weights and torch.bmm (Phase 1.5).

Exploits that all fixed-controller agents share one architecture: pack the
whole population's parameters into stacked (N, in, out) / (N, out) tensors
and run one matrix product per layer for everyone at once, instead of N
sequential per-agent nn.Module forward passes.

Layout of a flat genome (order, per layer): W weights then b biases.
Matches measures in neural.network.param_counts().
"""

from __future__ import annotations

import numpy as np
import torch

from neural.network import ARCHITECTURE, param_counts


StackedWeights = tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor, torch.Tensor, torch.Tensor]


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """
    Resolve a configured device name into the torch.device used for tensors.

    Strategy: one resolver is shared by the runner (tier tagging) and this
    module (tensor placement), so the two cannot disagree about where a run
    executed. "auto" prefers CUDA when present; an explicit "cuda" without
    CUDA support fails loudly rather than running CPU work under a GPU label.

    Args:
        requested: "cpu", "cuda", "auto", or an existing torch.device.

    Returns:
        The concrete torch.device ("cpu" or "cuda").

    Raises:
        ValueError: If the name is not one of the accepted values.
        RuntimeError: If CUDA was requested but is unavailable.
    """
    if isinstance(requested, torch.device):
        target = requested
    else:
        name = requested.strip().lower()
        if name == "cpu":
            target = torch.device("cpu")
        elif name == "cuda":
            target = torch.device("cuda")
        elif name == "auto":
            target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError(
                f"Unsupported device {requested!r}; expected 'cpu', 'cuda', or 'auto'"
            )

    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False; "
            "use --device cpu or --device auto"
        )
    return target


def _genome_slices() -> list[tuple[int, int, int, int]]:
    """
    Compute (start, weight_end, bias_end, bias_end) offset triples per layer.

    Each triple gives the start index of the layer's genes, the end index of
    the weight-but-not-bias block, and the end index of the whole layer block.
    """
    slices = []
    offset = 0
    for index in range(len(ARCHITECTURE) - 1):
        in_dim = ARCHITECTURE[index]
        out_dim = ARCHITECTURE[index + 1]
        weight_count = in_dim * out_dim
        layer_end = offset + weight_count + out_dim
        slices.append((offset, offset + weight_count, layer_end))
        offset = layer_end
    return slices  # type: ignore[return-value]


def stack_population_weights(
    genomes: np.ndarray,
    device: str | torch.device | None = None,
) -> StackedWeights:
    """
    Pack a population of flat genomes into stacked weight/bias tensors.

    Args:
        genomes: numpy float32 array of shape (N, total_params).
        device: Target device. None keeps the historical CPU-only behaviour
            so callers that have not opted into device selection stay on the
            cpu-deterministic tier.

    Returns:
        Tuple (W1, b1, W2, b2, W3, b3) where Wi is (N, in_i, out_i) and
        bi is (N, out_i).

    Raises:
        ValueError: If the genome length does not match the architecture.
    """
    if genomes.ndim != 2:
        raise ValueError(f"genomes must be 2D (N, total_params), got shape {genomes.shape}")
    if genomes.shape[1] != sum(w + b for w, b in param_counts()):
        raise ValueError(
            f"genome length {genomes.shape[1]} does not match architecture "
            f"{ARCHITECTURE} total params {sum(w + b for w, b in param_counts())}"
        )

    # None is an explicit CPU request, not "auto": every legacy caller and
    # cpu-deterministic run must stay on CPU when running on a GPU machine.
    target_device = resolve_device("cpu" if device is None else device)
    population_size = genomes.shape[0]
    stacked: list[torch.Tensor] = []

    for index, (offset, weight_end, layer_end) in enumerate(_genome_slices()):
        in_dim = ARCHITECTURE[index]
        out_dim = ARCHITECTURE[index + 1]
        weights_flat = torch.as_tensor(genomes[:, offset:weight_end],
                                       dtype=torch.float32, device=target_device)
        # (N, in*out) -> (N, in, out); contiguous ensures bmm's layout contract
        weights = weights_flat.reshape(population_size, in_dim, out_dim).contiguous()
        biases = torch.as_tensor(genomes[:, weight_end:layer_end],
                                 dtype=torch.float32, device=target_device)
        stacked.extend([weights, biases])

    return tuple(stacked)  # type: ignore[return-value]


def batched_forward(observations: torch.Tensor, weights: StackedWeights) -> torch.Tensor:
    """
    Run one batched forward pass for the whole population.

    Args:
        observations: tensor of shape (N, 12) -- one input row per agent.
        weights: output of stack_population_weights.

    Returns:
        Tensor of shape (N, 3): (steering, acceleration, eat_signal).
        steering in [-1, 1] (tanh), the other two in [0, 1] (sigmoid).
    """
    (w1, b1, w2, b2, w3, b3) = weights

    # Adapter: observations come from numpy ray casting on the host. Moving
    # them once to the weight device keeps bmm operands on the same device
    # without adding a device parameter to every inference caller.
    if observations.device != w1.device:
        observations = observations.to(device=w1.device)

    # hidden1: (N,1,32) = (N,1,12) x (N,12,32); squeeze the singleton bmm dim
    h1 = torch.bmm(observations.unsqueeze(1), w1).squeeze(1) + b1
    h1 = torch.relu(h1)

    # hidden2: (N,1,16) from (N,1,32) x (N,32,16)
    h2 = torch.bmm(h1.unsqueeze(1), w2).squeeze(1) + b2
    h2 = torch.relu(h2)

    # output: (N,1,3) from (N,1,16) x (N,16,3)
    output = torch.bmm(h2.unsqueeze(1), w3).squeeze(1) + b3

    # Activation split per design doc Section 12.3: steering is direction
    # (tanh), acceleration and eat are gated signals (sigmoid).
    steering = torch.tanh(output[:, 0:1])
    acceleration = torch.sigmoid(output[:, 1:2])
    eat_signal = torch.sigmoid(output[:, 2:3])
    return torch.cat([steering, acceleration, eat_signal], dim=1)