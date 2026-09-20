"""
Genome flatten/reshape utilities (Phase 2).

A genome is the flat concatenation of the fixed controller's parameters in
layer order, matching neural.network.param_counts(): for each layer, the
weight matrix row-major then the bias vector. Genome = [W1..., b1..., W2..., b2...,
W3..., b3...] per design doc Section 12.1.

These utilities are the single source of the flat<->shaped mapping so the
batched-inference stacker and the fitness/evolution layers in later phases
never disagree about the layout.
"""

from __future__ import annotations

import numpy as np

from neural.network import ARCHITECTURE, total_params


def genome_size() -> int:
    """Return the flat length of a genome (alias of network.total_params)."""
    return total_params()


def pack_weights(weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
    """
    Flatten per-layer weight and bias arrays into one flat genome.

    The flat genome encodes each layer as its (in, out) weight matrix in
    row-major order followed by its bias vector -- the canonical layout also
    read by neural.batched_inference.stack_population_weights.

    Args:
        weights: one array per layer, shape (in, out) [row-major weights].
        biases: one array per layer, shape (out,).

    Returns:
        Flat float32 genome vector in [W1, b1, W2, b2, ...] order.

    Raises:
        ValueError: If the number of layers or the flattened length does not
        match the architecture.
    """
    if len(weights) != len(ARCHITECTURE) - 1 or len(biases) != len(ARCHITECTURE) - 1:
        raise ValueError("weights/biases layer count must match architecture depth")
    blocks: list[np.ndarray] = []
    for weight, bias in zip(weights, biases):
        blocks.append(np.asarray(weight, dtype=np.float64).ravel())
        blocks.append(np.asarray(bias, dtype=np.float64).ravel())
    genome = np.concatenate(blocks)
    if genome.size != total_params():
        raise ValueError(
            f"packed genome length {genome.size} != {total_params()} (architecture {ARCHITECTURE})"
        )
    return genome.astype(np.float32)


def unpack_genome(genome: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Reshape a flat genome into per-layer weight and bias arrays.

    Args:
        genome: flat float array of length total_params().

    Returns:
        (weights, biases): lists over layers. weights are (in, out) matrices in
        row-major order (the canonical flat layout, matching batched
        inference). To load into an nn.Linear (which stores (out, in)),
        transpose the weight before copying.

    Raises:
        ValueError: If the genome length does not match the architecture.
    """
    genome = np.asarray(genome, dtype=np.float32)
    if genome.ndim != 1 or genome.size != total_params():
        raise ValueError(
            f"genome must be 1D of length {total_params()}, got shape {genome.shape}"
        )

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    offset = 0
    for index in range(len(ARCHITECTURE) - 1):
        in_dim = ARCHITECTURE[index]
        out_dim = ARCHITECTURE[index + 1]
        weight_block = genome[offset:offset + in_dim * out_dim]
        weights.append(weight_block.reshape(in_dim, out_dim))
        offset += in_dim * out_dim
        bias_block = genome[offset:offset + out_dim]
        biases.append(bias_block)
        offset += out_dim
    return weights, biases


def round_trip(genome: np.ndarray) -> np.ndarray:
    """
    Confirm a genome survives an unpack/pack cycle identically.

    Args:
        genome: flat float32 genome to round-trip.

    Returns:
        The re-packed genome, equal to the input within float tolerance if
        the layout is consistent.
    """
    weights, biases = unpack_genome(genome)
    return pack_weights(weights, biases)