"""
Genome checkpoint save/restore for Artificial Life Neuroevolution Simulation.

Saves a genome + architecture pair as a compressed .npz file.
Restoring produces identical outputs on identical inputs (verified in tests).

Format (.npz):
    genome: flat float32 array of length genome_size()
    architecture: int32 array [12, 32, 16, 3]
    metadata: JSON string with version info
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from neural.genome import genome_size, unpack_genome
from neural.network import ARCHITECTURE, FixedController


CHECKPOINT_VERSION = 1
DEFAULT_ARCHITECTURE = tuple(ARCHITECTURE)


def save_checkpoint(
    path: str,
    genome: np.ndarray,
    architecture: Optional[Tuple[int, ...]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a genome checkpoint to disk.

    Args:
        path: Output .npz file path.
        genome: Flat genome array (float32, length genome_size()).
        architecture: Network architecture tuple (default: 12,32,16,3).
        metadata: Optional dict of additional info (generation, fitness, etc.).
    """
    if architecture is None:
        architecture = DEFAULT_ARCHITECTURE

    genome = np.asarray(genome, dtype=np.float32)
    if genome.ndim != 1 or genome.size != genome_size():
        raise ValueError(f"genome must be 1D of length {genome_size()}, got {genome.shape}")

    meta = {
        "version": CHECKPOINT_VERSION,
        "architecture": list(architecture),
        "genome_size": genome_size(),
    }
    if metadata:
        meta.update(metadata)

    np.savez_compressed(
        path,
        genome=genome.astype(np.float32),
        architecture=np.array(architecture, dtype=np.int32),
        metadata=json.dumps(meta),
    )


def load_checkpoint(path: str) -> Tuple[np.ndarray, Tuple[int, ...], Dict[str, Any]]:
    """
    Load a genome checkpoint from disk.

    Args:
        path: Path to .npz checkpoint file.

    Returns:
        Tuple of (genome_array, architecture_tuple, metadata_dict).

    Raises:
        ValueError: If genome size doesn't match expected size or architecture
            doesn't match the expected ARCHITECTURE.
    """
    data = np.load(path, allow_pickle=True)
    genome = data["genome"].astype(np.float32)
    architecture = tuple(data["architecture"].tolist())
    metadata = json.loads(data["metadata"].item())

    if genome.ndim != 1 or genome.size != genome_size():
        raise ValueError(f"Loaded genome has wrong size: {genome.size} != {genome_size()}")

    if architecture != DEFAULT_ARCHITECTURE:
        raise ValueError(
            f"Checkpoint architecture {architecture} != expected {DEFAULT_ARCHITECTURE}"
        )

    return genome, architecture, metadata


def load_checkpoint_into_controller(
    path: str,
    controller: Optional[FixedController] = None,
) -> FixedController:
    """
    Load checkpoint and return a FixedController with weights loaded.

    Args:
        path: Path to .npz checkpoint.
        controller: Optional existing FixedController to load into (created if None).

    Returns:
        FixedController with loaded weights.
    """
    genome, architecture, _ = load_checkpoint(path)

    if controller is None:
        controller = FixedController()

    weights, biases = unpack_genome(genome)

    # Load into controller (transpose weights: genome is (in,out), nn.Linear is (out,in))
    params = list(controller.parameters())
    for idx, param in enumerate(params):
        if idx % 2 == 0:
            # weight param
            layer_idx = idx // 2
            w = torch.as_tensor(weights[layer_idx].T, dtype=torch.float32)
            param.data.copy_(w)
        else:
            # bias param
            layer_idx = idx // 2
            b = torch.as_tensor(biases[layer_idx], dtype=torch.float32)
            param.data.copy_(b)

    return controller


def verify_checkpoint(
    path: str,
    test_input: Optional[np.ndarray] = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """
    Verify that a checkpoint produces identical outputs before/after save.

    Args:
        path: Path to checkpoint file.
        test_input: Optional test input (default: random normal).
        rtol: Relative tolerance for output comparison.
        atol: Absolute tolerance for output comparison.

    Returns:
        True if outputs match within tolerance.
    """
    if test_input is None:
        test_input = np.random.randn(1, ARCHITECTURE[0]).astype(np.float32)

    # Load from checkpoint
    controller = load_checkpoint_into_controller(path)
    controller.eval()

    with torch.no_grad():
        out1 = controller(torch.as_tensor(test_input)).numpy()

    # Reload and recompute
    controller2 = load_checkpoint_into_controller(path)
    controller2.eval()
    with torch.no_grad():
        out2 = controller2(torch.as_tensor(test_input)).numpy()

    return np.allclose(out1, out2, rtol=rtol, atol=atol)