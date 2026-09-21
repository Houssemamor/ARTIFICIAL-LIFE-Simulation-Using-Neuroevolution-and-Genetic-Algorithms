"""
Determinism utilities for Artificial Life Neuroevolution Simulation.

Enforces identical behavior across runs on the CPU-deterministic tier.
Implements the determinism checklist from PLAN.md Phase 4.

Usage:
    from simulation.determinism import set_deterministic_seeds, DeterminismConfig
    
    # At program start, before any torch/numpy/random imports that use state
    config = DeterminismConfig(
        torch_seed=42,
        numpy_seed=123,
        random_seed=456,
        device="cpu",
        reproducibility_tier="cpu-deterministic",
        num_threads=1,
    )
    set_deterministic_seeds(config)
"""

from __future__ import annotations
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class DeterminismConfig:
    """
    Configuration for deterministic execution.

    All seed fields are REQUIRED for the cpu-deterministic tier.
    Never use a single combined "seed" — torch, numpy, and Python's
    random module have independent state and must be seeded separately.
    """
    torch_seed: int
    numpy_seed: int
    random_seed: int
    device: str = "cpu"
    reproducibility_tier: str = "cpu-deterministic"
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.reproducibility_tier == "cpu-deterministic" and self.device != "cpu":
            raise ValueError("cpu-deterministic tier requires device='cpu'")
        if self.torch_seed is None or self.numpy_seed is None or self.random_seed is None:
            raise ValueError("All three seeds (torch, numpy, random) must be provided for cpu-deterministic tier")


def set_deterministic_seeds(config: DeterminismConfig) -> None:
    """
    Set all random seeds and deterministic flags.

    Must be called BEFORE any other torch/numpy/random operations that
    depend on random state (including imports that trigger lazy initialization).

    Args:
        config: DeterminismConfig with all required seeds and settings.
    """
    # Python's built-in random
    random.seed(config.random_seed)

    # NumPy
    np.random.seed(config.numpy_seed)

    # PyTorch
    torch.manual_seed(config.torch_seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.torch_seed)

    # Deterministic algorithms (CPU and CUDA)
    # Note: torch.use_deterministic_algorithms(True) may raise on some ops
    # that lack deterministic implementations. We catch and warn.
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except RuntimeError as e:
        # Some operations (e.g., atomicAdd on CUDA) lack deterministic impl
        # Fall back to warn_only=True for those
        torch.use_deterministic_algorithms(True, warn_only=True)
        print(f"[determinism] Warning: {e}", file=sys.stderr)

    # Thread pinning for CPU reproducibility
    torch.set_num_threads(config.num_threads)
    try:
        torch.set_num_interop_threads(config.num_threads)
    except RuntimeError:
        # Already set or parallel work started; ignore
        pass

    # Environment variables that affect BLAS/MKL/OpenMP
    os.environ["OMP_NUM_THREADS"] = str(config.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(config.num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(config.num_threads)

    # CuDNN determinism (only relevant on CUDA)
    if config.device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Record config for metadata
    _SEED_CONFIG_STACK.append(config)


# Stack to track applied configs (for metadata recording)
_SEED_CONFIG_STACK: list[DeterminismConfig] = []


def get_current_determinism_config() -> Optional[DeterminismConfig]:
    """Return the most recently applied DeterminismConfig, if any."""
    return _SEED_CONFIG_STACK[-1] if _SEED_CONFIG_STACK else None


def set_all_seeds(
    torch_seed: int,
    numpy_seed: int,
    random_seed: int,
    device: str = "cpu",
    reproducibility_tier: str = "cpu-deterministic",
    num_threads: int = 1,
) -> DeterminismConfig:
    """
    Convenience function: create config and apply seeds in one call.

    Args:
        torch_seed: Seed for torch.manual_seed()
        numpy_seed: Seed for np.random.seed()
        random_seed: Seed for random.seed()
        device: "cpu" or "cuda"
        reproducibility_tier: "cpu-deterministic" or "gpu"
        num_threads: Thread count for torch CPU operations

    Returns:
        The DeterminismConfig that was applied.
    """
    config = DeterminismConfig(
        torch_seed=torch_seed,
        numpy_seed=numpy_seed,
        random_seed=random_seed,
        device=device,
        reproducibility_tier=reproducibility_tier,
        num_threads=num_threads,
    )
    set_deterministic_seeds(config)
    return config


def assert_cpu_deterministic(config: DeterminismConfig) -> None:
    """
    Assert that configuration is valid for cpu-deterministic tier.

    Raises:
        ValueError: If config is not suitable for bit-for-bit reproducibility.
    """
    if config.reproducibility_tier != "cpu-deterministic":
        raise ValueError(f"Expected cpu-deterministic tier, got {config.reproducibility_tier}")
    if config.device != "cpu":
        raise ValueError(f"cpu-deterministic tier requires device='cpu', got {config.device}")
    if config.torch_seed is None or config.numpy_seed is None or config.random_seed is None:
        raise ValueError("All three seeds must be provided")