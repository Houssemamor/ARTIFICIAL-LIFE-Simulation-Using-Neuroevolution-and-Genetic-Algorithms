"""
Phase 1.5 benchmark: batched vs. naive population inference.

Confirms (a) the batched forward pass computes the same outputs as the
per-agent reference path, and (b) it sustains the real-time target at 250
agents. The measured steps/sec numbers feed Phase 5's compute-budget math.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from neural import batched_inference
from neural._naive_inference import forward_population_naive
from neural.network import ARCHITECTURE, total_params

# Real-time target: the app's declared render rate (app/run.py:64 uses 30 Hz
# physics) -- batched inference must not be the bottleneck below this.
REAL_TIME_FPS = 30.0
# Population sizes mandated by PLAN.md Phase 1.5 for the benchmark.
POPULATION_SIZES = [50, 100, 250, 500]
# Warm-up forward passes so first-call autograd/layout costs are excluded.
WARMUP_STEPS = 5
# Timed forward passes per population size; enough to smooth scheduler noise
# while keeping the suite fast.
TIMED_STEPS = 50


def _dummy_genomes(population_size: int, rng: np.random.Generator) -> np.ndarray:
    """Return random float32 genomes of shape (N, total_params)."""
    return rng.uniform(-1.0, 1.0, size=(population_size, total_params())).astype(np.float32)


def _dummy_observations(population_size: int,
                        rng: np.random.Generator) -> torch.Tensor:
    """Return random observations of shape (N, input_size)."""
    return torch.as_tensor(
        rng.uniform(-1.0, 1.0, size=(population_size, ARCHITECTURE[0])),
        dtype=torch.float32,
    )


def _steps_per_second(path: str, genomes: np.ndarray,
                      observations: torch.Tensor) -> float:
    """
    Time the given inference path and return forward passes per second.

    Args:
        path: 'batched' or 'naive'. Dispatches to the matching implementation.
        genomes: (N, total_params) float32 array.
        observations: (N, input_size) tensor.

    Returns:
        Sustained forward passes per second (averaged over TIMED_STEPS runs).
    """
    weights = batched_inference.stack_population_weights(genomes)
    for _ in range(WARMUP_STEPS):
        if path == "batched":
            batched_inference.batched_forward(observations, weights)
        else:
            forward_population_naive(genomes, observations)

    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        if path == "batched":
            batched_inference.batched_forward(observations, weights)
        else:
            forward_population_naive(genomes, observations)
    elapsed = time.perf_counter() - start
    return TIMED_STEPS / elapsed


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("population_size", [1, 5, 50])
def test_batched_matches_naive(population_size: int) -> None:
    """Batched inference must equal per-agent inference within float tolerance."""
    rng = np.random.default_rng(42)
    genomes = _dummy_genomes(population_size, rng)
    observations = _dummy_observations(population_size, rng)

    stacked = batched_inference.stack_population_weights(genomes)
    batched_output = batched_inference.batched_forward(observations, stacked)
    naive_output = forward_population_naive(genomes, observations)

    assert batched_output.shape == naive_output.shape == (population_size, 3)
    torch.testing.assert_close(batched_output, naive_output, atol=1e-5, rtol=1e-5)


def test_output_bounds() -> None:
    """steering in [-1,1]; acceleration and eat_signal in [0,1]."""
    rng = np.random.default_rng(7)
    genomes = _dummy_genomes(100, rng)
    observations = _dummy_observations(100, rng)
    output = batched_inference.batched_forward(
        observations, batched_inference.stack_population_weights(genomes)
    )
    assert torch.all(output[:, 0] >= -1.0) and torch.all(output[:, 0] <= 1.0)
    assert torch.all(output[:, 1] >= 0.0) and torch.all(output[:, 1] <= 1.0)
    assert torch.all(output[:, 2] >= 0.0) and torch.all(output[:, 2] <= 1.0)


def test_stack_rejects_bad_genome_length() -> None:
    """Stacking must fail loudly on a genome length mismatch."""
    with pytest.raises(ValueError):
        batched_inference.stack_population_weights(np.zeros((4, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("population_size", POPULATION_SIZES)
def test_batched_benchmark_record(population_size: int) -> None:
    """
    Record batched and naive throughput; print for the Phase 1.5 report.

    The real-time assertion happens only at 250 agents (the mandated gate).
    """
    rng = np.random.default_rng(1234)
    genomes = _dummy_genomes(population_size, rng)
    observations = _dummy_observations(population_size, rng)

    batched_sps = _steps_per_second("batched", genomes, observations)
    naive_sps = _steps_per_second("naive", genomes, observations)

    print(f"\n[benchmark] population={population_size}: "
          f"batched={batched_sps:.1f} steps/s, "
          f"naive={naive_sps:.2f} steps/s, "
          f"speedup={batched_sps / naive_sps:.1f}x")

    # Throughput must beat naive at every size; at 250 it must hold real time.
    assert batched_sps > naive_sps
    if population_size == 250:
        assert batched_sps >= REAL_TIME_FPS, (
            f"batched inference at 250 agents ({batched_sps:.1f} steps/s) "
            f"below real-time target ({REAL_TIME_FPS} FPS) -- optimize before Phase 2"
        )