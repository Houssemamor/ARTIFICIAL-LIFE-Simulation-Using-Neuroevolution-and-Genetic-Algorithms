"""
Device-selection gate for the GPU patch.

Verifies that configured devices resolve to a single concrete device, that
only CPU runs stay on the cpu-deterministic tier, and that CUDA runs cannot
silently execute on CPU under a GPU label. CUDA-dependent assertions are
gated so the suite still runs on CPU-only machines.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.run import reproducibility_tier_for, resolve_run_device
from neural.batched_inference import (batched_forward, resolve_device,
                                      stack_population_weights)
from neural.genome import genome_size


def _genomes(count: int = 4) -> np.ndarray:
    """Return random genomes of the fixed architecture used by inference."""
    rng = np.random.default_rng(7)
    return rng.uniform(-1.0, 1.0, (count, genome_size())).astype(np.float32)


def test_cpu_stays_cpu_even_when_cuda_is_available(monkeypatch) -> None:
    """Explicit cpu must win over hardware availability (determinism rule)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu").type == "cpu"


def test_auto_prefers_cuda_only_when_available(monkeypatch) -> None:
    """Auto picks CUDA on a GPU host and CPU on a CPU-only host."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_explicit_cuda_fails_when_unavailable(monkeypatch) -> None:
    """Negative case: a GPU-labelled run must not quietly fall back to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_unknown_device_is_rejected() -> None:
    """Invalid config values fail at resolution instead of at tensor creation."""
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("mps")


def test_stack_weights_default_to_cpu(monkeypatch) -> None:
    """Existing callers keep CPU placement, including on a GPU machine."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    weights = stack_population_weights(_genomes())
    assert all(tensor.device.type == "cpu" for tensor in weights)


def test_run_device_helpers_tag_tiers(monkeypatch) -> None:
    """Runner and inference must agree on device, and CUDA gets its own tier."""
    assert resolve_run_device("cpu") == "cpu"
    assert reproducibility_tier_for("cpu") == "cpu-deterministic"
    assert reproducibility_tier_for("cuda") == "non-deterministic"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_run_device("auto") == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_forward_moves_observations_to_weight_device() -> None:
    """Host-side observations must reach CUDA weights without caller changes."""
    genomes = _genomes()
    observations = torch.as_tensor(
        np.random.default_rng(11).uniform(-1.0, 1.0, (genomes.shape[0], 12)),
        dtype=torch.float32,
    )

    cuda_out = batched_forward(
        observations, stack_population_weights(genomes, device="cuda"))
    cpu_out = batched_forward(
        observations, stack_population_weights(genomes, device="cpu"))

    assert cuda_out.device.type == "cuda"
    torch.testing.assert_close(cuda_out.cpu(), cpu_out, atol=1e-5, rtol=1e-5)
