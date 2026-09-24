"""
Integration test for checkpoint save/restore (Phase 4).

Verifies that a restored genome produces identical outputs to the original
before checkpointing.
"""

from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from neural.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_checkpoint_into_controller,
    verify_checkpoint,
)
from neural.network import FixedController, ARCHITECTURE
from neural.genome import genome_size, unpack_genome


class TestCheckpointRestore:
    """Test checkpoint save/restore preserves behavior exactly."""

    def test_save_load_roundtrip(self) -> None:
        """save_checkpoint -> load_checkpoint recovers genome exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_checkpoint.npz"

            # Create a known genome
            rng = np.random.default_rng(42)
            genome = rng.uniform(-1, 1, genome_size()).astype(np.float32)

            # Save
            save_checkpoint(str(path), genome, metadata={"generation": 5, "fitness": 1.23})

            # Load
            loaded_genome, architecture, metadata = load_checkpoint(str(path))

            # Verify
            assert np.array_equal(genome, loaded_genome)
            assert architecture == (12, 32, 16, 3)
            assert metadata["generation"] == 5
            assert metadata["fitness"] == 1.23

    def test_load_into_controller_produces_identical_outputs(self) -> None:
        """Loading checkpoint into controller produces identical forward pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "controller_checkpoint.npz"

            # Create controller and get baseline output
            rng = np.random.default_rng(42)
            genome = rng.uniform(-1, 1, genome_size()).astype(np.float32)

            controller = FixedController()
            weights, biases = unpack_genome(genome)
            params = list(controller.parameters())
            for idx, param in enumerate(params):
                if idx % 2 == 0:
                    param.data.copy_(torch.as_tensor(weights[idx // 2].T, dtype=torch.float32))
                else:
                    param.data.copy_(torch.as_tensor(biases[idx // 2], dtype=torch.float32))

            controller.eval()
            test_input = np.random.randn(1, ARCHITECTURE[0]).astype(np.float32)
            with torch.no_grad():
                original_output = controller(torch.as_tensor(test_input)).numpy()

            # Save checkpoint
            save_checkpoint(str(path), genome)

            # Load into fresh controller
            controller2 = load_checkpoint_into_controller(str(path))
            controller2.eval()

            with torch.no_grad():
                restored_output = controller2(torch.as_tensor(test_input)).numpy()

            assert np.allclose(original_output, restored_output, rtol=1e-5, atol=1e-6)

    def test_verify_checkpoint_utility(self) -> None:
        """verify_checkpoint utility returns True for valid checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "verify_test.npz"

            rng = np.random.default_rng(42)
            genome = rng.uniform(-1, 1, genome_size()).astype(np.float32)
            save_checkpoint(str(path), genome)

            assert verify_checkpoint(str(path))

    def test_batched_inference_after_restore(self) -> None:
        """Batched inference on restored genomes matches original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "batch_checkpoint.npz"

            rng = np.random.default_rng(123)
            population_size = 5
            genomes = np.stack([
                rng.uniform(-1, 1, genome_size()).astype(np.float32)
                for _ in range(population_size)
            ])

            # Save one genome as checkpoint
            save_checkpoint(str(path), genomes[0])

            # Restore and compare controller outputs
            restored_genome, _, _ = load_checkpoint(str(path))
            assert np.array_equal(genomes[0], restored_genome)

            # Test via controller
            controller = FixedController()
            weights, biases = unpack_genome(genomes[0])
            params = list(controller.parameters())
            for idx, param in enumerate(params):
                if idx % 2 == 0:
                    param.data.copy_(torch.as_tensor(weights[idx // 2].T, dtype=torch.float32))
                else:
                    param.data.copy_(torch.as_tensor(biases[idx // 2], dtype=torch.float32))

            controller.eval()
            test_input = torch.randn(1, ARCHITECTURE[0])
            with torch.no_grad():
                out1 = controller(test_input).numpy()

            # Restore and test
            controller2 = load_checkpoint_into_controller(str(path))
            controller2.eval()
            with torch.no_grad():
                out2 = controller2(test_input).numpy()

            assert np.allclose(out1, out2, rtol=1e-5, atol=1e-6)

    def test_checkpoint_metadata_preserved(self) -> None:
        """Custom metadata is preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta_checkpoint.npz"

            genome = np.random.default_rng(42).uniform(-1, 1, genome_size()).astype(np.float32)
            custom_meta = {
                "generation": 42,
                "fitness": 3.14159,
                "custom_field": "test_value",
                "nested": {"a": 1, "b": [1, 2, 3]},
            }

            save_checkpoint(str(path), genome, metadata=custom_meta)
            _, _, loaded_meta = load_checkpoint(str(path))

            assert loaded_meta["generation"] == 42
            assert loaded_meta["fitness"] == 3.14159
            assert loaded_meta["custom_field"] == "test_value"
            assert loaded_meta["nested"] == {"a": 1, "b": [1, 2, 3]}

    def test_architecture_mismatch_raises(self) -> None:
        """Loading checkpoint with wrong architecture raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "arch_mismatch.npz"

            genome = np.random.default_rng(42).uniform(-1, 1, genome_size()).astype(np.float32)
            save_checkpoint(str(path), genome, architecture=(12, 16, 3))  # wrong architecture

            with pytest.raises(ValueError):
                load_checkpoint(str(path))