#!/usr/bin/env python3
"""
Headless experiment entry point for Artificial Life Neuroevolution Simulation.

Loads a config through the same pydantic validation as the GUI
(simulation/world_config.load_config), resolves the configured device and
its reproducibility tier, and runs one GA training run, writing the
per-generation fitness history as JSON.

For the controlled multi-seed experiments (generalization, Experiment D,
Experiment E) prefer the dedicated runners in experiments/ - this entry
point is for single ad-hoc runs.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural.batched_inference import resolve_device
from neural.checkpoint import save_checkpoint
from simulation.world_config import load_config
from simulation.determinism import (DeterminismConfig,
                                    set_deterministic_seeds)
from experiments.experiment_runner import (train_best_genome,
                                           write_stats_summary)


def resolve_run_device(requested: str) -> str:
    """Return the concrete device name ('cpu' or 'cuda') for a config value."""
    return resolve_device(requested).type


def reproducibility_tier_for(device: str) -> str:
    """Only CPU runs are bit-reproducible; CUDA goes into its own tier."""
    return "cpu-deterministic" if device == "cpu" else "non-deterministic"


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless single-run mode")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration JSON file")
    parser.add_argument("--mode", type=str, choices=["headless", "gui"],
                        default="headless", help="Execution mode (headless only is implemented)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"],
                        default=None, help="Override config device")
    parser.add_argument("--output", type=str,
                        default="experiments/EXP-HEADLESS/run_history.json",
                        help="Output path for the fitness history")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Full-state checkpoint path for resume-enabled runs")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from --checkpoint instead of starting over")
    parser.add_argument("--history-csv", type=str, default=None,
                        help="Append-only per-generation CSV path")
    parser.add_argument("--save-interval", type=int, default=None,
                        help="Generations between checkpoint and CSV flushes")
    parser.add_argument("--best-genome", type=str, default=None,
                        help="Export the final best genome as a .npz checkpoint")
    args = parser.parse_args()

    # Config goes through the same pydantic validation as every other
    # entry point - no bespoke loaders that bypass the schema.
    config = load_config(args.config)
    if args.device is not None:
        config.device = args.device

    # A predation config silently degrades here: this runner trains a
    # solo prey population. Point the user at the Phase 6 driver instead
    # of letting predators be quietly ignored.
    if config.predation and config.predation.predator_count > 0:
        print("Warning: this config specifies predators; this runner "
              "trains a solo prey population. For co-evolution or "
              "ecosystem runs use experiments/run_coevolution.py.")

    # Resolve "auto" once and reuse the concrete device for seeding, tensor
    # placement (via config.device) and run metadata.
    resolved_device = resolve_run_device(config.device)
    tier = reproducibility_tier_for(resolved_device)
    config.device = resolved_device
    config.reproducibility_tier = tier

    # Seed all three RNGs from the config's seed fields (time-derived fallback,
    # same rule as the GUI). Seeding still applies on the GPU tier, but the
    # tier label records that bit-for-bit reproducibility is not claimed.
    import time
    base_seed = config.seed.get("numpy")
    if base_seed is None:
        base_seed = int(time.time())
        print(f"Config seeds are null; using time-derived base seed {base_seed}")
    torch_seed = config.seed.get("torch") or base_seed + 1
    random_seed = config.seed.get("random") or base_seed + 2
    set_deterministic_seeds(DeterminismConfig(
        torch_seed=torch_seed,
        numpy_seed=base_seed,
        random_seed=random_seed,
        device=resolved_device,
        reproducibility_tier=tier,
        num_threads=1,
    ))

    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {resolved_device} (tier: {tier})")
    print(f"Layout seed: {config.world.layout_seed}")
    print(f"Training {config.evolution.generations} generations, "
          f"population {config.population.size}")

    best_genome, history = train_best_genome(
        config, run_seed=base_seed,
        checkpoint_path=args.checkpoint, resume=args.resume,
        save_interval=args.save_interval, history_csv=args.history_csv)

    if args.best_genome is not None:
        save_checkpoint(args.best_genome, best_genome, metadata={
            "generation": len(history),
            "fitness": history[-1]["max_fitness"] if history else None,
            "experiment_name": config.experiment_name,
        })
        print(f"Best genome written to {args.best_genome}")

    summary = {
        "experiment_name": config.experiment_name,
        "seeds": {"torch": torch_seed, "numpy": base_seed,
                  "random": random_seed},
        "device": resolved_device,
        "reproducibility_tier": tier,
        "layout_seed": config.world.layout_seed,
        "n_generations": config.evolution.generations,
        "generations": history,
        "final_best_fitness": history[-1]["max_fitness"] if history else None,
    }
    write_stats_summary(Path(args.output), summary)
    print(f"Final best fitness: {summary['final_best_fitness']}")
    print(f"History written to {args.output}")


if __name__ == "__main__":
    main()