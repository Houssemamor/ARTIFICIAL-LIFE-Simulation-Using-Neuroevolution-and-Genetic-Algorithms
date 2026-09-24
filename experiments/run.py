#!/usr/bin/env python3
"""
Headless experiment entry point for Artificial Life Neuroevolution Simulation.

Loads a config through the same pydantic validation as the GUI
(simulation/world_config.load_config), enforces the cpu-deterministic
tier, and runs one GA training run, writing the per-generation fitness
history as JSON.

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

from simulation.world_config import load_config
from simulation.determinism import (DeterminismConfig,
                                    set_deterministic_seeds)
from experiments.experiment_runner import (train_best_genome,
                                           write_stats_summary)


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

    # Enforce the cpu-deterministic tier from the config's seed fields
    # (time-derived fallback, same rule as the GUI)
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
        device="cpu",
        reproducibility_tier="cpu-deterministic",
        num_threads=1,
    ))

    print(f"Experiment: {config.experiment_name}")
    print(f"Layout seed: {config.world.layout_seed}")
    print(f"Training {config.evolution.generations} generations, "
          f"population {config.population.size}")

    best_genome, history = train_best_genome(config, run_seed=base_seed)

    summary = {
        "experiment_name": config.experiment_name,
        "seeds": {"torch": torch_seed, "numpy": base_seed,
                  "random": random_seed},
        "device": "cpu",
        "reproducibility_tier": "cpu-deterministic",
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