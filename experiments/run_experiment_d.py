"""
Phase 5 Experiment D: fitness-weighting sensitivity.

Trains a GA under each of the four fitness-weighting conditions (D1 equal,
D2 resource-weighted, D3 exploration-weighted, D4 survival-weighted), then
scores every condition's frozen best genome on one common balanced metric so
the conditions are comparable. Results go through the Phase 4.5 statistical
pipeline (pairwise Mann-Whitney U + Holm-Bonferroni + rank-biserial).

Usage:
    python experiments/run_experiment_d.py --seeds 3        # exploratory
    python experiments/run_experiment_d.py --seeds 10        # confirmatory
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from analytics.statistics import compare_conditions_paired
from experiments.experiment_runner import (evaluate_genome,
                                           train_best_genome,
                                           write_stats_summary)
from simulation.determinism import DeterminismConfig, set_deterministic_seeds

CONDITION_CONFIGS = ["configs/fitness_weighting_d1.json",
                     "configs/fitness_weighting_d2.json",
                     "configs/fitness_weighting_d3.json",
                     "configs/fitness_weighting_d4.json"]
OUTPUT_DIR = Path("experiments/EXP-D")


def apply_seed_determinism(run_seed: int) -> None:
    """
    Enforce the cpu-deterministic tier for one per-seed run. Mirrors
    run_generalization.apply_seed_determinism (same derivation).
    """
    set_deterministic_seeds(DeterminismConfig(
        torch_seed=run_seed,
        numpy_seed=run_seed + 10_000,
        random_seed=run_seed + 20_000,
        device="cpu",
        reproducibility_tier="cpu-deterministic",
        num_threads=1,
    ))


def _load_condition(path: str):
    """Load a condition file ({"condition_name", "config"} wrapper)."""
    with open(path, "r") as f:
        data = json.load(f)
    from simulation.world_config import BaselineConfig
    return data["condition_name"], BaselineConfig(**data["config"])


def run_experiment_d(seeds: int) -> dict:
    """
    Run all four Experiment D conditions across seeds.

    Per condition and seed: train under the condition's weighting, freeze
    the best genome, evaluate it with the balanced weights on the shared
    layout. Conditions share the same run_seed (and therefore the same
    seeded initial population) per seed index, so the per-condition scores
    are blocked by seed and compared with the PAIRED pipeline
    (per-pair Wilcoxon signed-rank + Holm-Bonferroni), not the unpaired
    Mann-Whitney family.

    Returns:
        Dict with per-seed scores per condition and the paired comparison
        summary.
    """
    conditions = [_load_condition(p) for p in CONDITION_CONFIGS]

    per_condition = {}
    for cond_name, cond_config in conditions:
        scores = []
        for seed in range(seeds):
            t0 = time.time()
            apply_seed_determinism(seed)
            genome, _ = train_best_genome(cond_config, run_seed=seed)
            evaluation = evaluate_genome(
                cond_config, genome,
                run_seed=seed * 1000 + cond_config.world.layout_seed)
            scores.append(evaluation["mean_fitness"])
            print(f"{cond_name} seed {seed}: balanced="
                  f"{evaluation['mean_fitness']:.4f} "
                  f"food={evaluation['mean_food']:.2f} "
                  f"({time.time() - t0:.1f}s)")
        per_condition[cond_name] = scores

    comparison = compare_conditions_paired(
        {name: np.array(scores) for name, scores in per_condition.items()},
        metric="balanced_evaluation_fitness",
        alpha=0.05,
    )

    return {
        "experiment": "experiment_d_fitness_weighting",
        "n_seeds": seeds,
        "per_condition_scores": per_condition,
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Experiment D")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (3 exploratory / 10 confirmatory)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    results = run_experiment_d(args.seeds)
    out_dir = Path(args.output_dir)
    write_stats_summary(out_dir / "stats_summary.json", results)
    print(f"\nCondition means: "
          + ", ".join(f"{k}={np.mean(v):.4f}"
                      for k, v in results["per_condition_scores"].items()))
    print(f"Significant pairs (Holm-Bonferroni a=0.05): "
          f"{results['comparison']['significant_pairs']}")
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()