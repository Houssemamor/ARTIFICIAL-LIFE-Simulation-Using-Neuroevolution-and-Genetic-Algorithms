"""
Phase 5 generalization experiment: train on layouts A1-A3, freeze best
controllers, evaluate unmodified on held-out layouts B1-B3, and quantify
the train-vs-unseen gap through the Phase 4.5 statistical pipeline.

Usage:
    python experiments/run_generalization.py --seeds 3        # exploratory
    python experiments/run_generalization.py --seeds 10       # confirmatory
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np

from analytics.statistics import (bootstrap_ci, matched_pairs_rank_biserial,
                                  wilcoxon_signed_rank)
from experiments.experiment_runner import (evaluate_genome, load_layouts,
                                           train_best_genome,
                                           write_stats_summary)
from simulation.determinism import DeterminismConfig, set_deterministic_seeds

TRAIN_CONFIG = "configs/generalization_train.json"
TEST_CONFIG = "configs/generalization_test.json"
OUTPUT_DIR = Path("experiments/EXP-GEN")


def apply_seed_determinism(run_seed: int) -> None:
    """
    Enforce the cpu-deterministic tier for one per-seed run.

    The three seeds are derived from run_seed with fixed offsets so each
    seed index is a fully deterministic run (torch, numpy, and Python's
    random are seeded independently - never one combined seed).
    """
    set_deterministic_seeds(DeterminismConfig(
        torch_seed=run_seed,
        numpy_seed=run_seed + 10_000,
        random_seed=run_seed + 20_000,
        device="cpu",
        reproducibility_tier="cpu-deterministic",
        num_threads=1,
    ))


def run_generalization(seeds: int) -> Dict:
    """
    Run the full generalization experiment.

    Per seed: train one best genome per training layout (A1-A3), evaluate
    each frozen genome on its training layout (train score) and on every
    held-out layout (B1-B3, unseen score). Per-seed train/unseen scores are
    the means over layouts, giving paired samples across seeds.

    Returns:
        Dict with per-seed scores, gap statistics, and pairwise test results.
    """
    train_layouts = load_layouts(TRAIN_CONFIG)
    test_layouts = load_layouts(TEST_CONFIG)

    per_seed = []
    for seed in range(seeds):
        t0 = time.time()
        apply_seed_determinism(seed)
        train_scores, unseen_scores = [], []
        for layout_name, train_config in train_layouts:
            genome, _ = train_best_genome(train_config, run_seed=seed)
            # Train score: frozen genome back on its own training layout
            train_eval = evaluate_genome(train_config, genome,
                                         run_seed=seed * 1000 + train_config.world.layout_seed)
            train_scores.append(train_eval["mean_fitness"])
            # Unseen score: same frozen genome on every held-out layout
            for test_name, test_config in test_layouts:
                test_eval = evaluate_genome(test_config, genome,
                                            run_seed=seed * 1000 + test_config.world.layout_seed)
                unseen_scores.append(test_eval["mean_fitness"])
        per_seed.append({
            "seed": seed,
            "train_score": float(np.mean(train_scores)),
            "unseen_score": float(np.mean(unseen_scores)),
            "train_scores_per_layout": train_scores,
            "unseen_scores_per_layout": unseen_scores,
        })
        print(f"seed {seed}: train={per_seed[-1]['train_score']:.4f} "
              f"unseen={per_seed[-1]['unseen_score']:.4f} "
              f"({time.time() - t0:.1f}s)")

    train_arr = np.array([s["train_score"] for s in per_seed])
    unseen_arr = np.array([s["unseen_score"] for s in per_seed])
    gaps = train_arr - unseen_arr

    # Paired test: each seed's train and unseen scores come from the SAME
    # frozen genomes, so the samples are paired by seed. Wilcoxon
    # signed-rank on the differences (not unpaired Mann-Whitney) is the
    # correct test and removes the between-seed variance.
    w_stat, p_value = wilcoxon_signed_rank(train_arr, unseen_arr)
    effect = matched_pairs_rank_biserial(train_arr, unseen_arr)
    train_ci = bootstrap_ci(train_arr, n_resamples=10000,
                            rng=np.random.default_rng(42))
    unseen_ci = bootstrap_ci(unseen_arr, n_resamples=10000,
                             rng=np.random.default_rng(43))
    gap_ci = bootstrap_ci(gaps, n_resamples=10000,
                           rng=np.random.default_rng(44))

    return {
        "experiment": "generalization",
        "n_seeds": seeds,
        "train_layouts": [name for name, _ in train_layouts],
        "test_layouts": [name for name, _ in test_layouts],
        "per_seed": per_seed,
        "train_mean": float(train_arr.mean()),
        "unseen_mean": float(unseen_arr.mean()),
        "gap_mean": float(gaps.mean()),
        "train_ci95": train_ci,
        "unseen_ci95": unseen_ci,
        "gap_ci95": gap_ci,
        "wilcoxon_w": float(w_stat),
        "p_value": float(p_value),
        "matched_pairs_rank_biserial": effect,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 generalization experiment")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (3 exploratory / 10 confirmatory)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    results = run_generalization(args.seeds)
    out_dir = Path(args.output_dir)
    # stats_summary.json carries everything including per-seed scores;
    # no duplicate per_seed_results.json is written
    write_stats_summary(out_dir / "stats_summary.json", results)
    print(f"\nGeneralization gap (train - unseen): {results['gap_mean']:.4f}")
    print(f"Train 95% CI: {results['train_ci95']}")
    print(f"Unseen 95% CI: {results['unseen_ci95']}")
    print(f"Wilcoxon signed-rank p={results['p_value']:.4f}, "
          f"matched-pairs r={results['matched_pairs_rank_biserial']:.4f}")
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()