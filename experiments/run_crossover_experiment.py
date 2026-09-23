"""
Experiment E: crossover-method comparison (Phase 3's deferred deliverable).

Trains a GA under each crossover method (blend / uniform / none
i.e. mutation-only), freezes each condition's best genome, and scores all
conditions on one common balanced metric. Conditions share seeds and
initial populations per seed index, so comparisons use the PAIRED
pipeline. Also writes the fitness-over-generations curve as an SVG -
the Phase 3 "smoke-test evidence that fitness improves" deliverable.

Usage:
    python experiments/run_crossover_experiment.py --seeds 3    # exploratory
    python experiments/run_crossover_experiment.py --seeds 10   # confirmatory
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from analytics.statistics import compare_conditions_paired
from experiments.experiment_runner import (evaluate_genome, load_layouts,
                                           train_best_genome,
                                           write_stats_summary)
from simulation.determinism import (DeterminismConfig,
                                    set_deterministic_seeds)

BASE_CONFIG = "configs/generalization_train.json"
OUTPUT_DIR = Path("experiments/EXP-E")
CROSSOVER_METHODS = ["blend", "uniform", "none"]


def apply_seed_determinism(run_seed: int) -> None:
    """Enforce the cpu-deterministic tier (same derivation as other runners)."""
    set_deterministic_seeds(DeterminismConfig(
        torch_seed=run_seed,
        numpy_seed=run_seed + 10_000,
        random_seed=run_seed + 20_000,
        device="cpu",
        reproducibility_tier="cpu-deterministic",
        num_threads=1,
    ))


def _condition_configs():
    """One validated config per crossover method, layout A1 shared."""
    _, base = load_layouts(BASE_CONFIG)[0]
    for method in CROSSOVER_METHODS:
        config = base.model_copy(deep=True)
        config.crossover_method = method
        config.experiment_name = f"experiment_e_{method}"
        yield method, config


def run_experiment_e(seeds: int) -> Dict:
    """
    Run all three crossover conditions across seeds.

    Per condition and seed: train under that crossover method, freeze the
    best genome, evaluate with the balanced weights on the shared layout.
    Training histories are kept to draw the fitness-over-generations curve.

    Returns:
        Dict with per-seed scores per method, the paired comparison
        summary, and per-generation mean/max fitness per method.
    """
    per_condition: Dict[str, List[float]] = {m: [] for m in CROSSOVER_METHODS}
    histories: Dict[str, List[List[Dict]]] = {m: [] for m in CROSSOVER_METHODS}

    for method, cond_config in _condition_configs():
        for seed in range(seeds):
            t0 = time.time()
            apply_seed_determinism(seed)
            genome, history = train_best_genome(cond_config, run_seed=seed)
            evaluation = evaluate_genome(
                cond_config, genome,
                run_seed=seed * 1000 + cond_config.world.layout_seed)
            per_condition[method].append(evaluation["mean_fitness"])
            histories[method].append(history)
            print(f"{method} seed {seed}: balanced="
                  f"{evaluation['mean_fitness']:.4f} "
                  f"({time.time() - t0:.1f}s)")

    comparison = compare_conditions_paired(
        {m: np.array(scores) for m, scores in per_condition.items()},
        metric="balanced_evaluation_fitness",
        alpha=0.05,
    )

    # Per-generation aggregate curve per method: mean over seeds of the
    # per-generation population mean and max fitness. This is the Phase 3
    # "does fitness improve over generations" evidence.
    fitness_curves = {}
    for method in CROSSOVER_METHODS:
        gens = len(histories[method][0])
        mean_over_seeds = []
        max_over_seeds = []
        for g in range(gens):
            mean_over_seeds.append(float(np.mean(
                [h[g]["mean_fitness"] for h in histories[method]])))
            max_over_seeds.append(float(np.mean(
                [h[g]["max_fitness"] for h in histories[method]])))
        fitness_curves[method] = {"generation": list(range(gens)),
                                  "mean_fitness": mean_over_seeds,
                                  "max_fitness": max_over_seeds}

    return {
        "experiment": "experiment_e_crossover_method",
        "n_seeds": seeds,
        "per_condition_scores": per_condition,
        "comparison": comparison,
        "fitness_curves": fitness_curves,
    }


def write_fitness_curve_svg(curves: Dict, path: Path) -> None:
    """
    Render the fitness-over-generations curves as a standalone SVG.

    Hand-rolled polyline rendering: a plot is a Phase 3 deliverable, but
    adding matplotlib as a dependency for one static image is not
    justified. 30 lines of SVG cover it.
    """
    width, height, margin = 640, 400, 50
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    colors = {"blend": "#1f77b4", "uniform": "#2ca02c", "none": "#d62728"}
    all_vals = [v for c in curves.values() for k in ("mean_fitness", "max_fitness")
                for v in c[k]]
    y_min, y_max = min(all_vals), max(all_vals)
    y_range = (y_max - y_min) or 1.0
    n_gens = len(next(iter(curves.values()))["generation"])
    x_max = max(1, n_gens - 1)

    def x(g): return margin + (g / x_max) * plot_w
    def y(v): return margin + plot_h - ((v - y_min) / y_range) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="24" font-size="15" text-anchor="middle">'
                 f'Fitness over generations (mean over seeds)</text>')
    # Axes
    parts.append(f'<line x1="{margin}" y1="{margin+plot_h}" x2="{margin+plot_w}" '
                 f'y2="{margin+plot_h}" stroke="black"/>')
    parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
                 f'y2="{margin+plot_h}" stroke="black"/>')
    parts.append(f'<text x="{margin-8}" y="{y(y_min)+4}" font-size="11" '
                 f'text-anchor="end">{y_min:.2f}</text>')
    parts.append(f'<text x="{margin-8}" y="{y(y_max)+4}" font-size="11" '
                 f'text-anchor="end">{y_max:.2f}</text>')
    parts.append(f'<text x="{margin+plot_w}" y="{margin+plot_h+16}" font-size="11" '
                 f'text-anchor="middle">generation</text>')
    # Solid line = mean fitness, dashed = max fitness, per method
    for method, c in curves.items():
        color = colors.get(method, "gray")
        for key, dash in (("mean_fitness", None), ("max_fitness", "6 4")):
            pts = " ".join(f"{x(g):.1f},{y(v):.1f}"
                           for g, v in zip(c["generation"], c[key]))
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}"'
                         f'{dash_attr} stroke-width="2"/>')
    # Legend
    for i, method in enumerate(colors):
        parts.append(f'<line x1="{margin}" y1="{margin-30+i*16}" x2="{margin+24}" '
                     f'y2="{margin-30+i*16}" stroke="{colors[method]}" stroke-width="3"/>')
        parts.append(f'<text x="{margin+30}" y="{margin-26+i*16}" font-size="12">'
                     f'{method}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment E: crossover comparison")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (3 exploratory / 10 confirmatory)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    results = run_experiment_e(args.seeds)
    out_dir = Path(args.output_dir)
    write_stats_summary(out_dir / "stats_summary.json", results)
    write_fitness_curve_svg(results["fitness_curves"],
                            out_dir / "fitness_curve.svg")
    print(f"\nCondition means: "
          + ", ".join(f"{k}={np.mean(v):.4f}"
                      for k, v in results["per_condition_scores"].items()))
    print(f"Significant pairs (Holm-Bonferroni a=0.05): "
          f"{results['comparison']['significant_pairs']}")
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()