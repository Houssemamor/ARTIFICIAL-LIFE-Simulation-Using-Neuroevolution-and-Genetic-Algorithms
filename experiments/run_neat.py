"""
Phase 7 driver: NEAT topology evolution on the isolated food-seeking
benchmark.

Runs the full NEAT loop (evolution/neat/algorithm.py) for
config.evolution.generations, evaluating the whole population in one
shared world per generation through the depth-layered padded batch
(neural/sparse_inference - the strategy chosen in
docs/neat_extension_report.md).

Writes experiments/EXP-NEAT/:
  - neat_summary.json       per-generation metrics + benchmark results
  - complexity_over_generations.svg   the Phase 7 complexity plot
"""

from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from simulation.world_config import load_config
from evolution.neat.algorithm import initialize_population, run_generation
from evolution.neat.innovation import InnovationTracker
from experiments.experiment_runner import load_calibration_scales
from experiments.run_coevolution import apply_seed_determinism

OUTPUT_DIR = Path("experiments/EXP-NEAT")


def run_neat(config_path: str, seed: int = 1) -> Dict:
    """
    Run the full NEAT benchmark loop for one seed.

    Returns:
        Summary dict (also written to neat_summary.json).
    """
    config = load_config(config_path)
    if config.neat is None:
        raise ValueError("run_neat requires a config with a neat section")

    apply_seed_determinism(seed)
    rng = np.random.default_rng(seed + 40000)
    calibration_scales = load_calibration_scales()

    tracker = InnovationTracker()
    genomes = initialize_population(config.population.size, tracker,
                                    config.neat, rng)

    species = None
    champion = None
    champion_fitness = -float("inf")
    generation_rows: List[Dict] = []

    for generation in range(1, config.evolution.generations + 1):
        (genomes, species, champion, champion_fitness,
         metrics) = run_generation(
            config, genomes, generation, tracker, species, champion,
            champion_fitness, calibration_scales, rng)
        row = asdict(metrics)
        row["champion_complexity"] = champion.complexity()
        row["innovation_counter"] = tracker.counter
        generation_rows.append(row)
        print(f"gen {generation:3d}: species={metrics.species_count:3d} "
              f"fitness mean/max={metrics.mean_fitness:.4f}/"
              f"{metrics.max_fitness:.4f} complexity mean/max="
              f"{metrics.mean_complexity:.1f}/{metrics.max_complexity} "
              f"food={metrics.mean_food_eaten:.2f} alive="
              f"{metrics.num_alive_end}")

    return {
        "config": config.experiment_name,
        "seed": seed,
        "population_size": config.population.size,
        "generations": generation_rows,
        "final_champion_complexity": champion.complexity(),
        "final_species_count": len(species),
        "total_innovations": tracker.counter - 1,
    }


def write_complexity_svg(rows: List[Dict], path: Path) -> None:
    """
    Render the complexity-over-generations plot as a standalone SVG.

    Hand-rolled polyline rendering, same justification as the Phase 5
    fitness-curve SVG: one static image does not justify matplotlib.
    Mean complexity (solid) and max complexity (dashed).
    """
    width, height, margin = 640, 400, 50
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    generations = [row["generation"] for row in rows]
    mean_complexity = [row["mean_complexity"] for row in rows]
    max_complexity = [row["max_complexity"] for row in rows]
    y_min, y_max = min(mean_complexity + max_complexity), max(
        mean_complexity + max_complexity)
    y_range = (y_max - y_min) or 1.0
    x_max = max(1, generations[-1] - 1)

    def x(g): return margin + ((g - 1) / x_max) * plot_w
    def y(v): return margin + plot_h - ((v - y_min) / y_range) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}">']
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="24" font-size="15" '
                 f'text-anchor="middle">NEAT complexity over generations'
                 f'</text>')
    parts.append(f'<line x1="{margin}" y1="{margin+plot_h}" '
                 f'x2="{margin+plot_w}" y2="{margin+plot_h}" stroke="black"/>')
    parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
                 f'y2="{margin+plot_h}" stroke="black"/>')
    parts.append(f'<text x="{margin-8}" y="{y(y_min)+4}" font-size="11" '
                 f'text-anchor="end">{y_min:.0f}</text>')
    parts.append(f'<text x="{margin-8}" y="{y(y_max)+4}" font-size="11" '
                 f'text-anchor="end">{y_max:.0f}</text>')
    parts.append(f'<text x="{margin+plot_w}" y="{margin+plot_h+16}" '
                 f'font-size="11" text-anchor="middle">generation</text>')
    parts.append(f'<text x="14" y="{height/2}" font-size="11" '
                 f'text-anchor="middle" transform="rotate(-90 14 '
                 f'{height/2})">complexity</text>')
    for values, dash, color in ((mean_complexity, None, "#1f77b4"),
                                (max_complexity, "6 4", "#d62728")):
        points = " ".join(f"{x(g):.1f},{y(v):.1f}"
                          for g, v in zip(generations, values))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{points}" fill="none" '
                     f'stroke="{color}"{dash_attr} stroke-width="2"/>')
    for i, (label, color, dash) in enumerate(
            (("mean complexity", "#1f77b4", None),
             ("max complexity", "#d62728", "6 4"))):
        y_pos = margin - 30 + i * 16
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{margin}" y1="{y_pos}" x2="{margin+24}" '
                     f'y2="{y_pos}" stroke="{color}"{dash_attr} '
                     f'stroke-width="3"/>')
        parts.append(f'<text x="{margin+30}" y="{y_pos+4}" font-size="12">'
                     f'{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def benchmark_batching(config_path: str, seed: int = 1,
                       steps: int = 150) -> Dict:
    """
    Benchmark the three batching strategies for variable-topology
    inference (plan step 7): population-wide depth-layered batch,
    per-species batches, and the unbatched per-agent reference.

    Measures wall-clock steps/second on one episode of the real
    benchmark world, using the champion-ish population: a realistic
    mix of minimal and structurally-mutated genomes.
    """
    import time
    import torch
    from agents.sensors import RayCaster
    from agents.organism import Organism
    from simulation.environment import World
    from neural.sparse_inference import (compile_population, run_compiled,
                                         forward_single)
    from evolution.neat.mutation import add_node

    config = load_config(config_path)
    apply_seed_determinism(seed)
    rng = np.random.default_rng(seed + 50000)
    tracker = InnovationTracker()
    genomes = initialize_population(config.population.size, tracker,
                                    config.neat, rng)
    # Structural variety, as a mid-run population would have
    for index, genome in enumerate(genomes):
        for _ in range(index % 4):
            add_node(genome, tracker, rng)

    if config.world.layout_seed is not None:
        np.random.seed(config.world.layout_seed)
    world = World(config.world.width, config.world.height)
    world.load_from_config(config.model_dump())
    raycaster = RayCaster()
    agents = []
    for index in range(len(genomes)):
        x = rng.uniform(world.boundary_margin, world.width - world.boundary_margin)
        y = rng.uniform(world.boundary_margin, world.height - world.boundary_margin)
        agents.append(Organism(index, x, y, initial_energy=100.0))

    # A few species groups, as speciation would produce
    species_ids = [index % 5 for index in range(len(genomes))]
    observations = torch.as_tensor(
        raycaster.observe_population(agents, world), dtype=torch.float32)

    # Population-wide: compiled once, run per step (the algorithm's path)
    compiled_pop = compile_population(genomes)
    start = time.perf_counter()
    for _ in range(5):
        compile_population(genomes)
    compile_seconds = (time.perf_counter() - start) / 5

    start = time.perf_counter()
    for _ in range(steps):
        run_compiled(compiled_pop, observations)
    population_rate = steps / (time.perf_counter() - start)

    # Per-species: one compile per species group, then run per step
    groups = {}
    for index, species_id in enumerate(species_ids):
        groups.setdefault(species_id, []).append(index)
    compiled_species = {
        sid: compile_population([genomes[i] for i in indices])
        for sid, indices in groups.items()
    }
    start = time.perf_counter()
    for _ in range(steps):
        for sid, indices in groups.items():
            run_compiled(compiled_species[sid], observations[indices])
    species_rate = steps / (time.perf_counter() - start)

    # Unbatched per-agent reference
    start = time.perf_counter()
    for _ in range(steps):
        for index, genome in enumerate(genomes):
            forward_single(observations[index:index + 1], genome)
    single_rate = steps / (time.perf_counter() - start)

    mean_nodes = sum(len(g.nodes) for g in genomes) / len(genomes)
    return {
        "population": len(genomes),
        "mean_nodes_per_genome": mean_nodes,
        "steps": steps,
        "compile_seconds_per_episode": compile_seconds,
        "batched_population_steps_per_second": population_rate,
        "batched_per_species_steps_per_second": species_rate,
        "per_agent_steps_per_second": single_rate,
        "speedup_population_vs_per_agent": population_rate / single_rate,
        "speedup_species_vs_per_agent": species_rate / single_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7 NEAT food-seeking benchmark")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run only the batching benchmark")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.benchmark_only:
        result = benchmark_batching(args.config, seed=args.seed)
        with open(OUTPUT_DIR / "batching_benchmark.json", "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return

    summary = run_neat(args.config, seed=args.seed)
    summary["batching_benchmark"] = benchmark_batching(args.config,
                                                        seed=args.seed)
    with open(OUTPUT_DIR / "neat_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_complexity_svg(summary["generations"],
                         OUTPUT_DIR / "complexity_over_generations.svg")
    print(f"NEAT run: {len(summary['generations'])} generations -> "
          f"{OUTPUT_DIR}/neat_summary.json")
    print(f"Final champion complexity: "
          f"{summary['final_champion_complexity']}, "
          f"species: {summary['final_species_count']}, "
          f"innovations: {summary['total_innovations']}")


if __name__ == "__main__":
    main()
