"""
Phase 6 driver: co-evolution with hall-of-fame evaluation, plus the
long-horizon ecosystem stability mode.

Two modes:
  --mode coevolution  - alternating shared-world evaluation and per-role
      GA replacement. Every snapshot_interval generations the best
      genome per role is frozen into the hall-of-fame archive, and the
      current best controllers are evaluated against the frozen
      opponents (win_rate_against_archive). Writes
      experiments/EXP-COEV/coevolution_summary.json.
  --mode ecosystem  - no GA: a mixed population persists through
      consecutive 'days' (evaluation_steps each) with capture deaths
      and in-episode reproduction the only population dynamics. This is
      the long-horizon stability test harness from the plan's exit
      criterion. Writes experiments/EXP-COEV/ecosystem_summary.json.

Determinism: seeds are enforced per run (torch=s, numpy=s+10000,
random=s+20000), matching the Phase 5 runners.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from simulation.determinism import set_deterministic_seeds, DeterminismConfig
from simulation.environment import World
from simulation.engine import step_simulation, resolve_captures
from simulation.world_config import load_config, BaselineConfig, ReproductionConfig
from agents.organism import Organism
from agents.energy import energy_config_from_settings
from agents.sensors import RayCaster
from neural.genome import genome_size
from evolution.reproduction import reproduction_pass
from evolution.genetic_algorithm import run_coevolution_generation, compute_fitness
from analytics.hall_of_fame import HallOfFame, win_rate_against_archive
from experiments.experiment_runner import load_calibration_scales

OUTPUT_DIR = Path("experiments/EXP-COEV")


def apply_seed_determinism(run_seed: int) -> None:
    """Per-seed determinism, offset-matched to the Phase 5 runners."""
    set_deterministic_seeds(DeterminismConfig(
        torch_seed=run_seed,
        numpy_seed=run_seed + 10000,
        random_seed=run_seed + 20000,
        device="cpu",
        reproducibility_tier="cpu-deterministic",
        num_threads=1,
    ))


def _spawn_positions(world: World, count: int,
                     rng: np.random.Generator) -> List[tuple]:
    """Uniform in-bounds spawn positions (no corner-line artifact)."""
    m = world.boundary_margin
    return [(float(rng.uniform(m, world.width - m)),
             float(rng.uniform(m, world.height - m))) for _ in range(count)]


def _build_world(config: BaselineConfig) -> World:
    """World with the config's fixed layout (layout_seed applies)."""
    if config.world.layout_seed is not None:
        np.random.seed(config.world.layout_seed)
    world = World(config.world.width, config.world.height)
    world.load_from_config(config.model_dump())
    return world


def make_mixed_population(config: BaselineConfig, world: World,
                          rng: np.random.Generator) -> List[Organism]:
    """Random-genome mixed population: prey then predators."""
    size = genome_size()
    positions = _spawn_positions(
        world, config.population.size + config.predation.predator_count, rng)
    energy_config = energy_config_from_settings(config.energy)
    agents: List[Organism] = []
    for i in range(config.population.size):
        agent = Organism(i, *positions[i], role='prey',
                         energy_config=energy_config)
        agent.genome = rng.uniform(-1, 1, size).astype(np.float32)
        agents.append(agent)
    for j in range(config.predation.predator_count):
        agent = Organism(len(agents), *positions[config.population.size + j],
                         role='predator', energy_config=energy_config)
        agent.genome = rng.uniform(-1, 1, size).astype(np.float32)
        agents.append(agent)
    return agents


def run_ecosystem_days(config: BaselineConfig, days: int,
                       rng: Optional[np.random.Generator] = None,
                       seed: int = 1) -> List[dict]:
    """
    Ecosystem stability mode: no GA. Survivors carry over day to day;
    captures kill prey and reproduction births offspring in between.

    Args:
        config: Config with predation + reproduction sections.
        days: Number of consecutive days (evaluation_steps each).
        rng: Generator for spawns/reproduction.
        seed: Determinism seed for this run.

    Returns:
        List of per-day stats dicts (counts are live agents at the day
        boundary, after that day's deaths and births).
    """
    if config.predation is None or config.predation.predator_count <= 0:
        raise ValueError("ecosystem mode requires config.predation")
    repro = config.reproduction or ReproductionConfig(enabled=False)

    # The behavioral rng (spawns, reproduction jitter, mutation) must be
    # derived from the seed: default_rng() with no argument draws OS
    # entropy, which silently made every ecosystem run non-reproducible
    # regardless of --seed.
    if rng is None:
        rng = np.random.default_rng(seed + 30000)

    # Pin torch and the global numpy RNG (world layout reseeds the
    # global stream to layout_seed inside _build_world; food regrowth
    # draws from it thereafter)
    apply_seed_determinism(seed)
    world = _build_world(config)
    agents = make_mixed_population(config, world, rng)
    raycaster = RayCaster()
    next_id = len(agents)
    # Exploration = displacement from spawn, tracked per agent across the
    # whole run (an agent's spawn is its original position)
    spawn_positions = {agent.id: (agent.position.x, agent.position.y)
                       for agent in agents}
    # Reproduction eligibility gates on agent.fitness, which the GA
    # normally sets at generation end. Ecosystem mode has no GA, so
    # fitness is recomputed at every day boundary from the same
    # measured calibration scales the GA uses.
    calibration_scales = load_calibration_scales()

    day_stats: List[dict] = []
    for day in range(days):
        captures_day = 0
        births_day = 0
        for _ in range(config.evolution.evaluation_steps):
            live = [a for a in agents if a.is_alive]
            if not live:
                break
            step_simulation(world, agents, raycaster)
            captures_day += resolve_captures(
                agents,
                capture_radius=config.predation.capture_radius,
                energy_transfer=config.predation.capture_energy_transfer,
            )
            newborns, next_id = reproduction_pass(
                agents, world, repro, next_id, rng)
            # A newborn's spawn position is its birth site: without this
            # its exploration stays 0 for life, biasing both fitness and
            # the reproduction gate toward sedentary newborns.
            for newborn in newborns:
                spawn_positions[newborn.id] = (newborn.position.x,
                                               newborn.position.y)
            agents.extend(newborns)
            births_day += len(newborns)

        # Day boundary: refresh fitness so the reproduction gate sees
        # real achievement, then record the day's counts
        for agent in agents:
            spawn_x, spawn_y = spawn_positions[agent.id]
            exploration = float(np.hypot(agent.position.x - spawn_x,
                                         agent.position.y - spawn_y))
            agent.fitness = compute_fitness(
                agent, agent.age, agent.food_eaten, exploration,
                agent.collisions, calibration_scales, config.fitness_weights)

        prey_alive = sum(1 for a in agents if a.is_alive and a.role == 'prey')
        predator_alive = sum(1 for a in agents
                             if a.is_alive and a.role == 'predator')
        day_stats.append({
            "day": day,
            "prey_alive": prey_alive,
            "predator_alive": predator_alive,
            "captures": captures_day,
            "births": births_day,
            "total_agents": len(agents),
        })

        # Stop early on extinction: nothing left to observe
        if prey_alive == 0 or predator_alive == 0:
            break

    return day_stats


def run_coevolution(config: BaselineConfig, seed: int = 1) -> dict:
    """
    Full co-evolution run with hall-of-fame snapshots and frozen-
    opponent evaluations.

    Args:
        config: Config with predation section (reproduction is unused
            here; GA replacement manages population sizes).
        seed: Determinism seed for this run.

    Returns:
        Summary dict written to coevolution_summary.json.
    """
    if config.predation is None or config.predation.predator_count <= 0:
        raise ValueError("coevolution mode requires config.predation")
    # The hall-of-fame snapshot below takes population[0] as the
    # generation's best; evolve_genomes orders elites first, which is
    # only the true best when elitism is at least 1
    if config.evolution.elitism_count < 1:
        raise ValueError("coevolution mode requires elitism_count >= 1")

    apply_seed_determinism(seed)
    rng = np.random.default_rng(seed)
    world = _build_world(config)

    # Initial random populations, one organism per role genome
    size = genome_size()

    def _random_population(count: int, role: str) -> List[Organism]:
        pop = []
        positions = _spawn_positions(world, count, rng)
        for i in range(count):
            agent = Organism(i, *positions[i], role=role)
            agent.genome = rng.uniform(-1, 1, size).astype(np.float32)
            pop.append(agent)
        return pop

    prey = _random_population(config.population.size, 'prey')
    predators = _random_population(config.predation.predator_count, 'predator')

    calibration_scales = load_calibration_scales()
    # experiment.save_interval doubles as the hall-of-fame snapshot
    # interval: "how often the run freezes progress" is the same knob
    # for both, and adding a second interval field would only duplicate it
    hof = HallOfFame(
        snapshot_interval=config.experiment.save_interval,
        max_entries=10,
    )

    generations: List[dict] = []
    hof_evaluations: List[dict] = []
    for generation in range(1, config.evolution.generations + 1):
        predators, prey, metrics = run_coevolution_generation(
            config, predators, prey, generation,
            calibration_scales, config.fitness_weights, rng=rng,
        )
        generations.append({
            "generation": metrics.generation,
            "prey_fitness_mean": metrics.prey_fitness_mean,
            "prey_fitness_max": metrics.prey_fitness_max,
            "predator_fitness_mean": metrics.predator_fitness_mean,
            "predator_fitness_max": metrics.predator_fitness_max,
            "captures": metrics.captures,
            "prey_alive_end": metrics.prey_alive_end,
            "predator_alive_end": metrics.predator_alive_end,
        })

        # Freeze the generation's best per role at snapshot generations,
        # then evaluate both current bests against the frozen opponents.
        # evolve_genomes orders elites first, so population[0] carries
        # the best genome of the generation just evaluated.
        best_prey_genome = prey[0].genome
        best_predator_genome = predators[0].genome

        took_snapshot = hof.maybe_snapshot(
            generation, 'prey', best_prey_genome,
            metrics.prey_fitness_max)
        hof.maybe_snapshot(
            generation, 'predator', best_predator_genome,
            metrics.predator_fitness_max)

        if took_snapshot and generation > config.experiment.save_interval:
            # Both archives have at least one earlier entry to duel
            hof_evaluations.append({
                "generation": generation,
                "predator_win_rate_vs_archive": win_rate_against_archive(
                    'predator', best_predator_genome, hof, config, rng=rng),
                "prey_win_rate_vs_archive": win_rate_against_archive(
                    'prey', best_prey_genome, hof, config, rng=rng),
                "archive_size": len(hof.entries('prey')),
            })

    return {
        "config": config.experiment_name,
        "seed": seed,
        "generations": generations,
        "hof_evaluations": hof_evaluations,
    }


def run_stability_sweep(config: BaselineConfig, seeds: List[int],
                        days: int = 10) -> Dict:
    """
    Run the ecosystem stability harness across seeds; writes
    experiments/EXP-COEV/stability_sweep.json (see coevolution_notes).
    """
    per_seed: List[Dict] = []
    for seed in seeds:
        day_rows = run_ecosystem_days(config, days=days, seed=seed)
        min_prey = min((row["prey_alive"] for row in day_rows), default=0)
        min_predators = min((row["predator_alive"] for row in day_rows),
                            default=0)
        full_horizon = len(day_rows) == days
        survived = full_horizon and min_prey > 0 and min_predators > 0
        per_seed.append({
            "seed": seed,
            "min_prey_alive": min_prey,
            "min_predator_alive": min_predators,
            "full_horizon": full_horizon,
            "no_extinction": survived,
            "prey_alive_end": day_rows[-1]["prey_alive"] if day_rows else 0,
            "predator_alive_end": (day_rows[-1]["predator_alive"]
                                   if day_rows else 0),
        })
        print(f"seed {seed}: days={len(day_rows)} "
              f"min=({min_prey},{min_predators}) "
              f"{'PASS' if survived else 'FAIL'}")
    return {
        "days": days,
        "n_seeds": len(seeds),
        "collapsed": sum(1 for row in per_seed if not row["no_extinction"]),
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 co-evolution / ecosystem driver")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str,
                        choices=['coevolution', 'ecosystem', 'stability-sweep'],
                        default='coevolution')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--seeds', type=int, default=10,
                        help='stability-sweep: number of consecutive seeds '
                             'starting at --seed')
    parser.add_argument('--days', type=int, default=10,
                        help='ecosystem mode: consecutive days to run')
    args = parser.parse_args()

    config = load_config(args.config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == 'stability-sweep':
        seeds = list(range(args.seed, args.seed + args.seeds))
        summary = run_stability_sweep(config, seeds, days=args.days)
        out = OUTPUT_DIR / "stability_sweep.json"
        with open(out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Stability sweep: {summary['collapsed']} of "
              f"{summary['n_seeds']} seeds collapse -> {out}")
    elif args.mode == 'ecosystem':
        stats = run_ecosystem_days(config, args.days, seed=args.seed)
        out = OUTPUT_DIR / "ecosystem_summary.json"
        with open(out, 'w') as f:
            json.dump({"seed": args.seed, "days": stats}, f, indent=2)
        print(f"Ecosystem run: {len(stats)} days -> {out}")
        for day in stats:
            print(f"  day {day['day']}: prey={day['prey_alive']} "
                  f"predators={day['predator_alive']} "
                  f"captures={day['captures']} births={day['births']}")
    else:
        summary = run_coevolution(config, seed=args.seed)
        out = OUTPUT_DIR / "coevolution_summary.json"
        with open(out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Co-evolution run: {len(summary['generations'])} generations -> {out}")
        for entry in summary["hof_evaluations"]:
            print(f"  gen {entry['generation']}: "
                  f"predator win-rate vs archive "
                  f"{entry['predator_win_rate_vs_archive']:.2f}, "
                  f"prey win-rate vs archive "
                  f"{entry['prey_win_rate_vs_archive']:.2f}")


if __name__ == "__main__":
    main()
