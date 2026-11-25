"""
Showcase recorder: trains seeded controllers and records replays for the
static dashboard (webdash/build_static.py embeds the output).

Two scenarios, each recorded twice - once with untrained (generation 0)
random brains and once with the evolved ones - on an identical arena and
identical start positions, so the only difference is the controllers:

  coev  predator/prey co-evolution (configs/predator_prey.json): the
        evolved prey and predators are the top elites of the final
        generation; the untrained run uses fresh random genomes.
  solo  food-seeking prey alone (layout A1 of configs/generalization_train.json,
        the same setup as Experiments D/E): evolved = best genome of the
        final generation, untrained = the random genome the GA started with.

This script trains its OWN seeded runs; it does not read or overwrite the
committed experiment artifacts, and the simulation core is untouched.

Usage:
    python experiments/record_showcase.py
    python experiments/record_showcase.py --seed 1 --output experiments/EXP-SHOWCASE/recording.json
"""

from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Allow running as a plain script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.energy import energy_config_from_settings  # noqa: E402
from agents.organism import Organism  # noqa: E402
from analytics.replay import (recording_constants,  # noqa: E402
                              record_best_featured)
from evolution.genetic_algorithm import run_coevolution_generation  # noqa: E402
from experiments.experiment_runner import (load_calibration_scales,  # noqa: E402
                                           load_layouts,
                                           make_clone_population,
                                           make_population,
                                           train_best_genome)
from experiments.run_coevolution import (_build_world, _spawn_positions,  # noqa: E402
                                         apply_seed_determinism)
from neural.checkpoint import load_checkpoint  # noqa: E402
from neural.genome import genome_size  # noqa: E402
from simulation.geometry import AGENT_SPAWN_CLEARANCE  # noqa: E402
from simulation.world_config import BaselineConfig, load_config  # noqa: E402

COEV_CONFIG = "configs/predator_prey.json"
SOLO_BASE_CONFIG = "configs/generalization_train.json"
DEFAULT_OUTPUT = Path("experiments/EXP-SHOWCASE/recording.json")
RECORDING_VERSION = 2


def _spawn_role(genomes: List[np.ndarray], role: str, start_id: int,
                positions: List[tuple], energy_config,
                rng: np.random.Generator) -> List[Organism]:
    """One agent per genome at the given positions with randomized heading."""
    agents = []
    for offset, genome in enumerate(genomes):
        x, y = positions[offset]
        agent = Organism(start_id + offset, x, y, initial_energy=100.0,
                         role=role, energy_config=energy_config)
        agent.randomize_heading(rng)
        agent.genome = np.asarray(genome, dtype=np.float32).copy()
        agents.append(agent)
    return agents


def train_coevolution(config: BaselineConfig, seed: int):
    """
    Co-evolve predators and prey for config.evolution.generations.

    Mirrors run_coevolution's training loop minus the hall-of-fame
    evaluations (which only consume RNG state and are not needed for a
    replay), so these populations are this script's own seeded run.

    Returns:
        (final_prey_genomes, final_predator_genomes); index 0 of each is
        the elite, i.e. the best genome of the last evaluated generation.
    """
    apply_seed_determinism(seed)
    rng = np.random.default_rng(seed)
    world = _build_world(config)
    size = genome_size()
    energy_config = energy_config_from_settings(config.energy)

    def _random_population(count: int, role: str) -> List[Organism]:
        positions = _spawn_positions(world, count, rng)
        pop = []
        for i in range(count):
            agent = Organism(i, *positions[i], role=role,
                             energy_config=energy_config)
            agent.randomize_heading(rng)
            agent.genome = rng.uniform(-1, 1, size).astype(np.float32)
            pop.append(agent)
        return pop

    prey = _random_population(config.population.size, 'prey')
    predators = _random_population(config.predation.predator_count, 'predator')
    scales = load_calibration_scales()
    for generation in range(1, config.evolution.generations + 1):
        predators, prey, _ = run_coevolution_generation(
            config, predators, prey, generation, scales,
            config.fitness_weights, rng=rng)
    return [a.genome for a in prey], [a.genome for a in predators]


def replay_coevolution(config: BaselineConfig, prey_genomes: List[np.ndarray],
                       predator_genomes: List[np.ndarray], steps: int,
                       replay_seed: int) -> Dict:
    """Replay one shared-world day with the given genomes."""
    energy_config = energy_config_from_settings(config.energy)

    def prepare():
        apply_seed_determinism(replay_seed)
        world = _build_world(config)
        rng = np.random.default_rng(replay_seed + 50_000)
        positions = _spawn_positions(
            world, len(prey_genomes) + len(predator_genomes), rng)
        prey = _spawn_role(prey_genomes, 'prey', 0,
                           positions[:len(prey_genomes)], energy_config, rng)
        predators = _spawn_role(predator_genomes, 'predator', len(prey),
                                positions[len(prey_genomes):], energy_config, rng)
        # Regrowth draws from the global stream; pin it so replays repeat.
        np.random.seed(replay_seed)
        return world, prey + predators

    return record_best_featured(
        prepare, steps,
        capture_radius=config.predation.capture_radius,
        capture_energy_transfer=config.predation.capture_energy_transfer)


def replay_solo(config: BaselineConfig, genome: np.ndarray, steps: int,
                replay_seed: int) -> Dict:
    """Replay clones of one genome on the solo layout."""
    def prepare():
        apply_seed_determinism(replay_seed)
        world = _build_world(config)
        clones = make_clone_population(config, genome, replay_seed)
        # make_clone_population draws positions before the world exists,
        # so relocate any clone that landed inside a solid obstacle.
        for agent in clones:
            agent.position.x, agent.position.y = world.clear_position(
                agent.position.x, agent.position.y,
                AGENT_SPAWN_CLEARANCE, np.random)
        np.random.seed(replay_seed)
        return world, clones

    return record_best_featured(prepare, steps)


def build_scenarios(seed: int, steps: Optional[int], solo_agents: int,
                    coev_generations: Optional[int],
                    solo_generations: Optional[int],
                    solo_genome_path: Optional[Path] = None,
                    solo_config_path: Optional[Path] = None,
                    solo_only: bool = False,
                    coev_prey_genome_path: Optional[Path] = None,
                    coev_predator_genome_path: Optional[Path] = None) -> Dict:
    """Record untrained vs evolved replays using imported elite brains when given."""
    timings: Dict[str, float] = {}
    scenarios: List[Dict] = []

    # --- predator / prey -------------------------------------------------
    if not solo_only:
        coev_config = load_config(COEV_CONFIG)
        if coev_generations is not None:
            coev_config.evolution.generations = coev_generations
        coev_steps = steps or coev_config.evolution.evaluation_steps
        use_imported_coev = (coev_prey_genome_path is not None
                             and coev_predator_genome_path is not None)
        if use_imported_coev:
            started = time.perf_counter()
            prey_elite, _, prey_meta = load_checkpoint(
                str(coev_prey_genome_path))
            predator_elite, _, _ = load_checkpoint(
                str(coev_predator_genome_path))
            # The saved files hold one elite genome per role; replay clones
            # so the arena still shows a full population of that brain.
            prey_final = [np.asarray(prey_elite, dtype=np.float32).copy()
                          for _ in range(coev_config.population.size)]
            predators_final = [np.asarray(predator_elite, dtype=np.float32).copy()
                               for _ in range(coev_config.predation.predator_count)]
            reported_coev_generations = int(
                prey_meta.get("generation", coev_config.evolution.generations))
            timings["coev_import_seconds"] = time.perf_counter() - started
        else:
            started = time.perf_counter()
            prey_final, predators_final = train_coevolution(coev_config, seed)
            timings["coev_training_seconds"] = time.perf_counter() - started
            reported_coev_generations = coev_config.evolution.generations

        init_rng = np.random.default_rng(seed + 90_000)
        size = genome_size()
        prey_random = [init_rng.uniform(-1, 1, size).astype(np.float32)
                       for _ in prey_final]
        predators_random = [init_rng.uniform(-1, 1, size).astype(np.float32)
                            for _ in predators_final]

        scenarios.append({
            "id": "coev",
            "title": "Predator and prey",
            "description": ("Prey eat food and avoid predators; predators chase "
                            "and capture prey. Both populations evolve together."
                            if not use_imported_coev else
                            "Clones of the final-generation elite prey and "
                            "predator brains in the same shared world."),
            "generations": reported_coev_generations,
            "runs": {
                "untrained": replay_coevolution(
                    coev_config, prey_random, predators_random, coev_steps, seed),
                "evolved": replay_coevolution(
                    coev_config, prey_final, predators_final, coev_steps, seed),
            },
        })

    # --- solo food seeker ------------------------------------------------
    if solo_config_path is not None:
        solo_config = load_config(str(solo_config_path))
    else:
        _, solo_config = load_layouts(SOLO_BASE_CONFIG)[0]
    if solo_generations is not None and solo_genome_path is None:
        solo_config.evolution.generations = solo_generations
    solo_steps = steps or solo_config.evolution.evaluation_steps
    random_genome = make_population(solo_config, seed)[0].genome.copy()

    if solo_genome_path is not None:
        # Imported brain: skip local training entirely, which is the point
        # of running the expensive generations on Colab instead of here.
        started = time.perf_counter()
        imported, _, metadata = load_checkpoint(str(solo_genome_path))
        best_genome = np.asarray(imported, dtype=np.float32).copy()
        reported_generations = int(metadata.get("generation", 0) or 0)
        timings["solo_import_seconds"] = time.perf_counter() - started
    else:
        started = time.perf_counter()
        best_genome, _ = train_best_genome(solo_config, run_seed=seed)
        timings["solo_training_seconds"] = time.perf_counter() - started
        reported_generations = solo_config.evolution.generations

    replay_config = solo_config.model_copy(deep=True)
    replay_config.population.size = solo_agents
    scenarios.append({
        "id": "solo",
        "title": "Solo food seeker",
        "description": ("A single kind of agent that must find food before "
                        "it starves. No predators."),
        "generations": reported_generations,
        "runs": {
            "untrained": replay_solo(replay_config, random_genome,
                                     solo_steps, seed),
            "evolved": replay_solo(replay_config, best_genome,
                                   solo_steps, seed),
        },
    })
    return {"scenarios": scenarios, "timings": timings}


def build_recording(seed: int = 1, steps: Optional[int] = None,
                    solo_agents: int = 10,
                    coev_generations: Optional[int] = None,
                    solo_generations: Optional[int] = None,
                    solo_genome_path: Optional[Path] = None,
                    solo_config_path: Optional[Path] = None,
                    solo_only: bool = False,
                    coev_prey_genome_path: Optional[Path] = None,
                    coev_predator_genome_path: Optional[Path] = None) -> Dict:
    """Full recording document (JSON-serialisable)."""
    built = build_scenarios(seed, steps, solo_agents, coev_generations,
                            solo_generations, solo_genome_path,
                            solo_config_path, solo_only,
                            coev_prey_genome_path, coev_predator_genome_path)
    return {
        "version": RECORDING_VERSION,
        "seed": seed,
        "constants": recording_constants(),
        "scenarios": built["scenarios"],
        # Non-deterministic fields live apart from "scenarios" so two
        # recordings of the same seed can be compared on scenarios alone.
        "environment": {
            "training_seconds": {k: round(v, 1)
                                 for k, v in built["timings"].items()},
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
        },
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None,
                        help="replay length (default: the config's "
                             "evaluation_steps)")
    parser.add_argument("--solo-agents", type=int, default=10)
    parser.add_argument("--coev-generations", type=int, default=None)
    parser.add_argument("--solo-generations", type=int, default=None)
    parser.add_argument("--solo-genome", type=Path, default=None,
                        help="Imported best-genome .npz to replay as the evolved brain")
    parser.add_argument("--solo-config", type=Path, default=None,
                        help="Config used for the imported genome's training run")
    parser.add_argument("--solo-only", action="store_true",
                        help="Skip co-evolution training; record only the solo scenario")
    parser.add_argument("--coev-prey-genome", type=Path, default=None,
                        help="Imported elite prey .npz for the co-evolution replay")
    parser.add_argument("--coev-predator-genome", type=Path, default=None,
                        help="Imported elite predator .npz for the co-evolution replay")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    document = build_recording(args.seed, args.steps, args.solo_agents,
                               args.coev_generations, args.solo_generations,
                               args.solo_genome, args.solo_config,
                               args.solo_only, args.coev_prey_genome,
                               args.coev_predator_genome)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8")
    size_kb = args.output.stat().st_size // 1024
    print(f"Wrote {args.output} ({size_kb} KB) in "
          f"{time.perf_counter() - started:.1f}s")
    for scenario in document["scenarios"]:
        for name, run in scenario["runs"].items():
            s = run["summary"]
            print(f"  {scenario['id']}/{name}: steps={s['steps']} "
                  f"prey {s['prey_alive_end']}/{s['prey_total']} "
                  f"predators {s['predator_alive_end']}/{s['predator_total']} "
                  f"captures={s['captures']} prey_food={s['prey_food_eaten']} "
                  f"featured_alive={s['featured_alive_end']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
