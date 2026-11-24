"""
Full-state checkpointing for headless GA runs.

neural/checkpoint.py stores one genome; a run needs the whole population,
both RNG streams, the generation index and the accumulated history to resume
bit-identically after a Colab disconnect. This module owns that format and
the incremental per-generation CSV writer.

Format (.npz, compressed, no pickle):
    version, generation, run_seed, config_json, history_json
    genomes, ids, positions, velocities, headings, energy, health, age
    fitness, collisions, food_eaten, alive, roles
    legacy_state_name, legacy_state_keys, legacy_state_pos,
    legacy_state_has_gauss, legacy_state_cached_gauss, rng_state_json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agents.energy import energy_config_from_settings
from agents.organism import Organism
from neural.genome import genome_size
from simulation.world_config import BaselineConfig

STATE_VERSION = 1


def _rng_to_json(rng: np.random.Generator) -> str:
    """Encode a Generator's bit-generator state without pickle."""
    state = rng.bit_generator.state
    encoded = {
        "bit_generator": state["bit_generator"],
        "state": {key: int(value) for key, value in state["state"].items()},
        "has_uint32": int(state["has_uint32"]),
        "uinteger": int(state["uinteger"]),
    }
    return json.dumps(encoded)


def _rng_from_json(encoded: str) -> Dict[str, Any]:
    """Decode the bit-generator state produced by _rng_to_json."""
    state = json.loads(encoded)
    return {
        "bit_generator": state["bit_generator"],
        "state": dict(state["state"]),
        "has_uint32": state["has_uint32"],
        "uinteger": state["uinteger"],
    }


def save_run_state(
    path: str | Path,
    generation: int,
    population: List[Organism],
    rng: np.random.Generator,
    history: List[Dict],
    config: BaselineConfig,
    run_seed: int,
) -> None:
    """
    Write everything needed to continue a run at `generation`.

    Args:
        path: Destination .npz file; parent directories are created.
        generation: Index of the next generation to execute.
        population: Current organisms, including their genomes.
        rng: GA random generator whose bit state must be preserved.
        history: Per-generation metric dictionaries collected so far.
        config: Validated config, stored for a mismatch check on resume.
        run_seed: Seed that produced the initial population.
    """
    count = len(population)
    legacy = np.random.get_state()
    genome_length = genome_size()

    payload: Dict[str, np.ndarray] = {
        "version": np.array(STATE_VERSION, dtype=np.int32),
        "generation": np.array(generation, dtype=np.int64),
        "run_seed": np.array(run_seed, dtype=np.int64),
        "config_json": np.array(json.dumps(config.model_dump(), sort_keys=True)),
        "history_json": np.array(json.dumps(history)),
        "rng_state_json": np.array(_rng_to_json(rng)),
        "genomes": np.array(
            [agent.genome for agent in population], dtype=np.float32
        ).reshape(count, genome_length),
        "ids": np.array([agent.id for agent in population], dtype=np.int64),
        "positions": np.array(
            [[a.position.x, a.position.y] for a in population],
            dtype=np.float64).reshape(count, 2),
        "velocities": np.array(
            [[a.velocity.x, a.velocity.y] for a in population],
            dtype=np.float64).reshape(count, 2),
        "headings": np.array([a.heading for a in population], dtype=np.float64),
        "energy": np.array([a.energy for a in population], dtype=np.float64),
        "health": np.array([a.health for a in population], dtype=np.float64),
        "age": np.array([a.age for a in population], dtype=np.int64),
        "fitness": np.array([a.fitness for a in population], dtype=np.float64),
        "collisions": np.array([a.collisions for a in population], dtype=np.int64),
        "food_eaten": np.array([a.food_eaten for a in population], dtype=np.int64),
        "alive": np.array([a.is_alive for a in population], dtype=bool),
        "roles": np.array([a.role for a in population]),
        "legacy_state_name": np.array(str(legacy[0])),
        "legacy_state_keys": np.asarray(legacy[1], dtype=np.uint32),
        "legacy_state_pos": np.array(legacy[2], dtype=np.int64),
        "legacy_state_has_gauss": np.array(legacy[3], dtype=np.int64),
        "legacy_state_cached_gauss": np.array(legacy[4], dtype=np.float64),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def load_run_state(path: str | Path) -> Dict[str, Any]:
    """
    Load a checkpoint written by save_run_state.

    Raises:
        ValueError: If the version is unknown or the file is truncated.
    """
    with np.load(path, allow_pickle=False) as data:
        version = int(data["version"])
        if version != STATE_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {version}; expected {STATE_VERSION}")
        return {
            "generation": int(data["generation"]),
            "run_seed": int(data["run_seed"]),
            "config_json": str(data["config_json"].item()),
            "history": json.loads(str(data["history_json"].item())),
            "rng_state": _rng_from_json(str(data["rng_state_json"].item())),
            "genomes": data["genomes"].copy(),
            "ids": data["ids"].copy(),
            "positions": data["positions"].copy(),
            "velocities": data["velocities"].copy(),
            "headings": data["headings"].copy(),
            "energy": data["energy"].copy(),
            "health": data["health"].copy(),
            "age": data["age"].copy(),
            "fitness": data["fitness"].copy(),
            "collisions": data["collisions"].copy(),
            "food_eaten": data["food_eaten"].copy(),
            "alive": data["alive"].copy(),
            "roles": data["roles"].copy(),
            "legacy": (
                str(data["legacy_state_name"].item()),
                data["legacy_state_keys"].copy(),
                int(data["legacy_state_pos"]),
                int(data["legacy_state_has_gauss"]),
                float(data["legacy_state_cached_gauss"]),
            ),
        }


def restore_rng(state: Dict[str, Any], rng: np.random.Generator) -> None:
    """Restore both the GA generator and the global legacy RNG stream."""
    rng.bit_generator.state = state["rng_state"]
    legacy = state["legacy"]
    np.random.set_state(
        (legacy[0], legacy[1], legacy[2], legacy[3], legacy[4]))


def restore_population(state: Dict[str, Any],
                       config: BaselineConfig) -> List[Organism]:
    """Rebuild organisms from checkpoint arrays, preserving their state."""
    energy_config = energy_config_from_settings(config.energy)
    population = []
    for index in range(len(state["ids"])):
        agent = Organism(int(state["ids"][index]),
                         float(state["positions"][index, 0]),
                         float(state["positions"][index, 1]),
                         initial_energy=float(state["energy"][index]),
                         role=str(state["roles"][index]),
                         energy_config=energy_config)
        agent.velocity.x = float(state["velocities"][index, 0])
        agent.velocity.y = float(state["velocities"][index, 1])
        agent.heading = float(state["headings"][index])
        agent.energy = float(state["energy"][index])
        agent.health = float(state["health"][index])
        agent.age = int(state["age"][index])
        agent.fitness = float(state["fitness"][index])
        agent.collisions = int(state["collisions"][index])
        agent.food_eaten = int(state["food_eaten"][index])
        agent.is_alive = bool(state["alive"][index])
        agent.genome = state["genomes"][index].copy()
        population.append(agent)
    return population


def verify_config(state: Dict[str, Any], config: BaselineConfig) -> None:
    """Refuse to resume a run whose config would change the experiment.

    The generation budget is deliberately excluded: extending a run is the
    reason to resume, while population, world, weights, seeds and evaluation
    length must stay identical or the history is no longer comparable.
    """
    saved = json.loads(state["config_json"])
    current = config.model_dump()
    for document in (saved, current):
        document["evolution"].pop("generations", None)
        document["experiment"].pop("save_interval", None)
    if json.dumps(saved, sort_keys=True) != json.dumps(current, sort_keys=True):
        raise ValueError(
            "Checkpoint config does not match the supplied config; "
            "resume would mix two experimental regimes")


def append_history_csv(path: str | Path, history: List[Dict]) -> None:
    """
    Append generations to a CSV file, writing the header on first use.

    The caller passes only rows produced since the previous write, so a
    resumed run cannot duplicate rows already persisted.
    """
    if not history:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_header = not destination.exists() or destination.stat().st_size == 0
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(history)
