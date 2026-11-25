#!/usr/bin/env python3
"""Headless tensor-native predator/prey ecosystem runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.experiment_runner import load_calibration_scales  # noqa: E402
from neural.batched_inference import resolve_device  # noqa: E402
from neural.checkpoint import save_checkpoint  # noqa: E402
from simulation.determinism import (DeterminismConfig,  # noqa: E402
                                    set_deterministic_seeds)
from simulation.world_config import load_config  # noqa: E402
from tensor_ecosystem import EcosystemConfig, TensorEcosystem  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-dir", type=Path, default=None,
                        help="Save the final prey.npz and predator.npz genomes here")
    args = parser.parse_args()

    config = load_config(args.config)
    config.device = args.device
    device_name = resolve_device(args.device).type
    tier = "cpu-deterministic" if device_name == "cpu" else "non-deterministic"
    config.reproducibility_tier = tier
    set_deterministic_seeds(DeterminismConfig(
        torch_seed=11, numpy_seed=12, random_seed=13,
        device=device_name, reproducibility_tier=tier, num_threads=1))

    settings = EcosystemConfig.from_config(config)
    if args.generations is not None:
        object.__setattr__(settings, "generations", args.generations)
    scales = load_calibration_scales()
    ecosystem = TensorEcosystem(settings, torch.device(device_name), scales)

    started = time.perf_counter()
    prey, predators, history = ecosystem.train()
    elapsed = time.perf_counter() - started

    # Genomes are the trained controller weights. Index 0 is the elite of
    # the final generation for each role, so that is the brain worth keeping.
    weight_paths: dict[str, str] = {}
    if args.weights_dir is not None:
        args.weights_dir.mkdir(parents=True, exist_ok=True)
        for role, genomes, metric in (
                ("prey", prey, "prey_fitness_mean"),
                ("predator", predators, "predator_fitness_mean")):
            if len(genomes) == 0:
                continue
            path = args.weights_dir / f"{role}.npz"
            save_checkpoint(str(path), genomes[0], metadata={
                "generation": len(history),
                "mean_fitness": history[-1][metric] if history else None,
                "role": role,
                "device": device_name,
                "experiment_name": config.experiment_name,
            })
            weight_paths[role] = str(path)

    window = history[-50:] if len(history) >= 50 else history
    summary = {
        "experiment_name": config.experiment_name,
        "device": device_name,
        "reproducibility_tier": tier,
        "generations": len(history),
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_generation": round(elapsed / max(len(history), 1), 3),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        if device_name == "cuda" else None,
        "weights": weight_paths,
        "last_50": {
            "prey_alive_mean": round(sum(h["prey_alive"] for h in window)
                                     / len(window), 2),
            "predator_alive_mean": round(sum(h["predator_alive"] for h in window)
                                         / len(window), 2),
            "prey_alive_step_mean": round(
                sum(h["prey_alive_step_mean"] for h in window) / len(window), 2),
            "predator_alive_step_mean": round(
                sum(h["predator_alive_step_mean"] for h in window) / len(window), 2),
            "prey_food_mean": round(sum(h["prey_food"] for h in window)
                                    / len(window), 2),
            "captures_mean": round(sum(h["captures"] for h in window)
                                   / len(window), 2),
        },
        "history": history,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
