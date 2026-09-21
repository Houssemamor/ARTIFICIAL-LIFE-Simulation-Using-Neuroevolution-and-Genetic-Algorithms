"""
Experiment logger for Artificial Life Neuroevolution Simulation.

Writes the full `experiments/EXP-XXX/` folder structure:
    experiments/
        EXP-001/
            config.json           # full configuration used
            metadata.json         # seeds, device, reproducibility_tier, git commit
            generation_metrics.csv  # per-generation aggregate metrics
            agent_metrics.csv       # per-agent final metrics
            best_genome.json        # best genome + fitness
            checkpoints/
                gen-005.npz         # genome + architecture
                gen-010.npz
            stats_summary.json      # populated in Phase 4.5 (placeholder here)
"""

from __future__ import annotations
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from simulation.world_config import BaselineConfig
from neural.genome import genome_size


@dataclass
class ExperimentMetadata:
    """Metadata recorded at experiment start."""
    experiment_id: str
    experiment_name: str
    timestamp: str
    git_commit: str
    git_dirty: bool
    device: str
    reproducibility_tier: str  # "cpu-deterministic" | "gpu"
    seeds: Dict[str, Optional[int]]  # torch, numpy, random
    config: Dict[str, Any]
    python_version: str
    torch_version: str
    numpy_version: str


@dataclass
class GenerationMetricsRow:
    """One row in generation_metrics.csv."""
    generation: int
    population_size: int
    mean_fitness: float
    max_fitness: float
    min_fitness: float
    mean_survival: float
    mean_food: float
    mean_exploration: float
    mean_collisions: float
    num_alive_end: int
    elapsed_seconds: float


class ExperimentLogger:
    """
    Manages experiment folder creation and writing.

    Usage:
        logger = ExperimentLogger(config, seeds, device, tier)
        logger.log_generation(metrics, elapsed)
        logger.log_agent(agent, generation)
        logger.save_best_genome(genome, fitness)
        logger.save_checkpoint(genome, generation)
        logger.finalize()
    """

    GENERATION_METRICS_HEADERS = [
        "generation",
        "population_size",
        "mean_fitness",
        "max_fitness",
        "min_fitness",
        "mean_survival",
        "mean_food",
        "mean_exploration",
        "mean_collisions",
        "num_alive_end",
        "elapsed_seconds",
    ]

    AGENT_METRICS_HEADERS = [
        "generation",
        "agent_id",
        "fitness",
        "age",
        "energy",
        "food_eaten",
        "collisions",
        "survival_steps",
        "exploration_distance",
        "final_x",
        "final_y",
    ]

    def __init__(
            self,
            config: BaselineConfig,
            seeds: Dict[str, Optional[int]],
            device: str,
            reproducibility_tier: str,
            base_dir: str = "experiments",
            time_provider: Optional[callable] = None,
        ) -> None:
            """
            Initialize experiment logger and create folder structure.

            Args:
                config: Validated baseline configuration.
                seeds: Dict with keys 'torch', 'numpy', 'random' (separate seeds).
                device: "cpu" or "cuda".
                reproducibility_tier: "cpu-deterministic" or "gpu".
                base_dir: Base experiments directory.
                time_provider: Optional callable returning current time as float.
                    Defaults to time.time. For deterministic runs, provide a
                    counter-based function.
            """
            self.config = config
            self.seeds = seeds
            self.device = device
            self.reproducibility_tier = reproducibility_tier
            self.base_dir = Path(base_dir)
            self._time_provider = time_provider or time.time

            # Generate experiment ID: EXP-XXX
            existing = sorted([d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("EXP-")])
            next_num = len(existing) + 1
            self.experiment_id = f"EXP-{next_num:03d}"
            self.exp_dir = self.base_dir / self.experiment_id
            self.checkpoints_dir = self.exp_dir / "checkpoints"
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

            # Open CSV files for streaming writes
            self.gen_metrics_path = self.exp_dir / "generation_metrics.csv"
            self.agent_metrics_path = self.exp_dir / "agent_metrics.csv"
            self._gen_f = open(self.gen_metrics_path, "w", newline="")
            self._agent_f = open(self.agent_metrics_path, "w", newline="")
            self._gen_writer = csv.DictWriter(self._gen_f, fieldnames=self.GENERATION_METRICS_HEADERS)
            self._agent_writer = csv.DictWriter(self._agent_f, fieldnames=self.AGENT_METRICS_HEADERS)
            self._gen_writer.writeheader()
            self._agent_writer.writeheader()

            # Save config.json
            config_path = self.exp_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config.model_dump(), f, indent=2)

            # Save metadata.json
            self.metadata = ExperimentMetadata(
                experiment_id=self.experiment_id,
                experiment_name=config.experiment_name,
                timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                git_commit=self._get_git_commit(),
                git_dirty=self._get_git_dirty(),
                device=device,
                reproducibility_tier=reproducibility_tier,
                seeds=seeds,
                config=config.model_dump(),
                python_version=f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
                torch_version=torch.__version__,
                numpy_version=np.__version__,
            )
            meta_path = self.exp_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(asdict(self.metadata), f, indent=2)

            # Write placeholder stats_summary.json (Phase 4.5 will populate)
            stats_path = self.exp_dir / "stats_summary.json"
            with open(stats_path, "w") as f:
                json.dump({"status": "placeholder", "phase": "4.5"}, f, indent=2)

            self._start_time = self._time_provider()
            self._generation_start = self._time_provider()

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return "unknown"

    def _get_git_dirty(self) -> bool:
        try:
            return subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip() != ""
        except Exception:
            return False

    def log_generation(self, metrics: Dict[str, Any]) -> None:
        """
        Log one generation's aggregate metrics.

        Args:
            metrics: Dict with keys matching GenerationMetricsRow.
        """
        elapsed = self._time_provider() - self._start_time
        row = {**metrics, "elapsed_seconds": elapsed}
        self._gen_writer.writerow(row)
        self._gen_f.flush()

    def log_agent(self, agent: Any, generation: int) -> None:
        """
        Log one agent's final metrics.

        Args:
            agent: Organism instance.
            generation: Generation number this agent belongs to.
        """
        row = {
            "generation": generation,
            "agent_id": agent.id,
            "fitness": agent.fitness,
            "age": agent.age,
            "energy": agent.energy,
            "food_eaten": getattr(agent, "_food_eaten_total", 0),
            "collisions": agent.collisions,
            "survival_steps": agent.age,
            "exploration_distance": getattr(agent, "_exploration_total", 0.0),
            "final_x": agent.position.x,
            "final_y": agent.position.y,
        }
        self._agent_writer.writerow(row)
        self._agent_f.flush()

    def save_best_genome(self, genome: np.ndarray, fitness: float, generation: int) -> None:
        """
        Save the best genome of the experiment.

        Args:
            genome: Flat genome array.
            fitness: Its fitness value.
            generation: Generation it was found in.
        """
        path = self.exp_dir / "best_genome.json"
        data = {
            "generation": generation,
            "fitness": fitness,
            "genome": genome.tolist(),
            "genome_size": genome_size(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_checkpoint(self, genome: np.ndarray, generation: int) -> None:
        """
        Save a genome checkpoint.

        Args:
            genome: Flat genome array.
            generation: Generation number.
        """
        path = self.checkpoints_dir / f"gen-{generation:03d}.npz"
        np.savez_compressed(
            path,
            genome=genome,
            generation=generation,
            architecture=np.array([12, 32, 16, 3], dtype=np.int32),
        )

    def finalize(self) -> None:
        """Close files and ensure all data is written."""
        self._gen_f.close()
        self._agent_f.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finalize()