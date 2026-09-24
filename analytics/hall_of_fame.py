"""
Hall-of-fame archive for Artificial Life Neuroevolution Simulation.

Phase 6: within-generation fitness in a co-evolutionary ecosystem can
reflect an arms race between the two roles rather than absolute
improvement (a predator that gets better against the current prey is
not necessarily better against last month's prey). The archive freezes
best-per-role genomes every N generations and evaluates the current
best controller against these frozen opponents, giving an absolute
progress signal the relative generation fitness cannot provide.

Win condition, fixed by definition:
  - the predator wins a duel if it captures the prey within the
    evaluation horizon;
  - the prey wins if it survives the horizon uncaptured (including the
    case where the predator starves first).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from agents.organism import Organism
from agents.sensors import RayCaster
from simulation.environment import World
from simulation.engine import step_simulation, resolve_captures
from simulation.world_config import BaselineConfig


class HallOfFame:
    """
    Frozen best-per-role genome archive with a duel-based evaluation
    harness.

    Attributes:
        snapshot_interval (int): Snapshot the generation's best genome
            every this many generations.
        max_entries (int): Keep at most this many snapshots per role
            (oldest evicted first).
    """

    def __init__(self, snapshot_interval: int = 2, max_entries: int = 10):
        if snapshot_interval < 1:
            raise ValueError(f"snapshot_interval must be >= 1, got {snapshot_interval}")
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self.snapshot_interval = snapshot_interval
        self.max_entries = max_entries
        # role -> list of (generation, genome copy, fitness), oldest first
        self._archive: Dict[str, List[Tuple[int, np.ndarray, float]]] = {
            'prey': [], 'predator': []
        }

    def maybe_snapshot(self, generation: int, role: str,
                       genome: np.ndarray, fitness: float) -> bool:
        """
        Record the generation's best genome for a role if this is a
        snapshot generation.

        The caller supplies the best (genome, fitness) of the current
        generation; the archive only decides whether to freeze it.

        Args:
            generation: Current generation number.
            role: 'prey' or 'predator'.
            genome: Best genome of this generation for the role.
            fitness: Its fitness.

        Returns:
            bool: True if a snapshot was taken.
        """
        if role not in self._archive:
            raise ValueError(f"role must be 'prey' or 'predator', got {role!r}")
        if generation % self.snapshot_interval != 0:
            return False

        entry = (generation, np.asarray(genome, dtype=np.float32).copy(), float(fitness))
        archive = self._archive[role]
        archive.append(entry)
        if len(archive) > self.max_entries:
            archive.pop(0)  # evict the oldest snapshot
        return True

    def opponents(self, role: str) -> List[np.ndarray]:
        """Archived genomes for the given role (the duel opponents)."""
        if role not in self._archive:
            raise ValueError(f"role must be 'prey' or 'predator', got {role!r}")
        return [genome.copy() for _, genome, _ in self._archive[role]]

    def entries(self, role: str) -> List[Tuple[int, np.ndarray, float]]:
        """Archived (generation, genome, fitness) tuples for a role."""
        if role not in self._archive:
            raise ValueError(f"role must be 'prey' or 'predator', got {role!r}")
        return list(self._archive[role])


def run_duel(predator_genome: np.ndarray, prey_genome: np.ndarray,
             config: BaselineConfig,
             rng: Optional[np.random.Generator] = None) -> str:
    """
    One controlled duel: a single predator against a single prey.

    The world layout is built from config.world (layout_seed applies), so
    every duel in an evaluation runs on identical terrain - the only
    variation is spawn position jitter from the passed rng.

    Args:
        predator_genome: Predator controller genome.
        prey_genome: Prey controller genome.
        config: Baseline config (world + evaluation_steps + predation).
        rng: Random generator for spawn positions.

    Returns:
        str: 'predator' if the prey was captured, else 'prey'.
    """
    if rng is None:
        rng = np.random.default_rng()
    if config.predation is None:
        raise ValueError("run_duel requires a config with a predation section")

    if config.world.layout_seed is not None:
        np.random.seed(config.world.layout_seed)
    world = World(config.world.width, config.world.height)
    world.load_from_config(config.model_dump())
    raycaster = RayCaster()

    def _spawn(genome: np.ndarray, role: str, agent_id: int) -> Organism:
        x = rng.uniform(world.boundary_margin, world.width - world.boundary_margin)
        y = rng.uniform(world.boundary_margin, world.height - world.boundary_margin)
        agent = Organism(agent_id, x, y, initial_energy=100.0, role=role)
        agent.genome = np.asarray(genome, dtype=np.float32).copy()
        return agent

    prey_agent = _spawn(prey_genome, 'prey', 0)
    predator_agent = _spawn(predator_genome, 'predator', 1)
    agents = [prey_agent, predator_agent]

    for _ in range(config.evolution.evaluation_steps):
        if not prey_agent.is_alive:
            return 'predator'
        if not predator_agent.is_alive:
            # Predator starved before catching anything: prey wins
            return 'prey'
        step_simulation(world, agents, raycaster)
        resolve_captures(
            agents,
            capture_radius=config.predation.capture_radius,
            energy_transfer=config.predation.capture_energy_transfer,
        )

    # Horizon exhausted with the prey alive: prey wins by definition
    return 'prey'


def win_rate_against_archive(role: str, current_genome: np.ndarray,
                             hof: HallOfFame, config: BaselineConfig,
                             rng: Optional[np.random.Generator] = None) -> float:
    """
    Evaluate a role's current best controller against the frozen archive
    of the opposing role and return its win rate.

    Args:
        role: The role being evaluated ('prey' or 'predator').
        current_genome: Current best genome of that role.
        hof: The archive (opponents come from the other role).
        config: Duel configuration.
        rng: Random generator for spawn jitter.

    Returns:
        float: Wins / duels, in [0, 1].

    Raises:
        ValueError: If the opponent archive is empty (nothing to
            evaluate against yet).
    """
    opponent_role = 'prey' if role == 'predator' else 'predator'
    archive = hof.opponents(opponent_role)
    if not archive:
        raise ValueError(
            f"cannot evaluate {role!r}: the {opponent_role!r} archive is empty")

    wins = 0
    for opponent_genome in archive:
        if role == 'predator':
            winner = run_duel(current_genome, opponent_genome, config, rng)
        else:
            winner = run_duel(opponent_genome, current_genome, config, rng)
        if winner == role:
            wins += 1
    return wins / len(archive)
