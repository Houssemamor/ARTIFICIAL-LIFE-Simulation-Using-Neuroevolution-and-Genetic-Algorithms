"""Unit tests for the Phase 6 hall-of-fame archive and duels."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from analytics.hall_of_fame import (HallOfFame, run_duel,
                                     win_rate_against_archive)
from simulation.world_config import load_config
from neural.genome import genome_size


def make_genome(seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, genome_size()
                                               ).astype(np.float32)


def small_duel_config():
    """The real predator_prey config with a short evaluation horizon so
    duels stay fast."""
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        "configs", "predator_prey.json"))
    return config


def test_snapshot_only_at_interval_generations():
    hof = HallOfFame(snapshot_interval=2, max_entries=5)
    genome = make_genome()

    assert hof.maybe_snapshot(1, 'prey', genome, 0.5) is False
    assert hof.maybe_snapshot(2, 'prey', genome, 0.5) is True
    assert hof.maybe_snapshot(3, 'prey', genome, 0.5) is False
    assert hof.maybe_snapshot(4, 'prey', genome, 0.5) is True
    assert len(hof.opponents('prey')) == 2


def test_snapshot_evicts_oldest_beyond_max_entries():
    hof = HallOfFame(snapshot_interval=1, max_entries=3)
    for generation in range(1, 6):
        hof.maybe_snapshot(generation, 'predator', make_genome(generation),
                           0.5)
    entries = hof.entries('predator')
    assert len(entries) == 3
    assert [g for g, _, _ in entries] == [3, 4, 5]


def test_snapshot_rejects_invalid_role():
    hof = HallOfFame()
    try:
        hof.maybe_snapshot(2, 'parasite', make_genome(), 0.5)
        assert False, "invalid role must raise"
    except ValueError:
        pass


def test_snapshot_stores_copies_not_references():
    hof = HallOfFame(snapshot_interval=1)
    genome = make_genome()
    hof.maybe_snapshot(1, 'prey', genome, 0.5)
    genome[:] = 0.0  # mutate the caller's array
    assert not np.all(hof.opponents('prey')[0] == 0.0)


def test_win_rate_against_empty_archive_raises():
    hof = HallOfFame()
    config = small_duel_config()
    try:
        win_rate_against_archive('predator', make_genome(), hof, config)
        assert False, "empty archive must raise"
    except ValueError:
        pass


def test_duel_returns_a_valid_winner_deterministically():
    config = small_duel_config()
    predator = make_genome(10)
    prey = make_genome(11)

    first = run_duel(predator, prey, config, np.random.default_rng(1))
    second = run_duel(predator, prey, config, np.random.default_rng(1))

    assert first in ('predator', 'prey')
    assert first == second  # same seeds must duel identically


def test_win_rate_in_unit_interval():
    hof = HallOfFame(snapshot_interval=1)
    config = small_duel_config()
    for generation in (1, 2):
        hof.maybe_snapshot(generation, 'prey', make_genome(generation), 0.4)

    rate = win_rate_against_archive('predator', make_genome(99), hof,
                                    config, np.random.default_rng(1))
    assert 0.0 <= rate <= 1.0
