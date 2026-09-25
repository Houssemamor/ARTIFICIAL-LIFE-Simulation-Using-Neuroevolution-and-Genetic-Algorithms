"""
Integration tests for the Phase 6 predator/prey ecosystem.

The ecosystem is viability-marginal under the corrected dynamics
(newborns inherit the parent's energy equation, food regrows every
step): 7 of 10 seeds collapse within 10 days, every failure a
predator extinction (the committed
experiments/EXP-COEV/stability_sweep.json). These tests therefore
assert the honest property - the system is viable, not stable - on
the first three seeds (1-3, no cherry-picking): at least one must
complete the full horizon with both roles alive, and no seed may end
with both roles at zero. The 10-seed rate lives in the sweep
artifact; docs/coevolution_notes.md carries the analysis.

The negative test proves the fixture is not vacuous: extreme parameters
must produce an extinction.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from simulation.world_config import load_config
from experiments.run_coevolution import run_ecosystem_days

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "configs", "predator_prey.json")

# The first three sweep seeds (1, 2, 3): a deterministic, unselected
# sample of the sweep. Under the current dynamics seed 2 completes the
# horizon; seeds 1 and 3 collapse. Runtime: each seed is a 10-day run
# (~30s). Re-verify whenever dynamics-affecting parameters change
# (metabolic cost, consumption radius, calibration scales, predation or
# reproduction rates, newborn energy inheritance).
STABILITY_SEEDS = [1, 2, 3]


def test_ecosystem_is_viable_on_the_first_three_seeds():
    config = load_config(CONFIG_PATH)
    survivors = 0
    for seed in STABILITY_SEEDS:
        days = run_ecosystem_days(config, days=10, seed=seed)
        for day in days:
            assert not (day["prey_alive"] == 0
                        and day["predator_alive"] == 0), (
                f"seed {seed}: total extinction at day {day['day']}")
        if (len(days) == 10
                and all(day["prey_alive"] > 0 for day in days)
                and all(day["predator_alive"] > 0 for day in days)):
            survivors += 1
    assert survivors >= 1, (
        "the ecosystem must be viable on at least one of the first "
        f"three seeds; {survivors} survived (see "
        "experiments/EXP-COEV/stability_sweep.json for the 10-seed rate)")


def test_extreme_parameters_cause_extinction():
    # Negative/abuse case: 24 predators with no reproduction is far
    # beyond carrying capacity - the predator swarm overeats, then
    # starves to extinction (observed: extinct by day 7). The run ends
    # early on extinction, proving the stability fixture above can fail
    config = load_config(CONFIG_PATH)
    config = config.model_copy(update={
        "predation": config.predation.model_copy(
            update={"predator_count": 24}),
        "reproduction": config.reproduction.model_copy(
            update={"enabled": False}),
    })
    days = run_ecosystem_days(config, days=10, seed=21)

    prey_counts = [day["prey_alive"] for day in days]
    predator_counts = [day["predator_alive"] for day in days]
    assert len(days) < 10, (
        f"24 predators with no reproduction must break the run early, "
        f"got full horizon: {prey_counts}")
    assert min(prey_counts) == 0 or min(predator_counts) == 0, (
        f"expected an extinction under extreme parameters, got prey "
        f"{prey_counts} predators {predator_counts}")


def test_ecosystem_run_is_seed_reproducible():
    # Regression test: the behavioral rng was once created unseeded
    # (np.random.default_rng() with no argument), making every ecosystem
    # run non-reproducible regardless of --seed
    config = load_config(CONFIG_PATH)
    first = run_ecosystem_days(config, days=2, seed=99)
    second = run_ecosystem_days(config, days=2, seed=99)
    assert json.dumps(first) == json.dumps(second)

    other_seed = run_ecosystem_days(config, days=2, seed=100)
    assert json.dumps(first) != json.dumps(other_seed)
