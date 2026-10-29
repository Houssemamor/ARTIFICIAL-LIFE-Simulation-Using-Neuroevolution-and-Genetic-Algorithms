"""
End-to-end test of the showcase recorder with a tiny training budget.

Checks the property the dashboard depends on: recording the same seed
twice yields identical scenarios (only the environment/timing block may
differ).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.record_showcase import build_recording


def _tiny(seed):
    return build_recording(seed=seed, steps=12, solo_agents=3,
                           coev_generations=1, solo_generations=1)


def test_same_seed_records_identical_scenarios():
    first, second = _tiny(1), _tiny(1)
    assert json.dumps(first["scenarios"]) == json.dumps(second["scenarios"])
    assert set(first["environment"]["training_seconds"]) == {
        "coev_training_seconds", "solo_training_seconds"}


def test_different_seed_changes_the_recording():
    assert json.dumps(_tiny(1)["scenarios"]) != json.dumps(_tiny(2)["scenarios"])


def test_untrained_and_evolved_share_arena_and_start_positions():
    for scenario in _tiny(1)["scenarios"]:
        untrained = scenario["runs"]["untrained"]
        evolved = scenario["runs"]["evolved"]
        assert untrained["world"] == evolved["world"]
        assert untrained["frames"][0]["a"] == evolved["frames"][0]["a"]
