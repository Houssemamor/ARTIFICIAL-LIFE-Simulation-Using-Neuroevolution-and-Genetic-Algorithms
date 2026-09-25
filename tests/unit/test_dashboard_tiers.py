"""
Unit tests for the Phase 8 dashboard tier reconciliation (PLAN.md
Section 12).

The reconciliation requirement: the MVP build must never render a
Phase 6/7-only metric (predator/prey counts, hall-of-fame), because
the two dashboard variants are selected by config.phase_tier rather
than by one dashboard hiding fields.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from visualization.dashboard_advanced import (format_mvp_lines,
                                              format_ecosystem_lines,
                                              format_hof_lines,
                                              live_agent_count)


class _StubAgent:
    def __init__(self, is_alive):
        self.is_alive = is_alive


def test_live_agent_count_excludes_dead_organisms():
    # Deaths only clear is_alive; the dead stay in the list, so
    # len(agents) is the total population, not the living count
    agents = [_StubAgent(True), _StubAgent(False), _StubAgent(True),
              _StubAgent(False)]
    assert live_agent_count(agents) == 2
    assert len(agents) == 4


def test_mvp_lines_never_contain_phase6_or_7_metrics():
    text = " ".join(format_mvp_lines(30)).lower()
    for forbidden in ("predator", "prey", "hall-of-fame", "species",
                      "captures", "births", "win rate"):
        assert forbidden not in text, f"MVP panel leaked '{forbidden}'"
    assert "agents alive: 30" in text


def test_advanced_ecosystem_lines_report_live_counts():
    lines = format_ecosystem_lines(prey_alive=20, predator_alive=4,
                                   captures=3, births=7, total_agents=31)
    text = " ".join(lines)
    assert "prey: 20" in text
    assert "predators: 4" in text
    assert "captures: 3" in text
    assert "births: 7" in text


def test_hof_lines_without_run_say_so_instead_of_placeholder_numbers():
    lines = format_hof_lines(None)
    assert len(lines) == 1
    assert "no co-evolution run yet" in lines[0]
    empty = format_hof_lines([])
    assert "no co-evolution run yet" in empty[0]


def test_hof_lines_show_latest_evaluation():
    evaluations = [
        {"generation": 2, "predator_win_rate_vs_archive": 0.0,
         "prey_win_rate_vs_archive": 1.0},
        {"generation": 10, "predator_win_rate_vs_archive": 0.2,
         "prey_win_rate_vs_archive": 1.0},
    ]
    text = " ".join(format_hof_lines(evaluations))
    assert "gen 10" in text
    assert "predator 0.20" in text
    assert "gen 2" not in text


def test_phase_tier_defaults_to_mvp():
    from simulation.world_config import BaselineConfig, load_config
    # Configs without the field default to the MVP tier
    config = BaselineConfig(
        world={"width": 100, "height": 100},
        population={"size": 2, "agent": {"sensors": 7,
                    "brain": {"architecture": [12, 3]}}},
        evolution={}, experiment={})
    assert config.phase_tier == "mvp"
    # The advanced-tier configs opt in explicitly
    predator_prey = load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "configs", "predator_prey.json"))
    assert predator_prey.phase_tier == "advanced"
