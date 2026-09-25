"""
Unit tests for the web dashboard backend's filesystem layer.

The FastAPI routes are thin wrappers; the logic worth testing is
scan_experiments' discovery of the committed run artifacts.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from webdash.backend.main import scan_experiments


def test_scan_finds_committed_runs():
    runs = scan_experiments(Path("experiments"))
    ids = {run["id"] for run in runs}
    assert {"EXP-GEN", "EXP-D", "EXP-E", "EXP-NEAT"} <= ids
    for run in runs:
        assert run["n_seeds"] is None or run["n_seeds"] > 0
        assert isinstance(run["plots"], list)


def test_scan_skips_unreadable_or_missing_summaries():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "EXP-EMPTY").mkdir()            # no stats_summary
        (root / "EXP-BROKEN").mkdir()
        (root / "EXP-BROKEN" / "stats_summary.json").write_text("{not json")
        good = root / "EXP-GOOD"
        good.mkdir()
        (good / "stats_summary.json").write_text(
            json.dumps({"n_seeds": 3, "fitness_curves": {}}))
        runs = scan_experiments(root)
        ids = {run["id"] for run in runs}
        assert ids == {"EXP-GOOD"}
        assert runs[0]["n_seeds"] == 3


def test_scan_missing_root_is_empty():
    assert scan_experiments(Path("Z:/Temp/opencode/definitely-not-here")) == []


def test_app_exposes_the_documented_endpoints():
    from webdash.backend.main import app
    routes = {route.path for route in app.routes}
    assert "/experiments" in routes
    assert "/experiments/{run_id}/stats_summary" in routes
    assert "/experiments/{run_id}/metrics" in routes
    assert "/experiments/{run_id}/best_genome" in routes
