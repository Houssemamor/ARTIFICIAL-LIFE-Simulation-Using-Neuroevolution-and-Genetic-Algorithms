"""
Unit tests for the web dashboard backend's filesystem layer.

The FastAPI routes are thin wrappers; the logic worth testing is
scan_experiments' discovery of the committed run artifacts and the
containment rules on the run_id / plot-name path parameters.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from webdash.backend.main import load_summary, scan_experiments


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


def test_load_summary_rejects_traversal_and_unknown_ids():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = root / "EXP-GOOD"
        good.mkdir()
        (good / "stats_summary.json").write_text(json.dumps({"n_seeds": 1}))
        # A Windows-style relative id and an absolute id must not escape
        # the experiments root, even if a matching summary exists there.
        outside = root.parent / "stats_summary.json"
        outside.write_text(json.dumps({"n_seeds": 99}))
        try:
            for bad_id in ("..", "..\\..", str(good.resolve()),
                           "EXP-GOOD\\..", "EXP-MISSING"):
                with pytest.raises(HTTPException) as caught:
                    load_summary(bad_id, root)
                assert caught.value.status_code == 404
            assert load_summary("EXP-GOOD", root)["n_seeds"] == 1
        finally:
            outside.unlink(missing_ok=True)


def test_plot_route_serves_only_discovered_svgs(monkeypatch):
    from webdash.backend import main as backend
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = root / "EXP-GOOD"
        run.mkdir()
        (run / "stats_summary.json").write_text(json.dumps({"n_seeds": 1}))
        (run / "curve.svg").write_text("<svg/>")
        (run / "checkpoint.npz").write_bytes(b"binary")
        monkeypatch.setattr(backend, "EXPERIMENTS_ROOT", root)
        response = backend.plot("EXP-GOOD", "curve.svg")
        assert response.media_type == "image/svg+xml"
        for bad_name in ("checkpoint.npz", "stats_summary.json",
                         "..\\..\\secrets.svg"):
            with pytest.raises(HTTPException) as caught:
                backend.plot("EXP-GOOD", bad_name)
            assert caught.value.status_code == 404
        with pytest.raises(HTTPException):
            backend.plot("EXP-MISSING", "curve.svg")


def test_metrics_includes_generation_rows():
    # The committed NEAT and co-evolution summaries key their
    # per-generation rows as `generations`; the endpoint must pass them
    # through or those runs render as "no data".
    from webdash.backend import main as backend
    neat = backend.metrics("EXP-NEAT")
    assert neat["generations"], "EXP-NEAT stores rows under 'generations'"
    coev = backend.metrics("EXP-COEV")
    assert coev["generations"], "EXP-COEV stores rows under 'generations'"
