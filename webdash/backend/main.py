"""
Read-only web dashboard backend (PLAN.md Section 12.3, Phase 9).

Optional live mode. For a machine without the project's dependencies,
use the dependency-free static exporter instead:
    python webdash/build_static.py

Explicitly decoupled from the simulation core: this module only
reads the committed artifacts under `experiments/EXP-*/` (through
webdash.artifacts) and serves them as JSON plus the frontend page. It
never imports the simulation and never writes state.

Run: .venv\\Scripts\\python.exe -m uvicorn webdash.backend.main:app
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webdash.artifacts import (EXPERIMENTS_ROOT, FAMILY_BY_PREFIX,  # noqa: F401
                               SUMMARY_PREFERENCE, load_recording,
                               read_summary, scan_experiments)
from webdash.findings import build_findings

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Neuroevolution Experiment Viewer", version="1.1.0")


def load_summary(run_id: str, root: Optional[Path] = None) -> Dict:
    """Load a run's primary summary (404 for unknown or malformed ids)."""
    if root is None:
        root = EXPERIMENTS_ROOT
    # A run_id reaches the filesystem, so it must match a directory the
    # scan discovered: "..\\configs" and absolute paths are rejected here
    # instead of being sanitized piecemeal.
    entry = next((run for run in scan_experiments(root)
                  if run["id"] == run_id), None)
    summary = read_summary(entry, root) if entry else None
    if summary is None:
        raise HTTPException(status_code=404,
                            detail=f"no summary for {run_id}")
    return summary


@app.get("/showcase")
def showcase() -> Dict:
    """Findings cards and the recorded replay (None when not recorded)."""
    return {"findings": build_findings(EXPERIMENTS_ROOT),
            "recording": load_recording(EXPERIMENTS_ROOT)}


@app.get("/experiments")
def list_experiments() -> Dict:
    """List all runs with summary metadata."""
    return {"experiments": scan_experiments(EXPERIMENTS_ROOT)}


@app.get("/experiments/{run_id}/stats_summary")
def stats_summary(run_id: str) -> Dict:
    """
    The experiment's primary summary: per-condition scores, pairwise
    comparisons with p-values, confidence intervals and effect sizes.
    """
    return load_summary(run_id)


@app.get("/experiments/{run_id}/metrics")
def metrics(run_id: str) -> Dict:
    """
    Per-generation metrics. The solo runners embed their generation
    curves in stats_summary.json (fitness_curves); NEAT and
    co-evolution key their per-generation rows as `generations`.
    """
    summary = load_summary(run_id)
    return {
        "run_id": run_id,
        "fitness_curves": summary.get("fitness_curves", {}),
        "generations": summary.get("generations", []),
    }


@app.get("/experiments/{run_id}/plot/{name}")
def plot(run_id: str, name: str) -> FileResponse:
    """Serve a committed SVG plot (fitness curves, complexity growth)."""
    entry = next((r for r in scan_experiments(EXPERIMENTS_ROOT)
                  if r["id"] == run_id), None)
    if entry is None or name not in entry["plots"]:
        raise HTTPException(status_code=404, detail="plot not found")
    return FileResponse(EXPERIMENTS_ROOT / run_id / name,
                        media_type="image/svg+xml")


@app.get("/experiments/{run_id}/best_genome")
def best_genome(run_id: str) -> Dict:
    """
    Best controller for a run. The experiment pipeline freezes the
    best genome for balanced evaluation but does not persist it to
    disk, so this endpoint reports that honestly rather than
    fabricating a genome.
    """
    load_summary(run_id)  # 404s for unknown runs
    return {"run_id": run_id, "genome": None,
            "note": "best genomes are evaluated in-memory and not "
                    "persisted; re-run the experiment to obtain one"}


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True),
              name="frontend")
