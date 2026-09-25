"""
Read-only web dashboard backend (PLAN.md Section 12.3, Phase 9).

Explicitly decoupled from the simulation core: this module only
reads the committed artifacts under `experiments/EXP-*/` and serves
them as JSON plus the frontend page. It never imports the simulation,
never writes state, and the endpoints are adapted to the artifacts
that actually exist (stats_summary.json summaries and SVG plots)
rather than the CSV format the plan sketched before Phase 5 fixed
the output contract.

Run: .venv\Scripts\python.exe -m uvicorn webdash.backend.main:app
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

EXPERIMENTS_ROOT = Path("experiments")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Neuroevolution Experiment Viewer", version="1.0.0")

# Which family each experiment belongs to (filter on the plan's page)
FAMILY_BY_PREFIX = {
    "EXP-GEN": "A/B (generalization)",
    "EXP-D": "D (fitness weighting)",
    "EXP-E": "E (crossover)",
    "EXP-COEV": "Predator/prey co-evolution",
    "EXP-NEAT": "NEAT extension",
    "EXP-HEADLESS": "Headless smoke",
    "EXP-GUI": "GUI checkpoint",
}


# Which summary file to serve per run when a directory holds several
# (the solo runners write stats_summary.json; NEAT writes
# neat_summary.json; co-evolution writes coevolution_summary.json)
SUMMARY_PREFERENCE = ("stats_summary.json", "neat_summary.json",
                      "coevolution_summary.json")


def scan_experiments(root: Path = EXPERIMENTS_ROOT) -> List[Dict]:
    """
    Discover experiment runs and their summaries. Pure function over
    the filesystem - the endpoint layer is a thin wrapper, so this is
    what the tests exercise.
    """
    runs: List[Dict] = []
    if not root.is_dir():
        return runs
    for directory in sorted(root.glob("EXP-*")):
        if not directory.is_dir():
            continue
        summaries = [name for name in SUMMARY_PREFERENCE
                     if (directory / name).is_file()]
        if not summaries:
            continue
        try:
            summary = json.loads(
                (directory / summaries[0]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        plots = sorted(path.name for path in directory.glob("*.svg"))
        runs.append({
            "id": directory.name,
            "family": FAMILY_BY_PREFIX.get(directory.name, "Other"),
            "primary_summary": summaries[0],
            "summaries": summaries,
            "n_seeds": summary.get("n_seeds"),
            "plots": plots,
        })
    return runs


def load_summary(run_id: str, root: Path = EXPERIMENTS_ROOT) -> Dict:
    """Load a run's primary summary (404 when absent)."""
    for name in SUMMARY_PREFERENCE:
        path = root / run_id / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404,
                        detail=f"no summary for {run_id}")


@app.get("/experiments")
def list_experiments() -> Dict:
    """List all runs with summary metadata."""
    return {"experiments": scan_experiments()}


@app.get("/experiments/{run_id}/stats_summary")
def stats_summary(run_id: str) -> Dict:
    """
    The experiment's stats_summary.json: per-condition scores,
    pairwise comparisons with p-values, confidence intervals and
    effect sizes.
    """
    return load_summary(run_id)


@app.get("/experiments/{run_id}/metrics")
def metrics(run_id: str) -> Dict:
    """
    Per-generation metrics. The solo runners embed their generation
    curves in stats_summary.json (fitness_curves); the co-evolution
    and NEAT runs key theirs the same way.
    """
    summary = load_summary(run_id)
    return {
        "run_id": run_id,
        "fitness_curves": summary.get("fitness_curves", {}),
    }


@app.get("/experiments/{run_id}/plot/{name}")
def plot(run_id: str, name: str) -> FileResponse:
    """Serve a committed SVG plot (fitness curves, complexity growth)."""
    path = (EXPERIMENTS_ROOT / run_id / name).resolve()
    if not path.is_file() or not path.is_relative_to(
            EXPERIMENTS_ROOT.resolve()):
        raise HTTPException(status_code=404, detail="plot not found")
    return FileResponse(path, media_type="image/svg+xml")


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
