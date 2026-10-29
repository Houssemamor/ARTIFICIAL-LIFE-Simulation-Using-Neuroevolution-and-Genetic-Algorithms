"""
Filesystem layer shared by the live FastAPI viewer and the static exporter.

Standard library only, so the static exporter can run on a machine that
has none of the project's dependencies (no torch, pygame or fastapi).
It reads the committed artifacts under `experiments/EXP-*/` and never
writes anything.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

EXPERIMENTS_ROOT = Path("experiments")

# Which family each experiment belongs to (filter on the page)
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


def scan_experiments(root: Optional[Path] = None) -> List[Dict]:
    """
    Discover experiment runs and their summaries. Pure function over
    the filesystem. `root` defaults to EXPERIMENTS_ROOT.
    """
    if root is None:
        root = EXPERIMENTS_ROOT
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


# Replay written by experiments/record_showcase.py. Only this schema
# version and size cap are accepted, so a stray or oversized file cannot
# bloat or break the page.
RECORDING_PATH = Path("EXP-SHOWCASE") / "recording.json"
RECORDING_VERSION = 1
MAX_RECORDING_BYTES = 8 * 1024 * 1024


def load_recording(root: Optional[Path] = None) -> Optional[Dict]:
    """The showcase recording, or None when absent, oversized or invalid."""
    if root is None:
        root = EXPERIMENTS_ROOT
    path = root / RECORDING_PATH
    try:
        if path.stat().st_size > MAX_RECORDING_BYTES:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(data, dict) or data.get("version") != RECORDING_VERSION
            or not isinstance(data.get("scenarios"), list)):
        return None
    return data


def read_summary(run: Dict, root: Optional[Path] = None) -> Optional[Dict]:
    """
    Load the primary summary of a run entry returned by scan_experiments.
    Taking the entry (not a caller-supplied id) means the path always
    comes from a directory the scan itself discovered.
    """
    if root is None:
        root = EXPERIMENTS_ROOT
    try:
        return json.loads(
            (root / run["id"] / run["primary_summary"]).read_text(
                encoding="utf-8"))
    except (OSError, ValueError):
        return None
