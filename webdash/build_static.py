"""
Static dashboard exporter: bundles experiments/EXP-* into one HTML file.

Standard library only. The output needs no server, no network and no
Python on the viewing machine - copy it anywhere and open it in a
browser.

Usage:
    python webdash/build_static.py
    python webdash/build_static.py --experiments experiments --output dashboard.html
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Allow running as a plain script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webdash.artifacts import (load_recording, read_summary,  # noqa: E402
                               scan_experiments)
from webdash.findings import build_findings  # noqa: E402

TEMPLATE_PATH = Path(__file__).resolve().parent / "frontend" / "index.html"
PLACEHOLDER = "/*__EMBEDDED_DATA__*/ null"

# Plots are inlined into the page, so cap each one to keep the file small.
MAX_PLOT_BYTES = 2 * 1024 * 1024




def build_payload(root: Path) -> Dict:
    """
    Read every run once: the run list, each summary and each SVG plot.
    Plot names come from the scan (bare file names globbed from the run
    directory), so no caller-supplied path is ever joined here.
    """
    runs = scan_experiments(root)
    summaries: Dict[str, Dict] = {}
    plots: Dict[str, Dict[str, str]] = {}
    kept_runs = []
    for run in runs:
        summary = read_summary(run, root)
        if summary is None:
            continue  # unreadable between scan and read: skip the run
        summaries[run["id"]] = summary
        plots[run["id"]] = {}
        for name in run["plots"]:
            path = root / run["id"] / name
            try:
                if path.stat().st_size > MAX_PLOT_BYTES:
                    continue
                plots[run["id"]][name] = path.read_text(encoding="utf-8")
            except (OSError, ValueError):
                continue
        # Only advertise plots that were actually embedded.
        run = dict(run, plots=sorted(plots[run["id"]]))
        kept_runs.append(run)
    return {"experiments": kept_runs, "summaries": summaries, "plots": plots,
            "findings": build_findings(root),
            "recording": load_recording(root)}


def render_html(template: str, payload: Dict) -> str:
    """
    Inject the payload into the page template. The JSON is placed inside
    a <script> block, so "<" is escaped to keep data such as
    "</script>" from terminating the block; ensure_ascii keeps the file
    pure ASCII.
    """
    if PLACEHOLDER not in template:
        raise ValueError("template is missing the embedded-data placeholder")
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    data = data.replace("<", "\\u003c")
    return template.replace(PLACEHOLDER, data)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiments", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "experiments")
    parser.add_argument("--output", type=Path, default=Path("dashboard.html"))
    args = parser.parse_args(argv)

    payload = build_payload(args.experiments)
    if not payload["experiments"]:
        print(f"No experiment runs found under {args.experiments}",
              file=sys.stderr)
        return 1
    html = render_html(TEMPLATE_PATH.read_text(encoding="utf-8"), payload)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} ({len(html) // 1024} KB, "
          f"{len(payload['experiments'])} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
