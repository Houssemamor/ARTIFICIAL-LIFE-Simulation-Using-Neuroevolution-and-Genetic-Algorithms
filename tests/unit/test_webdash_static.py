"""
Unit tests for the dependency-free static dashboard exporter.

These import only webdash.artifacts and webdash.build_static, which use
the standard library, so they run without fastapi or an HTTP client.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from webdash import build_static
from webdash.build_static import (MAX_PLOT_BYTES, PLACEHOLDER, build_payload,
                                  render_html)


def _make_run(root: Path, name: str, summary: dict, svg: str = "") -> Path:
    run = root / name
    run.mkdir()
    (run / "stats_summary.json").write_text(json.dumps(summary))
    if svg:
        (run / "curve.svg").write_text(svg)
    return run


def _embedded_json(html: str) -> dict:
    """Extract and parse the object assigned to EMBEDDED."""
    match = re.search(r"const EMBEDDED = (\{.*?\});\n", html, re.S)
    assert match, "embedded payload not found"
    return json.loads(match.group(1))


def test_payload_bundles_summaries_and_plots(tmp_path):
    _make_run(tmp_path, "EXP-GOOD", {"n_seeds": 3}, "<svg/>")
    payload = build_payload(tmp_path)
    assert [r["id"] for r in payload["experiments"]] == ["EXP-GOOD"]
    assert payload["summaries"]["EXP-GOOD"]["n_seeds"] == 3
    assert payload["plots"]["EXP-GOOD"] == {"curve.svg": "<svg/>"}


def test_payload_skips_broken_and_empty_runs(tmp_path):
    (tmp_path / "EXP-EMPTY").mkdir()
    broken = tmp_path / "EXP-BROKEN"
    broken.mkdir()
    (broken / "stats_summary.json").write_text("{not json")
    _make_run(tmp_path, "EXP-GOOD", {"n_seeds": 1})
    payload = build_payload(tmp_path)
    assert list(payload["summaries"]) == ["EXP-GOOD"]


def test_payload_missing_root_is_empty(tmp_path):
    payload = build_payload(tmp_path / "does-not-exist")
    assert payload["experiments"] == []


def test_oversized_plot_is_skipped_and_not_advertised(tmp_path):
    _make_run(tmp_path, "EXP-BIG", {"n_seeds": 1},
              "x" * (MAX_PLOT_BYTES + 1))
    payload = build_payload(tmp_path)
    assert payload["plots"]["EXP-BIG"] == {}
    assert payload["experiments"][0]["plots"] == []


def test_non_svg_files_are_never_embedded(tmp_path):
    run = _make_run(tmp_path, "EXP-GOOD", {"n_seeds": 1})
    (run / "secret.txt").write_text("top secret")
    (run / "checkpoint.npz").write_bytes(b"binary")
    payload = build_payload(tmp_path)
    assert "top secret" not in json.dumps(payload)
    assert payload["plots"]["EXP-GOOD"] == {}


def test_hostile_strings_cannot_close_the_script_block(tmp_path):
    hostile = "</script><script>alert(1)</script>"
    _make_run(tmp_path, "EXP-X", {"n_seeds": 1, "note": hostile},
              "<svg>" + hostile + "</svg>")
    html = render_html("<script>const EMBEDDED = " + PLACEHOLDER + ";\n</script>",
                       build_payload(tmp_path))
    # Exactly the template's own closing tag - none smuggled in via data.
    assert html.count("</script>") == 1
    # The escaped data still round-trips to the original string.
    data = _embedded_json(html)
    assert data["summaries"]["EXP-X"]["note"] == hostile


def test_output_is_pure_ascii(tmp_path):
    _make_run(tmp_path, "EXP-U", {"n_seeds": 1, "label": "caf\u00e9"})
    html = render_html("const EMBEDDED = " + PLACEHOLDER + ";\n",
                       build_payload(tmp_path))
    html.encode("ascii")


def test_render_rejects_template_without_placeholder():
    with pytest.raises(ValueError):
        render_html("<html></html>", {"experiments": []})


def test_real_template_contains_placeholder_exactly_once():
    template = build_static.TEMPLATE_PATH.read_text(encoding="utf-8")
    assert template.count(PLACEHOLDER) == 1
    # Static export must work offline: no external resources.
    assert "cdn." not in template and "http://" not in template


def test_main_writes_file_and_fails_on_empty_root(tmp_path, capsys):
    _make_run(tmp_path, "EXP-GOOD", {"n_seeds": 2}, "<svg/>")
    out = tmp_path / "out.html"
    assert build_static.main(["--experiments", str(tmp_path),
                              "--output", str(out)]) == 0
    assert "EXP-GOOD" in out.read_text(encoding="utf-8")

    empty = tmp_path / "empty"
    empty.mkdir()
    assert build_static.main(["--experiments", str(empty),
                              "--output", str(tmp_path / "none.html")]) == 1
    assert not (tmp_path / "none.html").exists()


def test_export_of_committed_artifacts(tmp_path):
    out = tmp_path / "dashboard.html"
    assert build_static.main(["--output", str(out)]) == 0
    data = _embedded_json(out.read_text(encoding="utf-8"))
    ids = {r["id"] for r in data["experiments"]}
    assert {"EXP-GEN", "EXP-D", "EXP-E", "EXP-NEAT"} <= ids
    assert data["plots"]["EXP-E"]


# --- showcase recording and findings ------------------------------------

from webdash.artifacts import (MAX_RECORDING_BYTES, RECORDING_PATH,
                               RECORDING_VERSION, load_recording)


def _write_recording(root: Path, payload) -> Path:
    path = root / RECORDING_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_recording_is_embedded_with_findings_keys(tmp_path):
    _make_run(tmp_path, "EXP-GOOD", {"n_seeds": 1})
    _write_recording(tmp_path, {"version": RECORDING_VERSION, "scenarios": []})
    payload = build_payload(tmp_path)
    assert payload["recording"]["version"] == RECORDING_VERSION
    assert payload["findings"] == []   # no known experiment artifacts here


@pytest.mark.parametrize("bad", [
    "{not json",
    json.dumps({"version": RECORDING_VERSION + 1, "scenarios": []}),   # wrong schema
    json.dumps({"version": RECORDING_VERSION}),                        # no scenarios
    json.dumps([1, 2, 3]),                                             # not an object
])
def test_invalid_recording_is_ignored(tmp_path, bad):
    _write_recording(tmp_path, bad)
    assert load_recording(tmp_path) is None


def test_oversized_recording_is_ignored(tmp_path):
    path = _write_recording(tmp_path, "x")
    with open(path, "wb") as handle:
        handle.truncate(MAX_RECORDING_BYTES + 1)
    assert load_recording(tmp_path) is None


def test_missing_recording_is_none(tmp_path):
    assert load_recording(tmp_path) is None


def test_hostile_recording_text_cannot_close_the_script_block(tmp_path):
    hostile = "</script><img src=x onerror=alert(1)>"
    _make_run(tmp_path, "EXP-GOOD", {"n_seeds": 1})
    _write_recording(tmp_path, {"version": RECORDING_VERSION,
                                "scenarios": [{"title": hostile}]})
    html = render_html("<script>const EMBEDDED = " + PLACEHOLDER + ";\n</script>",
                       build_payload(tmp_path))
    assert html.count("</script>") == 1
    assert _embedded_json(html)["recording"]["scenarios"][0]["title"] == hostile


def test_committed_recording_is_well_formed():
    recording = load_recording(Path("experiments"))
    assert recording is not None, "run experiments/record_showcase.py"
    assert {s["id"] for s in recording["scenarios"]} == {"coev", "solo"}
    for scenario in recording["scenarios"]:
        assert set(scenario["runs"]) == {"untrained", "evolved"}
        for run in scenario["runs"].values():
            n_agents = len(run["agents"])
            assert run["frames"][0]["f"] is not None
            for frame in run["frames"]:
                assert len(frame["a"]) == n_agents
            for role, index in run["featured"].items():
                assert run["agents"][index]["role"] == role
                assert role in run["networks"]
            # First frame always has activations for the featured agents.
            assert set(run["frames"][0]["n"]) == set(run["featured"])
