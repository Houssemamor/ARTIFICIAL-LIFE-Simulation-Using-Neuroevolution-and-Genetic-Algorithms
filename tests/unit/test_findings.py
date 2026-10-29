"""
Tests for the dashboard findings builder (webdash/findings.py).

The builder is the page's guard against over-claiming, so most tests pin
down what it must NOT say: no "supported" verdict without a significant
result, and no crash on malformed artifacts.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from webdash.findings import (_verdict_for_p, build_findings)


def _write(root: Path, run: str, name: str, data) -> None:
    directory = root / run
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(data))


def _pairwise(a, b, p, diff=0.1):
    return {"condition_a": a, "condition_b": b, "mean_diff": diff,
            "p_value": p, "matched_pairs_rank_biserial": 0.3,
            "diff_ci95": [diff - 0.1, diff + 0.1]}


def _experiment_d(significant, p=0.5):
    return {"experiment": "d", "n_seeds": 3,
            "per_condition_scores": {"equal": [1.0, 1.1, 0.9],
                                     "other": [0.5, 0.6, 0.4]},
            "comparison": {"alpha": 0.05, "significant_pairs": significant,
                           "pairwise": [_pairwise("equal", "other", p, 0.5)],
                           "bootstrap_cis": {"equal": [0.9, 1.1],
                                             "other": [0.4, 0.6]}}}


def test_verdict_levels_follow_fixed_rules():
    assert _verdict_for_p(0.01) == "positive"
    assert _verdict_for_p(0.10) == "mixed"
    assert _verdict_for_p(0.50) == "null"
    # A raw p < alpha that fails the correction is never "positive".
    assert _verdict_for_p(0.01, survives_correction=False) == "mixed"
    assert _verdict_for_p(0.50, survives_correction=False) == "null"


def test_uncorrected_significance_is_not_reported_as_supported(tmp_path):
    # p=0.01 raw but not in significant_pairs (failed Holm): must not be positive.
    _write(tmp_path, "EXP-D", "stats_summary.json", _experiment_d([], p=0.01))
    finding = build_findings(tmp_path)[0]
    assert finding["verdict"]["level"] != "positive"


def test_corrected_significance_is_reported_as_supported(tmp_path):
    _write(tmp_path, "EXP-D", "stats_summary.json",
           _experiment_d(["equal_vs_other"], p=0.001))
    finding = build_findings(tmp_path)[0]
    assert finding["verdict"]["level"] == "positive"


def test_malformed_artifact_hides_only_its_own_card(tmp_path):
    _write(tmp_path, "EXP-D", "stats_summary.json", _experiment_d([]))
    _write(tmp_path, "EXP-GEN", "stats_summary.json", {"gap_mean": 0.1})  # missing keys
    (tmp_path / "EXP-E").mkdir()
    (tmp_path / "EXP-E" / "stats_summary.json").write_text("{not json")
    ids = [f["id"] for f in build_findings(tmp_path)]
    assert ids == ["EXP-D"]


def test_missing_root_gives_no_findings(tmp_path):
    assert build_findings(tmp_path / "nope") == []


def test_generalization_percentage_comes_from_the_data(tmp_path):
    _write(tmp_path, "EXP-GEN", "stats_summary.json", {
        "n_seeds": 2, "train_mean": 1.0, "unseen_mean": 0.8, "gap_mean": 0.2,
        "gap_ci95": [0.1, 0.3], "train_ci95": [0.9, 1.1],
        "unseen_ci95": [0.7, 0.9], "p_value": 0.5,
        "per_seed": [{"train_score": 1.0, "unseen_score": 0.8},
                     {"train_score": 1.0, "unseen_score": 1.2}]})
    finding = build_findings(tmp_path)[0]
    assert "20%" in finding["takeaway"]
    assert any(f["value"] == "1 of 2" for f in finding["facts"])
    assert finding["verdict"]["level"] == "null"


def test_committed_artifacts_produce_all_five_honest_cards():
    findings = build_findings(Path("experiments"))
    by_id = {f["id"]: f for f in findings}
    assert set(by_id) == {"EXP-GEN", "EXP-D", "EXP-E", "EXP-COEV", "EXP-NEAT"}
    # The committed results have no pair that survives correction, so
    # nothing statistical may be labelled "positive".
    for exp_id in ("EXP-GEN", "EXP-D", "EXP-E"):
        assert by_id[exp_id]["verdict"]["level"] != "positive"
    for finding in findings:
        assert finding["question"] and finding["method"] and finding["facts"]
        assert finding["chart"]["kind"] in {"bars", "lines", "strip"}
        # ASCII only, as everywhere else on the page.
        json.dumps(finding).encode("ascii")


def test_coevolution_collapse_is_never_called_supported(tmp_path):
    rows = [{"seed": s, "min_prey_alive": 1, "min_predator_alive": 0,
             "no_extinction": False} for s in range(10)]
    _write(tmp_path, "EXP-COEV", "stability_sweep.json",
           {"n_seeds": 10, "collapsed": 10, "per_seed": rows})
    _write(tmp_path, "EXP-COEV", "coevolution_summary.json",
           {"generations": [{"captures": 1}], "hof_evaluations": []})
    finding = build_findings(tmp_path)[0]
    assert finding["verdict"]["level"] == "mixed"
    assert "10 of 10" in finding["verdict"]["text"]
