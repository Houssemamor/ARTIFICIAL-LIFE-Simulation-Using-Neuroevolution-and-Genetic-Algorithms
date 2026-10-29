"""
Per-experiment findings for the showcase dashboard.

Standard library only. The prose describing each experiment (question and
method) is static and contains no numbers. Every number, chart, verdict
and takeaway is computed from the committed artifacts under
experiments/EXP-*, and the verdict level follows fixed statistical rules
(see _verdict_for_p), so the page cannot claim an effect the data does not
support.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

ALPHA = 0.05
# A p-value under this is reported as a "trend", not as a null result.
TREND_P = 0.2


def _load(path: Path) -> Optional[Dict]:
    """Read a JSON artifact; None when missing or malformed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _fmt(value: float, digits: int = 3, signed: bool = False) -> str:
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _verdict_for_p(p_value: float, survives_correction: Optional[bool] = None
                   ) -> str:
    """
    Verdict level from a p-value. 'positive' needs p < ALPHA and, when a
    multiple-comparison correction applies, surviving it. A visible but
    unproven trend is 'mixed'; everything else is 'null'.
    """
    if survives_correction is False:
        return "mixed" if p_value < TREND_P else "null"
    if p_value < ALPHA:
        return "positive"
    return "mixed" if p_value < TREND_P else "null"


def _finding(exp_id: str, title: str, question: str, method: str,
             facts: List[Dict], chart: Dict, level: str, verdict: str,
             takeaway: str) -> Dict:
    return {"id": exp_id, "title": title, "question": question,
            "method": method, "facts": facts, "chart": chart,
            "verdict": {"level": level, "text": verdict},
            "takeaway": takeaway}


def _generalization(directory: Path) -> Optional[Dict]:
    s = _load(directory / "stats_summary.json")
    if not s or "gap_mean" not in s:
        return None
    per_seed = s.get("per_seed", [])
    worse = sum(1 for row in per_seed
                if row["train_score"] > row["unseen_score"])
    level = _verdict_for_p(s["p_value"])
    percent = 100.0 * s["gap_mean"] / s["train_mean"] if s["train_mean"] else 0.0
    return _finding(
        "EXP-GEN", "Generalization to unseen worlds",
        "Do controllers evolved on a few training layouts still work on "
        "layouts they have never seen?",
        "Evolve on the training layouts, freeze the best controller, then "
        "test it unchanged on held-out layouts. Repeated over "
        f"{s['n_seeds']} independent seeds.",
        [
            {"label": "Train score", "value": _fmt(s["train_mean"])},
            {"label": "Unseen score", "value": _fmt(s["unseen_mean"])},
            {"label": "Gap (train - unseen)",
             "value": _fmt(s["gap_mean"], signed=True)},
            {"label": "95% interval of gap",
             "value": f"[{_fmt(s['gap_ci95'][0], signed=True)}, "
                      f"{_fmt(s['gap_ci95'][1], signed=True)}]"},
            {"label": "p-value (paired Wilcoxon)",
             "value": _fmt(s["p_value"], 4)},
            {"label": "Seeds where unseen was worse",
             "value": f"{worse} of {len(per_seed)}"},
        ],
        {"kind": "bars", "unit": "score", "items": [
            {"label": "train", "value": s["train_mean"],
             "lo": s["train_ci95"][0], "hi": s["train_ci95"][1]},
            {"label": "unseen", "value": s["unseen_mean"],
             "lo": s["unseen_ci95"][0], "hi": s["unseen_ci95"][1]},
        ]},
        level,
        ("The gap is statistically significant." if level == "positive" else
         f"Unseen layouts scored lower in {worse} of {len(per_seed)} "
         f"seeds, but the gap is not significant (p={_fmt(s['p_value'], 2)})."),
        f"Unseen layouts scored {percent:.0f}% lower on average than "
        "training layouts.",
    )


def _fitness_weighting(directory: Path) -> Optional[Dict]:
    s = _load(directory / "stats_summary.json")
    if not s or "comparison" not in s:
        return None
    comparison = s["comparison"]
    means = {name: _mean(values)
             for name, values in s["per_condition_scores"].items()}
    cis = comparison.get("bootstrap_cis", {})
    significant = comparison.get("significant_pairs", [])
    leader = max(means, key=means.get)
    others = [v for name, v in means.items() if name != leader]
    margins = [means[leader] - v for v in others]
    best_p = min((p["p_value"] for p in comparison["pairwise"]
                  if leader in (p["condition_a"], p["condition_b"])),
                 default=1.0)
    level = ("positive" if significant else
             _verdict_for_p(best_p, survives_correction=False))
    return _finding(
        "EXP-D", "Fitness weighting",
        "Does the way fitness trades off survival, food and exploration "
        "change what evolution produces?",
        "Four weightings (equal, resource, exploration, survival), each "
        f"over {s['n_seeds']} paired seeds, all scored on one common "
        "balanced metric and compared with Holm-corrected tests.",
        [{"label": f"Mean score: {name}", "value": _fmt(value)}
         for name, value in means.items()]
        + [{"label": "Pairs significant after correction",
            "value": str(len(significant))}],
        {"kind": "bars", "unit": "balanced score", "items": [
            {"label": name, "value": value,
             "lo": cis.get(name, [None, None])[0],
             "hi": cis.get(name, [None, None])[1]}
            for name, value in means.items()]},
        level,
        ("At least one weighting is significantly better." if significant
         else f"'{leader}' scored highest, but no pair survives the "
              "multiple-comparison correction."),
        f"'{leader}' led the others by {_fmt(min(margins), signed=True)} to "
        f"{_fmt(max(margins), signed=True)}.",
    )


def _crossover(directory: Path) -> Optional[Dict]:
    s = _load(directory / "stats_summary.json")
    if not s or "fitness_curves" not in s:
        return None
    curves = s["fitness_curves"]
    gains = {m: c["mean_fitness"][-1] - c["mean_fitness"][0]
             for m, c in curves.items()}
    comparison = s["comparison"]
    significant = comparison.get("significant_pairs", [])
    pairs = {(p["condition_a"], p["condition_b"]): p
             for p in comparison["pairwise"]}
    facts = [{"label": f"Fitness gained: {m}", "value": _fmt(g, signed=True)}
             for m, g in gains.items()]
    blend_uniform = pairs.get(("blend", "uniform"))
    if blend_uniform:
        facts.append({
            "label": "Blend vs uniform (difference, p)",
            "value": f"{_fmt(blend_uniform['mean_diff'], signed=True)}, "
                     f"p={_fmt(blend_uniform['p_value'], 2)}"})
    facts.append({"label": "Pairs significant after correction",
                  "value": str(len(significant))})
    smallest_p = min(p["p_value"] for p in pairs.values())
    level = ("positive" if significant else
             _verdict_for_p(smallest_p, survives_correction=False))
    return _finding(
        "EXP-E", "Crossover methods",
        "Does recombining parents help evolution, and does the recombination "
        "method matter?",
        "Blend, uniform and no crossover (mutation only) on the same seeds "
        f"and starting populations ({s['n_seeds']} paired seeds), tracked "
        "over the generations.",
        facts,
        {"kind": "lines", "xlabel": "generation", "ylabel": "mean fitness",
         "series": [{"label": m, "values": c["mean_fitness"]}
                    for m, c in curves.items()]},
        level,
        ("At least one crossover method is significantly better."
         if significant else
         "Every method gained fitness over the generations; no method is "
         "provably better than another at this sample size."),
        "Fitness gained over the run: " + ", ".join(
            f"{m} {_fmt(g, signed=True)}" for m, g in gains.items()) + ".",
    )


def _coevolution(directory: Path) -> Optional[Dict]:
    sweep = _load(directory / "stability_sweep.json")
    co = _load(directory / "coevolution_summary.json")
    if not sweep or not co:
        return None
    n_seeds, collapsed = sweep["n_seeds"], sweep["collapsed"]
    rows = sweep["per_seed"]
    predator_extinct = sum(1 for r in rows if r["min_predator_alive"] == 0)
    prey_extinct = sum(1 for r in rows if r["min_prey_alive"] == 0)
    captures = [g["captures"] for g in co["generations"]]
    hof = co.get("hof_evaluations", [])
    facts = [
        {"label": "Ecosystems that lasted the full run",
         "value": f"{n_seeds - collapsed} of {n_seeds}"},
        {"label": "Predator extinctions / prey extinctions",
         "value": f"{predator_extinct} / {prey_extinct}"},
        {"label": "Captures per generation (first -> last)",
         "value": f"{captures[0]} -> {captures[-1]}"},
    ]
    if hof:
        last = hof[-1]
        facts.append({
            "label": f"Win rate vs frozen past champions (generation "
                     f"{last['generation']})",
            "value": f"prey {_fmt(last['prey_win_rate_vs_archive'], 2)}, "
                     f"predators {_fmt(last['predator_win_rate_vs_archive'], 2)}"})
    # The collapse rate is a descriptive count, not a test: more than half
    # collapsing is reported as a limitation, never as a success.
    level = "positive" if collapsed * 2 <= n_seeds else "mixed"
    return _finding(
        "EXP-COEV", "Predator and prey ecosystem",
        "Can predators and prey coexist and keep improving against each "
        "other?",
        "Co-evolve both populations in a shared world, run a "
        f"{n_seeds}-seed stability sweep, and test current champions "
        "against frozen ones from earlier generations.",
        facts,
        {"kind": "strip", "title": "Seed outcome (green = no extinction)",
         "items": [{"label": f"s{r['seed']}", "ok": bool(r["no_extinction"])}
                   for r in rows]},
        level,
        f"{collapsed} of {n_seeds} ecosystems had a species go extinct "
        "before the run ended.",
        f"Of the {collapsed} collapses, {predator_extinct} runs lost "
        f"predators and {prey_extinct} lost prey.",
    )


def _neat(directory: Path) -> Optional[Dict]:
    s = _load(directory / "neat_summary.json")
    bench = _load(directory / "batching_benchmark.json") or {}
    if not s or not s.get("generations"):
        return None
    gens = s["generations"]
    start, end = gens[0]["mean_complexity"], gens[-1]["mean_complexity"]
    species = sorted({g["species_count"] for g in gens})
    facts = [
        {"label": "Mean complexity, nodes + connections (first -> last)",
         "value": f"{start:.0f} -> {end:.0f}"},
        {"label": "Species counts seen",
         "value": ", ".join(str(x) for x in species)},
        {"label": "Structural innovations",
         "value": str(s["total_innovations"])},
    ]
    if "speedup_population_vs_per_agent" in bench:
        facts.append({"label": "Batched inference vs per-agent",
                      "value": f"{bench['speedup_population_vs_per_agent']:.1f}x"})
    grew = end > start
    single = species == [1]
    return _finding(
        "EXP-NEAT", "Evolving network structure (NEAT)",
        "Can the network's structure itself evolve, not only its weights?",
        "NEAT-style evolution from minimal networks with innovation "
        "tracking, speciation and fitness sharing on a food-seeking task.",
        facts,
        {"kind": "lines", "xlabel": "generation", "ylabel": "mean complexity",
         "series": [{"label": "mean complexity",
                     "values": [g["mean_complexity"] for g in gens]}]},
        "mixed" if grew else "null",
        ("Networks grew in complexity, but the population stayed in a "
         "single species." if grew and single else
         "Networks grew in complexity." if grew else
         "Network structure did not grow."),
        "This run does not compare fitness against fixed-structure "
        "evolution, so the benefit of the added structure is not measured.",
    )


_BUILDERS = (
    ("EXP-GEN", _generalization),
    ("EXP-D", _fitness_weighting),
    ("EXP-E", _crossover),
    ("EXP-COEV", _coevolution),
    ("EXP-NEAT", _neat),
)


def build_findings(root: Path) -> List[Dict]:
    """Findings for every experiment whose artifacts are present."""
    findings = []
    for exp_id, builder in _BUILDERS:
        directory = root / exp_id
        if not directory.is_dir():
            continue
        try:
            finding = builder(directory)
        except (KeyError, IndexError, TypeError, ValueError,
                ZeroDivisionError):
            # A malformed artifact hides its card rather than breaking
            # the whole page.
            finding = None
        if finding is not None:
            findings.append(finding)
    return findings
