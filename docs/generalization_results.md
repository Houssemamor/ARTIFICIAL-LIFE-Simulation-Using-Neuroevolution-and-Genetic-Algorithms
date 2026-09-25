# Generalization Results (Phase 5)

**Experiment:** train on layouts A1-A3, freeze best controllers, evaluate
unmodified on held-out layouts B1-B3, quantify the train-vs-unseen gap.

**Status:** confirmatory run complete (10 seeds) under the
**energy-limited dynamics** (metabolic cost 0.7/step, food regrowth
0.3/step, live exploration component). No generalization *penalty*
claim is retired: the run shows a borderline **train > unseen
gap of +0.047** (paired p = 0.1055, r = +0.60, 7 of 10 seeds positive,
bootstrap CI [+0.002, +0.089]) - the direction is consistent and the
CI excludes zero, but the paired test does not clear 0.05 at n = 10.

## Method

| Parameter | Value |
|-----------|-------|
| Training layouts | A1, A2, A3 (placement seeds 101-103) |
| Held-out layouts | B1, B2, B3 (placement seeds 901-903) |
| Population | 40 agents, fixed-topology 12-32-16-3 |
| Training | 12 generations, 150 evaluation steps/generation |
| Per seed | Train one genome per training layout; evaluate each frozen genome on its training layout and on every held-out layout |
| Evaluation metric | Balanced fitness (survival 0.4 / food 0.3 / exploration 0.2 / collision 0.1) |
| Statistics | **Paired** Wilcoxon signed-rank on per-seed train vs. unseen scores (same frozen genomes measured on both layout sets), 95% bootstrap CIs (10,000 resamples), matched-pairs rank-biserial effect size |
| Reproducibility | cpu-deterministic tier, layout seeds fixed, per-seed GA seeding; agents spawn uniformly across the world from a seeded RNG |

The paired test matters: each seed's train and unseen scores come from the
same frozen genomes, so between-seed variance must be removed by testing
the within-seed differences. (An earlier draft of this experiment used
unpaired Mann-Whitney U; a code review caught the mismatch.)

Runner: `experiments/run_generalization.py`
Raw results: `experiments/EXP-GEN/stats_summary.json`

## Results

**Final run (Phase 9 corrected dynamics).** The numbers below
come from the full re-run after the dynamics fix: `metabolic_base_cost`
0.7/step, food regrowth 0.3/step actually applied by the evaluation
loop (an earlier iteration had the regrowth inert - a code review
caught it), and the exploration component computed per agent.
Calibration was re-measured against these dynamics (survival 133.94,
food 1.0, exploration 77.76, collision 1.20). All reported experiments
share this baseline.

| Measure | Value |
|---------|-------|
| Train score (mean over 10 seeds) | 0.8977 |
| Unseen score (mean over 10 seeds) | 0.8510 |
| Gap (train - unseen) | +0.0467 |
| Gap 95% bootstrap CI | [+0.002, +0.089] |
| Wilcoxon signed-rank p-value (two-sided) | 0.1055 |
| Matched-pairs rank-biserial | +0.600 |

Per-seed gaps: 7 of 10 favor train (+0.032 to +0.144; three seeds
favor unseen).

## Interpretation

**A consistent-direction generalization penalty, in the expected
direction.** Frozen controllers score ~0.047 lower on held-out
layouts than on their training layouts (p = 0.1055 at n = 10, effect
size +0.60, 7/10 seeds consistent, bootstrap CI barely excluding
zero). The controllers learn a real food-seeking policy under the
energy-limited landscape, and what transfers is a skill that
imperfectly fits a new placement.

**Read it as evidence of overfitting at this sample size, not proof.**
The paired test misses 0.05; n = 10 seeds cannot resolve an effect of
this size cleanly (a larger-n confirmatory run is the first thing
this result asks for). The direction is biologically expected:
controllers tuned to one obstacle/food placement transfer imperfectly
to new ones.

**History of this question, for calibration of expectations:** the
5 px dynamics gave a reverse gap (unseen easier, p = 0.0645,
r = -0.67); the 12 px dynamics at legacy metabolic cost gave a clean
null (+0.001, p = 0.846); the energy-limited dynamics give the
expected train > unseen penalty. Three dynamics regimes, three
different answers - which is itself the strongest argument for
treating any single run's generalization number as regime-dependent.

## Recommendations

- Re-run with more seeds (n = 20+) to resolve the +0.047 effect.
- Compare the gap between the train layouts and a *random-layout*
  control to separate true overfitting from layout-difficulty drift.
- Keep the energy-limited dynamics as the default experiment regime;
  the null/penalty distinction depends on it.