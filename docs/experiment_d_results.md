# Experiment D Results: Fitness-Weighting Sensitivity (Phase 5)

**Experiment:** does the choice of fitness weighting change which
controllers evolve? Four conditions (D1-D4) were trained and their frozen
best genomes compared on one common balanced metric.

**Status:** confirmatory run complete (10 seeds per condition). No
significant differences after Holm correction. Equal weighting has the
best point estimate, but it is statistically tied with
exploration-weighting (+0.015); survival-weighting is last by +0.18,
with a confidence interval that includes zero. The earlier
"equal leads every specialization by +0.07 to +0.12" reading does not
survive the solid-obstacle recalibration. An exploratory analysis had
suggested an ordering trend (survival > resource > exploration >
equal); the paired analysis below shows that trend was an artifact of
unpaired testing and non-uniform spawn positions, and it disappears
under the corrected method.

## Method

| Parameter | Value |
|-----------|-------|
| Conditions | D1 equal, D2 resource-weighted, D3 exploration-weighted, D4 survival-weighted |
| Training | 12 generations, 40 agents, 150 evaluation steps, shared layout (seed 101) |
| Seeds | 10 per condition; conditions share the same seeded initial population per seed index |
| Evaluation | Each condition's frozen best genome scored with the balanced weights (survival 0.4 / food 0.3 / exploration 0.2 / collision 0.1) |
| Statistics | **Paired** pipeline: per-pair Wilcoxon signed-rank (samples blocked by seed), Holm-Bonferroni step-down at alpha = 0.05 over the 6-comparison family, 95% bootstrap CIs, matched-pairs rank-biserial effect sizes |

Two methodology corrections versus the exploratory pass, both prompted by
code review: (1) because every condition reuses the same seeds and the
same seeded initial populations, samples across conditions are paired -
unpaired Mann-Whitney U discards the blocking and overstates within-family
variance; (2) agents now spawn uniformly across the world from a seeded
RNG rather than along a corner line, which had interacted with layout
placement and made evaluation scores partly an artifact of spawn
geometry.

Genomes are always compared on the balanced metric: each condition
*trains* under a different weighting, but comparing raw training fitness
across conditions would compare different objectives.

Runner: `experiments/run_experiment_d.py`
Raw results: `experiments/EXP-D/stats_summary.json`

## Results

**Final run (Phase 9 corrected dynamics).** `metabolic_base_cost`
0.7/step, food regrowth 0.3/step, live exploration component,
re-measured calibration. A code review found the re-run's first
iteration invalid: the hand-rolled solo evaluation loop never called
`world.regrow_food()`, so the configured regrowth was inert and the
numbers below (collected after that fix) are the real dynamics.
Newborn energy inheritance and newborn exploration tracking were
fixed in the same pass.

### Condition scores (balanced evaluation, 10 seeds)

| Condition | Mean |
|-----------|------|
| D1 equal | 1.3562 |
| D2 resource-weighted | 1.2831 |
| D3 exploration-weighted | 1.3409 |
| D4 survival-weighted | 1.1750 |

### Pairwise comparisons (paired Wilcoxon + Holm-Bonferroni)

| Comparison | Mean diff | p-value | Matched-pairs r | 95% CI | Significant (Holm, a=0.05) |
|------------|-----------|---------|-----------------|--------|----------------------------|
| equal vs survival-weighted | +0.1813 | 0.160 | +0.527 | [-0.013, +0.381] | No |
| exploration vs survival-weighted | +0.1659 | 0.322 | +0.382 | [-0.080, +0.421] | No |
| resource vs survival-weighted | +0.1081 | 0.131 | +0.564 | [-0.044, +0.245] | No |
| equal vs resource-weighted | +0.0731 | 0.625 | +0.200 | [-0.184, +0.300] | No |
| resource vs exploration-weighted | -0.0578 | 0.846 | -0.091 | [-0.300, +0.178] | No |
| equal vs exploration-weighted | +0.0153 | 0.625 | +0.200 | [-0.121, +0.138] | No |

**No pair survives Holm correction.** The largest contrast is equal
vs survival (+0.181, r = +0.53), consistent in direction but with a
confidence interval that includes zero (p = 0.160). Equal and
exploration-weighting are statistically tied (+0.015). Resource and
exploration are also indistinguishable (-0.058, r = -0.09). The one
stable ordering is that survival-weighting is last: down-weighting
food is the most costly choice on a replenished landscape.

Note on scales: scores normalize against the **measured** calibration
scales (`configs/calibration.json`), re-measured under these dynamics
(survival 146.44, food 1.44, exploration 61.26, collision 11.17).

## Interpretation

**With replenished food, equal weighting remains the best point
estimate, but the lead is smaller and less general than the previous
run suggested.** The balanced weights (survival 0.4 / food 0.3 /
exploration 0.2 / collision 0.1) sit close to the measured component
scales; pushing mass toward survival starves the food-seeking signal
that actually earns score on a replenished landscape, which is why
survival-weighting is last by +0.18. But equal is statistically tied
with exploration-weighting (+0.015), so the honest statement is that
equal is a good default, not that every specialization loses to it.
This is a genuine negative result: no re-weighting is measurably
better, and the one clear loser is the condition that down-weights
food.

**The earlier nulls were real, but they were about the landscape, not
about weighting.** Both the 5 px-era ordering and the 12 px-era
0.015-spread null were measured on survival-saturated dynamics. The
energy-limited regime spreads the conditions by 0.18 - the landscape
has leverage - and most of that spread is survival-weighting's
deficit. Fitness *weighting sensitivity* exists; it points away from
down-weighting food, but not measurably away from exploration.

## Recommendations

- The equal-weight baseline is the reference to beat; future weighting
  experiments should perturb around it, not replace it.
- Equal-vs-survival (+0.18, r = +0.53) is the contrast sized for
  n = 20-30 if anyone wants to confirm the direction; the CI already
  includes zero, so this is a candidate, not a finding.
- Revisit the weights themselves: a weight sweep *around* equal
  (e.g. food 0.2/0.4) is now the informative experiment, not a
  survival-dominant re-weighting.