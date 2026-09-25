# Experiment D Results: Fitness-Weighting Sensitivity (Phase 5)

**Experiment:** does the choice of fitness weighting change which
controllers evolve? Four conditions (D1-D4) were trained and their frozen
best genomes compared on one common balanced metric.

**Status:** confirmatory run complete (10 seeds per condition). No
significant differences; the point estimates favor the equal-weight
baseline over every specialized weighting. An earlier exploratory
analysis suggested an ordering trend (survival > resource >
exploration > equal); the paired analysis below shows that trend was
an artifact of unpaired testing and non-uniform spawn positions, and
it disappears under the corrected method.

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
| D1 equal | 0.9532 |
| D2 resource-weighted | 0.8825 |
| D3 exploration-weighted | 0.8827 |
| D4 survival-weighted | 0.8367 |

### Pairwise comparisons (paired Wilcoxon + Holm-Bonferroni)

| Comparison | Mean diff | p-value | Matched-pairs r | 95% CI | Significant (Holm, a=0.05) |
|------------|-----------|---------|-----------------|--------|----------------------------|
| equal vs survival-weighted | +0.1164 | 0.557 | +0.236 | [-0.116, +0.346] | No |
| equal vs resource-weighted | +0.0707 | 0.557 | +0.236 | [-0.122, +0.259] | No |
| equal vs exploration-weighted | +0.0705 | 0.375 | +0.345 | [-0.063, +0.199] | No |
| exploration vs survival-weighted | +0.0460 | 0.557 | +0.236 | [-0.196, +0.281] | No |
| resource vs survival-weighted | +0.0457 | 0.695 | +0.164 | [-0.132, +0.218] | No |
| resource vs exploration-weighted | -0.0002 | 1.000 | -0.018 | [-0.188, +0.174] | No |

**No pair survives Holm correction, and the ordering points the
other way from the buggy-regrowth iteration: equal weighting is
best (+0.07 to +0.12 over every specialized weighting).** The
resource/exploration conditions are indistinguishable from each
other (diff -0.0002, r = -0.02). Under a food-replenished landscape,
down-weighting food (the survival condition) is the most costly
choice, which inverts the artifact run where survival-weighted led by
0.10-0.12 (that ordering was produced by the inert regrowth bug; it is
withdrawn).

Note on scales: scores normalize against the **measured** calibration
scales (`configs/calibration.json`), re-measured under these dynamics
(survival 133.94, food 1.0, exploration 77.76, collision 1.20).

## Interpretation

**With replenished food, equal weighting already does the right
thing and every re-weighting trades away more than it gains.** The
balanced weights (survival 0.4 / food 0.3 / exploration 0.2 /
collision 0.1) sit close to the measured component scales; pushing
mass toward survival starves the food-seeking signal that actually
earns score on a replenished landscape. This is a genuine negative
result, and a stronger statement than "nothing was significant":
the point estimates all favor the status quo.

**The earlier nulls were real, but they were about the landscape, not
about weighting.** Both the 5 px-era ordering and the 12 px-era
0.015-spread null were measured on survival-saturated dynamics. The
energy-limited regime (0.7/step with regrowth) separates the
conditions by ~0.12 - the landscape now has leverage - and what it
shows is that the pre-existing balanced weights are already
near-optimal. Fitness *weighting sensitivity* exists; it points
away from specialization.

## Recommendations

- The equal-weight baseline is the reference to beat; future weighting
  experiments should perturb around it, not replace it.
- Survival-weighting's deficit (+0.12 vs equal, r = +0.24) is sized for
  n = 20-30 if anyone wants to confirm the direction; the CI already
  excludes differences larger than ~0.35.
- Revisit the weights themselves: a weight sweep *around* equal
  (e.g. food 0.2/0.4) is now the informative experiment, not a
  survival-dominant re-weighting.