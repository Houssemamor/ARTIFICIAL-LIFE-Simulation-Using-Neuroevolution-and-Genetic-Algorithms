# Experiment E Results: Crossover-Method Comparison (Phase 3, deferred deliverable)

**Experiment:** does crossover operator choice (blend / uniform /
mutation-only) change which controllers evolve? Three conditions were
trained and their frozen best genomes compared on one common balanced
metric. This also produces the Phase 3 "fitness-over-generations"
smoke-test evidence.

**Status:** confirmatory run complete (10 seeds per condition) under
the **energy-limited dynamics** (Phase 9: metabolic cost 0.7/step,
food regrowth 0.3/step, live exploration, 40 px solid obstacles). No
method separates after Holm. The fitness-over-generations curve climbs
steeply (+0.51 to +0.72 over 12 generations). Blend leads uniform by
+0.12 (p = 0.13) and mutation-only by +0.16 (p = 0.064); uniform and
mutation-only are indistinguishable (+0.04, p = 0.85).

## Method

| Parameter | Value |
|-----------|-------|
| Conditions | blend (default), uniform, none (mutation-only) |
| Training | 12 generations, 40 agents, 150 evaluation steps, layout A1 (seed 101) |
| Seeds | 10 per condition; conditions share the same seeded initial population per seed index |
| Fitness normalization | Measured scales from `configs/calibration.json` (regenerated per-agent; see below) |
| Evaluation | Each condition's frozen best genome scored with the balanced weights |
| Statistics | Paired pipeline: per-pair Wilcoxon signed-rank, Holm-Bonferroni at alpha = 0.05, matched-pairs rank-biserial |

Runner: `experiments/run_crossover_experiment.py`
Raw results: `experiments/EXP-E/stats_summary.json`, plot:
`experiments/EXP-E/fitness_curve.svg`

## Results

**Final run (Phase 9 corrected dynamics).** Full re-run after
the dynamics fix (`metabolic_base_cost` 0.7/step, food regrowth
0.3/step actually applied by the evaluation loop, exploration
component computed, calibration re-measured: survival 146.44, food
1.44, exploration 61.26, collision 11.17; obstacles solid 40 px
bodies). The 12 px-legacy-metabolic numbers (means 0.4474-0.4663) are
superseded, as are this document's first Phase 9 iteration (blend
0.9363 / uniform 0.8022), which was produced before the review found
the solo evaluation loop never regrew food.

### Condition scores (balanced evaluation, 10 seeds)

| Condition | Mean |
|-----------|------|
| blend | 1.2929 |
| uniform | 1.1710 |
| none (mutation-only) | 1.1315 |

### Pairwise comparisons (paired Wilcoxon + Holm-Bonferroni)

| Comparison | Mean diff | p-value | Matched-pairs r | 95% CI | Significant |
|------------|-----------|---------|-----------------|--------|-------------|
| blend vs uniform | +0.1218 | 0.131 | +0.564 | [+0.002, +0.230] | No |
| blend vs none | +0.1614 | 0.064 | +0.673 | [+0.024, +0.279] | No |
| uniform vs none | +0.0396 | 0.846 | -0.091 | [-0.077, +0.209] | No |

**No method separates after Holm correction.** Blend's lead is the
consistent one: +0.12 over uniform and +0.16 over mutation-only, with
bootstrap CIs that exclude zero, but p = 0.131 and p = 0.064 at
n = 10. Uniform and mutation-only are the same method as far as this
experiment can tell (+0.04, r = -0.09). The earlier "blend = uniform"
reading does not survive the recalibrated dynamics: the point estimate
now favors blend, but not at corrected significance.

### Fitness over generations (Phase 3 smoke-test evidence - far exceeded)

| Method | gen 0 mean | gen 11 mean | Change |
|--------|-----------|-------------|--------|
| blend | 0.580 | 1.296 | **+0.716** |
| uniform | 0.580 | 1.172 | +0.592 |
| none | 0.580 | 1.091 | +0.511 |

**The GA improves fitness by +0.51 to +0.72 mean over 12
generations** - 15-22x stronger than the 12 px regime managed
(+0.033-0.044) and far above the 5 px regime's flat zero. Every method
starts at 0.580 (random controllers) and ends with well-fed
populations; even mutation-only climbs +0.51, so most of the gain is
selection+culling rather than recombination. The Phase 3 exit
expectation is comfortably cleared.

## Recommendations

- The blend-vs-uniform question is not closed: the point estimate now
  favors blend by +0.12 with a CI excluding zero (p = 0.13). That is
  sized for n = 20-30. Blend-vs-mutation-only is +0.16 (p = 0.064) and
  uniform-vs-mutation-only is null (+0.04), so the informative contrast
  is blend against the other two, not uniform against mutation-only.
- The metabolic cost (0.7/step) is a load-bearing experimental
  parameter, not a tuning leftover: it sets how much of each episode
  is spent surviving versus foraging. Treat it as a factor to sweep
  (0.4 / 0.7 / 1.0) in a follow-up, since the whole result family
  moves with it.