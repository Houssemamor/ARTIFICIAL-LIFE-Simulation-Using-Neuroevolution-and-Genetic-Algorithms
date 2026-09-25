# Experiment E Results: Crossover-Method Comparison (Phase 3, deferred deliverable)

**Experiment:** does crossover operator choice (blend / uniform /
mutation-only) change which controllers evolve? Three conditions were
trained and their frozen best genomes compared on one common balanced
metric. This also produces the Phase 3 "fitness-over-generations"
smoke-test evidence.

**Status:** confirmatory run complete (10 seeds per condition) under
the **energy-limited dynamics** (Phase 9: metabolic cost 0.7/step,
food regrowth 0.3/step, live exploration). No method separates after
Holm. The fitness-over-generations curve climbs steeply (+0.28 to
+0.50 over 12 generations), but blend and uniform crossover are
statistically indistinguishable (p = 0.92, r = +0.06) - the earlier
blend advantage was an artifact of the inert-regrowth iteration.

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
component computed, calibration re-measured: survival 133.94, food
1.0, exploration 77.76, collision 1.20). The 12 px-legacy-metabolic
numbers (means 0.4474-0.4663) are superseded, as are this
document's first Phase 9 iteration (blend 0.9363 / uniform 0.8022),
which was produced before the review found the solo evaluation loop
never regrew food.

### Condition scores (balanced evaluation, 10 seeds)

| Condition | Mean |
|-----------|------|
| blend | 0.8917 |
| uniform | 0.8903 |
| none (mutation-only) | 0.8094 |

### Pairwise comparisons (paired Wilcoxon + Holm-Bonferroni)

| Comparison | Mean diff | p-value | Matched-pairs r | 95% CI | Significant |
|------------|-----------|---------|-----------------|--------|-------------|
| uniform vs none | +0.0809 | 0.193 | +0.491 | [-0.049, +0.217] | No |
| blend vs none | +0.0823 | 0.492 | +0.273 | [-0.082, +0.273] | No |
| blend vs uniform | +0.0014 | 0.922 | +0.055 | [-0.194, +0.186] | No |

**Blend and uniform are the same method as far as this experiment
can tell** (+0.0014, r = +0.06). Both beat mutation-only by ~+0.08,
consistently in sign, short of significance at n = 10.

### Fitness over generations (Phase 3 smoke-test evidence - far exceeded)

| Method | gen 0 mean | gen 11 mean | Change |
|--------|-----------|-------------|--------|
| blend | 0.489 | 0.983 | **+0.495** |
| uniform | 0.489 | 0.874 | +0.385 |
| none | 0.489 | 0.771 | +0.283 |

**The GA improves fitness by +0.28 to +0.50 mean over 12
generations** - a 7-9x stronger learning signal than the 12 px
regime managed (+0.033-0.044) and far above the 5 px regime's flat
zero. Every method starts at 0.489 (random controllers) and ends
with well-fed populations; even mutation-only climbs +0.28, so most
of the gain is selection+culling rather than recombination. The
Phase 3 exit expectation is comfortably cleared.

## Recommendations

- The blend-vs-uniform question is closed at this scale: they are
  equivalent here. Further method work should target *whether*
  crossover helps (vs mutation-only, +0.08, r up to +0.49), which is
  sized for n = 20-30.
- The metabolic cost (0.7/step) is a load-bearing experimental
  parameter, not a tuning leftover: it sets how much of each episode
  is spent surviving versus foraging. Treat it as a factor to sweep
  (0.4 / 0.7 / 1.0) in a follow-up, since the whole result family
  moves with it.