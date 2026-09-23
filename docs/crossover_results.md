# Experiment E Results: Crossover-Method Comparison (Phase 3, deferred deliverable)

**Experiment:** does crossover operator choice (blend / uniform /
mutation-only) change which controllers evolve? Three conditions were
trained and their frozen best genomes compared on one common balanced
metric. This also produces the Phase 3 "fitness-over-generations"
smoke-test evidence.

**Status:** confirmatory run complete (10 seeds per condition). No
significant differences. The fitness-over-generations curve is **flat**
- the GA does not measurably improve mean fitness under current
dynamics, for reasons the Phase 5 analyses already identified.

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

### Condition scores (balanced evaluation, 10 seeds)

| Condition | Mean |
|-----------|------|
| blend | 0.3160 |
| uniform | 0.3233 |
| none (mutation-only) | 0.3361 |

### Pairwise comparisons (paired Wilcoxon + Holm-Bonferroni)

| Comparison | p-value | Matched-pairs r | Significant |
|------------|---------|-----------------|-------------|
| blend vs uniform | 0.770 | +0.127 | No |
| blend vs none | 0.432 | -0.309 | No |
| uniform vs none | 0.432 | -0.309 | No |

Mutation-only is numerically highest, but nothing separates the three
methods statistically.

### Fitness over generations (Phase 3 smoke-test evidence)

The curve (`fitness_curve.svg`, mean over seeds of population mean/max
fitness per generation) is **flat for every method**: generation-0 mean
fitness is ~0.325 and generation-11 is ~0.31-0.32, with no upward trend;
max fitness fluctuates (0.64-0.86) without direction. **The GA does not
measurably improve fitness under the current dynamics.**

Why this contradicts the earlier Phase 3 telemetry (which showed
0.31 -> 0.60 within two generations): that improvement was real but its
*cause* was removed by the spawn-position fix. Agents previously spawned
in a corner line that crossed obstacle placements, so random genomes
collided heavily (mean 14.8 contacts) and selection had something to
learn (avoid obstacles). With uniform world-wide spawns, random genomes
rarely collide (calibration measured 3.1 contacts/agent) and already sit
near the fitness ceiling at generation 0. The learnable skill the GA
previously demonstrated was an artifact of evaluation geometry, not
general obstacle avoidance.

## Interpretation

Two findings, both consistent with the Phase 5 degenerate-landscape
analysis:

1. **Crossover method does not matter here.** With food rarely eaten,
   survival saturated, and exploration uncomputed, the effective
   objective is a nearly-flat function of the genome; there is little for
   any variation operator to exploit or disrupt. The competing-conventions
   concern that motivated this experiment cannot be evaluated on a flat
   landscape.
2. **The Phase 3 exit expectation ("fitness improves over controlled
   runs") is not met by the current evidence.** What earlier looked like
   learning was evaluation-geometry artifact + hand-picked calibration
   constants. This is the strongest argument yet for the recommended
   dynamics changes (live metabolic cost, achievable food acquisition,
   computed exploration) before any of the Phase 3/5 conclusions - positive
   or negative - are treated as final.

## Recommendations

Same as `docs/generalization_results.md` and
`docs/experiment_d_results.md`: make the fitness components live, then
rerun all three experiments (E included) through the same paired
pipelines. The infrastructure now exists end-to-end and is cheap to
re-execute.