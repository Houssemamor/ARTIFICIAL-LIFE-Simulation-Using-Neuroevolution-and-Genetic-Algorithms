# Generalization Results (Phase 5)

**Experiment:** train on layouts A1-A3, freeze best controllers, evaluate
unmodified on held-out layouts B1-B3, quantify the train-vs-unseen gap.

**Status:** confirmatory run complete (10 seeds). No generalization
penalty; a small borderline *reverse* gap (unseen slightly above train)
is documented below with its caveats.

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

| Measure | Value |
|---------|-------|
| Train score (mean over 10 seeds) | 0.3304 |
| Unseen score (mean over 10 seeds) | 0.3411 |
| Gap (train - unseen) | -0.0107 |
| Gap 95% bootstrap CI | [-0.021, -0.002] |
| Train 95% bootstrap CI | [0.312, 0.347] |
| Unseen 95% bootstrap CI | [0.325, 0.355] |
| Wilcoxon signed-rank W | 9.0 |
| Wilcoxon p-value (two-sided) | 0.0645 |
| Matched-pairs rank-biserial | -0.673 |

Per-seed gaps: 8 of 10 seeds score slightly higher on unseen layouts.

Note on scales: scores in this revision are normalized against the
**measured** calibration scales (`configs/calibration.json`), not the
hand-picked constants used in the first confirmatory pass; absolute
values therefore differ from earlier drafts, and the comparison
structure is unchanged.

## Interpretation

**No generalization penalty.** Frozen controllers transfer to held-out
layouts without performance loss; if anything, the B layout set is very
slightly *easier* for them (gap CI excludes zero, exact paired p = 0.0645
just misses the 0.05 threshold at n = 10).

**The reverse gap is almost certainly layout-set idiosyncrasy, not a
generalization effect.** Only three train and three test layouts exist,
and their placement seeds (101-103 vs 901-903) were picked arbitrarily.
A ~0.011 systematic difference between two fixed three-layout sets says
more about those particular obstacle/food placements than about
transfer. Reading it as "unseen layouts are easier" would be
over-interpretation; the defensible claim is that performance is
comparable across the two sets.

**The landscape caveat still applies.** Under the current (fixed)
evaluation geometry, training telemetry shows no measurable learning at
all: the fitness-over-generations curve is flat for every condition
(see `docs/crossover_results.md` and
`experiments/EXP-E/fitness_curve.svg`). Food is only rarely eaten
(typically 0-0.1 items per seed), survival saturates within the
150-step evaluation, and the exploration component is never computed.
An earlier telemetry table in a draft of this document showed apparent
collision-avoidance learning within 1-2 generations - that was an
artifact of corner-line spawning and is retracted. A controller with no
learned, layout-dependent skill trivially shows no gap: the measured
transfer is of an essentially static phenotype.

## Recommendations

Unchanged from the first run, and now partially verified: making spawn
positions uniform (done in this revision) already lets agents reach food
occasionally. Remaining, in order of leverage:

- Raise `metabolic_base_cost` so survival is energy-limited and food
  restores it (creates a genuine food-seeking gradient).
- Compute the exploration component (currently always 0) so its 0.2
  weight is live.
- Revisit consumption radius / eat-gate threshold so acquisition is
  achievable but non-trivial.

Rerun this experiment after those changes; a gap measured on a
multi-component behavior would be a far stronger claim than the current
null.