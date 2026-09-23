# Experiment D Results: Fitness-Weighting Sensitivity (Phase 5)

**Experiment:** does the choice of fitness weighting change which
controllers evolve? Four conditions (D1-D4) were trained and their frozen
best genomes compared on one common balanced metric.

**Status:** confirmatory run complete (10 seeds per condition). No
significant differences. An earlier exploratory analysis suggested an
ordering trend (survival > resource > exploration > equal); the paired
analysis below shows that trend was an artifact of unpaired testing and
non-uniform spawn positions, and it disappears under the corrected
method.

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

### Condition scores (balanced evaluation, 10 seeds)

| Condition | Mean | 95% bootstrap CI |
|-----------|------|------------------|
| D1 equal | 0.3081 | [0.270, 0.343] |
| D2 resource-weighted | 0.3287 | [0.283, 0.365] |
| D3 exploration-weighted | 0.3358 | [0.285, 0.385] |
| D4 survival-weighted | 0.3500 | [0.322, 0.375] |

### Pairwise comparisons (paired Wilcoxon + Holm-Bonferroni)

| Comparison | Mean diff | p-value | Matched-pairs r | Significant (Holm, a=0.05) |
|------------|-----------|---------|-----------------|----------------------------|
| equal vs resource-weighted | -0.021 | 0.625 | -0.200 | No |
| equal vs exploration-weighted | -0.028 | 0.232 | -0.455 | No |
| equal vs survival-weighted | -0.042 | 0.131 | -0.564 | No |
| resource vs exploration-weighted | -0.007 | 0.770 | -0.127 | No |
| resource vs survival-weighted | -0.021 | 0.557 | -0.236 | No |
| exploration vs survival-weighted | -0.014 | 0.193 | -0.491 | No |

**No pair is significant.** The closest comparison (equal vs
survival-weighted, p = 0.131, r = -0.564) has a medium point effect size
but does not survive correction at n = 10.

Note on scales: this revision normalizes against the **measured**
calibration scales (`configs/calibration.json`) rather than the
hand-picked constants of the first confirmatory pass. The condition
ordering also differs from that pass (survival > exploration > resource
> equal here) - under a degenerate landscape the ordering is noise, which
is itself informative.

## Interpretation

**Fitness weighting does not change which controllers evolve under the
current dynamics.** The four conditions produce statistically
indistinguishable genomes on the balanced metric.

**Why the exploratory "trend" vanished.** The earlier 3-seed pass showed
survival-weighted (0.34) apparently beating equal (0.20). With the
paired design and uniform spawns, all four conditions sit within 0.04 of
each other. Two artifacts produced the trend: unpaired testing counted
between-seed variance as noise against the effect (the paired test
removes it, *and* shrinks the apparent differences by measuring them
within-seed), and corner-line spawning made evaluation scores partly a
function of where obstacles happened to sit relative to the spawn line.
The condition ordering also changed between the two confirmatory passes
once measured calibration scales replaced hand-picked constants -
under a flat landscape the ordering is seed noise, which is itself
informative.

**The degenerate-landscape caveat remains the dominant explanation.**
Telemetry from the reruns: food is only occasionally eaten (0-0.12 items
per seed), survival saturates within the evaluation window for most
agents, and the exploration component is never computed. When three of
four fitness components are (nearly) inert, re-weighting them cannot
change selection pressure - the conditions are near-identical by
construction, so the null result is expected *for this dynamics*, and
says little about fitness weighting in a richer environment.

## Recommendations

Same as the generalization experiment: make the fitness components live
(metabolic cost, consumption radius, exploration computation) and rerun.
Only then does a null result here become evidence about weighting
robustness rather than about inert components. If differences still fail
to reach significance on a live landscape, that would be a meaningful
robustness finding for the design document's Experiment D.