# Generalization Results (Phase 5)

**Experiment:** train on layouts A1-A3, freeze best controllers, evaluate
unmodified on held-out layouts B1-B3, quantify the train-vs-unseen gap.

**Status:** confirmatory run complete (10 seeds) under the
**energy-limited dynamics** (metabolic cost 0.7/step, food regrowth
0.3/step, live exploration component, 40 px solid obstacles). The
earlier train-higher gap does **not** reproduce: train 1.2620 vs
unseen 1.2703, gap **-0.0083** (paired p = 0.7695, r = -0.127,
4 of 10 seeds train-higher, bootstrap CI [-0.043, +0.029]). This is a
clean null, not a penalty.

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
caught it), the exploration component computed per agent, and 40 px
obstacles that are now solid. Calibration was re-measured against
these dynamics (survival 146.44, food 1.44, exploration 61.26,
collision 11.17). All reported experiments share this baseline.

| Measure | Value |
|---------|-------|
| Train score (mean over 10 seeds) | 1.2620 |
| Unseen score (mean over 10 seeds) | 1.2703 |
| Gap (train - unseen) | -0.0083 |
| Gap 95% bootstrap CI | [-0.043, +0.029] |
| Wilcoxon signed-rank p-value (two-sided) | 0.7695 |
| Matched-pairs rank-biserial | -0.127 |

Per-seed gaps: 4 of 10 favor train (+0.001 to +0.094; six seeds
favor unseen, by -0.002 to -0.086).

## Interpretation

**No measurable generalization gap under the shipped dynamics.** Frozen
controllers score 0.008 *higher* on held-out layouts than on their
training layouts (p = 0.77 at n = 10, effect size -0.13, 4/10 seeds
train-higher, bootstrap CI spanning zero). The point estimate and the
interval both say the same thing: this run cannot distinguish train
from unseen performance.

**The previous +0.047 penalty was regime-dependent.** It was measured
before obstacles became solid and before the collision/exploration
recalibration. Solid walls lengthen paths around obstacle clusters,
and the recalibrated weights reward exploration far more heavily than
food intake, so the controllers are less tuned to a particular
food/obstacle placement. That is a plausible mechanism, not a measured
one; the honest claim is that the penalty did not survive the dynamics
change.

**History of this question, for calibration of expectations:** the
5 px dynamics gave a reverse gap (unseen easier, p = 0.0645,
r = -0.67); the 12 px dynamics at legacy metabolic cost gave a clean
null (+0.001, p = 0.846); the first energy-limited run gave a
borderline train-higher gap (+0.047, p = 0.1055); the solid-obstacle
recalibration returns a null (-0.008, p = 0.77). Four dynamics
regimes, four different answers - which is itself the strongest
argument for treating any single run's generalization number as
regime-dependent.

## Recommendations

- Do not spend n = 20-30 chasing the +0.047 penalty: it is not present
  under the shipped dynamics. The near-threshold effects worth
  confirming are now the weighting and crossover contrasts.
- If layout generalization is a priority, make layout diversity the
  experimental factor (train on one cluster, test on a distant one)
  rather than adding seeds to a null.
- Keep the energy-limited dynamics as the default experiment regime;
  the null/penalty distinction depends on it.