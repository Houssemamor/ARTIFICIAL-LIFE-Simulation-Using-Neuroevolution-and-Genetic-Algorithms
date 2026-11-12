# Co-evolution Notes (Phase 6)

Read this before interpreting any predator/prey generation metrics.
The central trap this document exists to defuse:

**Within-generation fitness in a co-evolutionary system can fall with
no absolute regression, and can rise with no absolute progress.**

## The arms race is relative

Predator and prey fitness are measured against the *current* opposing
population. When prey evolve better evasion, predator fitness drops -
not because predators got worse, but because their opponents got better.
A generation-by-generation fitness table from the shared-world
evaluation (`experiments/EXP-COEV/coevolution_summary.json`) therefore
tracks the *balance of the arms race*, not the absolute capability of
either side. Oscillating metrics are the expected signature of a healthy
arms race, not stalled evolution; monotone metrics against a co-evolving
opponent would be the surprising result.

This is exactly the failure mode the plan warns about: "within-generation
fitness may reflect an arms race, not absolute improvement".

## The hall-of-fame is the absolute yardstick

`analytics/hall_of_fame.py` freezes each role's best genome every
`experiment.save_interval` generations. The frozen archive never
changes, so `win_rate_against_archive` measures current controllers
against FIXED opponents - an absolute progress signal:

- predator win rate rising: predators genuinely improved at catching
  (old-era) prey
- prey win rate rising: prey genuinely improved at surviving
  (old-era) predators

One nuance: at the generation a snapshot is taken, the archive contains
that snapshot when the win rates are computed, so the freshest opponent
is from the same generation. The run gates the first evaluation on the
generation after the first snapshot, so at least one strictly older
opponent is always in the mix.

Both rates rising simultaneously is possible and meaningful: each side
improved against frozen opponents even while the live arms race kept
the balance roughly even.

Win conditions (fixed by definition, `run_duel`):

- predator wins a duel if it captures the prey within the horizon
- prey wins if it survives the horizon uncaptured (including the case
  where the predator starves first)

## Stability status under the shipped defaults

With randomized initial headings (prey and predators no longer all face
right at spawn, and newborns also start with random headings), fixed
(non-evolved) random controllers are **robust in the committed sweep**.
Under `configs/predator_prey.json` (6 apex predators, 24 prey, 20
coevolution generations, per-role carrying capacities 40/8, food regrowth
0.25/step, capture radius 30 px, `metabolic_base_cost` 0.25/step, 40 px
solid obstacles), **1 of 10 seeds collapses** within 10 days. The nine
survivors (seeds 1, 2, 3, 5, 6, 7, 8, 9 and 10) run the full horizon with
both roles alive; seed 4 is the only failure (predator extinction, days 8,
min prey/predator 5/0, ending 6/0). The claim traces to a committed,
regenerable artifact: `experiments/EXP-COEV/stability_sweep.json`,
produced by `experiments/run_coevolution.py --mode stability-sweep --seed
1 --seeds 10 --days 10` (deterministic per seed). This sweep includes the
Phase B predator-pursuit incentive fix: predators weighted toward captures
(`survival 0.2, food 0.55, exploration 0.2, collision 0.05`) and prey
weighted toward survival/foraging
(`survival 0.45, food 0.25, exploration 0.2, collision 0.1`).

**The single collapse is a predator extinction.** Seed 4 loses predators;
no seed loses prey. The committed seed-1 ecosystem run also survives the
full 10 days with prey/predators 13/5, 19/6, 17/6, 20/6, 9/6, 18/5, 9/3,
13/5, 5/3, 2/4 and captures 9, 8, 6, 4, 9, 0, 5, 4, 3, 5 (ending with low
prey but no extinction). Compared to the previous uniform-facing pass in
the same 6-predator/20-generation config (8/10 collapses, seed 1 extinct
on day 4), randomizing initial headings removed a shared initial-orientation
bias and materially improved fixed-controller viability.

The coevolution run (20 generations, seed 1) shows sustained pursuit under
the same initial-state change: captures per generation are
7, 12, 8, 14, 11, 13, 13, 8, 10, 13, 15, 14, 16, 10, 14, 13, 15, 16, 13, 15
and predators remain 4-6 alive at evaluation end in every generation
(mostly 6/6). Hall-of-fame win rates against the frozen archive are
predator/prey 0.50/0.50, 0.33/1.00, 0.50/1.00, 0.00/0.80, 0.17/0.67,
0.14/0.86, 0.38/0.88, 0.44/1.00 and 0.40/0.90 across the nine snapshots
(final 0.40/0.90). Single duels in a 1200x700 arena remain
detection-limited, but the in-episode capture and survival signal is strong.

The history now has six measured corrections. First, `create_offspring`
once constructed newborns without the parent's energy config (subsidy
removed, 3/10 -> 7/10). Second, predators received role-aware senses,
replacing a prey-centric observation (7/10 -> 6/10). Third, 40 px solid
obstacles stopped straight-line shortcuts through walls (6/10 -> 7/10).
Fourth, role-specific capture-weighted fitness improved pursuit incentives
(7/10, all predator extinctions in the 3-predator/10-generation regime).
Fifth, increasing to 6 predators and 20 coevolution generations improved
learned pursuit but worsened the uniform-facing fixed-controller sweep to
8/10. Sixth, randomizing initial headings for prey, predators, and newborns
removed the shared facing-right bias and improved the same-regime sweep to
1/10 (9/10 full-horizon survivors).

The 0.25 metabolic setting remains correct for the ecosystem (at 0.7 every
predator starves before its first capture and the ecosystem collapses on day
0): the solo experiments run 150-step episodes, the ecosystem runs 150-step
days in a persistent world. If a still sturdier basin is needed, the next
levers are capture normalization/pressure, predator count/carrying capacity,
and the 0.25/step clock. Hall-of-fame duels use the configured energy
equation and legal spawns, so frozen-checkpoint win rates are measured under
the same dynamics as the generation metrics.

The stability test (`tests/integration/test_predator_prey_stability.py`)
therefore asserts viability on a fixed contiguous sample of sweep seeds
(6, 7, 8), with no cherry-picking within the block: each must complete the
full horizon with both roles alive and no seed may reach total extinction.
The 10-seed rate (1/10 collapse, seed 4 only) lives in the sweep artifact.
The negative test proves the fixture is not vacuous (24 predators with
reproduction disabled extinguish themselves within 7 days).

Three structural findings from the tuning work, for whoever iterates:

1. **Prey viability requires food intake.** At the original 5 px
   consumption radius, random controllers almost never found food, prey
   populations starved on any horizon, and no parameter combination
   could stabilize the ecosystem. The radius was raised to 12 px (the
   top recommendation from `docs/generalization_results.md`), and
   `configs/calibration.json` was regenerated to match. The Phase 9
   energy-limited dynamics then made food *mandatory* rather than
   optional, which is what makes captures possible at all: a predator
   that never captures starves.
2. **Per-role carrying capacities are mandatory.** A shared population
   cap let predators overshoot to prey-crushing numbers; splitting
   the cap (`max_population_prey` / `max_population_predator`) is what
   made any stability window exist at all.
3. **Offspring must inherit the parent's energy equation.** The
   default-config fallback was a silent dynamics bug that flattered
   the stability numbers; every stability claim re-measured after
   the fix.

Tuning knobs for the stability basin (plan Phase 6 step 5): predation
(`predator_count`, `capture_radius`, `capture_energy_transfer`),
reproduction (`min_age`, `min_energy`, `min_fitness`,
`reproduction_cost`, per-role caps), and the food economy
(`food_count`, `food_regrowth_per_step`).

## Determinism

All runs are reproducible per seed: torch is seeded to the run seed,
the global numpy stream to seed+10000 (then re-seeded to
`world.layout_seed` for the layout), and the behavioral rng (spawns,
reproduction jitter, mutation) to seed+30000. Two bugs found and fixed
during this phase, both regression-tested:

- the ecosystem's behavioral rng was once created with
  `np.random.default_rng()` (OS entropy), silently ignoring `--seed`
- the reproduction cap once counted dead agents, permanently blocking
  births in long runs once enough agents had died

## Interaction with the solo-experiment dynamics, incentives, and initial headings

The energy-limited dynamics carry into the ecosystem, with the slower
clock described above. The solo experiments (`docs/crossover_results.md`)
show a GA fitness gain over 12 generations, and the ecosystem is the
ecological analogue of the same selection pressure. The stability basin
moved sharply across measured corrections: it narrowed under corrected
newborn energy inheritance (3/10 -> 7/10), shifted with role-aware senses
and solid 40 px obstacles (7/10 -> 6/10 -> 7/10), stayed fragile in the
uniform-facing 6-predator/20-generation regime (8/10 collapses, seed 1
extinct day 4) despite stronger coevolution pursuit, and then widened
substantially when initial headings were randomized (1/10 collapse, seed 4
only; seed 1 survives 10 days ending 2 prey / 4 predators).

The 20-generation coevolution run under randomized headings sustains
captures 7, 12, 8, 14, 11, 13, 13, 8, 10, 13, 15, 14, 16, 10, 14, 13, 15,
16, 13, 15 with predators 4-6 alive at evaluation end in every generation
(mostly 6/6), and HOF predator/prey win rates end 0.40/0.90 across
snapshots 0.50/0.50, 0.33/1.00, 0.50/1.00, 0.00/0.80, 0.17/0.67, 0.14/0.86,
0.38/0.88, 0.44/1.00, 0.40/0.90. The honest summary is: capture incentives,
a larger coevolution budget, and unbiased initial headings together produce
strong pursuit signals and a much wider fixed-controller viability basin in
this measured regime, while predator-prey balance can still be tight (seed 1
ends with only 2 prey alive) and single-duel HOF rates remain
detection-limited.

The downstream NEAT system still feels little divergence pressure
(`docs/neat_extension_report.md`), for a different reason than before:
the landscape is now sharply *bimodal* (a few well-fed foragers near
fitness 2.0, most of the population starving near 0.3) rather than
uniformly inert. A bimodal pool still yields one compatibility
cluster at the Appendix C threshold within 30 generations.
