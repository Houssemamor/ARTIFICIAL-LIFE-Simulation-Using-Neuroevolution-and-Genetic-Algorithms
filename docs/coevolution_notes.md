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

Honest finding: with fixed (non-evolved) random controllers, the
ecosystem is **viable but fragile**. Under
`configs/predator_prey.json` (3 apex predators, 24 prey, per-role
carrying capacities 40/8, food regrowth 0.25/step, capture radius
30 px, `metabolic_base_cost` 0.25/step, 40 px solid obstacles),
**7 of 10 seeds collapse** (one role extinct) within 10 days. The
three survivors (seeds 1, 3 and 5) run the full horizon with both
roles alive; their per-seed minima are 14/1, 18/1 and 2/3
(prey/predator). The claim traces to a committed, regenerable
artifact: `experiments/EXP-COEV/stability_sweep.json`, produced by
`experiments/run_coevolution.py --mode stability-sweep --seed 1
--seeds 10 --days 10` (the pipeline is deterministic per seed, so a
re-run reproduces the file). This sweep includes the Phase B
predator-pursuit incentive fix: role-specific fitness weights in the
config, with predators weighted toward captures
(`survival 0.2, food 0.55, exploration 0.2, collision 0.05`) and prey
weighted toward survival/foraging
(`survival 0.45, food 0.25, exploration 0.2, collision 0.1`), resolved
per role in co-evolution and ecosystem fitness updates.

**Every collapse is a predator extinction.** The 7 failing seeds (2, 4,
6, 7, 8, 9 and 10) all lose the predator role; none lose prey. This is
a different failure mode from the previous sweep under the same
collapse rate (7/10), which split 5 predator / 2 prey extinctions with
survivors 3, 4 and 5. The incentive fix made predators more
capture-seeking and improved seed 1 from an early predator die-off to a
full 10-day run (seed 1 is now a survivor, 14/1 minimum), but it did
not widen the stability basin: when captures do not arrive fast enough,
the predator population still starves.

The history now has four measured corrections. First, `create_offspring`
once constructed newborns without the parent's energy config, so every
predator born after day 0 silently ran the legacy 0.1/step equation
while its parent ran 0.25 - an easier landscape for offspring than for
parents. Fixing that removed the subsidy and raised the collapse rate
from 3/10 to 7/10. Second, the original 12-input observation was
prey-centric: input 7 always pointed at the nearest plant and the seven
rays always hit plants first, so a predator spent its life walking
toward food it cannot eat (the engine blocks plant consumption for
predators). Predators now receive role-aware senses - rays ignore
plants, input 7 points at the nearest live prey, input 8 at the nearest
rival predator - while prey observations remain byte-identical. With
effective predators the collapse rate fell to 6/10 and the failure mode
moved to prey overharvest. Third, the 40 px solid obstacles stopped
agents from driving through walls; the likely mechanism is energy, since
predators can no longer take straight-line shortcuts through obstacle
clusters (6/10 -> 7/10). Fourth, role-specific fitness weights now pay
predators much more for captures than for generic survival, which keeps
seed 1 viable through the full horizon but leaves the 10-seed collapse
count unchanged and predator starvation as the only observed failure.

The committed seed-1 ecosystem run illustrates the improved but still
fragile regime: it survives all 10 days with prey/predators 16/2,
14/5, 17/4, 24/3, 16/3, 19/1, 20/1, 20/6, 30/6, 21/6 and captures
6, 10, 2, 2, 8, 0, 4, 5, 11, 6. Predators dip to a single survivor on
days 5-6, then recover to 6 by day 7 through reproduction after later
captures. The 0.25 metabolic setting remains correct for the ecosystem
(at 0.7 every predator starves before its first capture and the
ecosystem collapses on day 0, because one missed day is death): the
solo experiments run 150-step episodes, the ecosystem runs 150-step
days in a persistent world, so it needs a slower clock. Lowering the
ecosystem clock further (0.15-0.20), reducing capture pressure (radius
30 -> 20 px or transfer 60 -> 40), or increasing predator
count/generations are the obvious next levers if a sturdier demo is
needed. The hall-of-fame duels were fixed earlier the same way:
`run_duel` builds its duelists with the configured energy equation and a
legal spawn clear of the solid bodies, so frozen-checkpoint win rates
are measured under the same dynamics as the generation metrics they are
compared against.

The stability test (`tests/integration/test_predator_prey_stability.py`)
therefore asserts the honest property - viability, not stability -
on the first three sweep seeds (1, 2, 3), with no cherry-picking:
at least one must complete the full horizon with both roles alive
(seeds 1 and 3 do), and no seed may reach total extinction. The 10-seed
rate lives in the sweep artifact. The negative test proves the
fixture is not vacuous (24 predators with reproduction disabled
extinguish themselves within 7 days).

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

## Interaction with the solo-experiment dynamics and incentive fix

The energy-limited dynamics carry into the ecosystem, with the slower
clock described above. The solo experiments (`docs/crossover_results.md`)
show a GA fitness gain over 12 generations, and the ecosystem is the
ecological analogue of the same selection pressure. The stability basin
*narrowed* under the corrected dynamics (3 of 10 -> 7 of 10 collapse)
once newborns inherited the parent's energy equation; role-aware
predator senses briefly traded predator starvation for prey overharvest
(7/10 -> 6/10), and the 40 px solid obstacles moved the failure mode
back to predator starvation (6/10 -> 7/10). Role-specific fitness
weights (the predator-pursuit incentive fix) then changed the character
of the failures without changing the rate: the committed sweep is still
7/10, but every collapse is now a predator extinction and seed 1 is a
full-horizon survivor. The committed seed-1 ecosystem run shows the
pattern directly: captures of 6, 10, 2, 2, 8, 0, 4, 5, 11, 6 over ten
days, with predators dipping to 1 on days 5-6 before recovering to 6 by
day 7. The honest summary is: learning and pursuit incentives are real,
viability is seed-dependent, and the fragile role remains the predator.

The downstream NEAT system still feels little divergence pressure
(`docs/neat_extension_report.md`), for a different reason than before:
the landscape is now sharply *bimodal* (a few well-fed foragers near
fitness 2.0, most of the population starving near 0.3) rather than
uniformly inert. A bimodal pool still yields one compatibility
cluster at the Appendix C threshold within 30 generations.
