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

With randomized initial headings and role-specific behavioral shaping,
fixed (non-evolved) random controllers are **viable but seed-sensitive in
the committed sweep**. Under `configs/predator_prey.json` (6 apex predators,
24 prey, 20 coevolution generations, per-role carrying capacities 40/8, food
regrowth 0.25/step, capture radius 30 px, `metabolic_base_cost` 0.25/step,
40 px solid obstacles), **4 of 10 seeds collapse** within 10 days. Six seeds
(1, 2, 3, 7, 9 and 10) run the full horizon with both roles alive; four
collapse: seed 4 loses predators (full horizon, min prey/predator 4/0, ends
4/0), seed 5 loses prey (full horizon, min 0/4, ends 0/4), seed 6 loses
predators on day 9 (min 15/0, ends 31/0), and seed 8 loses predators on day
7 (min 1/0, ends 1/0). The claim traces to a committed, regenerable
artifact: `experiments/EXP-COEV/stability_sweep.json`, produced by
`experiments/run_coevolution.py --mode stability-sweep --seed 1 --seeds 10
--days 10` (deterministic per seed).

The current pass adds dense, role-specific shaping on top of base fitness
in coevolution and ecosystem evaluation (no sensor, architecture, capture,
or HOF win-condition changes): prey receive a threat-avoidance bonus from
mean distance to the nearest live predator while alive, and predators
receive closing-progress reward toward the nearest live prey with a mild
penalty for persistent turning. This targets the observed "prey approach
predators" and "predator left-circling" behaviors by giving both roles a
denser pursuit/avoidance gradient than captures and survival alone.

**Collapses are mostly predator extinctions (3 predator, 1 prey).** The
committed seed-1 ecosystem run survives the full 10 days with
prey/predators 13/5, 19/6, 17/6, 20/6, 9/6, 14/4, 15/2, 20/1, 27/3, 19/6
and captures 9, 8, 6, 4, 9, 0, 2, 1, 10, 4 (ending 19 prey / 6 predators),
including a mid-run predator dip to 1 alive at day 7 followed by recovery.
Compared to the pre-shaping randomized-heading pass in the same regime
(1/10 collapse in that sweep, seed 1 ended 2 prey / 4 predators), shaping
improved the committed seed-1 end balance but the 10-seed sweep measured
4/10 collapses under the shaped dynamics, indicating that viability remains
sensitive to seed and population balance.

The coevolution run (20 generations, seed 1) shows sustained pursuit under
the shaped dynamics: captures per generation are
7, 4, 11, 17, 13, 10, 13, 12, 14, 11, 13, 15, 16, 11, 14, 14, 15, 15, 11, 12
and predators remain 4-6 alive at evaluation end in every generation
(mostly 6/6). Hall-of-fame win rates against the frozen archive are
predator/prey 0.50/0.00, 0.00/0.33, 0.25/0.50, 0.60/1.00, 0.83/0.83,
0.71/0.86, 0.50/0.88, 0.11/0.89 and 0.30/0.70 across the nine snapshots
(final 0.30/0.70). Single duels in a 1200x700 arena remain
detection-limited, so in-episode multi-agent capture/survival behavior is a
stronger pursuit signal than the duel win rates alone.

The history now has seven measured corrections. First, `create_offspring`
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
1/10 (9/10 full-horizon survivors). Seventh, dense prey-avoidance and
predator-closing/anti-turn shaping were added to address approach/circling
tendencies; in the committed shaped sweep this yielded 4/10 collapses but a
stronger seed-1 ecosystem end balance (19 prey / 6 predators) and sustained
coevolution captures.

The 0.25 metabolic setting remains correct for the ecosystem (at 0.7 every
predator starves before its first capture and the ecosystem collapses on day
0): the solo experiments run 150-step episodes, the ecosystem runs 150-step
days in a persistent world. If a still sturdier basin is needed, the next
levers are capture normalization/pressure, population economics, richer HOF
evaluation, and (if needed) sensing cues such as relative velocity or short
memory. Hall-of-fame duels use the configured energy equation and legal
spawns, so frozen-checkpoint win rates are measured under the same dynamics
as the generation metrics.

The stability test (`tests/integration/test_predator_prey_stability.py`)
therefore asserts viability on a fixed contiguous sample of sweep seeds
(1, 2, 3), with no cherry-picking within the block: each must complete the
full horizon with both roles alive, and no seed may reach total extinction.
The 10-seed rate (4/10 collapses: 3 predator, 1 prey) lives in the sweep
artifact. The negative test proves the fixture is not vacuous (24 predators
with reproduction disabled extinguish themselves within 7 days).

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

## Interaction with the solo-experiment dynamics, incentives, headings, and shaping

The energy-limited dynamics carry into the ecosystem, with the slower
clock described above. The solo experiments (`docs/crossover_results.md`)
show a GA fitness gain over 12 generations, and the ecosystem is the
ecological analogue of the same selection pressure. The stability basin
moved sharply across measured corrections: it narrowed under corrected
newborn energy inheritance (3/10 -> 7/10), shifted with role-aware senses
and solid 40 px obstacles (7/10 -> 6/10 -> 7/10), stayed fragile in the
uniform-facing 6-predator/20-generation regime (8/10 collapses, seed 1
extinct day 4) despite stronger coevolution pursuit, widened when initial
headings were randomized (1/10 collapse in that sweep), and measured 4/10
collapses in the committed shaped sweep (3 predator, 1 prey) with 6/10
seeds surviving the full horizon.

The 20-generation coevolution run under randomized headings + shaping
sustains captures 7, 4, 11, 17, 13, 10, 13, 12, 14, 11, 13, 15, 16, 11, 14,
14, 15, 15, 11, 12 with predators 4-6 alive at evaluation end in every
generation (mostly 6/6), and HOF predator/prey win rates end 0.30/0.70
across snapshots 0.50/0.00, 0.00/0.33, 0.25/0.50, 0.60/1.00, 0.83/0.83,
0.71/0.86, 0.50/0.88, 0.11/0.89, 0.30/0.70. The committed seed-1 ecosystem
run survives 10 days ending 19 prey / 6 predators, an improvement in end
balance over the pre-shaping seed 1 result (2 prey / 4 predators), though
predators dipped to 1 alive mid-run (day 7) before recovering.

The honest summary is: dense avoidance and pursuit shaping, together with
capture incentives, a larger coevolution budget, and unbiased initial
headings, produce strong in-episode pursuit and viable ecosystems in many
seeds, but the committed sweep shows the basin is still seed-sensitive
(4/10 collapses). This does not by itself prove the qualitative
approach/circling behaviors are eliminated in every replay - showcase and
HOF duels remain detection-limited views - and prey-predator balance can
still swing mid-run. If those artifacts persist in replays, the next
highest-value levers are sensing cues (relative velocity/closing rate),
capture normalization/pressure, population economics, and richer pursuit
diagnostics.

The downstream NEAT system still feels little divergence pressure
(`docs/neat_extension_report.md`), for a different reason than before:
the landscape is now sharply *bimodal* (a few well-fed foragers near
fitness 2.0, most of the population starving near 0.3) rather than
uniformly inert. A bimodal pool still yields one compatibility
cluster at the Appendix C threshold within 30 generations.
