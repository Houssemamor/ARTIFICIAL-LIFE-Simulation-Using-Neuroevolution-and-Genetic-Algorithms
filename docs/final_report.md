# Final Technical Report

Artificial Life Neuroevolution Simulation - fixed-topology
neuroevolution with a fully specified NEAT extension, built in eight
phases from `PLAN.md`. This report is assembled from the per-phase
documents and the frozen experiment artifacts under `experiments/`;
every number below traces to a committed JSON or SVG, and nothing is
re-derived here.

## 1. Abstract

A batched neuroevolution pipeline for a 2-D artificial-life world:
agents sense through a 7-ray sensor fan, decide through a neural
controller, and move under physics, paying an energy budget. The
project spans calibration-driven fitness design, a throughput-first
inference architecture, a confirmatory experiment program with a
paired statistical protocol, a two-population predator/prey
ecosystem with hall-of-fame evaluation against frozen checkpoints,
and a complete NEAT implementation with a resolved answer to
batched inference under variable topology.

Headline results, all measured under one final dynamics baseline
(Phase 9: energy-limited - `metabolic_base_cost` 0.7/step, food
regrowth 0.3/step applied in every evaluation loop, live exploration
component, newborn energy inheritance, 40 px solid obstacles;
calibration re-measured at survival 146.44, food 1.44, exploration
61.26, collision 11.17). The
batched fixed-topology evaluator runs at 10,762 steps/s with 250
agents (557x the per-agent reference, 358x the 30-FPS real-time
gate). The GA improves fitness by **+0.51 to +0.72 mean over 12
generations** - the strongest learning signal the project has
measured. Performance on unseen layouts is statistically
indistinguishable from training layouts (gap -0.008, paired
p = 0.7695, r = -0.127, 4/10 seeds train-higher; bootstrap CI
[-0.043, +0.029]); the earlier +0.047 penalty was regime-dependent.
Fitness weighting now has leverage, but no re-weighting is
measurably better: equal has the best point estimate, tied with
exploration-weighting (+0.015), and survival-weighting is last by
+0.18 (no pair survives Holm at n = 10). Crossover method
still has no Holm-significant winner (blend leads uniform by +0.12,
p = 0.13, and mutation-only by +0.16, p = 0.064). The ecosystem is viable
but fragile (8 of 10 random seeds collapse in the 6-predator/20-generation
fixed-controller sweep - seven via predator extinction, one via prey),
and NEAT topology grows (mean complexity 51 -> 107) with the batching
question resolved by measurement (3.1x the per-agent reference).
Role-specific capture incentives and a larger coevolution budget improved
in-episode pursuit, but fixed-controller ecosystem robustness worsened
from 7/10 to 8/10.

## 2. System

**World.** A 1200x700 continuous world with food, 40 px solid
obstacles, and bounded agents. The engine is step-based: observe -> batched
inference -> act -> physics -> metabolism. A fixed-step accumulator
decouples simulation rate from render rate, which is what makes both
the throughput benchmark and the demo story possible.

**Agents.** Seven ray sensors (each returning nearest-surface
distance plus left/right bearing) feed a 12-value observation vector.
The food and threat channels are role-aware: prey read plants and any
other live agent, while predators read live prey and rival predators
and never see plants, because they cannot eat them.
The controller is a 12-32-16-3 MLP decoded as tanh steering plus
sigmoid throttle and eat-gate. The energy equation charges base
metabolism, action costs, and collision contacts, and pays for food;
energy is clamped at its initial value (a documented deviation from
the plan equation: without the clamp, a fed agent could bank
unbounded reserves and never die).

**Evolution.** A tournament-selection GA with elitism, three
interchangeable crossover methods, and Gaussian weight mutation, plus
Phase 6's two-population co-evolution (predator/prey roles, capture
via the eat-gate, in-episode reproduction with per-role carrying
capacities) and Phase 7's full NEAT (innovation tracking,
innovation-aligned crossover, compatibility-distance speciation,
fitness sharing, stagnation removal, structural mutation with the
paper's weight-1.0 node-split initialization).

**Analytics.** Calibration by measurement (random-policy and
stand-still probes produce the reference scales; weights are never
picked blind), a confirmatory experiment program with seed-blocked
designs, paired Wilcoxon tests with Holm-Bonferroni correction,
bootstrap intervals, and matched-pairs effect sizes - reported
together, never p-values alone.

**Determinism.** Every entry point (GUI, headless, all experiment
runners) pins torch, the global numpy stream, Python's `random`, and
the behavioral rng from a per-run seed; the seeded-rerun integration
test asserts byte-identical trajectories. Two determinism bugs were
found and regression-tested during Phase 6 (an OS-entropy behavioral
rng; a reproduction cap that counted corpses).

## 3. Throughput (Phase 1.5)

| Path | Steps/s @250 agents |
|---|---|
| Batched (stacked weights, torch.bmm) | **10,762** |
| Per-agent loop (naive reference) | 19.3 |

The 557x speedup over the per-agent reference is the architectural
payoff of fixing one topology (358x over the 30-FPS real-time gate):
the whole population's parameters become stacked `(N, in, out)`
tensors, and one matmul per layer serves everyone. The
implementation detail that makes it real: the flat genome layout
(W then b per layer) must match the batched tensor slices, or the
speedup is fiction.

## 4. Experiments (all re-run under the energy-limited dynamics, 10 seeds)

### 4.1 Generalization (train layouts A1-A3 vs. held-out B1-B3)

| Measure | Value |
|---|---|
| Train / unseen means | 1.2620 / 1.2703 |
| Gap (train - unseen), 95% bootstrap CI | -0.0083 [-0.043, +0.029] |
| Paired Wilcoxon p / matched-pairs r | 0.7695 / -0.127 |

A clean null: frozen controllers score 0.008 *higher* on held-out
layouts, the interval spans zero, and only 4 of 10 seeds favor train.
The earlier +0.047 train-higher gap was measured before the obstacles
became solid and before the collision/exploration recalibration; it
does not reproduce under the shipped dynamics
(`docs/generalization_results.md`).

### 4.2 Fitness weighting (Experiment D) and crossover (Experiment E)

D1-D4 (equal, resource, exploration, survival weights) span
1.1750-1.3562: **equal weighting has the best point estimate, but it
is tied with exploration-weighting (+0.015)**. Survival-weighting is
last by +0.18 (r = +0.53, p = 0.160; no pair survives Holm at
n = 10). Resource- and exploration-weighting are indistinguishable
from each other (-0.058, r = -0.09). E (blend / uniform /
mutation-only) spans 1.1315-1.2929, with no pair surviving Holm:
blend leads uniform by +0.1218 (p = 0.131, r = +0.56) and
mutation-only by +0.1614 (p = 0.064, r = +0.67), while uniform and
mutation-only are indistinguishable (+0.0396, p = 0.846). The
fitness-over-generations curve rises **+0.51 to +0.72** mean over 12
generations - 15-22x the strongest previous signal, and the Phase 3
"fitness improves" exit expectation cleared decisively
(`docs/experiment_d_results.md`, `docs/crossover_results.md`).

### 4.3 Predator/prey and the hall of fame (Phase 6)

Capture-collision resolution plus in-episode reproduction produce a
viable-but-fragile ecosystem: 8 of 10 random seeds collapse within 10
days in the current 6-predator/20-generation regime. The number
reflects five measured corrections: the newborn-energy fix (offspring
now run the parent's 0.25/step equation instead of a silent 0.1
subsidy), the role-aware predator senses that replaced the original
prey-centric observation, the solid 40 px obstacles that stopped agents
driving through walls, role-specific fitness weights that reward
predators more strongly for captures than for generic survival, and the
increase to 6 predators / 20 coevolution generations. Seven failures
are predator extinctions (seeds 1, 2, 3, 5, 7, 9 and 10) and one is a
prey extinction (seed 4, ending 0 prey / 2 predators); the two
survivors (seeds 6 and 8) finish the horizon with per-seed minima of
15/4 and 13/1 prey/predator. The committed seed-1 fixed-controller
ecosystem run collapses on day 4 (14/5, 16/5, 15/6, 20/6, 17/0;
captures 9, 12, 8, 1, 0). The stability test asserts viability on a
fixed contiguous sweep sample (seeds 6-8, no cherry-picking within the
block; seeds 6 and 8 complete the horizon) and a negative test proves
the fixture can fail.

The 20-generation coevolution run shows materially stronger in-episode
pursuit than earlier passes: captures per generation are
5, 9, 9, 9, 10, 18, 12, 9, 11, 15, 14, 14, 13, 12, 13, 8, 14, 11, 15, 8,
and 4-6 predators remain alive at evaluation end in every generation.
Hall-of-fame win rates against the frozen archive are predator/prey
0.50/1.00, 0.33/1.00, 0.25/0.75, 0.20/0.80, 0.00/1.00, 0.00/0.86,
0.00/1.00, 0.00/1.00 and 0.10/1.00 across the nine snapshots (final
split 0.10 predator / 1.00 prey). Single duels in a 1200x700 arena
remain detection-limited, and the duel harness uses the same 0.25/step
energy equation and legal spawns as the generation metrics
(`docs/coevolution_notes.md`).

### 4.4 NEAT extension (Phase 7)

Topology grows: mean complexity 51 -> 107, max 51 -> 133 over 30
generations (`experiments/EXP-NEAT/complexity_over_generations.svg`).
The batching question is resolved by measurement: population-wide
depth-layered padded batching runs at 301 steps/s against 97 for the
per-agent reference (3.1x; the per-species prototype reached 299),
with compile-once-per-episode amortization. The evolutionary run
keeps a single species: the mechanism is unit-proven, but the
landscape is sharply bimodal (a few well-fed foragers near fitness
2.0, most agents starving near 0.3), and a bimodal pool still yields
one compatibility cluster within 30 generations
(`docs/neat_extension_report.md`).

## 5. Discussion: what the dynamics fix changed

The project's central scientific problem was a landscape whose fitness
was ~90% a saturated survival term, which made every behavioral
experiment read null. The Phase 9 fix - raising
`metabolic_base_cost` to 0.7/step so starvation bites inside the
150-step window, adding food regrowth, and computing the exploration
component - replaced that inert landscape with a live one, and the
nulls moved as a family:

- the GA's improvement signal grew 15-22x (+0.033 -> +0.51/+0.72);
- generalization returned a null (-0.008, p = 0.77) - no measurable
  layout overfitting under the shipped dynamics;
- fitness weighting acquired leverage, and the leverage favors the
  status quo only partly: equal is the best point estimate and is
  tied with exploration-weighting (+0.015), while survival-weighting
  is last by +0.18 (p = 0.16). The re-weighting experiments are now
  genuinely informative - down-weighting food is the one clear loser;
- crossover method still has no corrected-significant winner: blend
  leads uniform by +0.12 (p = 0.13) and mutation-only by +0.16
  (p = 0.064), with uniform and mutation-only indistinguishable;
- the ecosystem's fixed-controller collapse rate rose again in the
  6-predator/20-generation regime (7/10 -> 8/10: 7 predator extinctions,
  1 prey extinction) even though coevolution captures and in-episode
  predator survival improved substantially. The committed seed-1 run now
  collapses on day 4, while sweep survivors are seeds 6 and 8.

That last item deserves emphasis because it cuts against the earlier
framing: the first Phase 9 iteration *improved* every stability
number, and the improvement was a bug. The stability artifact was
regenerated after the newborn-energy fix, the fixture seeds were
replaced with an unselected sample, and the integration test now
asserts the weaker property the system actually has (viability, not
stability). Role-aware predator senses corrected a prey-centric
observation bug (6/10 collapse, prey overharvest), solid 40 px obstacles
moved the failure mode back toward predator starvation (7/10), and
role-specific capture incentives kept the 3-predator/10-generation
sweep at 7/10 with all predator extinctions. The 6-predator/20-generation
pass then improved evolved pursuit but worsened fixed-controller
robustness to 8/10 (7 predator, 1 prey extinction), showing that
coevolution progress and ecosystem stability are not the same claim.

What did *not* move: NEAT speciation still reads one species, and
Holm correction at n = 10 still holds every D and E contrast. Both
are honest limits of the current setup, not measurement failures -
the remaining work is statistical (larger n) and ecological (a graded
rather than bimodal fitness landscape).

A third finding survived every revision and belongs in the record:
the first apparent learning result (fitness 0.31 -> 0.60 in two
generations) was an evaluation-geometry artifact - corner-line spawns
funnelled random genomes through obstacle clusters. The spawn fix
removed the artifact; the honest improvement evidence arrived later,
twice - first from making food reachable (12 px radius, +0.04), then
from making survival energy-limited (+0.3). All three corrections
were documented in place rather than quietly overwritten.

## 6. Limitations and future work

1. **Statistical power.** The most interesting effects (the +0.18
   equal-vs-survival weighting gap, the +0.12 blend-vs-uniform and
   +0.16 blend-vs-mutation gaps) sit just below or just above
   Holm-corrected significance at n = 10. Re-running the solo
   experiments at n = 20-30 is the single highest-value next action -
   the effects are sized to resolve with more seeds.
2. **Bimodal fitness landscape.** The energy-limited dynamics
   separate foragers from starvers rather than grading between them
   (a few near fitness 2.0, most near 0.3). That is enough leverage
   for the GA but not for NEAT speciation, which still reads one
   species. A population with intermediate food availability (or a
   longer horizon before starvation) would give speciation a
   gradient to act on.
3. **Ecosystem robustness and predator economics.** The 8/10 collapse
   rate (7 predator, 1 prey extinction) coexists with much stronger
   6-predator/20-generation coevolution pursuit, which points at
   predator count/carrying capacity, capture normalization/pressure, and
   the 0.25/step ecosystem clock as the next levers. 0.7/step remains
   the load-bearing solo-experiment metabolic cost and still deserves a
   sweep (0.4 / 0.7 / 1.0).
4. **Dashboards.** The live view ships with tiered panels
   (`config.phase_tier` enforces MVP/Advanced separation so the MVP
   build cannot render a Phase 6/7-only metric), plus the Phase 9
   control bar and best-agent inspector. A presentation-mode
   two-column layout is the remaining cosmetic gap.
5. **GUI pacing.** The interactive demo runs the legacy metabolic
   cost by design (the GUI's 60 Hz step would starve agents in two
   seconds under experiment pacing); reported experiments never use
   the legacy value.

## 7. Reproducing the results

```bash
# environment
.venv\Scripts\python.exe -m pip install -e . --no-deps
.venv\Scripts\python.exe -m pip install -r requirements.txt

# tests (307)
.venv\Scripts\python.exe -m pytest tests/ -q

# experiments (each writes JSON + SVG under experiments/)
.venv\Scripts\python.exe experiments\run_generalization.py --seeds 10
.venv\Scripts\python.exe experiments\run_experiment_d.py --seeds 10
.venv\Scripts\python.exe experiments\run_crossover_experiment.py --seeds 10
.venv\Scripts\python.exe experiments\run_coevolution.py --config configs\predator_prey.json --mode ecosystem --seed 6 --days 10
.venv\Scripts\python.exe experiments\run_coevolution.py --config configs\predator_prey.json --mode stability-sweep --seed 1 --seeds 10 --days 10
.venv\Scripts\python.exe experiments\run_neat.py --config configs\neat_food_seeking.json --seed 1

# read-only results viewer (Phase 9)
.venv\Scripts\python.exe -m uvicorn webdash.backend.main:app

# GUI
.venv\Scripts\python.exe app\run.py --config configs\baseline.json --agent-mode neural
```

## 8. Artifact index

| Artifact | Contents |
|---|---|
| `docs/related_work.md` | Positioning against Sims (1994), Stanley & Miikkulainen (2002) |
| `docs/phase1_5_benchmark_report.md` | Throughput benchmark (10,762 steps/s) |
| `docs/generalization_results.md` | Experiment A/B results and interpretation |
| `docs/experiment_d_results.md` | Fitness-weighting results (D1-D4) |
| `docs/crossover_results.md` | Crossover results (E) + the improvement curve |
| `docs/coevolution_notes.md` | Arms-race caveat + ecosystem stability analysis |
| `docs/neat_extension_report.md` | NEAT implementation, batching decision, benchmark |
| `docs/demo_script.md`, `docs/slide_outline.md`, `docs/presentation_deck.pptx` | Presentation materials (14-slide deck, rehearsal script, author-side video checklist) |
| `webdash/` | Read-only FastAPI + Bootstrap viewer over the committed artifacts |
| `generate_deck.js` | Deck generator (pptxgenjs); re-run with `node generate_deck.js` |
| `experiments/EXP-*/` | Frozen run outputs (JSON summaries, SVGs) |
| `configs/` | All experiment configs + measured calibration |
