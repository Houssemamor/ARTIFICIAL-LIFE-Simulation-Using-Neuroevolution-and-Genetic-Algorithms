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
component, newborn energy inheritance; calibration re-measured at
survival 133.94, food 1.0, exploration 77.76, collision 1.20). The
batched fixed-topology evaluator runs at 10,762 steps/s with 250
agents (557x the per-agent reference, 358x the 30-FPS real-time
gate). The GA improves fitness by **+0.28 to +0.50 mean over 12
generations** - the strongest learning signal the project has
measured. Performance transfers to unseen layouts with a
consistent-direction train-higher gap of +0.047 (paired p = 0.1055,
r = +0.60, 7/10 seeds; bootstrap CI [+0.002, +0.089]). Fitness
weighting now has leverage, but it points *away* from
specialization: equal weighting leads every specialized condition by
+0.07 to +0.12 (no pair survives Holm at n = 10). Crossover method
does not matter at this scale (blend vs uniform +0.001, p = 0.92),
though both beat mutation-only by ~+0.08. The ecosystem is viable
but fragile (7 of 10 random seeds collapse, always via predator
extinction; captures climb 5 -> 22/day in survivors), and NEAT
topology grows (mean complexity 51 -> 102) with the batching
question resolved by measurement (2.9x the per-agent reference).

## 2. System

**World.** A 1200x700 continuous world with food, point obstacles,
and bounded agents. The engine is step-based: observe -> batched
inference -> act -> physics -> metabolism. A fixed-step accumulator
decouples simulation rate from render rate, which is what makes both
the throughput benchmark and the demo story possible.

**Agents.** Seven ray sensors (each returning nearest-surface
distance plus left/right bearing) feed a 12-value observation vector.
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
| Train / unseen means | 0.8977 / 0.8510 |
| Gap (train - unseen), 95% bootstrap CI | +0.0467 [+0.002, +0.089] |
| Paired Wilcoxon p / matched-pairs r | 0.1055 / +0.60 |

A consistent-direction generalization *penalty* (7 of 10 seeds favor
train; the bootstrap CI barely excludes zero) - enabled by
controllers that learn a real food-seeking policy. The paired test
misses 0.05 at n = 10; the first recommendation is a larger-n
confirmation (`docs/generalization_results.md`).

### 4.2 Fitness weighting (Experiment D) and crossover (Experiment E)

D1-D4 (equal, resource, exploration, survival weights) span
0.8367-0.9532: **equal weighting leads every specialized condition by
+0.07 to +0.12** (r = +0.24 to +0.35; no pair survives Holm at
n = 10). Resource- and exploration-weighting are indistinguishable
from each other (diff -0.0002, r = -0.02). E (blend / uniform /
mutation-only) spans 0.8094-0.8917, with blend and uniform
statistically identical (+0.0014, p = 0.92) and both ahead of
mutation-only by ~+0.08 (r up to +0.49, short of significance). The
fitness-over-generations curve rises **+0.28 to +0.50** mean over 12
generations - 7-9x the strongest previous signal, and the Phase 3
"fitness improves" exit expectation cleared decisively
(`docs/experiment_d_results.md`, `docs/crossover_results.md`).

### 4.3 Predator/prey and the hall of fame (Phase 6)

Capture-collision resolution plus in-episode reproduction produce a
viable-but-fragile ecosystem: 7 of 10 random seeds collapse within 10
days - every failure a predator extinction, a consequence of the
newborn-energy fix (offspring now run the parent's 0.25/step
equation instead of a silent 0.1 subsidy) - while the survivors
settle at carrying capacity with captures per day climbing 5 -> 22 as
selection favors capturable prey. The stability test asserts
viability on the first three sweep seeds (no cherry-picking) and a
negative test proves the fixture can fail. The co-evolution run's
hall-of-fame evaluation still shows prey win rates near 1.00 against
archived opponents (unevolved predators rarely catch anything inside
150-step duels)
(`docs/coevolution_notes.md`).

### 4.4 NEAT extension (Phase 7)

Topology grows: mean complexity 51 -> 102, max 51 -> 117 over 30
generations (`experiments/EXP-NEAT/complexity_over_generations.svg`).
The batching question is resolved by measurement: population-wide
depth-layered padded batching runs at 249 steps/s against 85 for the
per-agent reference (2.9x; the per-species prototype reached 221),
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

- the GA's improvement signal grew 7-9x (+0.033 -> +0.28/+0.50);
- generalization produced a consistent-direction penalty (+0.047,
  CI barely excluding zero) - controllers overfit their training
  layouts slightly;
- fitness weighting acquired leverage, and the leverage favors the
  status quo: equal weighting leads every specialization (+0.07 to
  +0.12). The re-weighting experiments are now genuinely informative
  - they say the balanced weights are already near-optimal;
- crossover method remains indistinguishable (blend = uniform), with
  both ahead of mutation-only by ~+0.08;
- the ecosystem's collapse rate rose (3/10 -> 7/10) once the newborn
  energy subsidy was removed: viable runs show captures climbing
  5 -> 22/day, but the predator role is fragile.

That last item deserves emphasis because it cuts against the earlier
framing: the first Phase 9 iteration *improved* every stability
number, and the improvement was a bug. The stability artifact was
regenerated after the newborn-energy fix, the fixture seeds were
replaced with an unselected sample, and the integration test now
asserts the weaker property the system actually has (viability, not
stability).

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

1. **Statistical power.** The most interesting effects (the +0.047
   generalization gap, the +0.12 equal-vs-survival weighting gap,
   the +0.08 crossover-vs-mutation gap) sit just below or just
   above Holm-corrected significance at n = 10. Re-running the three
   solo experiments at n = 20-30 is the single highest-value next
   action - the effects are sized to resolve with more seeds.
2. **Bimodal fitness landscape.** The energy-limited dynamics
   separate foragers from starvers rather than grading between them
   (a few near fitness 2.0, most near 0.3). That is enough leverage
   for the GA but not for NEAT speciation, which still reads one
   species. A population with intermediate food availability (or a
   longer horizon before starvation) would give speciation a
   gradient to act on.
3. **Metabolic cost as a factor.** 0.7/step is a load-bearing
   experimental parameter: it sets the survival-vs-forging split
   that every result above depends on. It deserves a sweep
   (0.4 / 0.7 / 1.0) rather than a single point. The ecosystem's
   0.25/step is the same knob set for a persistent world and is the
   first candidate if ecosystem robustness matters (its 7/10
   collapse rate is the weakest number in this report).
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

# tests (248)
.venv\Scripts\python.exe -m pytest tests/ -q

# experiments (each writes JSON + SVG under experiments/)
.venv\Scripts\python.exe experiments\run_generalization.py --seeds 10
.venv\Scripts\python.exe experiments\run_experiment_d.py --seeds 10
.venv\Scripts\python.exe experiments\run_crossover_experiment.py --seeds 10
.venv\Scripts\python.exe experiments\run_coevolution.py --config configs\predator_prey.json --mode ecosystem --seed 4 --days 10
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
