# Slide Deck Outline (Phase 8, Section 13.2)

~14 slides, one line of intent each. Numbers come from the committed
artifacts (`experiments/EXP-*/`, `docs/*_report.md`); regenerate
nothing by hand - if a number is challenged, open the JSON.

## 1. Title
Artificial Life: fixed-topology neuroevolution with a fully
specified NEAT extension. One-line framing per the plan.

## 2. Motivation
Rule-based controllers vs. evolutionary control: what changes when
the controller is not hand-written (Section 3 of the design).

## 3. Research questions
Primary: can a small brain learn an artificial-life world?
Secondary: does behavior depend on fitness weighting (Exp D/E), and
does it survive topology evolution (Phase 7)?

## 4. Related work
Sims 1994, Stanley & Miikkulainen 2002 (NEAT), prior ALife
platforms. What is precedented (evolution of controllers, energy
budgets) vs. novel here (the measured pipeline: calibration ->
weighted fitness -> paired statistical protocol -> frozen-checkpoint
co-evolution).

## 5. Architecture
One sentence per layer: sensors -> batched inference -> physics;
energy/organism; GA/NEAT; analytics; visualization. The detail is in
`docs/final_report.md` section 3.

## 6. Neuroevolution design + throughput
Fixed 12-32-16-3 topology, 557x faster than per-agent inference
(10,762 vs 19.3 steps/s at 250 agents, 358x the 30-FPS gate;
`docs/phase1_5_benchmark_report.md`). Genome packing is the
load-bearing detail: the flat-genome layout must match the batched
tensors or the speedup is fiction.

## 7. Fitness design
Calibration before weighting: measured component scales (survival
133.94, food 1.0, exploration 77.76, collision 1.20) then normalized
weighted sum. The scale-domination risk this fixed.

## 8. Live demo
`docs/demo_script.md` - GUI MVP tier, ecosystem advanced tier,
headless determinism, the experiments, topology growth.

## 9. Results: generalization (Exp A/B)
Train vs. unseen layouts under the final dynamics: +0.047 gap
(paired p = 0.1055, r = +0.60, 7/10 seeds, CI [+0.002, +0.089]) - a
consistent-direction penalty, not yet resolved at n = 10; see
`docs/generalization_results.md` for the paired numbers (CI + effect
size, no bare means).

## 10. Results: fitness weighting and crossover (Exp D/E)
Equal weighting leads every specialization by +0.07 to +0.12 (no pair
survives Holm); blend and uniform crossover are indistinguishable
(+0.001, p = 0.92), both +0.08 over mutation-only. The landscape is
live (the GA improves +0.28 to +0.50 over 12 generations) - the
weighting result says the balanced weights are already near-optimal
(`docs/experiment_d_results.md`, `docs/crossover_results.md`).

## 11. Results: predator/prey and hall-of-fame
The oscillating arms race chart, explained, not hidden
(`docs/coevolution_notes.md`): within-generation fitness is
relative; the frozen-archive win rate is the absolute measure.
Stability: 7 of 10 seeds collapse (5 predator extinctions,
2 prey) after the newborn-energy fix; viable runs climb
5 -> 22 captures/day.

## 12. Results: NEAT extension
Complexity-over-generations plot (51 -> 102 mean, max 117); the
batching decision (population-wide depth-layered padding, 2.9x
per-agent) and why the per-species prototype lost
(`docs/neat_extension_report.md`).

## 13. Limitations
One shape, honestly: the landscape is now *bimodal*, not graded - a
few well-fed foragers near fitness 2.0, most agents starving near
0.3. That shape is enough for the GA but not for NEAT speciation
(one species). The ecosystem's predator role is fragile (7/10
collapse). Every D/E contrast sits near but not past Holm at n = 10.

## 14. Future work
Re-run the solo experiments at n = 20-30; sweep metabolic cost
(0.4/0.7/1.0); soften the bimodality (intermediate food) to give
speciation a gradient; strengthen the ecosystem clock; presentation
rehearsal per Section 13.4.
