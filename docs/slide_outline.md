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
146.44, food 1.44, exploration 61.26, collision 11.17) then normalized
weighted sum. The scale-domination risk this fixed, and the collision
jump from 1.20 shows the cost of making the 40 px obstacles solid.

## 8. Live demo
`docs/demo_script.md` - GUI MVP tier, ecosystem advanced tier,
headless determinism, the experiments, topology growth.

## 9. Results: generalization (Exp A/B)
Train vs. unseen layouts under the final dynamics: no measurable gap
(-0.008; paired p = 0.7695, r = -0.127, 4/10 seeds train-higher,
CI [-0.043, +0.029]). The earlier +0.047 penalty did not survive the
solid-obstacle recalibration; see `docs/generalization_results.md`.

## 10. Results: fitness weighting and crossover (Exp D/E)
Equal weighting has the best point estimate but is tied with
exploration-weighting (+0.015); survival-weighting is last by +0.18
(p = 0.16; no pair survives Holm). Crossover has no
corrected-significant winner either:
blend leads uniform by +0.12 (p = 0.13) and mutation-only by +0.16
(p = 0.064), while uniform and mutation-only are indistinguishable
(+0.04, p = 0.85). The landscape is live (the GA improves +0.51 to
+0.72 over 12 generations) - the weighting result says the balanced
weights remain the best default
(`docs/experiment_d_results.md`, `docs/crossover_results.md`).

## 11. Results: predator/prey and hall-of-fame
The oscillating arms race chart, explained, not hidden
(`docs/coevolution_notes.md`): within-generation fitness is
relative; the frozen-archive win rate is the absolute measure.
Stability: 7 of 10 seeds collapse, and every collapse is a predator
extinction, after the newborn-energy fix, role-aware predator senses,
solid 40 px obstacles, and role-specific capture incentives. Survivors
are seeds 1, 3 and 5 (minima 14/1, 18/1, 2/3); seed 1 now survives 10
days (captures 6, 10, 2, 2, 8, 0, 4, 5, 11, 6) despite dipping to 1
predator mid-run.

## 12. Results: NEAT extension
Complexity-over-generations plot (51 -> 107 mean, max 133); the
batching decision (population-wide depth-layered padding, 3.1x
per-agent) and why the per-species prototype lost
(`docs/neat_extension_report.md`).

## 13. Limitations
One shape, honestly: the landscape is now *bimodal*, not graded - a
few well-fed foragers near fitness 2.0, most agents starving near
0.3. That shape is enough for the GA but not for NEAT speciation
(one species). The ecosystem is fragile (7/10 collapse, all predator
starvation despite stronger capture incentives). Every D/E contrast sits
near but not past Holm at n = 10.

## 14. Future work
Re-run the solo experiments at n = 20-30; sweep metabolic cost
(0.4/0.7/1.0); soften the bimodality (intermediate food) to give
speciation a gradient; tune the ecosystem clock and capture pressure;
presentation rehearsal per Section 13.4.
