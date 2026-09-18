# Artificial Life Neuroevolution Simulation

A 2D artificial-life simulation in which autonomous organisms learn and evolve behaviors through
neural-network controllers optimized by a genetic algorithm.

## Documents

| File | Version | Purpose |
|------|---------|---------|
| `docs\Artificial_Life_Neuroevolution_Design_Document.pdf` | v1.0 | Original design baseline |
| `docs\Artificial_Life_Neuroevolution_Design_Document_v2.pdf` | v2.0 (current) | Revised baseline integrating eleven fixes (C1-C11) from the independent Critical Review |

**Use v2.0 as the authoritative specification.** v1.0 is retained for history. v2.0 closes the
gaps between what v1.0 claimed and what it actually specified; it can be read standalone.

## Core Research Question

Can useful adaptive behavior emerge in autonomous agents when neural-network controllers are
optimized by evolutionary selection rather than directly programmed?

v2.0 adds two secondary questions:
- Does that behavior depend on how fitness is weighted? (Experiment D)
- Does it survive the transition from a fixed topology to full topology evolution? (Phase 7)

## System at a Glance

- **World:** 2D continuous space (1200 x 700), food, obstacles, boundaries.
- **Agents:** 100-250 organisms with 7 ray-based sensors (-90..+90 degrees).
- **Controller:** Fixed-topology MLP, `12-32-16-3`, ReLU hidden, tanh/sigmoid outputs
  (steering / accel / eat). Phases 1-6 only.
- **Evolution:** Real-coded GA - tournament selection + elitism, crossover (blend default,
  compared against uniform and mutation-only), Gaussian mutation.
- **Inference:** Population-wide batched forward pass (one matmul per layer across all
  agents), not per-agent calls. This is a Phase 1.5 exit criterion.
- **Fitness:** Components calibrated against random/stationary probes, then normalized and
  weighted before combining.
- **Reproducibility:** Every experiment tagged `cpu-deterministic` or `gpu`; CPU and GPU
  tiers are never pooled as equivalent.

## Stack

- Python (primary)
- PyGame (simulation engine / renderer)
- PyTorch (batched population inference)
- Optional: FastAPI + React web dashboard (presentation layer only, not coupled to core)

## Execution Modes

```bash
# Interactive GUI
python -m app.run --config configs/baseline.json --mode gui

# Headless scheduled experiment
python -m experiments.run --config configs/mutation_rate.json --seeds 10 --mode headless --device cpu
```

## Development Plan (summary)

| Phase | Work | Exit criterion |
|-------|------|----------------|
| Phase 1 | Environment + renderer + basic physics | Agents move and collide reliably |
| Phase 1.5 (NEW) | Prototype batched population inference; benchmark steps/sec | Measured steps/sec at 250 agents feeds the compute budget |
| Phase 2 | Sensors + fixed neural controller, batched from the start | Agents react to observations through batched NN outputs |
| Phase 3 | Fitness calibration/normalization + GA + crossover comparison arm | Fitness improves over controlled runs; crossover variants compared |
| Phase 4 | Analytics + checkpoints + determinism checklist | Experiments logged, repeatable, tagged by reproducibility tier |
| Phase 5 | Related-work write-up + statistical protocol implementation | Comparison pipeline ready before any headline experiment runs |
| Phase 6 | Generalization experiments + Experiment D (fitness weighting) | Unseen layouts and fitness-weighting sensitivity both evaluated |
| Phase 7 | Predator/prey + reproduction + hall-of-fame evaluation | Multi-agent ecosystem stable; co-evolution measured against frozen checkpoints |
| Phase 8 | NEAT-style extension (Appendix C) + packaging + final report | Topology evolution works on an isolated benchmark, matching the Appendix C spec |

See `PLAN.md` for the first-steps plan.

## Key Design Rules

1. **Batched inference is a Phase 1.5 exit criterion, not a late-stage fix.** ~250
   per-agent PyTorch calls per step become ~3 batched calls.
2. **Fitness is calibrated before it is trusted.** Each component's natural scale is measured
   with a random-policy and a stand-still probe; only then are coefficients locked.
3. **Scope honesty:** Phases 1-6 are fixed-topology neuroevolution. Phase 7 is a real,
   fully specified NEAT implementation (Appendix C), not a label applied to the whole system.
4. **Crossover is compared, not assumed.** Blend/arithmetic is the v2.0 default; uniform and
   mutation-only are comparison arms (Experiment E), because of the competing-conventions
   problem in real-valued weight vectors.
5. **Reproducibility tiers are separate.** CPU-deterministic and GPU runs are never compared
   as bit-identical.

## Experiments

- **A - Mutation rate:** 0.01 / 0.05 / 0.10 / 0.20
- **B - Population size:** 50 / 100 / 250 / 500
- **NC - Network capacity:** 12-8-3 / 12-16-8-3 / 12-32-16-3
- **D - Fitness weighting (NEW):** equal / resource-weighted / exploration-weighted / survival-weighted
- **E - Crossover method (NEW):** blend / uniform / mutation-only

Statistical comparison uses Mann-Whitney U on final fitness across 10 seeds, bootstrap CI on
the median, and Holm-Bonferroni correction (Appendix D).