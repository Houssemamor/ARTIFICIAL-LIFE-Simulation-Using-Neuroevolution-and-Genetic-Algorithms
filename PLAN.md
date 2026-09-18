# Development Plan — Artificial Life Neuroevolution Simulation

Source: `docs\Artificial_Life_Neuroevolution_Design_Document_v2.pdf` (v2.0, current).
This plan covers the first development steps only. Full phase detail is in the design doc.

## 0. Reading the spec

- v2.0 is authoritative. v1.0 is retained for history.
- Phases 1-6: fixed-topology neuroevolution (12-32-16-3 MLP).
- Phase 7: fully specified NEAT-style extension (Appendix C).
- Batched population inference is a Phase 1.5 exit criterion, not a late fix.

## 1. First steps (what to build first)

1. Project skeleton + dependency manifest (torch, pygame, numpy).
2. `simulation.environment` — World, Food, Obstacles, boundaries, spawn/reset.
3. `simulation.physics` — position, velocity, steering, fixed-timestep energy drain.
4. `simulation.engine` — main loop, fixed-timestep accumulator, headless/GUI modes.
5. `agents.organism` — agent state, genome binding, lifecycle, energy balance.
6. `agents.sensors` — 7 ray-based sensors, normalized observations.
7. `neural.network` + `neural.genome` — fixed MLP, deterministic packing/unpacking.
8. `neural.batched_inference` — population-wide batched forward pass.
9. `evolution.*` — selection, crossover (3 methods), mutation.
10. `analytics.*` — fitness calibration/normalization, metrics, statistical comparison.
11. `visualization.*` — renderer + dashboard.
12. `experiments.*` — config-driven headless runs, reproducibility tags.

## 2. Exit criteria per phase

| Phase | Exit criterion |
|-------|----------------|
| Phase 1 | Agents move and collide reliably |
| Phase 1.5 | Measured steps/sec at 250 agents feeds the compute budget |
| Phase 2 | Agents react to observations through batched NN outputs |
| Phase 3 | Fitness improves over controlled runs; crossover variants compared |
| Phase 4 | Experiments logged, repeatable, tagged by reproducibility tier |
| Phase 5 | Comparison pipeline ready before any headline experiment runs |
| Phase 6 | Unseen layouts and fitness-weighting sensitivity both evaluated |
| Phase 7 | Multi-agent ecosystem stable; co-evolution measured vs frozen checkpoints |
| Phase 8 | Topology evolution works on an isolated benchmark, matching Appendix C |

## 3. Immediate next action

Begin Phase 1: create the project skeleton and the `simulation.environment` module.