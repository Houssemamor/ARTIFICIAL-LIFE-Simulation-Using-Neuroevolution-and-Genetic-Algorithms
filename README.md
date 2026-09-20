# Artificial Life Neuroevolution Simulation

A 2D artificial-life simulation in which autonomous organisms learn and evolve behaviors through neural-network controllers optimized by a genetic algorithm.

## Documents

| File | Version | Purpose |
|------|---------|---------|
| `docs\Artificial_Life_Neuroevolution_Design_Document.pdf` | v1.0 | Original design baseline |
| `docs\Artificial_Life_Neuroevolution_Design_Document_v2.pdf` | v2.0 (current) | Revised baseline integrating eleven fixes (C1-C11) from the independent Critical Review |

**Use v2.0 as the authoritative specification.** v1.0 is retained for history. v2.0 closes the gaps between what v1.0 claimed and what it actually specified; it can be read standalone.

## Core Research Question

Can useful adaptive behavior emerge in autonomous agents when neural-network controllers are optimized by evolutionary selection rather than directly programmed?

v2.0 adds two secondary questions:
- Does that behavior depend on how fitness is weighted? (Experiment D)
- Does it survive the transition from a fixed topology to full topology evolution? (Phase 7)

## System at a Glance

- **World:** 2D continuous space (1200 x 700), food, obstacles, boundaries.
- **Agents:** 100-250 organisms with 7 ray-based sensors (-90..+90 degrees).
- **Controller:** Fixed-topology MLP, `12-32-16-3`, ReLU hidden, tanh/sigmoid outputs (steering / accel / eat). Phases 1-6 only.
- **Evolution:** Real-coded GA - tournament selection + elitism, crossover (blend default, compared against uniform and mutation-only), Gaussian mutation.
- **Inference:** Population-wide batched forward pass (one matmul per layer across all agents), not per-agent calls. This is a Phase 1.5 exit criterion.
- **Fitness:** Components calibrated against random/stationary probes, then normalized and weighted before combining.
- **Reproducibility:** Every experiment tagged `cpu-deterministic` or `gpu`; CPU and GPU tiers are never pooled as equivalent.

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

| Phase | Work | Exit criterion | Status |
|-------|------|----------------|---------|
| Phase 0 | Project setup & environment | Installable, empty project skeleton matching the Section 0 tree | ✅ Complete |
| Phase 1 | Environment + renderer + basic physics | Agents move and collide reliably | ✅ Completed |
| Phase 1.5 (NEW) | Prototype batched population inference; benchmark steps/sec | Measured steps/sec at 250 agents feeds the compute budget | ✅ Complete |
| Phase 2 | Sensors + fixed neural controller, batched from the start | Agents react to observations through batched NN outputs | ❌ Not Started |
| Phase 3 | Fitness calibration/normalization + GA + crossover comparison arm | Fitness improves over controlled runs; crossover variants compared | ❌ Not Started |
| Phase 4 | Analytics + checkpoints + determinism checklist | Experiments logged, repeatable, tagged by reproducibility tier | ❌ Not Started |
| Phase 5 | Related-work write-up + statistical protocol implementation | Comparison pipeline ready before any headline experiment runs | ❌ Not Started |
| Phase 6 | Generalization experiments + Experiment D (fitness weighting) | Unseen layouts and fitness-weighting sensitivity both evaluated | ❌ Not Started |
| Phase 7 | Predator/prey + reproduction + hall-of-fame evaluation | Multi-agent ecosystem stable; co-evolution measured against frozen checkpoints | ❌ Not Started |
| Phase 8 | NEAT-style extension (Appendix C) + packaging + final report | Topology evolution works on an isolated benchmark, matching the Appendix C spec | ❌ Not Started |

See [Plan](PLAN.md) for the first-steps plan.

## Key Design Rules

1. **Batched inference is a Phase 1.5 exit criterion, not a late-stage fix.** ~250 per-agent PyTorch calls per step become ~3 batched calls.
2. **Fitness is calibrated before it is trusted.** Each component's natural scale is measured with a random-policy and a stand-still probe; only then are coefficients locked.
3. **Scope honesty:** Phases 1-6 are fixed-topology neuroevolution. Phase 7 is a real, fully specified NEAT implementation (Appendix C), not a label applied to the whole system.
4. **Crossover is compared, not assumed.** Blend/arithmetic is the v2.0 default; uniform and mutation-only are comparison arms (Experiment E), because of the competing-conventions problem in real-valued weight vectors.
5. **Reproducibility tiers are separate.** CPU-deterministic and GPU runs are never compared as bit-identical.

## Experiments

- **A - Mutation rate:** 0.01 / 0.05 / 0.10 / 0.20
- **B - Population size:** 50 / 100 / 250 / 500
- **NC - Network capacity:** 12-8-3 / 12-16-8-3 / 12-32-16-3
- **D - Fitness weighting (NEW):** equal / resource-weighted / exploration-weighted / survival-weighted
- **E - Crossover method (NEW):** blend / uniform / mutation-only

Statistical comparison uses Mann-Whitney U on final fitness across 10 seeds, bootstrap CI on the median, and Holm-Bonferroni correction (Appendix D).

## Current Implementation Status

As of September 2026, the following components have been implemented:

### Configuration & Infrastructure (Phase 0 - Complete)
- ✅ Project scaffolding with `pyproject.toml`, `.devcontainer`, `pytest.ini`
- ✅ Configuration validation using Pydantic models (`simulation/world_config.py`)
- ✅ Baseline and experiment configuration files (`configs/baseline.json`, `configs/mutation_rate.json`)
- ✅ Empty package structure with `__init__.py` files for all modules

### Core Simulation Components (Phase 1 - In Progress)
- ✅ `World` class (`simulation/environment/world.py`) - manages food, obstacles, boundaries
- ✅ Physics mathematics (`simulation/physics/`) - Vector2D, AABB, SpatialHash implementations
- ✅ `Organism` class (`agents/organism.py`) - basic life cycle, energy management, placeholder neural/sensory systems
- ✅ `Renderer` class (`visualization/renderer.py`) - PyGame-based world rendering
- ✅ GUI entry point (`app/run.py`) - basic simulation loop with placeholder agent motion
- ✅ Headless entry point (`experiments/run.py`) - experiment framework skeleton
- ✅ Unit tests for physics and environment components

### Pending Implementation
- ❌ Neural network system (`neural/network.py`, `neural/batched_inference.py`, etc.)
- ❌ Sensor system (`agents/sensors.py`)
- ❌ Energy-balance equation (`agents/energy.py`)
- ❌ Genetic algorithm (`evolution/genetic_algorithm.py`, `selection.py`, `crossover.py`, `mutation.py`)
- ❌ Experiment logging and analytics (`analytics/experiment_logger.py`, `statistics.py`, etc.)
- ❌ Proper simulation engine (`simulation/engine.py`) - currently uses placeholder motion
- ❌ Checkpointing system (`neural/checkpoint.py`)
- ❌ Determinism utilities (`simulation/determinism.py`)
- ❌ Hall of fame tracking (`analytics/hall_of_fame.py`)
- ❌ NEAT topology evolution components (`evolution/neat/`)
- ❌ Web dashboard (`webdash/` - optional, Phase 8)

## Next Steps

To advance the simulation toward the Phase 1 exit criterion ("Agents move and collide reliably"), the following work is needed:

1. Implement proper neural network controllers in `neural/network.py`
2. Implement sensory ray-casting in `agents/sensors.py`
3. Connect neural outputs to physics inputs in the simulation engine
4. Implement the energy-balance equation in `agents/energy.py`
5. Replace placeholder motion in `app/run.py` with proper physics integration
6. Implement the genetic algorithm in the `evolution/` modules
7. Add experiment logging and analytics capabilities

Once these components are implemented and integrated, agents will be able to process sensory information through neural networks, convert that to physical actions, and exhibit emergent behaviors guided by evolutionary selection pressures.