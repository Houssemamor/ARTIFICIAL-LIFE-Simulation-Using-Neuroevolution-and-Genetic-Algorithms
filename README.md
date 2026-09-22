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

- Python 3.12+ (primary)
- PyGame-ce (simulation engine / renderer) — required on Python 3.12+
- PyTorch 2.14+ (batched population inference)
- Pydantic 2.12+ (config validation)
- Optional: FastAPI + React web dashboard (presentation layer only, not coupled to core)

## Installation

```bash
# 1. Clone and create virtual environment
git clone <repo-url>
cd ARTIFICIAL-LIFE-Simulation-Using-Neuroevolution-and-Genetic-Algorithms
python -m venv .venv

# 2. Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install pinned dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

**Pinned dependencies (from `requirements.txt`):**
- `torch==2.14.0`
- `pygame-ce==2.5.8` (required on Python 3.14; plain `pygame` has no wheel)
- `numpy==2.5.3`
- `pydantic==2.12.5`, `pydantic-settings==2.14.2`, `pydantic-core==2.41.5`
- `pytest==8.4.0`, `pytest-asyncio==1.4.0`, `pytest-cov==7.1.0`
- `flake8==7.3.0`, `black==23.12.1`, `isort==5.13.2`, `mypy==1.8.0`

**Note:** On Python 3.12+, plain `pygame` fails to build from source. This project uses `pygame-ce` (drop-in replacement) which provides Python 3.12+ wheels.

## Running

```bash
# Activate venv first (see Installation)

# Phase 1: Placeholder random-walk agents (no neural controller)
python app/run.py --config configs/baseline.json --agent-mode placeholder

# Phase 2+: Neural controller (batched inference)
python app/run.py --config configs/baseline.json --agent-mode neural

# Headless experiment (Phase 3+)
python -m experiments.run --config configs/baseline.json --mode headless --device cpu
```

**Agent modes:**
- `--agent-mode placeholder`: Phase 1 random-walk motion (validates physics/render loop)
- `--agent-mode neural`: Phase 2+ batched NN inference (12-32-16-3 MLP)

## Running Tests

```bash
# Full suite (103 tests)
pytest tests/ -q

# Specific test groups
pytest tests/unit/ -q
pytest tests/integration/ -q
pytest tests/performance/ -q
```

## Development Plan (summary)

| Phase | Work | Exit criterion | Status |
|-------|------|----------------|---------|
| Phase 0 | Project setup & environment | Installable project, pinned deps, green pytest | ✅ Complete |
| Phase 1 | Environment + renderer + basic physics | Agents move and collide reliably | ✅ Complete |
| Phase 1.5 (NEW) | Prototype batched population inference; benchmark steps/sec | Measured steps/sec at 250 agents feeds the compute budget | ✅ Complete |
| Phase 2 | Sensors + fixed neural controller, batched from the start | Agents react to observations through batched NN outputs | ✅ Complete |
| Phase 3 | Fitness calibration/normalization + GA + crossover comparison arm | Fitness improves over controlled runs; crossover variants compared | ✅ Complete |
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
- ✅ Pinned dependencies in `requirements.txt` and `pyproject.toml`
- ✅ `pip install -e .` works on Python 3.14

### Core Simulation Components (Phase 1 - Complete)
- ✅ `World` class (`simulation/environment.py`) - manages food, obstacles, boundaries
- ✅ Physics mathematics (`simulation/physics/`) - Vector2D, AABB, SpatialHash implementations
- ✅ `Organism` class (`agents/organism.py`) - life cycle, energy management, collision tracking
- ✅ `Renderer` class (`visualization/renderer.py`) - PyGame-based world rendering
- ✅ GUI entry point (`app/run.py`) - dual-mode: `--agent-mode placeholder` (Phase 1) / `--agent-mode neural` (Phase 2+)
- ✅ Headless entry point (`experiments/run.py`) - experiment framework
- ✅ Unit tests for physics and environment components

### Batched Inference (Phase 1.5 - Complete)
- ✅ `neural/batched_inference.py` - population-wide forward pass (`batched_forward`)
- ✅ `neural/network.py` - 12-32-16-3 MLP topology
- ✅ `neural/genome.py` - pack/unpack genome ↔ weights with round-trip tests
- ✅ Benchmark: 10,762 steps/sec at 250 agents (358× real-time FPS target)

### Sensors + Fixed Neural Controller (Phase 2 - Complete)
- ✅ `agents/sensors.py` - 7-ray casting (-90..+90°), 12-dim observation encoding
- ✅ `simulation/engine/__init__.py` - observe → batched_forward → act pipeline
- ✅ Batched-vs-per-agent equivalence test (≤1e-5 tolerance)

### Fitness Calibration + Genetic Algorithm (Phase 3 - Complete)
- ✅ `agents/energy.py` - energy balance equation (metabolism + action costs + food gain - collision penalty)
- ✅ `analytics/calibration.py` - random/stand-still calibration → `configs/calibration.json`
- ✅ `evolution/` - tournament selection + elitism, 3 crossover methods (blend/uniform/none), Gaussian mutation
- ✅ `evolution/genetic_algorithm.py` - `run_generation()` orchestration + normalized fitness
- ✅ 43 new unit/integration tests (energy, fitness, crossover, one-generation)

## Pending Implementation (Phase 4+)
- ❌ Experiment logging pipeline (`analytics/experiment_logger.py`)
- ❌ Determinism utilities (`simulation/determinism.py`)
- ❌ Checkpointing system (`neural/checkpoint.py`)
- ❌ Hall of fame tracking (`analytics/hall_of_fame.py`)
- ❌ NEAT topology evolution (`evolution/neat/`)
- ❌ Statistical comparison pipeline
- ❌ Web dashboard (`webdash/` - optional, Phase 8)

## Next Steps

Phase 4 begins with experiment logging, determinism enforcement, and checkpoint save/restore. See [Plan](PLAN.md) for the detailed first-steps plan.

Once these components are implemented and integrated, agents will be able to process sensory information through neural networks, convert that to physical actions, and exhibit emergent behaviors guided by evolutionary selection pressures.