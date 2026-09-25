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

For exact-environment reproducibility (all transitive dependencies pinned),
use `requirements.lock.txt` instead of `requirements.txt`.

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
# Full suite (251 tests)
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
| Phase 4 | Analytics + checkpoints + determinism checklist | Experiments logged, repeatable, tagged by reproducibility tier | ✅ Complete |
| Phase 4.5 | Related-work write-up + statistical protocol implementation | Comparison pipeline ready before any headline experiment runs | ✅ Complete |
| Phase 5 | Generalization experiments + Experiment D (fitness weighting) | Unseen layouts and fitness-weighting sensitivity both evaluated | ✅ Complete |
| Phase 6 | Predator/prey + reproduction + hall-of-fame evaluation | Multi-agent ecosystem stable; co-evolution measured against frozen checkpoints | ☑ Complete with caveat² |
| Phase 7 | NEAT-style extension (Appendix C) | Topology evolution works on an isolated benchmark, matching the Appendix C spec | ☑ Complete with caveat³ |
| Phase 8 | Packaging, final report, dashboard polish, presentation | Packaged, tested, installable application + complete presentation package | ☑ Complete |

See [Plan](PLAN.md) for the first-steps plan.

¹ Crossover variants were compared (Experiment E, `docs/crossover_results.md`)
and the fitness-over-generations plot was produced
(`experiments/EXP-E/fitness_curve.svg`). Under the final dynamics the
GA improves fitness (+0.28 to +0.50 mean over 12
generations), closing the Phase 3 "fitness improves" expectation; the
crossover methods themselves remain statistically indistinguishable
(blend vs uniform +0.001, p = 0.92).

² The ecosystem is viable on an unselected three-seed sample under fully
deterministic runs, but with fixed (non-evolved) controllers 7 of 10
random seeds collapse within 10 days - five of the seven failures are
predator extinctions, two are prey - after the newborn-energy fix
removed an inherited-dynamics subsidy (single-regime 10-seed
measurement under the final dynamics).
The stability test asserts viability on seeds 1-3, a negative test
proves the fixture can fail, and the full analysis lives in
`docs/coevolution_notes.md`.

³ The NEAT implementation matches the Appendix C operators and formula
(verified by unit tests), and topology grows on the benchmark (mean complexity
51 → 102 over 30 generations). The evolutionary run keeps a single species:
the energy-limited landscape is bimodal (a few foragers near fitness 2.0, most
agents starving near 0.3), which still yields one compatibility cluster at the
threshold. Speciation machinery is proven on constructed diverse populations.
Full analysis in `docs/neat_extension_report.md`.

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

### Analytics, Checkpoints, Determinism (Phase 4 - Complete)
- ✅ `analytics/experiment_logger.py` - experiment folder structure (config.json, metadata.json, generation_metrics.csv, agent_metrics.csv, best_genome.json, checkpoints/, stats_summary.json)
- ✅ `simulation/determinism.py` - `DeterminismConfig`, `set_deterministic_seeds()` (separate torch/numpy/random seeds, deterministic algorithms, thread pinning)
- ✅ `neural/checkpoint.py` - genome + architecture save/restore with bit-for-bit verification
- ✅ `tests/integration/test_seeded_rerun.py` - byte-for-bit reproducibility across runs
- ✅ `tests/integration/test_checkpoint_restore.py` - checkpoint save/restore verification

### Statistics + Comparison Pipeline (Phase 4.5 - Complete)
- ✅ `docs/related_work.md` - positioning against Sims (1994), Stanley & Miikkulainen NEAT (2002), AVIDA
- ✅ `analytics/statistics.py` - `mann_whitney_u()`, `wilcoxon_signed_rank()`, `bootstrap_ci()`, `holm_bonferroni()`, effect sizes, `compare_conditions()` (unpaired) and `compare_conditions_paired()` (seed-blocked designs)
- ✅ `tests/unit/test_statistics.py` - 40 tests cross-checking against scipy/statsmodels

### Generalization + Experiment D (Phase 5 - Complete)
- ✅ `configs/generalization_train.json` / `generalization_test.json` - layouts A1-A3 (train) / B1-B3 (held-out), varying placement seed only
- ✅ `configs/fitness_weighting_d1.json` ... `d4.json` - equal / resource / exploration / survival-weighted conditions
- ✅ `experiments/experiment_runner.py` - reusable train / freeze / balanced-evaluation primitives; fitness normalized against measured scales from `configs/calibration.json` (never hand-picked constants)
- ✅ `experiments/run_generalization.py` - train A1-A3, evaluate frozen genomes on B1-B3, paired Wilcoxon + bootstrap CIs + matched-pairs effect size
- ✅ `experiments/run_experiment_d.py` - D1-D4 x 10 seeds through the paired comparison pipeline (Wilcoxon + Holm-Bonferroni)
- ✅ `experiments/run_crossover_experiment.py` - Experiment E (blend/uniform/mutation-only) through the same paired pipeline; writes the fitness-over-generations SVG
- ✅ `docs/generalization_results.md` - consistent-direction generalization penalty: train-higher by +0.047 (p=0.1055, r=+0.60, 7/10 seeds, bootstrap CI [+0.002, +0.089]) under the energy-limited dynamics
- ✅ `docs/experiment_d_results.md` - equal weighting leads every specialization by +0.07 to +0.12 (no pair survives Holm at n=10); the balanced weights are already near-optimal
- ✅ `docs/crossover_results.md` - no pairs survive Holm; blend and uniform are indistinguishable (+0.001, p=0.92) and both +0.08 over mutation-only, while the fitness curve rises +0.28 to +0.50 over 12 generations (Phase 3 expectation decisively met)
- ✅ `tests/integration/test_experiment_runners.py` - runner smoke tests (8 tests)
- ✅ `WorldConfig.layout_seed` + `EvolutionConfig.evaluation_steps` schema extensions
- ✅ Determinism enforced at every entry point (GUI, headless, all experiment runners) via `set_deterministic_seeds`

### Predator/Prey + Hall-of-Fame (Phase 6 - Complete with caveat)
- ✅ `Organism.role` (`prey` / `predator`), validated; predators reuse the eat-gate as the capture action, captures score through the existing fitness `food` component
- ✅ `evolution/reproduction.py` - plan's eligibility rule (age / energy / fitness) + asexual clone+mutate offspring + parent energy cost + per-role carrying capacities
- ✅ Food regrowth (`WorldConfig.food_regrowth_per_step`, default 0 = legacy) - finite food made any ecosystem impossible; consumption radius raised 5 -> 12 px per the Phase 5 recommendation, `configs/calibration.json` regenerated
- ✅ `evolution/genetic_algorithm.run_coevolution_generation` - both roles evaluated in one shared world, standard GA machinery per role
- ✅ `analytics/hall_of_fame.py` - frozen best-per-role archive + duel harness (`run_duel`, `win_rate_against_archive`)
- ✅ `experiments/run_coevolution.py` - co-evolution mode (GA + HOF snapshots + frozen-opponent evaluations) and ecosystem mode (long-horizon stability harness)
- ✅ `visualization/dashboard_advanced.py` + F9 GUI toggle - live ecosystem counters + hall-of-fame delta; predators drawn red-family
- ✅ `docs/coevolution_notes.md` - the arms-race caveat; honest stability analysis: 7 of 10 random seeds collapse within 10 days (5 predator extinctions, 2 prey, after the newborn-energy fix), the test asserts viability on unselected seeds 1-3, and a negative test proves the fixture can fail
- ✅ `tests/unit/test_predation.py` (9), `tests/unit/test_reproduction.py` (8), `tests/unit/test_hall_of_fame.py` (7), `tests/integration/test_coevolution.py` (4), `tests/integration/test_predator_prey_stability.py` (3, incl. seed-reproducibility regression)
- ⚠️ Caveat: the ecosystem is viable but fragile with fixed (non-evolved) controllers - see `docs/coevolution_notes.md` before interpreting ecosystem metrics; the ecosystem metabolic clock (0.25/step) is the first lever if robustness matters

### NEAT Topology Evolution (Phase 7 - Complete with caveat)
- ✅ `evolution/neat/` package - `genome.py` (node + connection genes with innovations), `innovation.py` (global counter with per-generation reuse), `mutation.py` (`add_node` with weight-1.0 split init, `add_connection`, Gaussian weight mutation with re-enable), `crossover.py` (innovation-aligned: matching random from either parent, disjoint/excess from the fitter), `compatibility.py` (δ = c1·E/N + c2·D/N + c3·Ŵ, Appendix C constants), `speciation.py` (assignment, fitness sharing, stagnation removal with champion protection), `algorithm.py` (full generation loop)
- ✅ `neural/sparse_inference.py` - the batching question resolved: population-wide depth-layered padded/masked batch, **2.9x the per-agent reference** (249 vs 85 steps/s at 100 genomes), verified equivalent to per-agent on minimal, variable-depth, and orphaned-node genomes
- ✅ `experiments/run_neat.py` + `configs/neat_food_seeking.json` - 30-generation food-seeking benchmark; topology grows (mean complexity 51 → 102, max 51 → 117; 571 innovations); complexity SVG + summary JSON
- ✅ `docs/neat_extension_report.md` - batching decision with measured numbers, benchmark results, and the Appendix C provenance caveat
- ✅ `tests/unit/test_neat_primitives.py` (20), `tests/unit/test_sparse_inference.py` (6), `tests/integration/test_neat_speciation.py` (3)
- ⚠️ Caveat: the evolutionary run keeps a single species - the mechanism is unit-proven, but the landscape is bimodal (few foragers near fitness 2.0, most near 0.3), which still yields one compatibility cluster (`docs/neat_extension_report.md`)

### Packaging, Final Report, Dashboards, Presentation (Phase 8 - Complete)
- ✅ `docs/final_report.md` - the capstone, assembled from the frozen artifacts; every Phase 5-7 experiment re-run under the final dynamics in this phase
- ✅ Dashboard reconciliation - `config.phase_tier` selects MVP vs. Advanced (the MVP build can never render a Phase 6/7-only metric); `tests/unit/test_dashboard_tiers.py` enforces it
- ✅ Section 12 build-out (Phase 9): control bar (Pause/x1/x10/Save/Reset) + best-agent inspector (vitals, seven-ray overlay, controller graph with live activations) - `visualization/control_bar.py`, `visualization/best_agent_inspector.py`
- ✅ Web dashboard - FastAPI + Bootstrap/vanilla JS read-only viewer over the committed artifacts (`webdash/`; run `uvicorn webdash.backend.main:app`)
- ✅ Packaging verified - editable install, both entry points, 251-test suite, lint gate clean
- ✅ `docs/presentation_deck.pptx` (14 slides, measured numbers, speaker notes) + `docs/demo_script.md` (rehearsable, with the author-side video checklist) + `docs/slide_outline.md`
- ✅ Energy-limited dynamics (Phase 9): `metabolic_base_cost` config-tunable (0.7/step in experiments, legacy pacing in the GUI), food regrowth applied in every evaluation loop, live exploration component, newborn energy inheritance; every experiment re-run - +0.28 to +0.50 fitness gain, consistent-direction generalization penalty, equal-weighting leads its specializations

## Next Steps

All eight plan phases plus the Phase 9 dynamics fix are complete. The
highest-leverage next action, consistent across every result document:
re-run the solo experiments at n = 20-30 seeds to resolve the three
near-threshold effects (generalization +0.047, equal-vs-survival
weighting +0.12, crossover-vs-mutation-only +0.08), and sweep
`metabolic_base_cost` (0.4 / 0.7 / 1.0) since it sets the
survival-vs-forging split every result depends on. The ecosystem's
7/10 collapse rate points at its own clock (0.25/step) as the next
robustness lever, and a graded food landscape is what NEAT speciation
needs. See [Plan](PLAN.md) and `docs/final_report.md`.

Agents process sensory information through neural networks, convert
that to physical actions, and exhibit emergent behaviors guided by
evolutionary selection pressures - with every number in this README
traceable to a committed artifact.