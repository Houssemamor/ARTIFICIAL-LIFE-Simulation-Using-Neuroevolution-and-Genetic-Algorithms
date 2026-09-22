# Artificial Life Neuroevolution Simulation — Implementation Plan

**Companion to:** *Artificial Life Simulation — Revised Design Document v2.0*
**Purpose of this file:** a build-order-accurate, file-by-file, step-by-step execution plan for turning the v2.0 design into working software. Every phase below maps 1:1 to the phase of the same name/number in the design document's Section 21 (Development Plan), so exit criteria match exactly.

**How to use this plan:** work top to bottom. Do not start a phase until the previous phase's Deliverables checklist is fully checked off — the phases are ordered specifically so that expensive mistakes (a slow inference path, an under-specified fitness function, a fudged reproducibility claim) get caught while they're still cheap to fix.

---

## Table of Contents

0. [Repository Layout (Final State)](#0-repository-layout-final-state)
1. [Phase 0 — Project Setup & Environment](#phase-0--project-setup--environment)
2. [Phase 1 — Environment, Renderer, Basic Physics](#phase-1--environment-renderer-basic-physics)
3. [Phase 1.5 — Batched Population Inference Prototype & Benchmark](#phase-15--batched-population-inference-prototype--benchmark)
4. [Phase 2 — Sensors + Fixed Neural Controller](#phase-2--sensors--fixed-neural-controller-batched-from-the-start)
5. [Phase 3 — Fitness Calibration + Genetic Algorithm](#phase-3--fitness-calibration-normalization-and-the-genetic-algorithm)
6. [Phase 4 — Analytics, Checkpoints, Determinism](#phase-4--analytics-checkpoints-determinism)
7. [Phase 4.5 — Related Work + Statistical Protocol](#phase-45--related-work--statistical-protocol-implementation)
8. [Phase 5 — Generalization + Fitness-Weighting Experiment](#phase-5--generalization-experiments--experiment-d-fitness-weighting)
9. [Phase 6 — Predator/Prey + Reproduction + Hall of Fame](#phase-6--predatorprey-reproduction-and-hall-of-fame-evaluation)
10. [Phase 7 — NEAT-Style Extension](#phase-7--neat-style-extension-appendix-c)
11. [Phase 8 — Packaging, Final Report, Presentation](#phase-8--packaging-final-report-dashboard-polish-presentation)
12. [Dashboard — Full Specification](#12-dashboard--full-specification)
13. [Final Presentation of All the Work](#13-final-presentation-of-all-the-work)
14. [Master File Checklist](#14-master-file-checklist)

---

## 0. Repository Layout (Final State)

This is the complete file tree as it will exist after Phase 8. Every phase section below tells you exactly which of these paths it creates or modifies, so you always know what "done" looks like for a given file.

```
artificial-life-sim/
├── README.md
├── pyproject.toml
├── .gitignore
├── pytest.ini
│
├── configs/
│   ├── baseline.json
│   ├── calibration.json
│   ├── mutation_rate_a1.json ... a4.json
│   ├── population_size_b1.json ... b4.json
│   ├── network_capacity_nc1.json ... nc3.json
│   ├── fitness_weighting_d1.json ... d4.json
│   ├── crossover_method_e1.json ... e3.json
│   ├── generalization_train.json
│   └── generalization_test.json
│
├── app/
│   └── run.py                        # GUI entrypoint
│
├── experiments/
│   ├── run.py                        # headless experiment runner entrypoint
│   ├── run_generalization.py
│   ├── run_experiment_d.py
│   └── run_experiment_e.py
│
├── simulation/
│   ├── __init__.py
│   ├── engine.py                     # main loop, GUI/headless orchestration
│   ├── environment.py                # World, Food, Obstacle
│   ├── physics.py                    # fixed-timestep physics, collisions
│   ├── world_config.py               # config schema/validation
│   └── determinism.py                # seed + reproducibility-tier management
│
├── agents/
│   ├── __init__.py
│   ├── organism.py                   # agent state, lifecycle, role (prey/predator)
│   ├── sensors.py                    # ray casting, observation encoding
│   └── energy.py                     # energy-balance equation
│
├── neural/
│   ├── __init__.py
│   ├── network.py                    # fixed 12-32-16-3 architecture definition
│   ├── batched_inference.py          # population-level batched forward pass
│   ├── genome.py                     # fixed-length genome pack/unpack
│   ├── checkpoint.py                 # save/restore genome + architecture
│   └── sparse_inference.py           # Phase 7: batching under variable topology
│
├── evolution/
│   ├── __init__.py
│   ├── genetic_algorithm.py          # run_generation orchestration
│   ├── selection.py                  # tournament + elitism
│   ├── crossover.py                  # blend / uniform / none
│   ├── mutation.py                   # Gaussian mutation + clipping
│   ├── reproduction.py               # explicit reproduction (Phase 6)
│   └── neat/
│       ├── __init__.py
│       ├── genome.py                 # variable-topology genome
│       ├── innovation.py             # innovation-number tracking
│       ├── mutation.py               # add_connection / add_node
│       ├── crossover.py              # innovation-aligned crossover
│       ├── compatibility.py          # δ compatibility-distance formula
│       └── speciation.py             # species assignment, fitness sharing
│
├── analytics/
│   ├── __init__.py
│   ├── calibration.py                # fitness reference-scale calibration
│   ├── experiment_logger.py          # experiments/EXP-XXX/ writer
│   ├── statistics.py                 # Mann-Whitney U, bootstrap CI, Holm-Bonferroni
│   ├── compare_conditions.py         # pairwise comparison orchestration
│   ├── hall_of_fame.py               # frozen-opponent archive (Phase 6)
│   └── reports.py                    # plot + summary generation
│
├── visualization/
│   ├── __init__.py
│   ├── renderer.py                   # PyGame world renderer
│   ├── dashboard_mvp.py              # Phases 1-5 dashboard
│   ├── dashboard_advanced.py         # Phase 6+ dashboard
│   └── best_agent_inspector.py       # detail panel
│
├── webdash/                          # optional, Phase 8
│   ├── backend/
│   │   ├── main.py                   # FastAPI app
│   │   └── routes.py
│   └── frontend/
│       ├── src/
│       │   ├── pages/ExperimentList.jsx
│       │   ├── pages/ExperimentDetail.jsx
│       │   ├── pages/CompareConditions.jsx
│       │   └── pages/BestAgentViewer.jsx
│       └── package.json
│
├── tests/
│   ├── unit/
│   │   ├── test_physics.py
│   │   ├── test_environment.py
│   │   ├── test_sensors.py
│   │   ├── test_genome.py
│   │   ├── test_batched_vs_per_agent.py
│   │   ├── test_energy_balance.py
│   │   ├── test_fitness_normalization.py
│   │   ├── test_crossover_all_methods.py
│   │   ├── test_statistics.py
│   │   ├── test_innovation_numbers.py
│   │   ├── test_compatibility_distance.py
│   │   └── test_neat_crossover.py
│   ├── integration/
│   │   ├── test_reset.py
│   │   ├── test_one_generation.py
│   │   ├── test_seeded_rerun.py
│   │   ├── test_checkpoint_restore.py
│   │   ├── test_predator_prey_stability.py
│   │   └── test_speciation_stability.py
│   └── performance/
│       └── test_batched_inference_benchmark.py
│
├── experiments_data/                 # runtime output, gitignored
│   └── EXP-001/ ... EXP-NNN/
│
└── docs/
    ├── related_work.md
    ├── phase1_5_benchmark_report.md
    ├── generalization_results.md
    ├── experiment_d_results.md
    ├── experiment_e_results.md
    ├── coevolution_notes.md
    ├── neat_extension_report.md
    ├── final_report.md
    └── demo_script.md
```

---

## Phase 0 — Project Setup & Environment

**Goal:** a working, reproducible Python environment and a structured, empty-but-correct repository before any simulation code is written.

**New files**

| Path | Purpose |
|---|---|
| `pyproject.toml` | Pins `pygame`, `torch`, `numpy`, `scipy`, `pytest`, `matplotlib`, `pydantic`; optional `fastapi`+`uvicorn` for Phase 8 |
| `.gitignore` | Excludes `experiments_data/`, `__pycache__/`, `*.pt` checkpoints |
| `pytest.ini` | Points pytest at `tests/unit`, `tests/integration`, `tests/performance` |
| `README.md` | Project overview, quick-start commands (stubbed until real entrypoints exist) |
| `configs/baseline.json` | Appendix-A-default parameters (population 250, world 1200×700, etc.) |
| `simulation/__init__.py`, `agents/__init__.py`, `neural/__init__.py`, `evolution/__init__.py`, `analytics/__init__.py`, `visualization/__init__.py` | Empty package markers |

**Steps**

1. Initialize the git repository; add `.gitignore` first so nothing runtime-generated gets committed by accident.
2. Create a virtual environment and pin all dependency versions in `pyproject.toml` — pin, don't range, since Section 15.4/16.3's determinism claims depend on identical library versions across runs.
3. Scaffold every package directory listed in the Section 0 tree with an empty `__init__.py`.
4. Write `configs/baseline.json` using Appendix A's defaults verbatim, including the newer `crossover_method`, `fitness_weights`, `seed` (as an object, not a single int), `reproducibility_tier`, and `device` fields.
5. Configure `pytest.ini` and confirm `pytest` runs (with zero tests) cleanly.
6. Write a `README.md` stub describing the two eventual entry points (`app/run.py --mode gui`, `experiments/run.py --mode headless`) even before they exist, so the command surface is decided up front.
7. Commit: `"Phase 0: project scaffold"`.

**Deliverables**

- [ ] Installable, empty project skeleton matching the Section 0 tree
- [ ] Pinned dependency lockfile
- [ ] `configs/baseline.json` containing every field the v2.0 design document's Section 18.2 configuration example specifies
- [ ] A green (zero-test) `pytest` run, proving the test scaffold itself works

---

## Phase 1 — Environment, Renderer, Basic Physics

**Goal:** agents move and collide reliably inside a rendered 2D world. No neural network yet — motion is a placeholder random walk, purely to prove the physics/render loop is solid before anything is built on top of it.

**New / modified files**

| Path | Purpose |
|---|---|
| `simulation/environment.py` | `World`, `Food`, `Obstacle` classes (design doc §10.1–10.3) |
| `simulation/physics.py` | Fixed-timestep-with-accumulator loop; boundary clamping + collision penalty |
| `simulation/world_config.py` | Pydantic model validating `configs/*.json` against the schema |
| `agents/organism.py` | Agent state dataclass (§11.1 fields): id, position, velocity, heading, energy, health, age, fitness, genome placeholder, alive |
| `visualization/renderer.py` | PyGame renderer: world bounds, food, obstacles, agents |
| `app/run.py` | GUI entrypoint: load config → build World → render loop → pause/quit controls |
| `tests/unit/test_physics.py` | Boundary clamping, collision penalty |
| `tests/unit/test_environment.py` | World construction from config |
| `tests/integration/test_reset.py` | Reset the environment N times; verify entity counts match config every time |

**Steps**

1. Implement `World`, `Food`, `Obstacle` exactly per the design doc's field tables.
2. Implement the fixed-timestep-with-accumulator pattern so physics steps are decoupled from render FPS (this is what makes headless fast-forward correct later).
3. Implement boundary clamping plus a small collision penalty on contact (the recommended baseline — avoids the instability of velocity reflection).
4. Implement placeholder agent motion: random steering, constant low acceleration — just enough to prove collisions and rendering work.
5. Build `visualization/renderer.py`: draw world bounds, food (green circles), obstacles (gray polygons), agents (triangles oriented by heading).
6. Wire `app/run.py`: load `configs/baseline.json` → construct `World` → run the render loop → keyboard controls for pause/quit.
7. Write and pass `test_physics.py`, `test_environment.py`, `test_reset.py`.
8. Manual check: run the GUI with 250 placeholder agents for 60 seconds with no crash, no agents escaping the world bounds.

**Exit criterion:** agents move and collide reliably.

### Deliverables — Phase 1

- [ ] Running GUI app showing a live, bounded 2D world with food, obstacles, and moving placeholder agents
- [ ] Fixed-timestep physics fully decoupled from render FPS
- [ ] Passing unit tests for boundary collision and world reset
- [ ] `configs/baseline.json` validated end-to-end for the first time via `world_config.py`

---

## Phase 1.5 — Batched Population Inference Prototype & Benchmark

**Goal:** prove that population-scale neural inference is fast enough to hit the real-time NFR *before* any real sensor or GA code is written on top of it. This is the highest-leverage phase in the entire plan — v1.0 discovered this problem in a Phase 8 risk note; v2.0 resolves it here, first.

**New files**

| Path | Purpose |
|---|---|
| `neural/network.py` | 12-32-16-3 architecture as pure tensor shapes, not yet wired to real sensors |
| `neural/batched_inference.py` | Stacked-weights + `torch.bmm` batched forward pass |
| `tests/performance/test_batched_inference_benchmark.py` | Benchmarks batched vs. naive per-agent inference at population 50/100/250/500 |
| `docs/phase1_5_benchmark_report.md` | Records measured steps/sec — a hard input to Phase 5's compute-budget math |

**Steps**

1. Implement `stack_population_weights(genomes) -> (W1,b1,W2,b2,W3,b3)`, each shaped `(N, in, out)` / `(N, out)`.
2. Implement `batched_forward(observations, weights)`: one `bmm` per layer, ReLU between hidden layers, `tanh`/`sigmoid` split on the 3-dim output — exactly the design document's §12.3 pseudocode.
3. Implement a naive per-agent `nn.Module.forward()`-in-a-loop baseline **purely for the benchmark comparison**; quarantine it clearly so it can never accidentally ship past this phase.
4. Write the benchmark test: random dummy weights and observations, time both paths at each population size, assert the batched path meets the configured real-time FPS target at population 250.
5. If the target isn't met: profile first (weight-stacking overhead? device placement? per-step vs. per-generation stacking?) and optimize before moving on — do not proceed to Phase 2 on an unverified performance assumption.
6. Write `docs/phase1_5_benchmark_report.md` with the actual measured steps/sec numbers at all four population sizes.
7. Delete or clearly quarantine the naive per-agent code path.

**Exit criterion:** measured steps/sec at 250 agents feeds the compute-budget estimate used from Phase 5 onward.

### Deliverables — Phase 1.5

- [x] `neural/batched_inference.py`, proven correct and fast, before any real sensor/GA code depends on it
- [x] A written benchmark report with concrete steps/sec numbers for 50/100/250/500 agents
- [x] A performance test that will catch any future regression in inference throughput
- [x] Confirmed evidence the real-time NFR is achievable, before further code is built on a potentially slow foundation

---

## Phase 2 — Sensors + Fixed Neural Controller (Batched From the Start)

**Goal:** agents react to real sensor observations through the Phase 1.5 batched network, replacing the Phase 1 random-motion placeholder.

**New / modified files**

| Path | Purpose |
|---|---|
| `agents/sensors.py` | 7-ray casting; normalized observation encoding (food/threat/obstacle distance+bearing, energy, speed) |
| `neural/genome.py` | Genome flatten/reshape (`Genome = [W1,b1,W2,b2,W3,b3]`) |
| `simulation/engine.py` | Main loop: `batch_observe → batched_forward → apply_actions → physics.update` |
| `agents/organism.py` (modified) | Wires decoded actions (steering/accel/eat_signal) into physics |
| `tests/unit/test_sensors.py` | Ray intersection and normalization correctness |
| `tests/unit/test_genome.py` | Pack/unpack round-trip correctness |
| `tests/unit/test_batched_vs_per_agent.py` | **Critical:** batched output matches an independent per-agent reference within floating-point tolerance, for a random sample of genomes |

**Steps**

1. Implement 7-ray casting against food/obstacles/other agents; return nearest-object distance + bearing per ray.
2. Implement the fixed observation encoding table (food distance `[0,1]`, food bearing `[-1,1]`, threat distance/bearing, obstacle distance, energy `[0,1]`, speed `[0,1]`).
3. Implement genome flatten/reshape utilities matching the fixed layout.
4. Replace Phase 1's random-motion stub in `simulation/engine.py` with the real loop: observe → `batched_forward` → decode `steering = tanh(out[0])`, `accel = sigmoid(out[1])`, `eat_signal = sigmoid(out[2])` → apply to physics.
5. Write the batched-vs-per-agent equivalence test — this is the correctness gate that validates Phase 1.5's speed optimization didn't change behavior.
6. Visual sanity check in the GUI: agents should visibly turn toward food and away from obstacles even with random (untrained) weights reacting sensibly to gradients in observation, before trusting the pipeline further.

**Exit criterion:** agents react to observations through batched NN outputs.

### Deliverables — Phase 2

- [x] Working ray-based sensor system with a fixed, documented observation encoding
- [x] Genome pack/unpack utilities with passing round-trip tests
- [x] Fully wired observe → batched-inference → act loop, replacing the Phase 1 placeholder
- [x] Passing batched-vs-per-agent equivalence test

---

## Phase 3 — Fitness Calibration, Normalization, and the Genetic Algorithm

**Goal:** fitness improves over controlled runs, using calibrated/normalized components; crossover methods are compared rather than assumed.

**New / modified files**

| Path | Purpose |
|---|---|
| `agents/energy.py` | Energy-balance equation: metabolic cost, acceleration/steering cost, eat-event gain, death condition |
| `analytics/calibration.py` | Runs random-policy and stand-still calibration episodes; writes `configs/calibration.json` |
| `evolution/genetic_algorithm.py` | Top-level `run_generation` orchestration |
| `evolution/selection.py` | Tournament selection + elitism |
| `evolution/crossover.py` | `blend()` (default), `uniform()`, `none()` — chosen by `config.crossover_method` |
| `evolution/mutation.py` | Gaussian mutation + clipping |
| `configs/calibration.json` | Output of the calibration pass; consumed at runtime for fitness normalization |
| `tests/unit/test_energy_balance.py`, `test_fitness_normalization.py`, `test_crossover_all_methods.py` | Correctness of each new module |
| `tests/integration/test_one_generation.py` | Run one full generation end-to-end with a tiny population |

**Steps**

1. Implement the energy-balance equation exactly:
   `energy[t+1] = energy[t] - metabolic_base_cost - k_accel·accel² - k_steer·|steering| + eaten_energy_value·eat_event`, with death at `energy <= 0 or health <= 0`.
2. Implement the calibration pass: run a random policy and a stand-still policy for N episodes each, record each raw fitness component's mean value, write `configs/calibration.json`.
3. Implement fitness as normalized components (`raw / reference_scale`) combined with configurable weights — this replaces any hand-picked raw-count coefficients outright.
4. Implement tournament selection with elite carry-over.
5. Implement all three crossover methods behind one config switch; document the competing-conventions risk directly in the module docstring of `crossover.py`, not just in the design doc.
6. Implement Gaussian mutation with clipping.
7. Wire the full `run_generation` loop (observe/act loop from Phase 2, plus fitness accumulation, selection, crossover, mutation, replacement).
8. Write and pass all listed tests.
9. Run a multi-generation smoke test on a tiny population and confirm fitness trends upward before trusting the pipeline at full scale.

**Exit criterion:** fitness improves over controlled runs; crossover variants compared.

### Deliverables — Phase 3

- [x] Explicit, tested energy-balance model
- [x] A calibration pipeline producing `configs/calibration.json`, so fitness weights are never chosen blind
- [x] A working genetic algorithm with three interchangeable, tested crossover methods
- [x] Smoke-test evidence (a fitness-over-generations plot) that fitness improves

---

## Phase 4 — Analytics, Checkpoints, Determinism

**Goal:** experiments are logged, repeatable, and tagged by reproducibility tier.

**New files**

| Path | Purpose |
|---|---|
| `analytics/experiment_logger.py` | Writes the full `experiments/EXP-XXX/` folder structure |
| `simulation/determinism.py` | Seeds torch/numpy/random; enables deterministic algorithms; pins thread count; records `device` + `reproducibility_tier` |
| `neural/checkpoint.py` | Save/restore a genome + architecture pair without changing outputs |
| `tests/integration/test_seeded_rerun.py` | Same-seed CPU-deterministic run reproduces identical metrics across two runs |
| `tests/integration/test_checkpoint_restore.py` | Restored genome produces identical outputs pre/post checkpoint |

**Steps**

1. Implement the experiment-folder writer: `config.json`, `metadata.json`, `generation_metrics.csv`, `agent_metrics.csv`, `best_genome.json`, `checkpoints/`, and a `stats_summary.json` placeholder (populated in Phase 4.5).
2. Implement `metadata.json` with **separate** `torch`, `numpy`, `random` seed fields (never one combined "seed"), plus `reproducibility_tier` and `device`.
3. Implement the determinism checklist as real code: seed-setting utility, `torch.use_deterministic_algorithms(True)`, thread pinning via `torch.set_num_threads()`.
4. Implement genome checkpoint save/restore; verify identical outputs on identical inputs before and after a save/restore cycle.
5. Run the same seeded config twice on the CPU-deterministic tier and diff `generation_metrics.csv` byte-for-byte — this must pass before Phase 4.5 begins.
6. Document explicitly (in code comments and `metadata.json` itself) that GPU runs are a separate, non-comparable reproducibility tier.

**Exit criterion:** experiments are logged, repeatable, and tagged by reproducibility tier.

### Deliverables — Phase 4

- [x] Full experiment-folder logging pipeline
- [x] A determinism module enforced identically in GUI and headless modes
- [x] Verified bit-for-bit reproducibility on the CPU-deterministic tier
- [x] Working checkpoint save/restore

---

## Phase 4.5 — Related Work + Statistical Protocol Implementation

**Goal:** the condition-comparison pipeline is fully ready before any headline experiment is run.

**New files**

| Path | Purpose |
|---|---|
| `docs/related_work.md` | Positions the project against Sims (1994), Stanley & Miikkulainen NEAT (2002), and prior artificial-life platforms |
| `analytics/statistics.py` | `mann_whitney_u()`, `bootstrap_ci()`, `holm_bonferroni()`, `rank_biserial_correlation()` |
| `analytics/compare_conditions.py` | Orchestrates all pairwise comparisons for an experiment; writes `stats_summary.json` |
| `tests/unit/test_statistics.py` | Cross-checks statistical functions against `scipy`/`statsmodels` reference implementations |

**Steps**

1. Draft `docs/related_work.md` (300–500 words): what's precedented (tournament selection, speciation via compatibility distance) vs. this project's own experimental questions (fitness weighting, crossover-method comparison).
2. Implement `mann_whitney_u()`, `bootstrap_ci()` (resampling with replacement, configurable resample count and alpha), `holm_bonferroni()` (step-down correction across a family of p-values), and `rank_biserial_correlation()` (effect size).
3. Implement `compare_conditions.py`: given seed-level results per condition, run every pairwise comparison, correct p-values, and write the full record to `stats_summary.json`.
4. Cross-check every statistical function against a reference implementation (e.g., `scipy.stats.mannwhitneyu`) in unit tests — do not trust a from-scratch implementation of a significance test without this.
5. Dry-run the whole pipeline on Phase 3's smoke-test data to confirm the plumbing works end-to-end before it's trusted with real experimental conclusions.

**Exit criterion:** the comparison pipeline is ready before any headline experiment is run.

### Deliverables — Phase 4.5

- [x] `docs/related_work.md`
- [x] A tested statistics module (Mann-Whitney U, bootstrap CI, Holm-Bonferroni, effect size)
- [x] An end-to-end condition-comparison pipeline producing `stats_summary.json`
- [x] Confirmed statistical machinery correctness before real experiments depend on it

---

## Phase 5 — Generalization Experiments + Experiment D (Fitness Weighting)

**Goal:** unseen layouts and fitness-weighting sensitivity are both evaluated, with results run through the Phase 4.5 statistical pipeline.

**New files**

| Path | Purpose |
|---|---|
| `configs/generalization_train.json`, `configs/generalization_test.json` | Training layouts A1–A3 / held-out layouts B1–B3 |
| `configs/fitness_weighting_d1.json` … `d4.json` | The four Experiment D conditions |
| `experiments/run_generalization.py` | Train on A1–A3, freeze best controllers, evaluate on B1–B3 |
| `experiments/run_experiment_d.py` | Runs D1–D4, 10 seeds each, through `compare_conditions.py` |
| `docs/generalization_results.md`, `docs/experiment_d_results.md` | Write-ups including effect sizes, not just significance |

**Steps**

1. Build three training-layout configs (A1–A3) and three held-out layouts (B1–B3), varying food/obstacle placement seeds only.
2. Implement the freeze-and-evaluate pipeline: train to convergence on A1–A3, snapshot best genomes, evaluate unmodified on B1–B3, log the train-vs-unseen delta.
3. Build the four Experiment D configs (equal weights / resource-weighted / exploration-weighted / survival-weighted).
4. Run the exploratory pass first (3 seeds/condition) to sanity-check before committing to the full confirmatory run.
5. Run the confirmatory 10-seed pass for both the generalization test and Experiment D.
6. Feed both result sets through `compare_conditions.py`; write up findings including effect sizes.

**Exit criterion:** unseen layouts and fitness-weighting sensitivity are both evaluated.

### Deliverables — Phase 5

- [ ] Documented generalization-gap result (train vs. unseen performance)
- [ ] Documented, statistically-compared fitness-weighting result (Experiment D)
- [ ] Reusable generalization-evaluation and experiment-runner scripts for later phases

---

## Phase 6 — Predator/Prey, Reproduction, and Hall-of-Fame Evaluation

**Goal:** a stable multi-agent ecosystem, with co-evolution progress measured against frozen checkpoints rather than only same-generation relative fitness.

**New / modified files**

| Path | Purpose |
|---|---|
| `agents/organism.py` (modified) | Adds a `role` field (`prey` / `predator`) and predator capture action |
| `simulation/environment.py` (modified) | Predator/prey spawning, capture-collision resolution |
| `evolution/reproduction.py` | Explicit reproduction eligibility and offspring creation |
| `analytics/hall_of_fame.py` | Archives best-of-generation snapshots; evaluates current-best controllers against the frozen archive |
| `visualization/dashboard_advanced.py` (modified) | Wires the hall-of-fame delta panel to real data |
| `tests/integration/test_predator_prey_stability.py` | No extinction under default parameters over N generations |
| `docs/coevolution_notes.md` | Documents that within-generation fitness may reflect an arms race, not absolute improvement |

**Steps**

1. Extend agent state with a role field and predator-specific capture action/energy transfer.
2. Implement reproduction eligibility (`age >= MIN_AGE and energy >= MIN_ENERGY and fitness >= MIN_FITNESS`) and offspring creation, separate from generation-level replacement.
3. Implement the hall-of-fame archive: snapshot best genomes every N generations; build an evaluation harness pitting current-best controllers against archived opponents.
4. Wire the hall-of-fame delta into the Advanced dashboard as a live metric.
5. Run a long-horizon stability test; tune capture efficiency/energy transfer if either population collapses.
6. Write `docs/coevolution_notes.md` so anyone reading the generation-metrics CSVs later doesn't mistake an oscillating arms race for stalled evolution.

**Exit criterion:** the multi-agent ecosystem is stable; co-evolution progress is measured against frozen checkpoints.

### Deliverables — Phase 6

- [ ] Stable predator/prey ecosystem with explicit reproduction
- [ ] A working hall-of-fame evaluation harness and dashboard panel
- [ ] A documented co-evolution caveat protecting later readers from misreading the metrics

---

## Phase 7 — NEAT-Style Extension (Appendix C)

**Goal:** topology evolution works on an isolated benchmark, matching the Appendix C specification exactly — innovation tracking, compatibility-distance speciation, fitness sharing, and complexification.

**New files**

| Path | Purpose |
|---|---|
| `evolution/neat/genome.py` | Variable-topology genome: node genes + connection genes with innovation numbers |
| `evolution/neat/innovation.py` | Global innovation counter with per-generation reuse logic |
| `evolution/neat/mutation.py` | `add_connection()`, `add_node()` structural operators |
| `evolution/neat/crossover.py` | Innovation-number-aligned crossover |
| `evolution/neat/compatibility.py` | δ = c1·E/N + c2·D/N + c3·W̄ |
| `evolution/neat/speciation.py` | Species assignment, fitness sharing, stagnation removal |
| `neural/sparse_inference.py` | Resolves the batching-under-variable-topology question |
| `tests/unit/test_innovation_numbers.py`, `test_compatibility_distance.py`, `test_neat_crossover.py` | Correctness of each NEAT primitive |
| `tests/integration/test_speciation_stability.py` | Species count doesn't collapse to 1 or explode unboundedly |
| `docs/neat_extension_report.md` | Documents the chosen batching strategy and measured performance |

**Steps**

1. Implement the variable-topology genome representation (node genes; connection genes with weight/enabled/innovation fields).
2. Implement the global innovation-number counter with per-generation reuse: two independent, identical structural mutations occurring in the same generation must receive the same innovation number — write a unit test proving this directly.
3. Implement `add_connection` and `add_node`, including the new-node weight-1.0 initialization that minimizes immediate functional disruption.
4. Implement innovation-aligned crossover: matching genes inherited randomly from either parent, disjoint/excess genes from the fitter parent.
5. Implement the compatibility-distance formula exactly as specified, with configurable `c1`, `c2`, `c3`, and species threshold (baseline `c1=1.0, c2=1.0, c3=0.4`, threshold `3.0`).
6. Implement speciation assignment, fitness sharing (divide by species size), and stagnation-based species removal.
7. **Resolve the open batching question early**: prototype per-species batching first; if species are too small/uneven to batch efficiently, fall back to a sparse/masked-weight-matrix representation. Benchmark both against the Phase 1.5 fixed-topology baseline and record the decision.
8. Initialize Phase 7 populations from a minimal topology (direct input→output only, no hidden nodes).
9. Run the full NEAT loop on an isolated benchmark task (basic food-seeking, no predator/prey) and confirm topology visibly grows in complexity over generations.
10. Write `docs/neat_extension_report.md`.

**Exit criterion:** topology evolution works on an isolated benchmark, fully matching the Appendix C specification.

### Deliverables — Phase 7

- [ ] A complete, literature-accurate NEAT implementation (innovation tracking, speciation, fitness sharing, complexification)
- [ ] A resolved and documented answer to the batched-inference-under-variable-topology question
- [ ] A complexity-over-generations plot showing structure is actually evolving, not just weights

---

## Phase 8 — Packaging, Final Report, Dashboard Polish, Presentation

**Goal:** demo, documentation, tests, and final experiment results complete; dashboards reconciled with the actually-delivered scope.

**New / modified files**

| Path | Purpose |
|---|---|
| `webdash/backend/`, `webdash/frontend/` | Optional read-only web dashboard (FastAPI + React) |
| `docs/final_report.md` | Full technical report |
| `docs/demo_script.md` | Live-demo walkthrough (see Section 13 below) |
| `pyproject.toml` (finalized) | Pinned, release-ready dependency versions |

**Steps**

1. Freeze all experiment configs and results; regenerate every plot and statistics table from the final, tagged data — no hand-edited numbers.
2. Write `docs/final_report.md`, pulling directly from `docs/related_work.md`, `docs/*_results.md`, `docs/neat_extension_report.md`, and `docs/coevolution_notes.md` rather than re-deriving anything.
3. Reconcile the dashboards: confirm the MVP dashboard never shows a Phase-7-only metric, and the Advanced dashboard correctly reflects the final predator/prey + NEAT state (see Section 12 below for the full spec).
4. (Optional) build the FastAPI + React web dashboard as a read-only viewer over `experiments_data/`, explicitly decoupled from the simulation core.
5. Run the full test suite one final time on a clean checkout; fix any flaky or skipped tests.
6. Package the desktop application; verify both `--mode gui` and `--mode headless` entry points work from a fresh install.
7. Prepare the final presentation materials (Section 13 below).

**Exit criterion:** demo, documentation, tests, and final experiment results complete.

### Deliverables — Phase 8

- [ ] Final technical report (`docs/final_report.md`)
- [ ] Fully reconciled MVP + Advanced dashboards
- [ ] Optional web dashboard
- [ ] Packaged, tested, installable application
- [ ] Complete presentation package (Section 13)

---

## 12. Dashboard — Full Specification

The dashboard exists in two forms that share one data source (`experiments_data/EXP-XXX/`) and one rendering technology (PyGame, in-process, real-time) plus one optional secondary technology (FastAPI + React, out-of-process, for after-the-fact browsing). This section is the complete spec for both.

### 12.1  In-Simulation Dashboard (PyGame) — Primary

**Data flow.** The dashboard never writes to simulation state and never blocks the simulation loop. Each render tick, it reads a read-only snapshot object (`DashboardSnapshot`) that `simulation/engine.py` populates once per frame from the live `World`/population state — the dashboard itself has no simulation logic in it.

```
simulation.engine  --(per frame)-->  DashboardSnapshot  --(read-only)-->  dashboard_mvp / dashboard_advanced
```

**Layout — MVP variant (Phases 1–5).** Fixed two-column layout: a live world view (left, ~65% width) and a statistics column (right, ~35% width), with a fitness-history strip and control bar along the bottom.

| Region | Contents | Data source |
|---|---|---|
| Header bar | `GENERATION <n>`, `POPULATION <n>` | `World.generation`, `len(World.agents)` |
| Live world panel | Rendered food (green), obstacles (gray), agents (colored by role, oriented by heading) | `visualization/renderer.py` |
| Statistics panel | Best / mean / median fitness, alive count (bar), food consumed, average lifespan | `analytics/experiment_logger.py` running totals |
| Fitness history strip | Line chart, one line per generation-level statistic (best/mean/median), scrollable/zoomable | `generation_metrics.csv`, tailed live |
| Control bar | `[Pause] [x1] [x10] [Next Gen] [Save] [Reset]` | see interaction table below |

**Control bar behavior**

| Control | Behavior |
|---|---|
| Pause | Freezes the physics/inference loop; rendering continues so the frozen frame stays visible |
| x1 / x10 | Sets the fast-forward multiplier for the *simulation* update rate only (render FPS is unaffected — this is the whole point of decoupling them in Phase 1) |
| Next Gen | Skips to the end of the current generation immediately (used for demoing many generations quickly without full fast-forward) |
| Save | Triggers `neural/checkpoint.py` to snapshot the current best genome and `analytics/experiment_logger.py` to flush current metrics to disk |
| Reset | Reinitializes the `World` from the active config, generation counter back to zero |

**Layout — Advanced variant (Phase 6+).** Everything in the MVP variant, plus:

| Region | Contents | Data source |
|---|---|---|
| Header bar (extended) | Adds `SPECIES <n>` | `evolution/neat/speciation.py` species count (Phase 7) or predator/prey population split (Phase 6) |
| Live world panel (extended) | Adds predator/prey rendering (distinct color/shape per role) | `agents/organism.py` role field |
| Statistics panel (extended) | Adds **hall-of-fame delta** (current-best vs. frozen-archive performance) and per-species fitness breakdown | `analytics/hall_of_fame.py`, `evolution/neat/speciation.py` |
| Fitness history (extended) | One line per species instead of one line for the whole population | `generation_metrics.csv` grouped by `species_id` |

**Why two variants, not one configurable one:** this directly resolves fix C6 from the design review — the MVP build must never be able to render a species count that doesn't exist yet. The two dashboard modules are selected by `config.phase_tier` (`"mvp"` or `"advanced"`), not by a single dashboard trying to conditionally hide fields.

### 12.2  Best-Agent Inspector (both variants)

A modal/side panel opened by clicking any agent in the live world view.

| Element | Detail |
|---|---|
| Vitals | Current fitness, age, energy, health, movement speed — numeric readout, updated live |
| Sensor overlay | The 7 sensor rays drawn directly on top of the live world view, colored by what each ray currently detects |
| Neural network graph | Node-and-edge diagram of the controller; active output nodes (steering/accel/eat) highlighted by current activation magnitude. In Phase 7, this is the actual variable topology, not a fixed 12-32-16-3 diagram |
| Genome identifier + lineage | The genome's unique ID and a small parent-lineage tree (up to 3 generations back), pulled from `best_genome.json` history |
| Behavior timeline | A scrubbable strip showing this agent's fitness-component contributions (survival/food/exploration/collisions) accumulating over the current generation |

### 12.3  Optional Web Dashboard (FastAPI + React) — Secondary, Phase 8

Explicitly decoupled from the simulation core: a **read-only** viewer over `experiments_data/`, used for after-the-fact browsing and cross-experiment comparison rather than live demonstration.

**Backend (`webdash/backend/`) — REST endpoints**

| Endpoint | Returns |
|---|---|
| `GET /experiments` | List of all `experiments_data/EXP-XXX/` runs with summary metadata |
| `GET /experiments/{id}/metrics` | Full `generation_metrics.csv`, as JSON |
| `GET /experiments/{id}/stats_summary` | The Phase 4.5 `stats_summary.json` — test statistics, CIs, corrected p-values |
| `GET /experiments/{id}/best_genome` | The best genome and its lineage |

**Frontend (`webdash/frontend/`) — pages**

| Page | Purpose |
|---|---|
| Experiment List | Browse all runs, filter by experiment family (A/B/NC/D/E) |
| Experiment Detail | Fitness-over-generations charts (recharts/d3), generation metrics table |
| Compare Conditions | Reads `stats_summary.json`; shows each pairwise comparison with its corrected p-value, confidence interval, and effect size — **never significance alone** |
| Best-Agent Viewer | Static replay of a saved best-genome's behavior timeline and lineage |

**Design constraints:** color-blind-safe palette across all charts; every fitness chart offers a log-scale toggle (generation-scale fitness growth is frequently log-shaped); every "condition A beat condition B" claim on the Compare Conditions page must display its confidence interval and effect size next to the p-value, never the p-value alone.

---

## 13. Final Presentation of All the Work

This section is the Phase 8 deliverable in full: what to prepare, in what order, and how to run the actual presentation/demo session.

### 13.1  Presentation package checklist

- [ ] Slide deck (outline in 13.2)
- [ ] Live demo, rehearsed against `docs/demo_script.md` (13.3)
- [ ] Pre-recorded demo video as a fallback, in case the live demo fails during the actual session
- [ ] `docs/final_report.md`, exported to PDF for anyone who wants the full written detail
- [ ] Repository link with an up-to-date `README.md` quick-start
- [ ] All raw experiment data (`experiments_data/`) archived and linked, not just the summarized plots

### 13.2  Suggested slide-deck outline (~12–14 slides)

1. **Title** — project name, one-line framing (fixed-topology neuroevolution + a fully specified NEAT extension)
2. **Motivation / Problem Definition** — rule-based vs. evolutionary control (the Section 3 comparison)
3. **Core Research Question** — plus the secondary question this revision added (does behavior depend on fitness weighting; does it survive topology evolution)
4. **Related Work** — Sims, NEAT, prior artificial-life platforms; what's precedented vs. novel here
5. **System Architecture** — the layered diagram (Section 9), one sentence per layer
6. **Neuroevolution Design** — fixed topology + batched inference, with the throughput numbers from the Phase 1.5 benchmark report as a concrete, credible number
7. **Fitness Design** — calibration + normalization, and why that mattered (the scale-domination risk it fixed)
8. **Live Demo** — see 13.3
9. **Headline Results: Mutation Rate / Population / Network Capacity** — with statistical comparisons (CI + effect size), not bare means
10. **Headline Results: Fitness Weighting & Crossover Method (Experiments D, E)** — the two new experiments this revision added
11. **Generalization Results** — train vs. unseen-layout performance
12. **Predator/Prey & Hall-of-Fame Results** — the co-evolution caveat, shown as an oscillating chart *explained*, not hidden
13. **NEAT Extension (Phase 7)** — the complexity-over-generations plot; the batching-under-variable-topology decision and why
14. **Limitations & Future Work** — pulled directly from the design document's Section 26

### 13.3  Live demo script (mirrors `docs/demo_script.md`)

1. Launch `app/run.py --mode gui` with `configs/baseline.json` at generation 0 — show visibly random behavior.
2. Pause; load a Phase 3 checkpoint from a late generation via **Save/Load**; resume — show visibly food-seeking, obstacle-avoiding behavior for contrast.
3. Click an agent to open the **Best-Agent Inspector** (Section 12.2); walk through the sensor overlay and the live neural-network activation graph.
4. Switch to a terminal and kick off a short **headless** run (`experiments/run.py --mode headless --seeds 3`) to show the GUI/headless split from Section 16.1 concretely, not just described.
5. Open the **web dashboard**'s Compare Conditions page (Section 12.3) on a completed experiment; walk through one pairwise comparison, pointing explicitly at the confidence interval and effect size, not just the p-value.
6. If Phase 7 is complete: switch to the isolated NEAT benchmark and show the complexity-over-generations plot alongside a rendered snapshot of an evolved (non-fixed) network topology.
7. Return to the slide deck for the results and limitations sections.

### 13.4  Presentation risk management

- Rehearse the live demo at least twice on the exact hardware that will be used for the session — Phase 1.5's benchmark numbers are the fallback talking point if live performance looks different from what was measured.
- Keep the pre-recorded video cued to the same six demo steps above, so a fallback doesn't require re-narrating a different sequence.
- Have `docs/final_report.md` and `stats_summary.json` for every headline experiment open and ready in tabs, in case a question asks for a number not on a slide.

---

## 14. Master File Checklist

Every file this plan creates, in the order it first appears, for a final pre-submission pass:

**Config & setup:** `pyproject.toml`, `.gitignore`, `pytest.ini`, `README.md`, `configs/baseline.json`, `configs/calibration.json`, all `configs/*_a*.json` / `*_b*.json` / `*_nc*.json` / `*_d*.json` / `*_e*.json`, `configs/generalization_train.json`, `configs/generalization_test.json`

**Simulation core:** `simulation/environment.py`, `simulation/physics.py`, `simulation/world_config.py`, `simulation/engine.py`, `simulation/determinism.py`

**Agents:** `agents/organism.py`, `agents/sensors.py`, `agents/energy.py`

**Neural:** `neural/network.py`, `neural/batched_inference.py`, `neural/genome.py`, `neural/checkpoint.py`, `neural/sparse_inference.py`

**Evolution:** `evolution/genetic_algorithm.py`, `evolution/selection.py`, `evolution/crossover.py`, `evolution/mutation.py`, `evolution/reproduction.py`, `evolution/neat/*.py` (six files)

**Analytics:** `analytics/calibration.py`, `analytics/experiment_logger.py`, `analytics/statistics.py`, `analytics/compare_conditions.py`, `analytics/hall_of_fame.py`, `analytics/reports.py`

**Visualization:** `visualization/renderer.py`, `visualization/dashboard_mvp.py`, `visualization/dashboard_advanced.py`, `visualization/best_agent_inspector.py`

**Entry points:** `app/run.py`, `experiments/run.py`, `experiments/run_generalization.py`, `experiments/run_experiment_d.py`, `experiments/run_experiment_e.py`

**Web dashboard (optional):** `webdash/backend/main.py`, `webdash/backend/routes.py`, `webdash/frontend/src/pages/*.jsx` (four files)

**Tests:** all files under `tests/unit/`, `tests/integration/`, `tests/performance/` listed in Section 0

**Docs:** `docs/related_work.md`, `docs/phase1_5_benchmark_report.md`, `docs/generalization_results.md`, `docs/experiment_d_results.md`, `docs/experiment_e_results.md`, `docs/coevolution_notes.md`, `docs/neat_extension_report.md`, `docs/final_report.md`, `docs/demo_script.md`
