# Live Demo Script (Phase 8)

A rehearsed walkthrough of the finished system. Every step below maps
to something that actually runs in this repository; steps the plan's
Section 13.3 imagined but the build does not implement are listed at
the end with what to say instead.

Rehearse twice on the presentation hardware. The Phase 1.5 benchmark
numbers (10,762 steps/s at 250 agents) are the fallback talking point
if live performance differs from the measured values.

## 0. Pre-flight (before the session starts)

- Terminal tabs open at the repo root: one for the GUI, one for
  headless runs, one for the report.
- `docs/final_report.md`, `docs/neat_extension_report.md`, and
  `experiments/EXP-D/stats_summary.json` open and ready (Section 13.4).
- Slide deck at slide 8 (the demo-divider slide).

## 1. The fixed-topology simulation (MVP tier, ~90 s)

```bash
.venv\Scripts\python.exe app\run.py --config configs\baseline.json --agent-mode neural
```

- Agents move under a 12-32-16-3 neural controller at ~10,000 steps/s
  (Phase 1.5 report).
- Press SPACE to pause: rendering continues, physics freezes.
- Press F9: the MVP-tier panel appears showing only MVP metrics
  (alive count). Say: "the panel is tiered by config; the advanced
  tier, shown next, adds predator/prey and hall-of-fame metrics that
  do not exist in this run - the MVP build cannot render a metric
  that isn't there."

## 2. The ecosystem (advanced tier, ~90 s)

```bash
.venv\Scripts\python.exe app\run.py --config configs\predator_prey.json --agent-mode neural
```

- Red triangles are predators, green are prey; captures transfer
  energy and birth offspring (fitness refresh every evaluation window).
- Press F9: the advanced panel shows live prey/predator counts,
  captures, births, and the hall-of-fame win rates from the last
  co-evolution run (`experiments/EXP-COEV/coevolution_summary.json`).
- Click any agent: its seven sensor rays appear over the world, and
  the inspector opens on the right (vitals, genome fingerprint,
  controller graph with live output activations).
- Control bar (bottom): Pause / x1 / x10 / Save (snapshots the best
  controller to `experiments/EXP-GUI/checkpoint.npz`) / Reset.

## 3. Headless determinism (terminal, ~60 s)

```bash
.venv\Scripts\python.exe experiments\run.py --config configs\baseline.json --output Z:\Temp\demo_headless.json
```

- Point at the printed seed and note: same seed, byte-identical run
  (the Phase 4 seeded-rerun test asserts it in CI).

## 4. The experiments behind the claims (web viewer, ~2 min)

```bash
.venv\Scripts\python.exe -m uvicorn webdash.backend.main:app
```

Then open http://127.0.0.1:8000/ : browse the run list, open EXP-D's
**Compare Conditions** tab, and walk through one pairwise comparison,
pointing explicitly at the effect size and interval next to the
p-value - never the p-value alone (the Phase 4.5 design constraint).

## 5. Topology evolution (Phase 7, ~2 min)

Open `experiments/EXP-NEAT/complexity_over_generations.svg`.

- Mean complexity 51 -> 102 over 30 generations: structure is
  evolving, not just weights.
- Say honestly: the run keeps a single species because the landscape
  is bimodal rather than graded (a few well-fed foragers near fitness
  2.0, most starving near 0.3), so selection has too weak a pull
  toward divergence to cross the compatibility threshold; the
  speciation machinery is unit-proven on constructed diverse
  populations (`docs/neat_extension_report.md`).
- Optional: `experiments/run_neat.py --config configs/neat_food_seeking.json --benchmark-only` reprints the
  batching decision numbers (249 vs 85 steps/s, 2.9x).

## 6. Back to the deck

Return to the results and limitations slides (9-14).

## Steps the plan imagined that this build does not implement

Say what is real instead of pretending:

| Plan's 13.3 step | Reality | What to say |
|---|---|---|
| Load a Phase 3 checkpoint into the GUI via Save/Load | The GUI *saves* checkpoints (control bar -> Save, best controller to `experiments/EXP-GUI/checkpoint.npz`); loading into a live GUI is not wired | "Checkpointing is exercised through the headless runner and the GUI save path; loading a checkpoint into a live GUI is future work." |
| Best-Agent Inspector (sensor overlay, NN graph, lineage) | Built in Phase 9: click an agent for vitals, the seven-ray overlay, and the controller graph; lineage is a genome fingerprint (fixed topology has no breeding history) | "This is the plan's inspector with a lineage stand-in; NEAT genomes would carry real lineage." |
| x1/x10/Next Gen/Save/Reset control bar | Built in Phase 9 except "Next Gen" (the GUI is a sandbox, not a generation loop - the headless runners advance generations) | "Physics and rendering are decoupled - the control bar's x10 is that separation made visible." |
| Web dashboard (FastAPI + React) | Built in Phase 9 as FastAPI + Bootstrap/vanilla JS: `uvicorn webdash.backend.main:app` | "It's a read-only viewer over the committed artifacts - it never touches the simulation core." |
| Control bar fast-forward | x1/x10 implemented | The headless runner in step 3 is the many-generations story. |

## Recording the fallback video (author-side, ~4 minutes)

I cannot record a live session from the build environment - the
fallback video is an author task. Record with the same six beats as
above, in this order, at the same pace as the live demo:

1. GUI MVP tier: launch, pause, F9 panel (~40 s)
2. GUI advanced tier: ecosystem with captures and births (~40 s)
3. Click an agent: sensor overlay + inspector; Save via the control bar (~40 s)
4. Terminal: headless run with a fixed seed (~40 s)
5. Web viewer: experiments list, EXP-D compare conditions - point at
   the interval and effect size next to the p-value (~40 s)
6. NEAT: open `experiments/EXP-NEAT/complexity_over_generations.svg`
   beside a GUI snapshot of the evolved populations (~40 s)

Keep the fixed-seed run's printed seed on screen; it is the
reproducibility proof.
