# Related Work

This project builds on three foundational lines of artificial life and neuroevolution research.

## 1. Artificial Life Foundations

**Sims (1994)** — *Evolving Virtual Creatures* established the paradigm of evolving both morphology and control in a physics simulator. Our work inherits the core insight that embodied agents in a physics environment produce qualitatively different behaviors than disembodied function optimization. Unlike Sims, we fix morphology (all agents are identical triangles) and evolve only neural controllers, isolating the controller-evolution question.

**Ray (1991)** — *Tierra* introduced digital organisms competing for CPU cycles and memory. Our energy model (metabolism + action costs + food reward) echoes Tierra's resource competition, but in a spatial 2D world rather than a linear memory array.

## 2. Neuroevolution and NEAT

**Stanley & Miikkulainen (2002)** — *NEAT* introduced speciation via compatibility distance, historical markings for crossover, and incremental complexification from minimal topologies. Our Phase 7 implements NEAT per Appendix C of the design document. Key differences:

- **Fixed topology (Phases 1–6):** We deliberately constrain to fixed 12-32-16-3 MLPs for Phases 1–6 to isolate weight evolution before topology evolution. NEAT's historical markings and speciation add complexity we defer.
- **Crossover comparison (Phase 3, Experiment E):** NEAT uses historical markings to align genomes for crossover. Our fixed-topology GA tests three crossover methods (blend, uniform, mutation-only) without alignment, testing whether the "competing conventions" problem manifests even without structural alignment issues.
- **Speciation vs. tournament selection:** NEAT protects innovations via speciation. We use tournament selection + elitism (Phase 3), which is simpler but risks premature convergence. The comparison is an open experimental question (Experiment E).

## 3. Artificial Life Platforms

**AVIDA (Ofria & Wilke, 2004)** — Digital evolution platform with configurable instruction sets and population structures. Our tournament selection + energy model is conceptually similar to AVIDA's merit-based reproduction, but in continuous space with neural controllers rather than discrete instruction sequences.

**GReaNs (Garcia et al., 2009)** — Grammar-based neural network evolution. Our fixed-topology approach is simpler but less expressive; the NEAT extension (Phase 7) recovers some expressiveness.

## What This Project Adds

| Precedented | Novel in This Project |
|-------------|----------------------|
| Tournament selection + elitism | Fixed-topology MLP with batched population inference (1 matmul/layer vs N per-agent calls) |
| Energy budget (metabolism + action costs) | Fitness calibration pipeline: random/stand-still probes → reference scales → normalized weighted sum |
| Tournament selection | Crossover-method comparison (blend vs. uniform vs. mutation-only) as a first-class experimental arm |
| NEAT speciation | Fixed-topology GA + Phase 7 NEAT as *optional extension*, not default |
| AVIDA energy model | Energy = survival + food + exploration − collision, calibrated before weighting |

The central methodological contribution is **fitness calibration before weighting**: rather than hand-tuning coefficients, we measure each component's natural scale via random-policy and stand-still probes, then combine normalized components. This prevents arbitrary coefficient choices from driving results.