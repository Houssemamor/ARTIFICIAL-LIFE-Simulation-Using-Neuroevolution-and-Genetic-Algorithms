# NEAT Extension Report (Phase 7)

Implements Stanley & Miikkulainen (2002), "Evolving Topologies through
Complexification", per PLAN.md Phase 7: innovation tracking,
innovation-aligned crossover, compatibility-distance speciation with
fitness sharing and stagnation removal, and structural mutation.

One caveat stated up front: Appendix C of the v2.0 design document is
not part of this repository, so the implementation follows the
specification embedded in PLAN.md (exact compatibility formula, the
c1=1.0 / c2=1.0 / c3=0.4 / threshold=3.0 baseline constants, the
operators, and the weight-1.0 node-split initialization) and the
literature where PLAN.md is silent. If the design document deviates
from either, that deviation is not reflected here.

## The batching question, resolved (plan step 7)

The open question was how to forward a population of variable-topology
genomes, given that the Phase 1.5 stack-(N, in, out) trick assumes one
shared architecture. Two strategies were implemented and benchmarked
against the unbatched per-agent reference, on a 100-genome
population with realistic structural variety (16.5 nodes per genome
on average, 4 depth layers):

| Strategy | steps/s | vs per-agent |
|---|---|---|
| Population-wide depth-layered padded/masked batch | **320** | **3.1x** |
| Per-species batches (5 groups) | 271 | 2.6x |
| Per-agent topological loop | 104 | 1.0x |

For reference, the fixed-topology Phase 1.5 baseline ran 10,762
steps/s at 250 agents; variable-topology batching is ~30x slower than
that, which is the honest price of topology evolution: the padded
blocks are rebuilt in Python whenever the population's topologies
change, and every depth layer pays a matmul even when a genome has no
node at that depth.

**Decision: population-wide depth-layered padded batching.** It
outperforms the per-species prototype (the plan's suggested first
try), because species groups here are small and uneven - exactly the
condition under which per-species batching loses. Padding waste grows
with topology variance, not group count, so one batch over the whole
population amortizes the tensor work best.

Implementation note: a population's topologies are fixed for a whole
evaluation episode, so the padded blocks are compiled once
(`compile_population`, ~20 ms) and executed once per step
(`run_compiled`, ~3 ms) with row indexing for the shrinking live set.
Benchmarking the convenience wrapper (compile + run per step) instead
of the compiled path made batching look 2.5x slower than the per-agent
loop; the compiled path is the algorithm's actual usage and is 3.1x
faster. Both batched strategies are verified against the per-agent
reference in `tests/unit/test_sparse_inference.py`.

## Benchmark run (plan step 9)

`configs/neat_food_seeking.json`, seed 1, 30 generations, 100
genomes, 150-step shared-world episodes (`experiments/run_neat.py`):

| Quantity | Gen 1 | Gen 30 |
|---|---|---|
| Mean complexity (nodes + enabled genes) | 51.0 | 108.6 |
| Max complexity | 51 | 124 |
| Mean fitness | 0.407 | 0.386 |
| Max fitness | 0.709 | 0.977 |
| Species count | 1 | 1 |
| Distinct innovations created | - | 581 |

**Topology grows**: mean complexity more than doubles, and the
complexity-over-generations plot
(`experiments/EXP-NEAT/complexity_over_generations.svg`) shows the
growth directly. This is the plan's exit criterion, met at the
population level.

**Species count stays 1** - the honest, measured outcome, and the one
number that does not match the plan's expectation. The mechanism is
correct - `tests/unit/test_neat_primitives.py` proves that structurally
and weight-wise divergent genomes split into separate species under
the Appendix C formula - but this run never generates such divergence:
the fitness landscape is degenerate (see below), so there is no
selection pressure to drive lineages apart, and the compatibility
distance never reaches the 3.0 threshold. The same root cause runs
through every experiment from Phase 3 on: under the current dynamics,
fitness barely responds to controller behavior, so selection cannot
shape anything - weights, behavior, or lineage identity.

The plan's speciation-stability expectation ("species count doesn't
collapse to 1 or explode unboundedly") is therefore recorded as
measured-but-unmet for the evolutionary run, with the machinery
independently verified. Speciation pressure requires the same
live-fitness-landscape fix the Phase 5/6 documents recommend
(`docs/crossover_results.md`, `docs/coevolution_notes.md`): raise
`metabolic_base_cost` so food actually matters, then rerun.

## Complexity over generations

`experiments/EXP-NEAT/complexity_over_generations.svg` - mean
(solid) and max (dashed) complexity per generation, the Phase 7
complexity-plot deliverable. Growth is driven by neutral drift:
structural mutations that do not break the (flat) fitness landscape
survive, so complexity accumulates. Selection's contribution to
topology is not yet observable - which is the honest reading until
the fitness landscape is live.

## Fitness signal

Max fitness rises 0.709 -> 0.977 while mean drifts down
(0.407 -> 0.386): a few lucky agents eat more per episode, and the
champion elitism carries them. Mean food intake stays near 0.1-0.15
items per episode, the same degenerate food-seeking dynamic documented
throughout Phases 3-6. The NEAT machinery is correct; the benchmark's
difficulty is the outstanding limit.
