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
| Population-wide depth-layered padded/masked batch | **249** | **2.94x** |
| Per-species batches (5 groups) | 221 | 2.61x |
| Per-agent topological loop | 85 | 1.0x |

For reference, the fixed-topology Phase 1.5 baseline ran 10,762
steps/s at 250 agents; variable-topology batching is ~40x slower than
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
(`compile_population`, ~30 ms) and executed once per step
(`run_compiled`) with row indexing for the shrinking live set.
Benchmarking the convenience wrapper (compile + run per step) instead
of the compiled path made batching look 2.5x slower than the per-agent
loop; the compiled path is the algorithm's actual usage and is 2.9x
faster. Both batched strategies are verified against the per-agent
reference in `tests/unit/test_sparse_inference.py`.

## Benchmark run (plan step 9)

`configs/neat_food_seeking.json`, seed 1, 30 generations, 100
genomes, 150-step shared-world episodes (`experiments/run_neat.py`):

| Quantity | Gen 1 | Gen 30 |
|---|---|---|
| Mean complexity (nodes + enabled genes) | 51.0 | 102.0 |
| Max complexity | 51 | 117 |
| Mean fitness | 0.481 | 0.429 |
| Max fitness | 2.21 | 1.59 |
| Live agents at episode end | 17 | 5 |
| Species count | 1 | 1 |
| Distinct innovations created | - | 571 |
| Final champion complexity | - | 58 |

**Topology grows**: mean complexity more than doubles, and the
complexity-over-generations plot
(`experiments/EXP-NEAT/complexity_over_generations.svg`) shows the
growth directly. This is the plan's exit criterion, met at the
population level.

**Species count stays 1** - the honest, measured outcome, and the one
number that does not match the plan's expectation. The mechanism is
correct - `tests/unit/test_neat_primitives.py` proves that structurally
and weight-wise divergent genomes split into separate species under
the Appendix C formula - but this run never generates such divergence.
Under the energy-limited dynamics the landscape is sharply *bimodal*:
a few well-fed foragers reach fitness near 2.0 while most of the
population starves near 0.3 (only 8-12 of 100 agents are alive at
the end of an episode). A pool split between starving and surviving
still yields a single compatibility cluster within 30 generations -
the Appendix C threshold (3.0) needs weight divergence the bimodality
does not produce. Extending to more generations, or a population
whose fitness is graded rather than bimodal (intermediate food
availability), is the route to observable speciation.

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
complexity-plot deliverable. Growth is driven by near-neutral drift:
structural mutations that do not break the bimodal fitness
landscape survive, so complexity accumulates. Selection's contribution
to topology is not yet observable - the stronger landscape the report
recommends is what would make complex structure pay.

## Fitness signal

Mean fitness drifts down (0.481 -> 0.429) while max fitness sits
near 1.6-2.2 and the live-agent count falls from 17 to 5: under the
energy-limited dynamics most agents starve inside the episode and the
population carries a few well-fed foragers. The NEAT machinery is
correct, and the landscape now has genuine selective signal (the
sibling GA improves by +0.28 to +0.50 over 12 generations on the same
world; see `docs/crossover_results.md`) - what the signal selects
*for* is survival of the already-fortunate, and that is not yet rich
enough behavior to pin speciation apart.
