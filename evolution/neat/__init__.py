"""
NEAT extension package (Phase 7): variable-topology neuroevolution.

Implements Stanley & Miikkulainen (2002) per PLAN.md Phase 7:
innovation tracking, innovation-aligned crossover, compatibility-distance
speciation with fitness sharing and stagnation removal, and structural
mutation (complexification).
"""

from evolution.neat.genome import (ConnectionGene, NeatGenome, NodeGene,
                                   minimal_genome)
from evolution.neat.innovation import InnovationTracker
from evolution.neat.compatibility import compatibility_distance
from evolution.neat.mutation import (add_connection, add_node,
                                     mutate_structure, mutate_weights)
from evolution.neat.crossover import crossover
from evolution.neat.speciation import Species, SpeciationResult, assign_species
from evolution.neat.algorithm import (NeatGenerationMetrics,
                                      initialize_population, run_generation)

__all__ = [
    "ConnectionGene",
    "NeatGenome",
    "NodeGene",
    "minimal_genome",
    "InnovationTracker",
    "compatibility_distance",
    "add_connection",
    "add_node",
    "mutate_structure",
    "mutate_weights",
    "crossover",
    "Species",
    "SpeciationResult",
    "assign_species",
    "NeatGenerationMetrics",
    "initialize_population",
    "run_generation",
]
